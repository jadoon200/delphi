"""Azure LLM inference trace parsing and the GPU serving model."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from delphi.control.serving import (
    ServingProfile,
    gpu_seconds_demand,
    phase_shares,
    proxy_demand,
    rescale_to,
    signal_tracking_error,
    tokens_per_request,
)
from delphi.data.llm_inference import (
    BinnedInference,
    _parse_timestamp,
    bin_trace,
    load_requests,
    stream_rows,
    to_demand_series,
)

HEADER = "TIMESTAMP,ContextTokens,GeneratedTokens\n"


def _write(path: Path, rows: list[tuple[str, int, int]]) -> Path:
    path.write_text(HEADER + "".join(f"{t},{c},{g}\n" for t, c, g in rows))
    return path


def test_fast_timestamp_parser_matches_fromisoformat() -> None:
    """The hand-rolled parser exists for speed; it must agree with the correct one."""
    for raw in (
        "2024-05-10 00:00:00.009930+00:00",
        "2024-05-12 23:59:59.999999+00:00",
        "2024-05-11 12:34:56.000001+00:00",
    ):
        assert _parse_timestamp(raw) == datetime.fromisoformat(raw)


def test_stream_rows_rejects_an_unexpected_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("time,tokens\n1,2\n")
    with pytest.raises(ValueError, match="unexpected LLM trace header"):
        list(stream_rows(path))


def test_binning_aggregates_all_three_signals(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "t.csv",
        [
            ("2024-05-10 00:00:01.000000+00:00", 100, 10),
            ("2024-05-10 00:00:30.000000+00:00", 200, 20),
            ("2024-05-10 00:01:05.000000+00:00", 400, 40),
        ],
    )
    binned = bin_trace(path, bin_seconds=60)
    assert binned.requests.tolist() == [2, 1]
    assert binned.prefill_tokens.tolist() == [300, 40 * 10]
    assert binned.decode_tokens.tolist() == [30, 40]
    assert binned.start == datetime(2024, 5, 10, 0, 0, tzinfo=UTC)


def test_binning_leaves_empty_bins_as_zero_not_missing(tmp_path: Path) -> None:
    """A quiet minute is real demand information, not a gap to be skipped."""
    path = _write(
        tmp_path / "t.csv",
        [
            ("2024-05-10 00:00:01.000000+00:00", 10, 1),
            ("2024-05-10 00:05:01.000000+00:00", 10, 1),
        ],
    )
    binned = bin_trace(path, bin_seconds=60)
    assert binned.requests.tolist() == [1, 0, 0, 0, 0, 1]


def test_binning_rejects_an_unsorted_trace(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "t.csv",
        [
            ("2024-05-10 00:05:00.000000+00:00", 10, 1),
            ("2024-05-10 00:00:00.000000+00:00", 10, 1),
        ],
    )
    with pytest.raises(ValueError, match="not sorted"):
        bin_trace(path, bin_seconds=60)


def test_load_requests_returns_only_the_asked_window(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "t.csv",
        [
            ("2024-05-10 00:00:10.000000+00:00", 10, 1),
            ("2024-05-10 00:01:10.000000+00:00", 20, 2),
            ("2024-05-10 00:02:10.000000+00:00", 30, 3),
        ],
    )
    import delphi.data.llm_inference as module

    original = module.TRACES.copy()
    module.TRACES["unit"] = path.name
    try:
        requests = load_requests(
            "unit",
            window_start=datetime(2024, 5, 10, 0, 1, tzinfo=UTC),
            window_seconds=60,
            raw_dir=tmp_path,
        )
    finally:
        module.TRACES.clear()
        module.TRACES.update(original)
    assert len(requests) == 1
    assert requests.context_tokens.tolist() == [20]
    assert requests.arrival_seconds.tolist() == [10.0]


def _binned(prefill: list[int], decode: list[int], requests: list[int]) -> BinnedInference:
    return BinnedInference(
        start=datetime(2024, 5, 10, tzinfo=UTC),
        bin_seconds=60,
        requests=np.asarray(requests, dtype=np.int64),
        prefill_tokens=np.asarray(prefill, dtype=np.int64),
        decode_tokens=np.asarray(decode, dtype=np.int64),
    )


def test_demand_series_projection_carries_the_right_units() -> None:
    binned = _binned([1000, 2000], [10, 20], [5, 6])
    series = to_demand_series(binned, trace="code", kind="prefill_tokens")
    assert series.values.tolist() == [1000.0, 2000.0]
    assert series.step_seconds == 60
    assert series.resource_kind == "tokens"
    assert series.timestamps[1] - series.timestamps[0] == timedelta(seconds=60)


# --- serving model -----------------------------------------------------------------


def test_gpu_seconds_sums_both_contending_phases() -> None:
    profile = ServingProfile(prefill_tokens_per_second=1000.0, decode_tokens_per_second=100.0)
    binned = _binned(prefill=[2000], decode=[300], requests=[10])
    # 2000/1000 + 300/100 = 2 + 3
    assert gpu_seconds_demand(binned, profile).tolist() == [5.0]


def test_phase_shares_sum_to_one_and_reflect_the_cost_ratio() -> None:
    profile = ServingProfile(prefill_tokens_per_second=1000.0, decode_tokens_per_second=100.0)
    binned = _binned(prefill=[2000], decode=[300], requests=[10])
    prefill_share, decode_share = phase_shares(binned, profile)
    assert prefill_share + decode_share == pytest.approx(1.0)
    assert prefill_share == pytest.approx(0.4)


def test_a_replica_supplies_exactly_one_bin_of_gpu_time() -> None:
    """The unit trick that lets the validated simulator score the GPU lane unchanged."""
    profile = ServingProfile()
    capacity = profile.capacity_profile(workload_id="w", bin_seconds=60)
    assert capacity.capacity_per_replica == 60.0
    assert capacity.step_seconds == 60


def test_rescaling_removes_calibration_so_only_shape_error_remains() -> None:
    truth = np.asarray([10.0, 20.0, 30.0])
    proxy = truth * 7.5  # perfectly shaped, badly calibrated
    scaled = rescale_to(truth, proxy)
    assert np.allclose(scaled, truth)
    stats = signal_tracking_error(truth, proxy)
    assert stats["mean_abs_relative_error"] == pytest.approx(0.0, abs=1e-12)
    assert stats["correlation"] == pytest.approx(1.0)


def test_a_misshapen_proxy_is_penalised_even_after_rescaling() -> None:
    truth = np.asarray([10.0, 10.0, 100.0])
    proxy = np.asarray([10.0, 100.0, 10.0])  # right mean, wrong shape
    stats = signal_tracking_error(truth, proxy)
    assert stats["mean_abs_relative_error"] > 0.5
    assert stats["worst_under_provision"] < 0


def test_tokens_per_request_is_flat_when_work_per_request_is_constant() -> None:
    """If this were flat on real data, request-rate scaling would be fine and the lane moot."""
    binned = _binned(prefill=[100, 200, 300], decode=[10, 20, 30], requests=[1, 2, 3])
    context, generated = tokens_per_request(binned)
    assert np.allclose(context, 100.0)
    assert np.allclose(generated, 10.0)


def test_proxy_signal_names_are_validated() -> None:
    binned = _binned([1], [1], [1])
    with pytest.raises(ValueError, match="unknown proxy signal"):
        proxy_demand(binned, signal="cpu")


def test_serving_profile_rejects_impossible_rates() -> None:
    with pytest.raises(ValueError, match="token throughput"):
        ServingProfile(prefill_tokens_per_second=0.0)
    with pytest.raises(ValueError, match="utilisation_target"):
        ServingProfile(utilisation_target=1.5)

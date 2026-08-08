"""LLM inference serving model — what a GPU replica can actually absorb.

The premise every CPU-shaped autoscaler rests on breaks here. A GPU serving continuous
batches saturates on *token work*, not on request count and not on CPU utilisation, and the
two phases of that work have different costs:

* **prefill** processes the whole prompt at once and is compute bound — it scales with
  ``ContextTokens``;
* **decode** emits one token per sequence per step and is memory-bandwidth bound — it
  scales with ``GeneratedTokens``.

Both phases contend for the same device, so the honest capacity currency is **GPU-seconds
of work per bin**, and a replica supplies exactly ``bin_seconds`` of them. Expressing it
that way lets the already-validated replay simulator score inference workloads unchanged:
demand becomes required GPU-seconds, ``capacity_per_replica`` becomes ``bin_seconds``.

Rates are *modelled parameters*, not measurements from these traces — the Azure traces
publish tokens, not hardware timings. Defaults are order-of-magnitude figures for a mid-size
model on one datacentre GPU, and every conclusion drawn from them is swept in the
sensitivity analysis rather than asserted.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from delphi.control.simulator import CapacityProfile
from delphi.data.llm_inference import BinnedInference

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ServingProfile:
    """Throughput characteristics of one GPU replica.

    Defaults are deliberately round: they are a modelling assumption, and pretending to
    four significant figures would imply a measurement that was never taken.
    """

    #: Prompt tokens per second during prefill (compute bound).
    prefill_tokens_per_second: float = 12_000.0
    #: Generated tokens per second across the whole batch (memory-bandwidth bound).
    decode_tokens_per_second: float = 2_500.0
    #: Seconds to load weights and become ready. Minutes, not seconds — this is the term
    #: that makes reactive scaling structurally unable to hold TTFT.
    startup_seconds: float = 240.0
    teardown_seconds: float = 30.0
    #: Fraction of wall-clock a replica can spend on useful work before latency degrades.
    utilisation_target: float = 0.85

    def __post_init__(self) -> None:
        if self.prefill_tokens_per_second <= 0 or self.decode_tokens_per_second <= 0:
            raise ValueError("token throughput rates must be positive")
        if self.startup_seconds < 0 or self.teardown_seconds < 0:
            raise ValueError("actuation delays cannot be negative")
        if not 0 < self.utilisation_target <= 1:
            raise ValueError("utilisation_target must lie within (0, 1]")

    @property
    def prefill_decode_cost_ratio(self) -> float:
        """How many decode tokens cost the same GPU time as one prefill token."""
        return self.decode_tokens_per_second / self.prefill_tokens_per_second

    def capacity_profile(self, *, workload_id: str, bin_seconds: int) -> CapacityProfile:
        """Express the replica as the simulator's generic capacity contract.

        One replica supplies ``bin_seconds`` of GPU time per bin, so demand measured in
        required GPU-seconds is directly comparable to it. This is what lets the validated
        replay machinery score the GPU lane without a second, unvalidated simulator.
        """
        return CapacityProfile(
            workload_id=workload_id,
            step_seconds=bin_seconds,
            capacity_per_replica=float(bin_seconds),
            startup_seconds=self.startup_seconds,
            teardown_seconds=self.teardown_seconds,
            utilisation_target=self.utilisation_target,
            min_replicas=0,
            max_replicas=100_000,
            scale_to_zero=True,
        )


def gpu_seconds_demand(binned: BinnedInference, profile: ServingProfile) -> FloatArray:
    """The true work each bin imposes, in GPU-seconds.

    ``prefill_tokens / prefill_rate + decode_tokens / decode_rate``. Summing the two phases
    is the work-conservation assumption: they contend for one device, so time spent on
    prefill is time unavailable for decode. It ignores scheduling effects — batching
    efficiency, prefill chunking, preemption — which is why it is the *primary* model but
    not the only one, and why its parameters get swept.
    """
    prefill = binned.prefill_tokens.astype(np.float64) / profile.prefill_tokens_per_second
    decode = binned.decode_tokens.astype(np.float64) / profile.decode_tokens_per_second
    return prefill + decode


def phase_shares(binned: BinnedInference, profile: ServingProfile) -> tuple[float, float]:
    """Fraction of total GPU work spent in prefill and in decode."""
    prefill = float(binned.prefill_tokens.sum()) / profile.prefill_tokens_per_second
    decode = float(binned.decode_tokens.sum()) / profile.decode_tokens_per_second
    total = prefill + decode
    if total <= 0:
        raise ValueError("trace imposes no GPU work")
    return prefill / total, decode / total


def tokens_per_request(binned: BinnedInference) -> tuple[FloatArray, FloatArray]:
    """Per-bin mean context and generated tokens per request.

    The variability of these two series is exactly what a request-rate autoscaler is blind
    to: it sees a stable request count while the work behind each request moves.
    """
    counts = np.maximum(binned.requests.astype(np.float64), 1.0)
    return (
        binned.prefill_tokens.astype(np.float64) / counts,
        binned.decode_tokens.astype(np.float64) / counts,
    )


def proxy_demand(binned: BinnedInference, *, signal: str) -> FloatArray:
    """A demand signal an autoscaler might track instead of the true work.

    ``requests`` is the KEDA/HPA-style proxy — count arrivals and scale on them. ``prefill``
    and ``decode`` are single-phase proxies that ignore the other half of the contention.
    Each is rescaled to the same mean as the true GPU-second demand so the comparison is
    about *shape*, not about someone picking a bad constant: a proxy that merely needed
    recalibrating would be a tuning problem, not a structural one.
    """
    if signal == "requests":
        raw = binned.requests.astype(np.float64)
    elif signal == "prefill":
        raw = binned.prefill_tokens.astype(np.float64)
    elif signal == "decode":
        raw = binned.decode_tokens.astype(np.float64)
    elif signal == "total_tokens":
        raw = (binned.prefill_tokens + binned.decode_tokens).astype(np.float64)
    else:
        raise ValueError(f"unknown proxy signal {signal!r}")
    return raw


def rescale_to(reference: FloatArray, proxy: FloatArray) -> FloatArray:
    """Scale ``proxy`` to share ``reference``'s mean, isolating shape from calibration."""
    proxy_mean = float(np.mean(proxy))
    if proxy_mean <= 0:
        raise ValueError("cannot rescale a proxy with non-positive mean")
    return proxy * (float(np.mean(reference)) / proxy_mean)


def signal_tracking_error(reference: FloatArray, proxy: FloatArray) -> dict[str, float]:
    """How badly a rescaled proxy tracks true GPU work.

    Correlation alone is not enough — a proxy can correlate well and still mis-size badly at
    the peaks that actually breach SLOs, so the tail error is reported separately.
    """
    scaled = rescale_to(reference, proxy)
    error = scaled - reference
    relative = error / np.maximum(reference, 1e-9)
    peak_mask = reference >= np.quantile(reference, 0.95)
    return {
        "correlation": float(np.corrcoef(reference, scaled)[0, 1]),
        "mean_abs_relative_error": float(np.mean(np.abs(relative))),
        "p95_abs_relative_error": float(np.quantile(np.abs(relative), 0.95)),
        "under_provision_at_peak": float(np.mean(relative[peak_mask])),
        "worst_under_provision": float(np.min(relative)),
    }

"""Azure LLM inference traces — request-level arrivals with token counts.

These are the flagship lane's dataset, and they are unusual in two ways that matter.

**They carry token counts, not just arrivals.** ``ContextTokens`` drives prefill (compute
bound) and ``GeneratedTokens`` drives decode (memory-bandwidth bound). A trace with
arrivals alone cannot tell a burst of short prompts from a trickle of long generations, and
those have completely different capacity implications. That distinction is the whole reason
the GPU lane is worth building.

**The timestamps are absolute UTC.** Unlike the anonymized Azure Functions trace, there is
no epoch to assume and no nominal calendar anchor — day-of-week here is a fact, not a
convention.

Licence: **CC-BY 4.0**. Citation required — see ``CITATION_2024``.

The files are large (660 MB and 1.1 GB, ~44M requests combined), so parsing is streamed and
binned aggregates are cached to ``data/cache``. Nothing is held in memory that does not have
to be.
"""

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from delphi.data.series import DemandSeries

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

SOURCE_ID = "azure-llm-inference-2024"
LICENCE = "CC-BY-4.0"
CITATION_2024 = (
    "Jovan Stojkovic, Chaojie Zhang, Inigo Goiri, Josep Torrellas, Esha Choukse. "
    "DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency, "
    "HPCA 2025."
)
RELEASE_BASE = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/"
TRACES: dict[str, str] = {
    "code": "AzureLLMInferenceTrace_code_1week.csv",
    "conv": "AzureLLMInferenceTrace_conv_1week.csv",
}
SHA256: dict[str, str] = {
    "code": "71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f",
    "conv": "a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941",
}

DemandKind = Literal["requests", "prefill_tokens", "decode_tokens", "total_tokens"]


def _parse_timestamp(raw: str) -> datetime:
    """Parse ``2024-05-10 00:00:00.009930+00:00`` without the datetime machinery.

    ``fromisoformat`` is correct but costs roughly a microsecond per call, which is a full
    minute of wall clock over 44M rows. The format is fixed-width and always UTC, so the
    fields are sliced directly and the result is asserted to be timezone-aware by
    construction.
    """
    return datetime(
        int(raw[0:4]),
        int(raw[5:7]),
        int(raw[8:10]),
        int(raw[11:13]),
        int(raw[14:16]),
        int(raw[17:19]),
        int(raw[20:26]) if len(raw) > 20 and raw[19] == "." else 0,
        tzinfo=UTC,
    )


@dataclass(frozen=True)
class BinnedInference:
    """Per-bin aggregates: the four demand signals a serving cluster actually faces."""

    start: datetime
    bin_seconds: int
    requests: IntArray
    prefill_tokens: IntArray
    decode_tokens: IntArray

    def __post_init__(self) -> None:
        shapes = {self.requests.shape, self.prefill_tokens.shape, self.decode_tokens.shape}
        if len(shapes) != 1:
            raise ValueError("binned inference arrays must be parallel")
        if self.start.tzinfo is None or self.start.utcoffset() != timedelta(0):
            raise ValueError("binned inference start must be UTC")

    def __len__(self) -> int:
        return len(self.requests)

    def signal(self, kind: DemandKind) -> IntArray:
        if kind == "requests":
            return self.requests
        if kind == "prefill_tokens":
            return self.prefill_tokens
        if kind == "decode_tokens":
            return self.decode_tokens
        return self.prefill_tokens + self.decode_tokens


def stream_rows(path: Path) -> Iterator[tuple[datetime, int, int]]:
    """Yield ``(timestamp, context_tokens, generated_tokens)`` one request at a time."""
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expected = ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]
        if [column.strip() for column in header] != expected:
            raise ValueError(f"unexpected LLM trace header: {header}")
        for row in reader:
            if len(row) != 3:
                continue
            yield _parse_timestamp(row[0]), int(row[1]), int(row[2])


def bin_trace(path: Path, *, bin_seconds: int = 60) -> BinnedInference:
    """Stream the whole file into fixed bins. Memory stays flat regardless of file size."""
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive")
    requests: list[int] = []
    prefill: list[int] = []
    decode: list[int] = []
    start: datetime | None = None

    for timestamp, context_tokens, generated_tokens in stream_rows(path):
        if start is None:
            start = timestamp.replace(second=0, microsecond=0)
        index = int((timestamp - start).total_seconds() // bin_seconds)
        if index < 0:
            raise ValueError("LLM trace is not sorted by timestamp")
        while len(requests) <= index:
            requests.append(0)
            prefill.append(0)
            decode.append(0)
        requests[index] += 1
        prefill[index] += context_tokens
        decode[index] += generated_tokens

    if start is None:
        raise ValueError(f"no rows parsed from {path}")
    return BinnedInference(
        start=start,
        bin_seconds=bin_seconds,
        requests=np.asarray(requests, dtype=np.int64),
        prefill_tokens=np.asarray(prefill, dtype=np.int64),
        decode_tokens=np.asarray(decode, dtype=np.int64),
    )


def cache_path(trace: str, bin_seconds: int, directory: Path) -> Path:
    return directory / f"llm-{trace}-{bin_seconds}s.npz"


def load_binned(
    trace: str,
    *,
    raw_dir: Path = Path("data/raw"),
    cache_dir: Path = Path("data/cache"),
    bin_seconds: int = 60,
    refresh: bool = False,
) -> BinnedInference:
    """Load binned aggregates, parsing the raw CSV once and caching the result.

    Re-parsing 44M rows for every experiment would make the eval loop unusable, so the
    binned form is cached. The cache key includes the bin width, so a different binning
    never silently reuses the wrong aggregate.
    """
    if trace not in TRACES:
        raise ValueError(f"unknown LLM trace {trace!r}; expected one of {sorted(TRACES)}")
    cache = cache_path(trace, bin_seconds, cache_dir)
    if cache.exists() and not refresh:
        payload = np.load(cache, allow_pickle=False)
        return BinnedInference(
            start=datetime.fromtimestamp(float(payload["start"]), tz=UTC),
            bin_seconds=int(payload["bin_seconds"]),
            requests=payload["requests"],
            prefill_tokens=payload["prefill_tokens"],
            decode_tokens=payload["decode_tokens"],
        )
    binned = bin_trace(raw_dir / TRACES[trace], bin_seconds=bin_seconds)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        start=binned.start.timestamp(),
        bin_seconds=binned.bin_seconds,
        requests=binned.requests,
        prefill_tokens=binned.prefill_tokens,
        decode_tokens=binned.decode_tokens,
    )
    return binned


def to_demand_series(
    binned: BinnedInference,
    *,
    trace: str,
    kind: DemandKind = "requests",
) -> DemandSeries:
    """Project one of the four signals into the canonical demand contract."""
    values = binned.signal(kind).astype(np.float64)
    units = {
        "requests": "requests/bin",
        "prefill_tokens": "context tokens/bin",
        "decode_tokens": "generated tokens/bin",
        "total_tokens": "tokens/bin",
    }
    return DemandSeries(
        workload_id=f"{SOURCE_ID}:{trace}:{kind}",
        source_id=SOURCE_ID,
        resource_kind="tokens" if kind.endswith("tokens") else "requests",
        unit=units[kind],
        step_seconds=binned.bin_seconds,
        timestamps=tuple(
            binned.start + timedelta(seconds=binned.bin_seconds * index)
            for index in range(len(binned))
        ),
        values=values,
        is_imputed=np.zeros(len(values), dtype=np.bool_),
        quality=np.ones(len(values), dtype=np.float64),
    )


@dataclass(frozen=True)
class InferenceRequests:
    """Request-level arrivals for one window — what the serving model replays."""

    start: datetime
    arrival_seconds: FloatArray
    context_tokens: IntArray
    generated_tokens: IntArray

    def __post_init__(self) -> None:
        shapes = {
            self.arrival_seconds.shape,
            self.context_tokens.shape,
            self.generated_tokens.shape,
        }
        if len(shapes) != 1:
            raise ValueError("request arrays must be parallel")
        if len(self.arrival_seconds) and not np.all(np.diff(self.arrival_seconds) >= 0):
            raise ValueError("requests must be sorted by arrival")

    def __len__(self) -> int:
        return len(self.arrival_seconds)


def load_requests(
    trace: str,
    *,
    window_start: datetime,
    window_seconds: int,
    raw_dir: Path = Path("data/raw"),
    max_requests: int | None = None,
) -> InferenceRequests:
    """Load one time window at request level, streaming past everything outside it."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if window_start.tzinfo is None:
        raise ValueError("window_start must be timezone-aware")
    window_end = window_start + timedelta(seconds=window_seconds)
    arrivals: list[float] = []
    context: list[int] = []
    generated: list[int] = []
    for timestamp, context_tokens, generated_tokens in stream_rows(raw_dir / TRACES[trace]):
        if timestamp < window_start:
            continue
        if timestamp >= window_end:
            break
        arrivals.append((timestamp - window_start).total_seconds())
        context.append(context_tokens)
        generated.append(generated_tokens)
        if max_requests is not None and len(arrivals) >= max_requests:
            break
    return InferenceRequests(
        start=window_start,
        arrival_seconds=np.asarray(arrivals, dtype=np.float64),
        context_tokens=np.asarray(context, dtype=np.int64),
        generated_tokens=np.asarray(generated, dtype=np.int64),
    )

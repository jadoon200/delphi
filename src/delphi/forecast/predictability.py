"""Training-free measures of whether a demand series is worth forecasting.

The project shipped a diagnostic built on **daily autocorrelation**, and Q13 established
that it does not generalise: on individual Azure Functions workloads the 0.50 cutoff is a
coin flip and the relationship is not even monotone. The forecastability literature predicts
exactly that failure. A single lagged correlation interrogates one frequency; a workload can
be highly structured at a period the chosen lag does not look at, or carry a strong day-lag
correlation whose power sits at frequencies too fast to help a six-hour commitment.

The established alternative is **spectral entropy** — the Shannon entropy of the normalised
power spectral density, which reads all frequencies at once. Near 0 means the power is
concentrated in a few components (a near-sinusoid, highly predictable); near 1 means it is
spread evenly (white noise, unforecastable). It is the standard forecastability feature and
recent work proposes it specifically as a fast, training-free indicator of whether
forecasting will beat a simple baseline — the same job DELPHI's diagnostic does.

This module also carries a third measure that the project's own Q1 result motivates.
Forecasting pays over a commitment window only when demand carries structure *slower* than
that window; power at frequencies faster than the window averages out inside it and cannot
be exploited by a decision that is fixed for its duration. ``low_frequency_power_fraction``
measures that share directly, and unlike the other two it is a function of the horizon being
committed to rather than of the series alone.

Numpy only, deliberately: a diagnostic that must run inside a web request should not pull a
transitive scipy import, and a periodogram is fifteen lines.
"""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _validate(values: FloatArray, minimum: int) -> FloatArray:
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("demand must be one-dimensional")
    if len(series) < minimum:
        raise ValueError(f"need at least {minimum} points, got {len(series)}")
    if not np.isfinite(series).all():
        raise ValueError("demand must be finite")
    if float(series.std()) <= 0:
        raise ValueError("a constant series has no spectrum to measure")
    return series


def lag_autocorrelation(values: FloatArray, lag: int) -> float:
    """Pearson correlation between the series and itself at ``lag`` steps.

    The single definition of the project's headline diagnostic. It previously existed twice
    — once in the API and once in the evaluation script — which is the same drift risk that
    let the shipped verdict contradict `docs/EVAL.md` for a day.
    """
    if lag < 1:
        raise ValueError("lag must be positive")
    series = _validate(values, lag + 9)
    lagged, leading = series[:-lag], series[lag:]
    if lagged.std() <= 0 or leading.std() <= 0:
        raise ValueError("one half of the lagged pair is constant")
    return float(np.corrcoef(lagged, leading)[0, 1])


def optional_lag_autocorrelation(values: FloatArray, lag: int) -> float | None:
    """``lag_autocorrelation``, returning ``None`` where it is not measurable.

    The API reports "could not be computed" rather than "no structure" for a series too
    short to carry the lag, and those two must never be confused: one is missing evidence,
    the other is evidence of absence.
    """
    try:
        return lag_autocorrelation(values, lag)
    except ValueError:
        return None


def daily_autocorrelation(values: FloatArray, day_steps: int) -> float:
    """Correlation at a one-day lag — the measure the shipped diagnostic uses."""
    return lag_autocorrelation(values, day_steps)


def power_spectrum(values: FloatArray, *, segments: int = 8) -> tuple[FloatArray, FloatArray]:
    """Welch-style averaged periodogram: returns (frequencies in cycles/step, power).

    A raw periodogram is an inconsistent estimator — its variance does not fall as the
    series grows — so entropy computed from one is dominated by noise. Averaging Hann-
    windowed half-overlapping segments trades frequency resolution for the variance
    reduction that makes the entropy stable. The DC term is dropped: a series' mean level
    says nothing about whether its *shape* is predictable.
    """
    series = _validate(values, 32)
    if segments < 1:
        raise ValueError("segments must be positive")

    # Half-overlapping segments, each at least 16 points; fall back to one segment if short.
    length = max(len(series) // max(segments, 1) * 2, 16)
    length = min(length, len(series))
    step = max(length // 2, 1)
    starts = range(0, len(series) - length + 1, step)

    window = np.hanning(length)
    correction = float(np.sum(window**2))
    accumulated = np.zeros(length // 2 + 1, dtype=np.float64)
    count = 0
    for start in starts:
        chunk = series[start : start + length]
        chunk = chunk - chunk.mean()
        spectrum = np.abs(np.fft.rfft(chunk * window)) ** 2 / correction
        accumulated += spectrum
        count += 1
    if count == 0:  # pragma: no cover - length is clamped to len(series) above
        raise ValueError("series too short to segment")

    power = accumulated[1:] / count
    freqs = np.fft.rfftfreq(length, d=1.0)[1:]
    return freqs, power


def spectral_entropy(values: FloatArray, *, segments: int = 8) -> float:
    """Normalised Shannon entropy of the power spectrum, in [0, 1].

    0 is a pure sinusoid; 1 is white noise. This is the field's standard forecastability
    feature, and it is reported here in its raw orientation — **lower means more
    predictable** — so it cannot be silently confused with an autocorrelation.
    """
    _, power = power_spectrum(values, segments=segments)
    total = float(power.sum())
    if total <= 0:
        raise ValueError("spectrum carries no power")
    density = power / total
    density = density[density > 0]
    if len(density) < 2:
        return 0.0
    entropy = float(-(density * np.log(density)).sum())
    return entropy / float(np.log(len(power)))


def spectral_predictability(values: FloatArray, *, segments: int = 8) -> float:
    """``1 - spectral_entropy``, so that higher means more predictable.

    Provided because every other number in the diagnostic points the same way, and a metric
    whose direction is inverted relative to its neighbours is a reporting bug waiting to
    happen.
    """
    return 1.0 - spectral_entropy(values, segments=segments)


def low_frequency_power_fraction(
    values: FloatArray, window_steps: int, *, segments: int = 8
) -> float:
    """Share of spectral power at periods **longer than** ``window_steps``.

    Motivated by this project's own Q1 result: what a commitment controller can exploit is
    structure that persists across the window it is committing to. Variation faster than the
    window averages out inside it, and a capacity level fixed for the whole window cannot
    track it however well it was predicted. Unlike entropy and autocorrelation this is a
    function of the decision horizon, not of the series alone — which is the point.
    """
    if window_steps < 2:
        raise ValueError("window_steps must be at least two")
    freqs, power = power_spectrum(values, segments=segments)
    total = float(power.sum())
    if total <= 0:
        raise ValueError("spectrum carries no power")
    slow = freqs < (1.0 / float(window_steps))
    return float(power[slow].sum() / total)

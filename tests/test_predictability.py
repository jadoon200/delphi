"""Training-free predictability measures, checked against signals with known answers."""

import numpy as np
import pytest

from delphi.forecast.predictability import (
    daily_autocorrelation,
    low_frequency_power_fraction,
    power_spectrum,
    spectral_entropy,
    spectral_predictability,
)

DAY = 288  # 5-minute bins


def _sine(periods: float = 10.0, day_steps: int = DAY, noise: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(20260808)
    t = np.arange(int(periods * day_steps), dtype=np.float64)
    values = 50 + 20 * np.sin(2 * np.pi * t / day_steps)
    if noise:
        values = values + rng.normal(0.0, noise, size=len(t))
    return values


def _white_noise(length: int = DAY * 10) -> np.ndarray:
    rng = np.random.default_rng(7)
    return 50 + rng.normal(0.0, 10.0, size=length)


def test_spectral_entropy_separates_a_sinusoid_from_white_noise() -> None:
    """The defining property: near 0 for one frequency, near 1 for all of them."""
    assert spectral_entropy(_sine()) < 0.25
    assert spectral_entropy(_white_noise()) > 0.90


def test_spectral_entropy_is_bounded_and_ordered_by_noise_level() -> None:
    previous = spectral_entropy(_sine(noise=0.0))
    for noise in (2.0, 5.0, 10.0, 20.0):
        current = spectral_entropy(_sine(noise=noise))
        assert 0.0 <= current <= 1.0
        assert current > previous, f"entropy must rise with noise (at sigma={noise})"
        previous = current


def test_spectral_entropy_is_invariant_to_shift_and_scale() -> None:
    """A forecastability measure must not depend on the units demand is reported in."""
    values = _sine(noise=4.0)
    base = spectral_entropy(values)
    assert spectral_entropy(values * 7.5) == pytest.approx(base, abs=1e-9)
    assert spectral_entropy(values + 1000.0) == pytest.approx(base, abs=1e-9)


def test_spectral_predictability_is_the_complement() -> None:
    values = _sine(noise=4.0)
    assert spectral_predictability(values) == pytest.approx(1.0 - spectral_entropy(values))


def test_low_frequency_fraction_tracks_the_commitment_window() -> None:
    """A daily cycle is slow relative to six hours and fast relative to three days."""
    daily = _sine(noise=3.0)
    assert low_frequency_power_fraction(daily, window_steps=DAY // 4) > 0.8
    assert low_frequency_power_fraction(daily, window_steps=DAY * 3) < 0.2


def test_low_frequency_fraction_separates_fast_from_slow_structure() -> None:
    """Two series, equally predictable, differing only in whether a window can use it."""
    t = np.arange(DAY * 10, dtype=np.float64)
    slow = 50 + 20 * np.sin(2 * np.pi * t / DAY)
    fast = 50 + 20 * np.sin(2 * np.pi * t / (DAY // 24))
    window = DAY // 4

    # Both are near-pure tones, so spectral entropy cannot tell them apart...
    assert abs(spectral_entropy(slow) - spectral_entropy(fast)) < 0.15
    # ...but only one carries structure a six-hour commitment can exploit.
    assert low_frequency_power_fraction(slow, window) > 0.8
    assert low_frequency_power_fraction(fast, window) < 0.05


def test_daily_autocorrelation_matches_the_shipped_definition() -> None:
    values = _sine(noise=4.0)
    expected = float(np.corrcoef(values[:-DAY], values[DAY:])[0, 1])
    assert daily_autocorrelation(values, DAY) == pytest.approx(expected)


def test_power_spectrum_drops_dc_and_returns_positive_frequencies() -> None:
    freqs, power = power_spectrum(_sine(noise=2.0))
    assert len(freqs) == len(power)
    assert (freqs > 0).all(), "the mean level is not a statement about predictability"
    assert (power >= 0).all()


def test_measures_reject_degenerate_input() -> None:
    for bad in (np.zeros(DAY * 2), np.full(DAY * 2, 5.0)):
        with pytest.raises(ValueError):
            spectral_entropy(bad)
    with pytest.raises(ValueError):
        spectral_entropy(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        spectral_entropy(np.array([1.0, np.nan] * 100))
    with pytest.raises(ValueError):
        low_frequency_power_fraction(_sine(), window_steps=1)

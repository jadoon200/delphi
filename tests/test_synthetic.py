import numpy as np

from delphi.data.synthetic import generate_multi_resource, generate_synthetic


def test_synthetic_generation_is_seeded_and_labelled() -> None:
    first = generate_synthetic("burst_known", periods=480, seed=7)
    second = generate_synthetic("burst_known", periods=480, seed=7)
    assert np.array_equal(first.values, second.values)
    assert [annotation.label for annotation in first.annotations] == ["burst", "burst"]


def test_level_shift_and_near_noise_are_distinct_regimes() -> None:
    shifted = generate_synthetic("level_shift", periods=1000, seed=9)
    annotation = shifted.annotations[0]
    before = shifted.values[: annotation.start].mean()
    after = shifted.values[annotation.start :].mean()
    assert after - before > 25

    noise = generate_synthetic("near_noise", periods=3000, seed=11)
    lag_one = float(np.corrcoef(noise.values[:-1], noise.values[1:])[0, 1])
    assert abs(lag_one) < 0.1


def test_multi_resource_peaks_are_deliberately_decorrelated() -> None:
    cpu, memory = generate_multi_resource(periods=1000, seed=13)
    cpu_peak = cpu.annotations[0]
    memory_peak = memory.annotations[0]
    assert cpu_peak.end <= memory_peak.start
    assert int(np.argmax(cpu.values)) in range(cpu_peak.start, cpu_peak.end)
    assert int(np.argmax(memory.values)) in range(memory_peak.start, memory_peak.end)

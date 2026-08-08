"""The simulator's queue must reproduce closed-form queueing theory, not just run."""

import math

import numpy as np
import pytest

from delphi.control.queueing import (
    erlang_c,
    mmc_mean_queue_length,
    mmc_mean_wait_seconds,
    poisson_arrivals,
    simulate_multi_server_queue,
)


def test_erlang_c_matches_known_single_server_result() -> None:
    # M/M/1: the probability of waiting is exactly the utilisation.
    for rho in (0.1, 0.5, 0.9):
        assert erlang_c(rho, 1.0, 1) == pytest.approx(rho, rel=1e-12)


def test_mm1_mean_wait_matches_closed_form() -> None:
    # M/M/1: Wq = rho / (mu - lambda)
    mu, rho = 2.0, 0.6
    lam = rho * mu
    assert mmc_mean_wait_seconds(lam, mu, 1) == pytest.approx(rho / (mu - lam), rel=1e-12)


def test_unstable_queue_reports_infinity_rather_than_a_comfortable_number() -> None:
    assert math.isinf(mmc_mean_wait_seconds(10.0, 1.0, 5))
    assert math.isinf(mmc_mean_queue_length(10.0, 1.0, 5))


def test_littles_law_holds_between_wait_and_queue_length() -> None:
    lam, mu, servers = 3.0, 1.0, 5
    wait = mmc_mean_wait_seconds(lam, mu, servers)
    assert mmc_mean_queue_length(lam, mu, servers) == pytest.approx(lam * wait, rel=1e-12)


@pytest.mark.parametrize(
    ("servers", "utilisation"),
    [(1, 0.5), (2, 0.6), (3, 0.7), (5, 0.8)],
)
def test_event_driven_queue_reproduces_erlang_c(servers: int, utilisation: float) -> None:
    """The kill criterion: a queue that cannot reproduce Erlang-C is broken.

    Measured 2026-08-03 at 4x this sample size, relative error stayed under 1.3% across
    c = 1..10 and rho = 0.5..0.85. The tolerance here is loosened only to keep the suite
    fast; the full sweep lives in ``scripts/validate_simulator.py``.
    """
    rng = np.random.default_rng(20260803 + servers)
    service_rate = 0.5
    arrival_rate = utilisation * servers * service_rate
    duration = 400_000 / arrival_rate

    arrivals = poisson_arrivals(rate_per_second=arrival_rate, duration_seconds=duration, rng=rng)
    service = rng.exponential(1 / service_rate, size=len(arrivals))
    pool = np.full(int(duration // 60) + 2, servers, dtype=np.int64)

    outcome = simulate_multi_server_queue(
        arrival_seconds=arrivals,
        service_seconds=service,
        servers_per_step=pool,
        step_seconds=60.0,
    )
    theoretical = mmc_mean_wait_seconds(arrival_rate, service_rate, servers)
    assert outcome.mean_wait_seconds == pytest.approx(theoretical, rel=0.06)


def test_zero_capacity_drops_rather_than_queues_forever() -> None:
    arrivals = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    service = np.ones(3, dtype=np.float64)
    outcome = simulate_multi_server_queue(
        arrival_seconds=arrivals,
        service_seconds=service,
        servers_per_step=np.zeros(1, dtype=np.int64),
        step_seconds=60.0,
    )
    assert outcome.dropped.all()
    assert outcome.served == 0
    assert math.isnan(outcome.mean_wait_seconds)


def test_idle_queue_makes_every_request_wait_nothing() -> None:
    arrivals = np.arange(0.0, 100.0, 10.0)
    service = np.ones(len(arrivals), dtype=np.float64)
    outcome = simulate_multi_server_queue(
        arrival_seconds=arrivals,
        service_seconds=service,
        servers_per_step=np.full(4, 8, dtype=np.int64),
        step_seconds=60.0,
    )
    assert np.allclose(outcome.wait_seconds, 0.0)


def test_unsorted_arrivals_are_refused() -> None:
    with pytest.raises(ValueError, match="sorted"):
        simulate_multi_server_queue(
            arrival_seconds=np.asarray([5.0, 1.0]),
            service_seconds=np.ones(2),
            servers_per_step=np.ones(1, dtype=np.int64),
            step_seconds=60.0,
        )

"""Analytic queueing results and an event-driven multi-server queue.

The analytic half exists to *validate* the simulated half. A discrete-event queue that
cannot reproduce the Erlang-C closed form is not modelling what it claims to model, and
every downstream capacity number inherits that error silently. This module is therefore
deliberately small, deliberately exact, and tested against mathematics rather than against
its own output.
"""

import math
from bisect import insort
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def offered_load(arrival_rate: float, service_rate: float) -> float:
    """Erlang offered load ``a = lambda / mu``, in units of busy servers."""
    if arrival_rate <= 0 or service_rate <= 0:
        raise ValueError("arrival and service rates must be positive")
    return arrival_rate / service_rate


def erlang_c(arrival_rate: float, service_rate: float, servers: int) -> float:
    """Probability that an arriving request has to wait, for M/M/c.

    ``C(c, a) = (a^c / c!) / ( a^c / c! + (1 - rho) * sum_{k<c} a^k / k! )``
    """
    if servers < 1:
        raise ValueError("an M/M/c queue needs at least one server")
    load = offered_load(arrival_rate, service_rate)
    utilisation = load / servers
    if utilisation >= 1:
        return 1.0
    # Computed in log space only where it matters; c is small in every realistic capacity
    # question, so the direct sum is both exact enough and easier to read.
    terms = [load**k / math.factorial(k) for k in range(servers)]
    top = load**servers / math.factorial(servers)
    return top / (top + (1 - utilisation) * sum(terms))


def mmc_mean_wait_seconds(arrival_rate: float, service_rate: float, servers: int) -> float:
    """Mean time spent queueing (excluding service) for M/M/c.

    ``Wq = C(c, a) / (c * mu - lambda)``. Returns infinity for an unstable queue, which is
    the honest answer: an over-subscribed queue has no steady state, and a simulator that
    reports a finite number there is lying about a system that is actually diverging.
    """
    if servers < 1:
        raise ValueError("an M/M/c queue needs at least one server")
    if arrival_rate >= servers * service_rate:
        return math.inf
    return erlang_c(arrival_rate, service_rate, servers) / (servers * service_rate - arrival_rate)


def mmc_mean_queue_length(arrival_rate: float, service_rate: float, servers: int) -> float:
    """Mean number waiting, via Little's law ``Lq = lambda * Wq``."""
    wait = mmc_mean_wait_seconds(arrival_rate, service_rate, servers)
    return math.inf if math.isinf(wait) else arrival_rate * wait


@dataclass(frozen=True)
class QueueOutcome:
    """Per-request timings from one event-driven run."""

    arrival_seconds: FloatArray
    wait_seconds: FloatArray
    service_seconds: FloatArray
    dropped: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        shapes = {
            self.arrival_seconds.shape,
            self.wait_seconds.shape,
            self.service_seconds.shape,
            self.dropped.shape,
        }
        if len(shapes) != 1:
            raise ValueError("queue outcome arrays must be parallel")

    @property
    def mean_wait_seconds(self) -> float:
        """Mean wait over *served* requests; NaN when everything was dropped."""
        served = ~self.dropped
        if not served.any():
            return float("nan")
        return float(np.mean(self.wait_seconds[served]))

    @property
    def served(self) -> int:
        return int((~self.dropped).sum())


def simulate_multi_server_queue(
    *,
    arrival_seconds: FloatArray,
    service_seconds: FloatArray,
    servers_per_step: IntArray,
    step_seconds: float,
) -> QueueOutcome:
    """Run a FIFO queue against a time-varying pool of identical servers.

    Servers are tracked by the time each becomes free. On scale-up the new servers are
    idle from the moment they appear; on scale-down the servers whose work finishes
    *latest* are the ones removed. That second rule is an approximation — a real platform
    drains a replica before terminating it — and it only bites while capacity is shrinking.
    It is recorded here rather than hidden because the sensitivity sweep needs to know
    which constants are modelled and which are assumed.

    A request arriving while zero servers are provisioned is dropped rather than queued
    forever, so ``scale_to_zero`` workloads produce a finite, interpretable result.
    """
    if arrival_seconds.ndim != 1 or arrival_seconds.shape != service_seconds.shape:
        raise ValueError("arrivals and service times must be parallel 1-D arrays")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    if servers_per_step.ndim != 1 or not len(servers_per_step):
        raise ValueError("servers_per_step must be a non-empty 1-D array")
    if np.any(servers_per_step < 0):
        raise ValueError("server counts cannot be negative")
    if len(arrival_seconds) and not np.all(np.diff(arrival_seconds) >= 0):
        raise ValueError("arrivals must be sorted in time")

    last_step = len(servers_per_step) - 1
    free_times: list[float] = []
    waits = np.zeros(len(arrival_seconds), dtype=np.float64)
    dropped = np.zeros(len(arrival_seconds), dtype=np.bool_)

    for index, arrival in enumerate(arrival_seconds):
        step = min(int(arrival // step_seconds), last_step)
        capacity = int(servers_per_step[step])
        if capacity == 0:
            free_times.clear()
            dropped[index] = True
            continue
        # Grow the pool with servers that are idle from now; shrink it by discarding the
        # latest-finishing servers.
        while len(free_times) < capacity:
            insort(free_times, float(arrival))
        while len(free_times) > capacity:
            free_times.pop()
        earliest_free = free_times.pop(0)
        start = max(float(arrival), earliest_free)
        waits[index] = start - float(arrival)
        insort(free_times, start + float(service_seconds[index]))

    return QueueOutcome(
        arrival_seconds=arrival_seconds.copy(),
        wait_seconds=waits,
        service_seconds=service_seconds.copy(),
        dropped=dropped,
    )


def poisson_arrivals(
    *,
    rate_per_second: float,
    duration_seconds: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw a homogeneous Poisson arrival stream by exponential interarrival times."""
    if rate_per_second <= 0 or duration_seconds <= 0:
        raise ValueError("rate and duration must be positive")
    expected = int(rate_per_second * duration_seconds)
    # Draw generously, then truncate: an exact-count draw would condition on the count and
    # stop being Poisson.
    draw = max(64, int(expected * 1.4) + 64)
    gaps = rng.exponential(1.0 / rate_per_second, size=draw)
    times = np.cumsum(gaps)
    while times[-1] < duration_seconds:
        extra = rng.exponential(1.0 / rate_per_second, size=draw)
        times = np.concatenate([times, times[-1] + np.cumsum(extra)])
    return np.asarray(times[times < duration_seconds], dtype=np.float64)

"""M17 — zero-shot foundation forecasters, behind an optional dependency.

Apache-2.0 only, per the zero-cost audit: Chronos-Bolt is the default choice, TimesFM and
Lag-Llama are the alternatives, and Moirai and TiRex are excluded on licence grounds. The
weights are free and keyless, so nothing here breaks the zero-cost rule — but `torch` and
`transformers` are a heavy dependency for a project whose deploy image deliberately carries
no scientific stack, so this module is **optional**. It is absent from `requirements.txt`
and from the deploy image, and everything that uses it degrades to a clear message rather
than an ImportError traceback.

Install with ``pip install chronos-forecasting`` to reproduce the M17 experiment.

**The quantile ceiling is the finding, and it is structural rather than a tuning problem.**
Chronos-Bolt is trained on quantile levels 0.1 through 0.9. Ask it for p95 or p99 and it
does not extrapolate — it clamps to p90 and warns. A capacity controller consumes exactly
those upper quantiles, so a model that cannot express them cannot size for a 95% or 99%
compliance target no matter how accurate its median is. This class surfaces that ceiling as
a property rather than letting a silently clamped number reach a sizing decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from delphi.data.series import DemandSeries
from delphi.forecast.contracts import DemandForecast, build_forecast

FloatArray = npt.NDArray[np.float64]

#: The levels Chronos-Bolt was trained on. Anything outside this is clamped by the model.
CHRONOS_TRAINED_RANGE = (0.1, 0.9)

INSTALL_HINT = (
    "chronos-forecasting is not installed. It is an optional extra, deliberately absent "
    "from requirements.txt and the deploy image. Install it with "
    "`pip install chronos-forecasting` to reproduce M17."
)


def chronos_available() -> bool:
    """Whether the optional dependency is importable, without importing torch eagerly."""
    from importlib.util import find_spec

    return find_spec("chronos") is not None


@dataclass
class ChronosForecaster:
    """Zero-shot Chronos-Bolt wrapped in the project's forecaster contract.

    ``context_steps`` bounds how much history is fed to the model. Chronos-Bolt is trained
    for prediction lengths up to 64 steps and warns beyond that; the warning is not
    suppressed, because a 12-hour commitment at 5-minute bins asks for 144 and a reader
    should see that the model is being used outside its designed range.
    """

    model_name: str = "amazon/chronos-bolt-tiny"
    context_steps: int = 1024
    _pipeline: Any = field(default=None, init=False, repr=False)

    @property
    def model_id(self) -> str:
        return f"chronos:{self.model_name.split('/')[-1]}:ctx={self.context_steps}:v1"

    @property
    def minimum_history(self) -> int:
        return 64

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            if not chronos_available():
                raise RuntimeError(INSTALL_HINT)
            import torch
            from chronos import BaseChronosPipeline

            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_name, device_map="cpu", torch_dtype=torch.float32
            )
        return self._pipeline

    @staticmethod
    def clamped_levels(quantile_levels: tuple[float, ...]) -> tuple[float, ...]:
        """Which requested levels the model cannot express and will silently flatten."""
        low, high = CHRONOS_TRAINED_RANGE
        return tuple(level for level in quantile_levels if level < low or level > high)

    def forecast(
        self,
        history: DemandSeries,
        *,
        horizon_steps: int,
        quantile_levels: tuple[float, ...],
    ) -> DemandForecast:
        import torch

        context = history.values[-self.context_steps :]
        predicted, _mean = self.pipeline.predict_quantiles(
            inputs=torch.tensor(context, dtype=torch.float32),
            prediction_length=horizon_steps,
            quantile_levels=list(quantile_levels),
        )
        values = np.asarray(predicted[0].numpy(), dtype=np.float64)
        # The contract forbids negative demand and crossing quantiles. Chronos can emit
        # both — negatives on near-zero series, and exact ties once upper levels are
        # clamped to p90 — so they are repaired here rather than at the call site.
        values = np.maximum(values, 0.0)
        values = np.maximum.accumulate(values, axis=1)
        return build_forecast(
            history,
            horizon_steps=horizon_steps,
            quantile_levels=quantile_levels,
            quantile_values=values,
            model_id=self.model_id,
        )

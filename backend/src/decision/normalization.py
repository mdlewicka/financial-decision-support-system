"""Explicit min-max normalization used by the deterministic scorer."""

from __future__ import annotations

import math

from backend.src.decision.models import MetricNormalization

NEUTRAL_CONSTANT_SCORE = 0.5


def build_normalization(
    metric: str,
    values: tuple[float, ...],
    *,
    higher_is_better: bool,
) -> MetricNormalization:
    if not metric.strip():
        raise ValueError("metric must not be empty.")
    if not values:
        raise ValueError("values must not be empty.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("normalization values must be finite.")
    return MetricNormalization(
        metric=metric,
        minimum=min(values),
        maximum=max(values),
        higher_is_better=higher_is_better,
    )


def normalize_value(value: float, normalization: MetricNormalization) -> float:
    if not math.isfinite(value):
        raise ValueError("normalized value must be finite.")
    spread = normalization.maximum - normalization.minimum
    if math.isclose(spread, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return NEUTRAL_CONSTANT_SCORE
    scaled = (value - normalization.minimum) / spread
    if not normalization.higher_is_better:
        scaled = 1.0 - scaled
    return float(min(1.0, max(0.0, scaled)))

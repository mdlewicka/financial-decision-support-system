import pandas as pd
import pytest

from backend.src.forecast.forecast_evaluator import ForecastEvaluator
from backend.src.forecast.forecast_metrics import ForecastMetrics


def test_evaluate_returns_metrics_suitable_for_returns() -> None:
    actual = pd.Series(
        [
            0.02,
            -0.01,
            0.00,
            0.03,
        ],
        dtype=float,
    )
    predictions = pd.Series(
        [
            0.01,
            -0.02,
            0.00,
            -0.01,
        ],
        dtype=float,
    )

    evaluator = ForecastEvaluator()

    metrics = evaluator.evaluate(
        actual=actual,
        predictions=predictions,
    )

    assert isinstance(metrics, ForecastMetrics)
    assert metrics.observations == 4
    assert metrics.mae == pytest.approx(0.015)
    assert metrics.mse == pytest.approx(0.00045)
    assert metrics.rmse == pytest.approx(0.00045**0.5)
    assert metrics.r2 == pytest.approx(-0.8)
    assert metrics.directional_accuracy == pytest.approx(0.75)

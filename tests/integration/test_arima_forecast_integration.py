import numpy as np
import pandas as pd

from backend.src.forecast.forecast_evaluator import ForecastEvaluator
from backend.src.forecast.forecast_result import ForecastResult
from backend.src.forecast.forecast_service import ForecastService
from backend.src.models.arima_forecast_model import (
    ArimaForecastModel,
)


def create_training_dataset() -> pd.DataFrame:
    observations_per_ticker = 50
    dates = pd.date_range(
        start="2024-01-01",
        periods=observations_per_ticker,
        freq="D",
    )
    observation_numbers = np.arange(
        observations_per_ticker,
        dtype=float,
    )

    return pd.concat(
        [
            pd.DataFrame(
                {
                    "ticker": "SPY",
                    "date": dates,
                    "target": (0.008 * np.sin(observation_numbers / 3.0)),
                }
            ),
            pd.DataFrame(
                {
                    "ticker": "QQQ",
                    "date": dates,
                    "target": (0.012 * np.cos(observation_numbers / 4.0)),
                }
            ),
        ],
        ignore_index=True,
    )


def create_evaluation_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [
                "SPY",
                "SPY",
                "SPY",
                "QQQ",
                "QQQ",
                "QQQ",
            ],
            "date": pd.to_datetime(
                [
                    "2024-02-20",
                    "2024-02-21",
                    "2024-02-22",
                    "2024-02-20",
                    "2024-02-21",
                    "2024-02-22",
                ]
            ),
            "target_date": pd.to_datetime(
                [
                    "2024-02-21",
                    "2024-02-22",
                    "2024-02-23",
                    "2024-02-21",
                    "2024-02-22",
                    "2024-02-23",
                ]
            ),
            "target": [
                0.004,
                -0.003,
                0.005,
                -0.006,
                0.007,
                -0.002,
            ],
        }
    )


def test_arima_runs_walk_forward_forecast_for_each_ticker() -> None:
    model = ArimaForecastModel(
        order=(1, 0, 0),
        minimum_observations=20,
    )
    service = ForecastService(
        model=model,
        evaluator=ForecastEvaluator(),
    )

    result = service.forecast(
        training_dataset=create_training_dataset(),
        evaluation_dataset=create_evaluation_dataset(),
    )

    assert isinstance(
        result,
        ForecastResult,
    )
    assert len(result.observations) == 6
    assert np.isfinite(result.observations["prediction"].to_numpy()).all()

    assert result.overall_metrics.observations == 6
    assert set(result.metrics_by_ticker) == {
        "QQQ",
        "SPY",
    }
    assert result.metrics_by_ticker["SPY"].observations == 3
    assert result.metrics_by_ticker["QQQ"].observations == 3

    assert model.observation_counts == {
        "QQQ": 53,
        "SPY": 53,
    }

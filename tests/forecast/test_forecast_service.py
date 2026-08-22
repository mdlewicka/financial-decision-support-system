import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.forecast.forecast_evaluator import ForecastEvaluator
from backend.src.forecast.forecast_result import ForecastResult
from backend.src.forecast.forecast_service import ForecastService
from backend.src.models.base_forecast_model import (
    BaseForecastModel,
    BaseOnlineForecastModel,
)


class StubForecastModel(BaseForecastModel):
    def __init__(self) -> None:
        self.training_dataset: pd.DataFrame | None = None

    def fit(
        self,
        training_dataset: pd.DataFrame,
    ) -> None:
        self.training_dataset = training_dataset

    def predict(
        self,
        dataset: pd.DataFrame,
    ) -> pd.Series:
        return pd.Series(
            [
                0.01,
                -0.02,
                0.01,
                0.03,
            ],
            index=dataset.index,
            dtype=float,
        )


def create_training_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [
                "SPY",
                "QQQ",
            ],
            "target": [
                0.01,
                -0.01,
            ],
        }
    )


def create_evaluation_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [
                "SPY",
                "SPY",
                "QQQ",
                "QQQ",
            ],
            "date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-05",
                    "2024-01-08",
                ]
            ),
            "target_date": pd.to_datetime(
                [
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-08",
                    "2024-01-09",
                ]
            ),
            "target": [
                0.02,
                -0.01,
                -0.03,
                0.04,
            ],
        }
    )


def test_forecast_returns_detailed_result_and_metrics_per_ticker() -> None:
    training_dataset = create_training_dataset()
    evaluation_dataset = create_evaluation_dataset()

    model = StubForecastModel()
    service = ForecastService(
        model=model,
        evaluator=ForecastEvaluator(),
    )

    result = service.forecast(
        training_dataset=training_dataset,
        evaluation_dataset=evaluation_dataset,
    )

    assert isinstance(result, ForecastResult)
    assert model.training_dataset is training_dataset

    assert result.observations.columns.tolist() == [
        "ticker",
        "date",
        "target_date",
        "actual",
        "prediction",
        "absolute_error",
        "actual_direction",
        "predicted_direction",
        "direction_correct",
    ]

    expected_predictions = pd.Series(
        [
            0.01,
            -0.02,
            0.01,
            0.03,
        ],
        name="prediction",
        dtype=float,
    )
    expected_absolute_errors = pd.Series(
        [
            0.01,
            0.01,
            0.04,
            0.01,
        ],
        name="absolute_error",
        dtype=float,
    )

    pdt.assert_series_equal(
        result.predictions,
        expected_predictions,
    )
    pdt.assert_series_equal(
        result.observations["absolute_error"],
        expected_absolute_errors,
        check_exact=False,
        atol=1e-12,
    )

    assert result.observations["actual_direction"].tolist() == [
        "up",
        "down",
        "down",
        "up",
    ]
    assert result.observations["predicted_direction"].tolist() == [
        "up",
        "down",
        "up",
        "up",
    ]
    assert result.observations["direction_correct"].tolist() == [
        True,
        True,
        False,
        True,
    ]

    assert result.overall_metrics.observations == 4
    assert result.overall_metrics.mae == pytest.approx(0.0175)
    assert result.overall_metrics.directional_accuracy == pytest.approx(0.75)

    assert set(result.metrics_by_ticker) == {
        "QQQ",
        "SPY",
    }
    assert result.metrics_by_ticker["SPY"].observations == 2
    assert result.metrics_by_ticker["SPY"].directional_accuracy == pytest.approx(1.0)
    assert result.metrics_by_ticker["QQQ"].observations == 2
    assert result.metrics_by_ticker["QQQ"].directional_accuracy == pytest.approx(0.5)


class StubOnlineForecastModel(BaseOnlineForecastModel):
    def __init__(self) -> None:
        self.last_targets: dict[str, float] = {}
        self.events: list[tuple[str, str, float]] = []

    def fit(
        self,
        training_dataset: pd.DataFrame,
    ) -> None:
        self.last_targets = {
            str(ticker): float(ticker_data.iloc[-1]["target"])
            for ticker, ticker_data in training_dataset.groupby(
                "ticker",
                sort=False,
            )
        }

    def predict(
        self,
        dataset: pd.DataFrame,
    ) -> pd.Series:
        assert len(dataset) == 1
        assert "target" not in dataset.columns

        ticker = str(dataset.iloc[0]["ticker"])
        prediction = self.last_targets[ticker]

        self.events.append(
            (
                "predict",
                ticker,
                prediction,
            )
        )

        return pd.Series(
            [prediction],
            index=dataset.index,
            dtype=float,
        )

    def update(
        self,
        observed_dataset: pd.DataFrame,
    ) -> None:
        assert len(observed_dataset) == 1
        assert "target" in observed_dataset.columns

        ticker = str(observed_dataset.iloc[0]["ticker"])
        actual = float(observed_dataset.iloc[0]["target"])

        self.events.append(
            (
                "update",
                ticker,
                actual,
            )
        )
        self.last_targets[ticker] = actual


def test_forecast_updates_online_model_only_after_each_prediction() -> None:
    training_dataset = create_training_dataset()
    evaluation_dataset = create_evaluation_dataset()

    model = StubOnlineForecastModel()
    service = ForecastService(
        model=model,
        evaluator=ForecastEvaluator(),
    )

    result = service.forecast(
        training_dataset=training_dataset,
        evaluation_dataset=evaluation_dataset,
    )

    expected_predictions = pd.Series(
        [
            0.01,
            0.02,
            -0.01,
            -0.03,
        ],
        name="prediction",
        dtype=float,
    )

    pdt.assert_series_equal(
        result.predictions,
        expected_predictions,
    )

    assert model.events == [
        ("predict", "SPY", 0.01),
        ("update", "SPY", 0.02),
        ("predict", "SPY", 0.02),
        ("update", "SPY", -0.01),
        ("predict", "QQQ", -0.01),
        ("update", "QQQ", -0.03),
        ("predict", "QQQ", -0.03),
        ("update", "QQQ", 0.04),
    ]

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.forecast.forecast_evaluator import ForecastEvaluator
from backend.src.forecast.forecast_pipeline import ForecastPipeline
from backend.src.forecast.forecast_result import ForecastResult
from backend.src.forecast.forecast_service import ForecastService
from backend.src.models.naive_forecast_model import NaiveForecastModel
from backend.src.storage.market_dataset_reader import MarketDatasetReader


def write_market_datasets(directory: Path) -> None:
    training_dataset = pd.DataFrame(
        {
            "ticker": [
                "SPY",
                "SPY",
                "QQQ",
                "QQQ",
            ],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "target_date": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "target": [
                0.01,
                -0.01,
                0.005,
                -0.005,
            ],
        }
    )
    validation_dataset = pd.DataFrame(
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
    test_dataset = pd.DataFrame(
        {
            "ticker": [
                "SPY",
                "QQQ",
            ],
            "date": pd.to_datetime(
                [
                    "2024-01-10",
                    "2024-01-10",
                ]
            ),
            "target_date": pd.to_datetime(
                [
                    "2024-01-11",
                    "2024-01-11",
                ]
            ),
            "target": [
                0.50,
                -0.50,
            ],
        }
    )

    training_dataset.to_csv(
        directory / "train.csv",
        index=False,
    )
    validation_dataset.to_csv(
        directory / "validation.csv",
        index=False,
    )
    test_dataset.to_csv(
        directory / "test.csv",
        index=False,
    )


def test_forecast_pipeline_runs_complete_naive_forecast_flow(
    tmp_path: Path,
) -> None:
    write_market_datasets(tmp_path)

    pipeline = ForecastPipeline(
        dataset_reader=MarketDatasetReader(
            input_directory=tmp_path,
        ),
        forecast_service=ForecastService(
            model=NaiveForecastModel(),
            evaluator=ForecastEvaluator(),
        ),
    )

    result = pipeline.run()

    assert isinstance(result, ForecastResult)

    expected_predictions = pd.Series(
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
        name="prediction",
    )

    pdt.assert_series_equal(
        result.predictions.reset_index(drop=True),
        expected_predictions,
    )

    expected_columns = [
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

    assert result.observations.columns.tolist() == expected_columns
    assert result.observations["ticker"].tolist() == [
        "SPY",
        "SPY",
        "QQQ",
        "QQQ",
    ]
    assert result.observations["actual"].tolist() == [
        0.02,
        -0.01,
        -0.03,
        0.04,
    ]
    assert result.observations["actual_direction"].tolist() == [
        "up",
        "down",
        "down",
        "up",
    ]
    assert result.observations["predicted_direction"].tolist() == [
        "flat",
        "flat",
        "flat",
        "flat",
    ]
    assert not result.observations["direction_correct"].any()

    metrics = result.overall_metrics

    assert metrics.observations == 4
    assert metrics.mae == pytest.approx(0.025)
    assert metrics.mse == pytest.approx(0.00075)
    assert metrics.rmse == pytest.approx(0.0273861279)
    assert metrics.r2 == pytest.approx(-1 / 29)
    assert metrics.directional_accuracy == pytest.approx(0.0)

    assert set(result.metrics_by_ticker) == {
        "QQQ",
        "SPY",
    }
    assert result.metrics_by_ticker["SPY"].observations == 2
    assert result.metrics_by_ticker["SPY"].mae == pytest.approx(0.015)
    assert result.metrics_by_ticker["QQQ"].observations == 2
    assert result.metrics_by_ticker["QQQ"].mae == pytest.approx(0.035)

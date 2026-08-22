import json
from pathlib import Path

import pandas as pd
import pytest

from backend.src.forecast.forecast_exporter import (
    ForecastExportPaths,
    export_forecast_result,
)
from backend.src.forecast.forecast_metrics import ForecastMetrics
from backend.src.forecast.forecast_result import ForecastResult


def create_forecast_result() -> ForecastResult:
    overall_metrics = ForecastMetrics(
        observations=2,
        mae=0.015,
        mse=0.00025,
        rmse=0.015811,
        r2=0.0,
        directional_accuracy=0.5,
    )
    spy_metrics = ForecastMetrics(
        observations=1,
        mae=0.01,
        mse=0.0001,
        rmse=0.01,
        r2=0.0,
        directional_accuracy=1.0,
    )
    qqq_metrics = ForecastMetrics(
        observations=1,
        mae=0.02,
        mse=0.0004,
        rmse=0.02,
        r2=0.0,
        directional_accuracy=0.0,
    )

    observations = pd.DataFrame(
        {
            "ticker": ["SPY", "QQQ"],
            "date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-05",
                ]
            ),
            "target_date": pd.to_datetime(
                [
                    "2024-01-08",
                    "2024-01-08",
                ]
            ),
            "actual": [0.02, -0.02],
            "prediction": [0.01, 0.0],
            "absolute_error": [0.01, 0.02],
            "actual_direction": ["up", "down"],
            "predicted_direction": ["up", "flat"],
            "direction_correct": [True, False],
        }
    )

    return ForecastResult(
        observations=observations,
        overall_metrics=overall_metrics,
        metrics_by_ticker={
            "SPY": spy_metrics,
            "QQQ": qqq_metrics,
        },
    )


def test_export_forecast_result_writes_csv_and_json_files(
    tmp_path: Path,
) -> None:
    result = create_forecast_result()

    paths = export_forecast_result(
        result=result,
        output_directory=tmp_path,
        model_name="naive_zero_return",
    )

    assert isinstance(paths, ForecastExportPaths)
    assert paths.observations_csv == tmp_path / "forecast_observations.csv"
    assert paths.metrics_csv == tmp_path / "forecast_metrics.csv"
    assert paths.result_json == tmp_path / "forecast_result.json"

    assert paths.observations_csv.is_file()
    assert paths.metrics_csv.is_file()
    assert paths.result_json.is_file()

    observations = pd.read_csv(paths.observations_csv)

    assert observations.columns.tolist() == result.observations.columns.tolist()
    assert observations["ticker"].tolist() == ["SPY", "QQQ"]
    assert observations["date"].tolist() == [
        "2024-01-05",
        "2024-01-05",
    ]
    assert observations["target_date"].tolist() == [
        "2024-01-08",
        "2024-01-08",
    ]

    metrics = pd.read_csv(
        paths.metrics_csv,
        keep_default_na=False,
    )

    assert metrics["scope"].tolist() == [
        "overall",
        "ticker",
        "ticker",
    ]
    assert metrics["ticker"].tolist() == [
        "",
        "QQQ",
        "SPY",
    ]
    assert metrics.loc[0, "observations"] == 2
    assert metrics.loc[0, "mae"] == pytest.approx(0.015)
    assert metrics.loc[0, "directional_accuracy"] == pytest.approx(0.5)

    payload = json.loads(paths.result_json.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["model_name"] == "naive_zero_return"
    assert payload["evaluation_split"] == "validation"
    assert payload["target_definition"] == "next_session_simple_return"
    assert payload["metrics"]["overall"]["observations"] == 2
    assert set(payload["metrics"]["by_ticker"]) == {
        "QQQ",
        "SPY",
    }
    assert payload["observations"][0]["date"] == "2024-01-05"
    assert payload["observations"][0]["target_date"] == "2024-01-08"

import json
from pathlib import Path

import pandas as pd

from backend.src.forecast.forecast_metrics import (
    ForecastMetrics,
)
from backend.src.forecast.forecast_pipeline import (
    ForecastPipeline,
)
from backend.src.forecast.forecast_result import (
    ForecastResult,
)
from backend.src.forecast.random_forest_cli import (
    RANDOM_FOREST_MODEL_NAME,
    RandomForestBenchmarkResult,
    build_forecast_pipeline,
    parse_arguments,
    run_random_forest_benchmark,
)
from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)


class StubForecastPipeline(ForecastPipeline):
    def __init__(
        self,
        result: ForecastResult,
    ) -> None:
        self.result = result
        self.was_called = False

    def run(self) -> ForecastResult:
        self.was_called = True

        return self.result


def create_forecast_result() -> ForecastResult:
    observations = pd.DataFrame(
        {
            "ticker": [
                "SPY",
                "QQQ",
            ],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "target_date": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "actual": [
                0.02,
                -0.01,
            ],
            "prediction": [
                0.01,
                0.005,
            ],
            "absolute_error": [
                0.01,
                0.015,
            ],
            "actual_direction": [
                1,
                -1,
            ],
            "predicted_direction": [
                1,
                1,
            ],
            "direction_correct": [
                True,
                False,
            ],
        }
    )
    overall_metrics = ForecastMetrics(
        observations=2,
        mae=0.0125,
        mse=0.0001625,
        rmse=0.01274755,
        r2=0.0,
        directional_accuracy=0.5,
    )
    metrics_by_ticker = {
        "QQQ": ForecastMetrics(
            observations=1,
            mae=0.015,
            mse=0.000225,
            rmse=0.015,
            r2=0.0,
            directional_accuracy=0.0,
        ),
        "SPY": ForecastMetrics(
            observations=1,
            mae=0.01,
            mse=0.0001,
            rmse=0.01,
            r2=0.0,
            directional_accuracy=1.0,
        ),
    }

    return ForecastResult(
        observations=observations,
        overall_metrics=overall_metrics,
        metrics_by_ticker=metrics_by_ticker,
    )


def test_parse_arguments_accepts_custom_directories(
    tmp_path: Path,
) -> None:
    dataset_directory = tmp_path / "datasets"
    output_directory = tmp_path / "benchmark"

    arguments = parse_arguments(
        [
            "--dataset-directory",
            str(dataset_directory),
            "--output-directory",
            str(output_directory),
        ]
    )

    assert arguments.dataset_directory == dataset_directory
    assert arguments.output_directory == output_directory


def test_build_pipeline_uses_random_forest_model(
    tmp_path: Path,
) -> None:
    pipeline = build_forecast_pipeline(
        dataset_directory=tmp_path,
    )

    assert isinstance(
        pipeline.forecast_service.model,
        RandomForestForecastModel,
    )
    assert pipeline.dataset_reader.input_directory == tmp_path


def test_run_benchmark_exports_validation_results(
    tmp_path: Path,
) -> None:
    expected_result = create_forecast_result()
    pipeline = StubForecastPipeline(
        result=expected_result,
    )

    benchmark = run_random_forest_benchmark(
        output_directory=tmp_path,
        pipeline=pipeline,
    )

    assert isinstance(
        benchmark,
        RandomForestBenchmarkResult,
    )
    assert pipeline.was_called
    assert benchmark.forecast_result is expected_result

    assert benchmark.export_paths.observations_csv.is_file()
    assert benchmark.export_paths.metrics_csv.is_file()
    assert benchmark.export_paths.result_json.is_file()

    payload = json.loads(
        benchmark.export_paths.result_json.read_text(
            encoding="utf-8",
        )
    )

    assert payload["model_name"] == RANDOM_FOREST_MODEL_NAME
    assert payload["evaluation_split"] == "validation"
    assert payload["metrics"]["overall"]["observations"] == 2

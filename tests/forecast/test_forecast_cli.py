from pathlib import Path

import pandas as pd

from backend.src.forecast.cli import (
    ForecastBenchmarkResult,
    parse_arguments,
    run_forecast_benchmark,
)
from backend.src.forecast.forecast_metrics import ForecastMetrics
from backend.src.forecast.forecast_pipeline import ForecastPipeline
from backend.src.forecast.forecast_result import ForecastResult


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
    metrics = ForecastMetrics(
        observations=2,
        mae=0.015,
        mse=0.00025,
        rmse=0.015811,
        r2=0.0,
        directional_accuracy=0.0,
    )

    return ForecastResult(
        observations=pd.DataFrame(
            {
                "ticker": ["SPY", "SPY"],
                "date": pd.to_datetime(
                    [
                        "2024-01-05",
                        "2024-01-08",
                    ]
                ),
                "target_date": pd.to_datetime(
                    [
                        "2024-01-08",
                        "2024-01-09",
                    ]
                ),
                "actual": [0.02, -0.01],
                "prediction": [0.0, 0.0],
                "absolute_error": [0.02, 0.01],
                "actual_direction": ["up", "down"],
                "predicted_direction": ["flat", "flat"],
                "direction_correct": [False, False],
            }
        ),
        overall_metrics=metrics,
        metrics_by_ticker={
            "SPY": metrics,
        },
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


def test_run_forecast_benchmark_exports_validation_result(
    tmp_path: Path,
) -> None:
    expected_result = create_forecast_result()
    pipeline = StubForecastPipeline(
        result=expected_result,
    )
    output_directory = tmp_path / "benchmark"

    benchmark = run_forecast_benchmark(
        dataset_directory=tmp_path / "datasets",
        output_directory=output_directory,
        pipeline=pipeline,
    )

    assert isinstance(benchmark, ForecastBenchmarkResult)
    assert pipeline.was_called
    assert benchmark.forecast_result is expected_result
    assert benchmark.export_paths.observations_csv.is_file()
    assert benchmark.export_paths.metrics_csv.is_file()
    assert benchmark.export_paths.result_json.is_file()

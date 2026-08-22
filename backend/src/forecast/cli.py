import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.src.config import MARKET_DATASETS_DIR, PROJECT_ROOT
from backend.src.forecast.forecast_evaluator import ForecastEvaluator
from backend.src.forecast.forecast_exporter import (
    ForecastExportPaths,
    export_forecast_result,
)
from backend.src.forecast.forecast_pipeline import ForecastPipeline
from backend.src.forecast.forecast_result import ForecastResult
from backend.src.forecast.forecast_service import ForecastService
from backend.src.models.naive_forecast_model import NaiveForecastModel
from backend.src.storage.market_dataset_reader import MarketDatasetReader

DEFAULT_FORECAST_BENCHMARK_DIR = (
    PROJECT_ROOT / "benchmark_results" / "forecast" / "naive"
)

NAIVE_MODEL_NAME = "naive_zero_return"


@dataclass(frozen=True, slots=True)
class ForecastBenchmarkResult:
    forecast_result: ForecastResult
    export_paths: ForecastExportPaths


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the zero-return naive forecast baseline "
            "on the validation market dataset."
        )
    )
    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=MARKET_DATASETS_DIR,
        help=(
            "Directory containing train.csv, validation.csv "
            f"and test.csv. Default: {MARKET_DATASETS_DIR}."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_FORECAST_BENCHMARK_DIR,
        help=(
            "Directory for validation benchmark results. "
            f"Default: {DEFAULT_FORECAST_BENCHMARK_DIR}."
        ),
    )

    return parser


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    return create_argument_parser().parse_args(arguments)


def build_forecast_pipeline(
    dataset_directory: Path,
) -> ForecastPipeline:
    return ForecastPipeline(
        dataset_reader=MarketDatasetReader(
            input_directory=dataset_directory,
        ),
        forecast_service=ForecastService(
            model=NaiveForecastModel(),
            evaluator=ForecastEvaluator(),
        ),
    )


def run_forecast_benchmark(
    dataset_directory: Path = MARKET_DATASETS_DIR,
    output_directory: Path = DEFAULT_FORECAST_BENCHMARK_DIR,
    pipeline: ForecastPipeline | None = None,
) -> ForecastBenchmarkResult:
    selected_pipeline = pipeline

    if selected_pipeline is None:
        selected_pipeline = build_forecast_pipeline(
            dataset_directory=dataset_directory,
        )

    forecast_result = selected_pipeline.run()

    export_paths = export_forecast_result(
        result=forecast_result,
        output_directory=output_directory,
        model_name=NAIVE_MODEL_NAME,
    )

    return ForecastBenchmarkResult(
        forecast_result=forecast_result,
        export_paths=export_paths,
    )


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    parsed_arguments = parse_arguments(arguments)

    benchmark = run_forecast_benchmark(
        dataset_directory=parsed_arguments.dataset_directory,
        output_directory=parsed_arguments.output_directory,
    )

    metrics = benchmark.forecast_result.overall_metrics

    print(f"Model: {NAIVE_MODEL_NAME}")
    print("Evaluation split: validation")
    print(f"Observations: {metrics.observations}")
    print(f"MAE: {metrics.mae:.8f}")
    print(f"MSE: {metrics.mse:.8f}")
    print(f"RMSE: {metrics.rmse:.8f}")
    print(f"R2: {metrics.r2:.8f}")
    print(f"Directional accuracy: {metrics.directional_accuracy:.4%}")
    print(f"Detailed observations: {benchmark.export_paths.observations_csv}")
    print(f"Metrics: {benchmark.export_paths.metrics_csv}")
    print(f"JSON result: {benchmark.export_paths.result_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

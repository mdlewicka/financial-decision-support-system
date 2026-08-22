import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.src.config import (
    MARKET_DATASETS_DIR,
    PROJECT_ROOT,
)
from backend.src.forecast.forecast_evaluator import (
    ForecastEvaluator,
)
from backend.src.forecast.forecast_exporter import (
    ForecastExportPaths,
    export_forecast_result,
)
from backend.src.forecast.forecast_pipeline import (
    ForecastPipeline,
)
from backend.src.forecast.forecast_result import (
    ForecastResult,
)
from backend.src.forecast.forecast_service import (
    ForecastService,
)
from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)
from backend.src.storage.market_dataset_reader import (
    MarketDatasetReader,
)

DEFAULT_RANDOM_FOREST_BENCHMARK_DIR = (
    PROJECT_ROOT / "benchmark_results" / "forecast" / "random_forest"
)

RANDOM_FOREST_MODEL_NAME = "random_forest"


@dataclass(frozen=True, slots=True)
class RandomForestBenchmarkResult:
    forecast_result: ForecastResult
    export_paths: ForecastExportPaths


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the compact Random Forest model "
            "on the training market dataset and evaluate "
            "it on the validation dataset."
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
        default=DEFAULT_RANDOM_FOREST_BENCHMARK_DIR,
        help=(
            "Directory for Random Forest validation "
            f"results. Default: "
            f"{DEFAULT_RANDOM_FOREST_BENCHMARK_DIR}."
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
            model=RandomForestForecastModel(),
            evaluator=ForecastEvaluator(),
        ),
    )


def run_random_forest_benchmark(
    dataset_directory: Path = MARKET_DATASETS_DIR,
    output_directory: Path = (DEFAULT_RANDOM_FOREST_BENCHMARK_DIR),
    pipeline: ForecastPipeline | None = None,
) -> RandomForestBenchmarkResult:
    selected_pipeline = pipeline

    if selected_pipeline is None:
        selected_pipeline = build_forecast_pipeline(
            dataset_directory=dataset_directory,
        )

    forecast_result = selected_pipeline.run()

    export_paths = export_forecast_result(
        result=forecast_result,
        output_directory=output_directory,
        model_name=RANDOM_FOREST_MODEL_NAME,
    )

    return RandomForestBenchmarkResult(
        forecast_result=forecast_result,
        export_paths=export_paths,
    )


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    parsed_arguments = parse_arguments(arguments)

    benchmark = run_random_forest_benchmark(
        dataset_directory=(parsed_arguments.dataset_directory),
        output_directory=(parsed_arguments.output_directory),
    )

    metrics = benchmark.forecast_result.overall_metrics

    print(f"Model: {RANDOM_FOREST_MODEL_NAME}")
    print("Training split: training")
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

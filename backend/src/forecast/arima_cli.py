import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.src.config import MARKET_DATASETS_DIR, PROJECT_ROOT
from backend.src.forecast.arima_order_selection import (
    DEFAULT_ARIMA_CANDIDATE_ORDERS,
    ArimaOrder,
    ArimaOrderSelectionResult,
    select_arima_orders,
)
from backend.src.forecast.forecast_evaluator import ForecastEvaluator
from backend.src.forecast.forecast_exporter import (
    ForecastExportPaths,
    export_forecast_result,
)
from backend.src.forecast.forecast_result import ForecastResult
from backend.src.forecast.forecast_service import ForecastService
from backend.src.models.arima_forecast_model import ArimaForecastModel
from backend.src.storage.market_dataset_reader import MarketDatasetReader

DEFAULT_ARIMA_BENCHMARK_DIR = PROJECT_ROOT / "benchmark_results" / "forecast" / "arima"

ARIMA_MODEL_NAME = "arima_walk_forward"

ARIMA_SELECTION_SCORES_FILENAME = "arima_order_selection_scores.csv"
ARIMA_SELECTION_SUMMARY_FILENAME = "arima_order_selection.json"


@dataclass(frozen=True, slots=True)
class ArimaBenchmarkResult:
    forecast_result: ForecastResult
    order_selection: ArimaOrderSelectionResult
    export_paths: ForecastExportPaths
    selection_scores_csv: Path
    selection_summary_json: Path


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select ticker-specific ARIMA orders on the training dataset "
            "and evaluate walk-forward forecasts on the validation dataset."
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
        default=DEFAULT_ARIMA_BENCHMARK_DIR,
        help=(
            "Directory for ARIMA validation benchmark results. "
            f"Default: {DEFAULT_ARIMA_BENCHMARK_DIR}."
        ),
    )
    parser.add_argument(
        "--criterion",
        choices=(
            "aic",
            "bic",
        ),
        default="aic",
        help="Information criterion used for training-only order selection.",
    )
    parser.add_argument(
        "--minimum-observations",
        type=_positive_integer,
        default=30,
        help=(
            "Minimum number of training observations required "
            "for every ticker. Default: 30."
        ),
    )

    return parser


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    return create_argument_parser().parse_args(arguments)


def run_arima_benchmark(
    dataset_directory: Path = MARKET_DATASETS_DIR,
    output_directory: Path = DEFAULT_ARIMA_BENCHMARK_DIR,
    candidate_orders: tuple[
        ArimaOrder,
        ...,
    ] = DEFAULT_ARIMA_CANDIDATE_ORDERS,
    criterion: str = "aic",
    minimum_observations: int = 30,
) -> ArimaBenchmarkResult:
    dataset_reader = MarketDatasetReader(
        input_directory=dataset_directory,
    )
    (
        training_dataset,
        validation_dataset,
        _,
    ) = dataset_reader.read()

    order_selection = select_arima_orders(
        training_dataset=training_dataset,
        candidate_orders=candidate_orders,
        criterion=criterion,
    )

    forecast_service = ForecastService(
        model=ArimaForecastModel(
            order=order_selection.selected_orders,
            minimum_observations=minimum_observations,
        ),
        evaluator=ForecastEvaluator(),
    )
    forecast_result = forecast_service.forecast(
        training_dataset=training_dataset,
        evaluation_dataset=validation_dataset,
    )

    export_paths = export_forecast_result(
        result=forecast_result,
        output_directory=output_directory,
        model_name=ARIMA_MODEL_NAME,
    )
    (
        selection_scores_csv,
        selection_summary_json,
    ) = export_order_selection(
        result=order_selection,
        output_directory=output_directory,
        criterion=criterion,
    )

    return ArimaBenchmarkResult(
        forecast_result=forecast_result,
        order_selection=order_selection,
        export_paths=export_paths,
        selection_scores_csv=selection_scores_csv,
        selection_summary_json=selection_summary_json,
    )


def export_order_selection(
    result: ArimaOrderSelectionResult,
    output_directory: Path,
    criterion: str,
) -> tuple[Path, Path]:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores_path = output_directory / ARIMA_SELECTION_SCORES_FILENAME
    summary_path = output_directory / ARIMA_SELECTION_SUMMARY_FILENAME

    result.scores.to_csv(
        scores_path,
        index=False,
    )

    candidate_scores = json.loads(
        result.scores.to_json(
            orient="records",
        )
    )
    selected_orders = {
        ticker: list(order)
        for ticker, order in sorted(
            result.selected_orders.items(),
        )
    }

    payload = {
        "schema_version": 1,
        "selection_dataset": "training",
        "criterion": criterion,
        "selected_orders": selected_orders,
        "candidate_scores": candidate_scores,
    }

    summary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return (
        scores_path,
        summary_path,
    )


def _positive_integer(
    value: str,
) -> int:
    parsed_value = int(value)

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            "minimum-observations must be greater than zero."
        )

    return parsed_value


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    parsed_arguments = parse_arguments(arguments)

    benchmark = run_arima_benchmark(
        dataset_directory=parsed_arguments.dataset_directory,
        output_directory=parsed_arguments.output_directory,
        criterion=parsed_arguments.criterion,
        minimum_observations=parsed_arguments.minimum_observations,
    )

    metrics = benchmark.forecast_result.overall_metrics

    print(f"Model: {ARIMA_MODEL_NAME}")
    print("Order-selection dataset: training")
    print(f"Selection criterion: {parsed_arguments.criterion.upper()}")
    print("Selected orders:")

    for ticker, order in sorted(
        benchmark.order_selection.selected_orders.items(),
    ):
        print(f"  {ticker}: {order}")

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
    print(f"Order-selection scores: {benchmark.selection_scores_csv}")
    print(f"Order-selection summary: {benchmark.selection_summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

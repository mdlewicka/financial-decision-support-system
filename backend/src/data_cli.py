import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from backend.src.config import (
    DATA_INTERVAL,
    HISTORY_START_DATE,
    MARKET_DATASETS_DIR,
    MINIMUM_MODEL_OBSERVATIONS_PER_TICKER,
    PROJECT_ROOT,
    SUPPORTED_TICKERS,
)
from backend.src.pipelines.market_data_pipeline import MarketDataPipeline
from backend.src.pipelines.market_dataset_pipeline import MarketDatasetPipeline
from backend.src.services.historical_market_data_service import (
    HistoricalMarketDataService,
)
from backend.src.services.market_data_validator import MarketDataValidator
from backend.src.services.market_dataset_builder import MarketDatasetBuilder
from backend.src.services.market_dataset_preparation_service import (
    MarketDatasetPreparationService,
)
from backend.src.services.market_dataset_splitter import MarketDatasetSplitter
from backend.src.services.market_target_builder import MarketTargetBuilder
from backend.src.storage.market_dataset_writer import MarketDatasetWriter

DEFAULT_DATA_MANIFEST_PATH = MARKET_DATASETS_DIR / "manifest.json"


@dataclass(frozen=True, slots=True)
class DataWorkflowResult:
    processed_data_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Date must use the YYYY-MM-DD format."
        ) from error


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download market data and prepare chronological "
            "train, validation and test datasets."
        )
    )
    parser.add_argument(
        "--start-date",
        type=parse_iso_date,
        default=HISTORY_START_DATE,
        help=(
            "First requested market date in YYYY-MM-DD format. "
            f"Default: {HISTORY_START_DATE.isoformat()}."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=parse_iso_date,
        required=True,
        help=(
            "Exclusive end date in YYYY-MM-DD format. "
            "This argument is required for reproducibility."
        ),
    )
    parser.add_argument(
        "--minimum-observations",
        type=int,
        default=MINIMUM_MODEL_OBSERVATIONS_PER_TICKER,
        help=(
            "Minimum number of observations required for every ticker. "
            f"Default: {MINIMUM_MODEL_OBSERVATIONS_PER_TICKER}."
        ),
    )

    return parser


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = create_argument_parser()
    parsed_arguments = parser.parse_args(arguments)

    if parsed_arguments.end_date <= parsed_arguments.start_date:
        parser.error("--end-date must be later than --start-date.")

    if parsed_arguments.minimum_observations <= 0:
        parser.error("--minimum-observations must be greater than zero.")

    return parsed_arguments


def build_market_data_pipeline(
    minimum_observations: int,
) -> MarketDataPipeline:
    historical_data_service = HistoricalMarketDataService(
        validator=MarketDataValidator(
            minimum_observations=minimum_observations,
        )
    )

    return MarketDataPipeline(
        historical_data_service=historical_data_service,
    )


def build_market_dataset_pipeline() -> MarketDatasetPipeline:
    preparation_service = MarketDatasetPreparationService(
        dataset_builder=MarketDatasetBuilder(),
        target_builder=MarketTargetBuilder(),
        dataset_splitter=MarketDatasetSplitter(),
        dataset_writer=MarketDatasetWriter(),
    )

    return MarketDatasetPipeline(
        preparation_service=preparation_service,
    )


def run_data_workflow(
    start_date: date,
    end_date: date,
    minimum_observations: int,
    market_data_pipeline: MarketDataPipeline | None = None,
    dataset_pipeline: MarketDatasetPipeline | None = None,
) -> DataWorkflowResult:
    selected_market_data_pipeline = market_data_pipeline

    if selected_market_data_pipeline is None:
        selected_market_data_pipeline = build_market_data_pipeline(
            minimum_observations=minimum_observations,
        )

    selected_dataset_pipeline = dataset_pipeline

    if selected_dataset_pipeline is None:
        selected_dataset_pipeline = build_market_dataset_pipeline()

    processed_data_path = selected_market_data_pipeline.run(
        tickers=SUPPORTED_TICKERS,
        start_date=start_date,
        end_date=end_date,
    )

    (
        train_path,
        validation_path,
        test_path,
    ) = selected_dataset_pipeline.run(
        processed_data_path=processed_data_path,
    )

    return DataWorkflowResult(
        processed_data_path=processed_data_path,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
    )


def _display_path(path: Path) -> str:
    resolved_path = path.resolve()

    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _summarize_dataset(path: Path) -> dict[str, object]:
    dataset = pd.read_csv(path)

    required_columns = {
        "date",
        "ticker",
    }
    missing_columns = sorted(required_columns.difference(dataset.columns))

    if missing_columns:
        joined_columns = ", ".join(missing_columns)
        raise ValueError(f"Cannot summarize {path}: missing columns: {joined_columns}.")

    if dataset.empty:
        raise ValueError(f"Cannot summarize an empty dataset: {path}.")

    dataset["date"] = pd.to_datetime(
        dataset["date"],
        errors="raise",
    )

    rows_per_ticker = {
        str(ticker): int(count)
        for ticker, count in (
            dataset["ticker"].astype(str).value_counts().sort_index().items()
        )
    }

    minimum_date = pd.Timestamp(dataset["date"].min())
    maximum_date = pd.Timestamp(dataset["date"].max())

    summary: dict[str, object] = {
        "path": _display_path(path),
        "rows": int(len(dataset)),
        "rows_per_ticker": rows_per_ticker,
        "date_min": minimum_date.date().isoformat(),
        "date_max": maximum_date.date().isoformat(),
    }

    if "target_date" in dataset.columns:
        if dataset["target_date"].isna().any():
            raise ValueError(
                f"Cannot summarize {path}: target_date contains missing values."
            )

        target_dates = pd.to_datetime(
            dataset["target_date"],
            errors="raise",
        )
        minimum_target_date = pd.Timestamp(target_dates.min())
        maximum_target_date = pd.Timestamp(target_dates.max())

        summary["target_date_min"] = minimum_target_date.date().isoformat()
        summary["target_date_max"] = maximum_target_date.date().isoformat()

    return summary


def write_data_manifest(
    result: DataWorkflowResult,
    start_date: date,
    end_date: date,
    minimum_observations: int,
    output_path: Path = DEFAULT_DATA_MANIFEST_PATH,
) -> Path:
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "request": {
            "start_date": start_date.isoformat(),
            "end_date_exclusive": end_date.isoformat(),
            "interval": DATA_INTERVAL,
            "minimum_observations_per_ticker": minimum_observations,
            "tickers": list(SUPPORTED_TICKERS),
        },
        "target": {
            "column": "target",
            "definition": "next_session_simple_return",
            "target_date_column": "target_date",
        },
        "outputs": {
            "processed": _summarize_dataset(result.processed_data_path),
            "train": _summarize_dataset(result.train_path),
            "validation": _summarize_dataset(result.validation_path),
            "test": _summarize_dataset(result.test_path),
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    parsed_arguments = parse_arguments(arguments)

    result = run_data_workflow(
        start_date=parsed_arguments.start_date,
        end_date=parsed_arguments.end_date,
        minimum_observations=parsed_arguments.minimum_observations,
    )

    manifest_path = write_data_manifest(
        result=result,
        start_date=parsed_arguments.start_date,
        end_date=parsed_arguments.end_date,
        minimum_observations=parsed_arguments.minimum_observations,
    )

    print(f"Processed data: {result.processed_data_path}")
    print(f"Training dataset: {result.train_path}")
    print(f"Validation dataset: {result.validation_path}")
    print(f"Test dataset: {result.test_path}")
    print(f"Data manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

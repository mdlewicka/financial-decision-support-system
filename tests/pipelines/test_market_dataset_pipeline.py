from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pandas.testing as pdt

from backend.src.pipelines.market_dataset_pipeline import (
    MarketDatasetPipeline,
)


def create_processed_data() -> pd.DataFrame:
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
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "adjusted_close": [
                100.0,
                101.0,
                200.0,
                202.0,
            ],
        }
    )


def test_run_returns_preparation_service_result(
    tmp_path: Path,
) -> None:
    preparation_service = Mock()

    expected = (
        Path("train.csv"),
        Path("validation.csv"),
        Path("test.csv"),
    )

    preparation_service.prepare.return_value = expected

    processed_data_path = tmp_path / "processed_market_data.csv"

    create_processed_data().to_csv(
        processed_data_path,
        index=False,
    )

    pipeline = MarketDatasetPipeline(
        preparation_service=preparation_service,
    )

    result = pipeline.run(processed_data_path)

    assert result == expected


def test_run_reads_processed_csv(
    tmp_path: Path,
) -> None:
    preparation_service = Mock()

    preparation_service.prepare.return_value = (
        Path("train.csv"),
        Path("validation.csv"),
        Path("test.csv"),
    )

    expected_data = create_processed_data()

    processed_data_path = tmp_path / "processed_market_data.csv"

    expected_data.to_csv(
        processed_data_path,
        index=False,
    )

    pipeline = MarketDatasetPipeline(
        preparation_service=preparation_service,
    )

    pipeline.run(processed_data_path)

    data_passed_to_service = preparation_service.prepare.call_args.args[0]

    pdt.assert_frame_equal(
        data_passed_to_service,
        expected_data,
    )


def test_run_parses_date_column_as_datetime(
    tmp_path: Path,
) -> None:
    preparation_service = Mock()

    preparation_service.prepare.return_value = (
        Path("train.csv"),
        Path("validation.csv"),
        Path("test.csv"),
    )

    processed_data_path = tmp_path / "processed_market_data.csv"

    create_processed_data().to_csv(
        processed_data_path,
        index=False,
    )

    pipeline = MarketDatasetPipeline(
        preparation_service=preparation_service,
    )

    pipeline.run(processed_data_path)

    data_passed_to_service = preparation_service.prepare.call_args.args[0]

    assert pd.api.types.is_datetime64_any_dtype(data_passed_to_service["date"])


def test_run_calls_preparation_service_once(
    tmp_path: Path,
) -> None:
    preparation_service = Mock()

    preparation_service.prepare.return_value = (
        Path("train.csv"),
        Path("validation.csv"),
        Path("test.csv"),
    )

    processed_data_path = tmp_path / "processed_market_data.csv"

    create_processed_data().to_csv(
        processed_data_path,
        index=False,
    )

    pipeline = MarketDatasetPipeline(
        preparation_service=preparation_service,
    )

    pipeline.run(processed_data_path)

    preparation_service.prepare.assert_called_once()

from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from backend.src.config import (
    TEST_DATASET_FILENAME,
    TRAIN_DATASET_FILENAME,
    VALIDATION_DATASET_FILENAME,
)
from backend.src.storage.market_dataset_writer import (
    MarketDatasetWriter,
)


def create_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [
                "SPY",
                "SPY",
            ],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "adjusted_close": [
                100.0,
                101.0,
            ],
            "target": [
                101.0,
                102.0,
            ],
        }
    )


def test_write_creates_output_directory(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "datasets" / "market"

    writer = MarketDatasetWriter(
        output_directory=output_directory,
    )

    data = create_dataset()

    writer.write(
        train_data=data,
        validation_data=data,
        test_data=data,
    )

    assert output_directory.exists()
    assert output_directory.is_dir()


def test_write_creates_all_dataset_files(
    tmp_path: Path,
) -> None:
    writer = MarketDatasetWriter(
        output_directory=tmp_path,
    )

    data = create_dataset()

    writer.write(
        train_data=data,
        validation_data=data,
        test_data=data,
    )

    assert (tmp_path / TRAIN_DATASET_FILENAME).exists()

    assert (tmp_path / VALIDATION_DATASET_FILENAME).exists()

    assert (tmp_path / TEST_DATASET_FILENAME).exists()


def test_write_returns_dataset_paths(
    tmp_path: Path,
) -> None:
    writer = MarketDatasetWriter(
        output_directory=tmp_path,
    )

    data = create_dataset()

    result = writer.write(
        train_data=data,
        validation_data=data,
        test_data=data,
    )

    expected = (
        tmp_path / TRAIN_DATASET_FILENAME,
        tmp_path / VALIDATION_DATASET_FILENAME,
        tmp_path / TEST_DATASET_FILENAME,
    )

    assert result == expected


def test_write_does_not_save_dataframe_index(
    tmp_path: Path,
) -> None:
    writer = MarketDatasetWriter(
        output_directory=tmp_path,
    )

    data = create_dataset()
    data.index = [10, 20]

    train_path, _, _ = writer.write(
        train_data=data,
        validation_data=data,
        test_data=data,
    )

    saved_data = pd.read_csv(train_path)

    assert "Unnamed: 0" not in saved_data.columns
    assert saved_data.columns.tolist() == (data.columns.tolist())


def test_write_saves_expected_dataset_content(
    tmp_path: Path,
) -> None:
    writer = MarketDatasetWriter(
        output_directory=tmp_path,
    )

    train_data = create_dataset()

    validation_data = create_dataset().iloc[:1].copy()

    test_data = create_dataset().iloc[1:].copy()

    (
        train_path,
        validation_path,
        test_path,
    ) = writer.write(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
    )

    saved_train = pd.read_csv(
        train_path,
        parse_dates=["date"],
    )
    saved_validation = pd.read_csv(
        validation_path,
        parse_dates=["date"],
    )
    saved_test = pd.read_csv(
        test_path,
        parse_dates=["date"],
    )

    pdt.assert_frame_equal(
        saved_train,
        train_data.reset_index(drop=True),
    )
    pdt.assert_frame_equal(
        saved_validation,
        validation_data.reset_index(drop=True),
    )
    pdt.assert_frame_equal(
        saved_test,
        test_data.reset_index(drop=True),
    )

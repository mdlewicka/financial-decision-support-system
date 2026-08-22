from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from backend.src.storage.market_dataset_reader import MarketDatasetReader


def create_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [
                "SPY",
                "QQQ",
            ],
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
            "adjusted_close": [
                100.0,
                200.0,
            ],
            "target": [
                0.10,
                -0.05,
            ],
        }
    )


def write_training_and_validation_datasets(
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dataset = create_dataset()
    validation_dataset = create_dataset()

    train_dataset.to_csv(
        directory / "train.csv",
        index=False,
    )
    validation_dataset.to_csv(
        directory / "validation.csv",
        index=False,
    )

    return (
        train_dataset,
        validation_dataset,
    )


def write_datasets(
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dataset, validation_dataset = write_training_and_validation_datasets(
        directory
    )
    test_dataset = create_dataset()

    test_dataset.to_csv(
        directory / "test.csv",
        index=False,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def test_read_returns_three_dataframes(
    tmp_path: Path,
) -> None:
    write_datasets(tmp_path)

    reader = MarketDatasetReader(
        input_directory=tmp_path,
    )

    result = reader.read()

    assert len(result) == 3
    assert all(isinstance(dataset, pd.DataFrame) for dataset in result)


def test_read_returns_expected_datasets(
    tmp_path: Path,
) -> None:
    expected = write_datasets(tmp_path)

    reader = MarketDatasetReader(
        input_directory=tmp_path,
    )

    result = reader.read()

    for actual_dataset, expected_dataset in zip(
        result,
        expected,
        strict=True,
    ):
        pdt.assert_frame_equal(
            actual_dataset,
            expected_dataset,
        )


def test_read_parses_date_columns_as_datetime(
    tmp_path: Path,
) -> None:
    write_datasets(tmp_path)

    reader = MarketDatasetReader(
        input_directory=tmp_path,
    )

    datasets = reader.read()

    for dataset in datasets:
        assert pd.api.types.is_datetime64_any_dtype(dataset["date"])
        assert pd.api.types.is_datetime64_any_dtype(dataset["target_date"])


def test_read_training_and_validation_returns_two_dataframes(
    tmp_path: Path,
) -> None:
    write_training_and_validation_datasets(tmp_path)

    reader = MarketDatasetReader(
        input_directory=tmp_path,
    )

    result = reader.read_training_and_validation()

    assert len(result) == 2
    assert all(isinstance(dataset, pd.DataFrame) for dataset in result)


def test_read_training_and_validation_returns_expected_datasets(
    tmp_path: Path,
) -> None:
    expected = write_training_and_validation_datasets(tmp_path)

    reader = MarketDatasetReader(
        input_directory=tmp_path,
    )

    result = reader.read_training_and_validation()

    for actual_dataset, expected_dataset in zip(
        result,
        expected,
        strict=True,
    ):
        pdt.assert_frame_equal(
            actual_dataset,
            expected_dataset,
        )


def test_read_training_and_validation_does_not_require_test_dataset(
    tmp_path: Path,
) -> None:
    write_training_and_validation_datasets(tmp_path)

    assert not (tmp_path / "test.csv").exists()

    reader = MarketDatasetReader(
        input_directory=tmp_path,
    )

    train_dataset, validation_dataset = reader.read_training_and_validation()

    assert not train_dataset.empty
    assert not validation_dataset.empty
    assert pd.api.types.is_datetime64_any_dtype(train_dataset["date"])
    assert pd.api.types.is_datetime64_any_dtype(validation_dataset["target_date"])

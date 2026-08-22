from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pandas.testing as pdt

from backend.src.services.market_dataset_preparation_service import (
    MarketDatasetPreparationService,
)


def create_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [
                "SPY",
                "SPY",
                "SPY",
            ],
            "date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-09",
                ]
            ),
            "adjusted_close": [
                100.0,
                110.0,
                99.0,
            ],
        }
    )


def create_target_dataset() -> pd.DataFrame:
    dataset = create_dataset()

    dataset["target_date"] = pd.Series(
        [
            pd.Timestamp("2024-01-08"),
            pd.Timestamp("2024-01-09"),
            pd.NaT,
        ],
        dtype=dataset["date"].dtype,
    )
    dataset["target"] = [
        0.10,
        -0.10,
        None,
    ]

    return dataset


def test_prepare_returns_writer_result() -> None:
    builder = Mock()
    target_builder = Mock()
    splitter = Mock()
    writer = Mock()

    data = create_dataset()

    builder.build.return_value = data
    target_builder.build.return_value = create_target_dataset()

    train = pd.DataFrame({"a": [1]})
    validation = pd.DataFrame({"a": [2]})
    test = pd.DataFrame({"a": [3]})

    splitter.split.return_value = (
        train,
        validation,
        test,
    )

    expected = (
        Path("train.csv"),
        Path("validation.csv"),
        Path("test.csv"),
    )
    writer.write.return_value = expected

    service = MarketDatasetPreparationService(
        dataset_builder=builder,
        target_builder=target_builder,
        dataset_splitter=splitter,
        dataset_writer=writer,
    )

    result = service.prepare(data)

    assert result == expected


def test_prepare_calls_all_components() -> None:
    builder = Mock()
    target_builder = Mock()
    splitter = Mock()
    writer = Mock()

    data = create_dataset()

    builder.build.return_value = data
    target_builder.build.return_value = create_target_dataset()

    splitter.split.return_value = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    writer.write.return_value = (
        Path("train.csv"),
        Path("validation.csv"),
        Path("test.csv"),
    )

    service = MarketDatasetPreparationService(
        builder,
        target_builder,
        splitter,
        writer,
    )

    service.prepare(data)

    builder.build.assert_called_once()
    target_builder.build.assert_called_once()
    splitter.split.assert_called_once()
    writer.write.assert_called_once()


def test_prepare_removes_rows_without_complete_target() -> None:
    builder = Mock()
    target_builder = Mock()
    splitter = Mock()
    writer = Mock()

    data = create_dataset()
    dataset_with_target = create_target_dataset()

    # Symulacja niekompletnego rekordu: target istnieje, ale brakuje jego daty.
    dataset_with_target.loc[1, "target_date"] = pd.NaT

    builder.build.return_value = data
    target_builder.build.return_value = dataset_with_target

    splitter.split.return_value = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    writer.write.return_value = (
        Path("train.csv"),
        Path("validation.csv"),
        Path("test.csv"),
    )

    service = MarketDatasetPreparationService(
        builder,
        target_builder,
        splitter,
        writer,
    )

    service.prepare(data)

    dataset_passed = splitter.split.call_args.args[0]

    assert len(dataset_passed) == 1
    assert dataset_passed["target"].isna().sum() == 0
    assert dataset_passed["target_date"].isna().sum() == 0


def test_prepare_passes_split_result_to_writer() -> None:
    builder = Mock()
    target_builder = Mock()
    splitter = Mock()
    writer = Mock()

    data = create_dataset()

    builder.build.return_value = data
    target_builder.build.return_value = create_target_dataset()

    train = pd.DataFrame({"train": [1]})
    validation = pd.DataFrame({"validation": [2]})
    test = pd.DataFrame({"test": [3]})

    splitter.split.return_value = (
        train,
        validation,
        test,
    )
    writer.write.return_value = (
        Path("train.csv"),
        Path("validation.csv"),
        Path("test.csv"),
    )

    service = MarketDatasetPreparationService(
        builder,
        target_builder,
        splitter,
        writer,
    )

    service.prepare(data)

    writer.write.assert_called_once_with(
        train_data=train,
        validation_data=validation,
        test_data=test,
    )


def test_prepare_does_not_modify_input_dataframe() -> None:
    builder = Mock()
    target_builder = Mock()
    splitter = Mock()
    writer = Mock()

    data = create_dataset()
    original = data.copy(deep=True)

    builder.build.return_value = data
    target_builder.build.return_value = create_target_dataset()

    splitter.split.return_value = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    writer.write.return_value = (
        Path("train.csv"),
        Path("validation.csv"),
        Path("test.csv"),
    )

    service = MarketDatasetPreparationService(
        builder,
        target_builder,
        splitter,
        writer,
    )

    service.prepare(data)

    pdt.assert_frame_equal(
        data,
        original,
    )

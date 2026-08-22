from pathlib import Path

import pandas as pd
import pytest

from backend.src.storage.processed_market_data_writer import (
    ProcessedMarketDataWriter,
)


def test_write_creates_output_file(
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
            ],
            "daily_return": [
                0.01,
                0.02,
            ],
        }
    )

    writer = ProcessedMarketDataWriter(
        output_directory=tmp_path,
    )

    output_path = writer.write(data)

    assert output_path.exists()


def test_write_returns_output_path(
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "ticker": [
                "SPY",
            ],
            "daily_return": [
                0.01,
            ],
        }
    )

    writer = ProcessedMarketDataWriter(
        output_directory=tmp_path,
    )

    output_path = writer.write(data)

    assert output_path == (tmp_path / "market_features.csv")


def test_write_creates_output_directory(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "processed"

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "ticker": [
                "SPY",
            ],
            "daily_return": [
                0.01,
            ],
        }
    )

    writer = ProcessedMarketDataWriter(
        output_directory=output_directory,
    )

    writer.write(data)

    assert output_directory.exists()
    assert output_directory.is_dir()


def test_write_raises_value_error_for_empty_dataframe(
    tmp_path: Path,
) -> None:
    writer = ProcessedMarketDataWriter(
        output_directory=tmp_path,
    )

    empty_data = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Cannot write an empty DataFrame.",
    ):
        writer.write(empty_data)


def test_write_uses_custom_file_name(
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "ticker": [
                "SPY",
            ],
            "daily_return": [
                0.01,
            ],
        }
    )

    writer = ProcessedMarketDataWriter(
        output_directory=tmp_path,
    )

    output_path = writer.write(
        data,
        file_name="custom_features.csv",
    )

    assert output_path == (tmp_path / "custom_features.csv")
    assert output_path.exists()

from pathlib import Path

import pandas as pd
import pytest

from backend.src.storage.interim_market_data_writer import (
    InterimMarketDataWriter,
)


def test_write_creates_csv_file(tmp_path: Path) -> None:
    writer = InterimMarketDataWriter(tmp_path)

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
            ],
            "close": [
                593.0,
                594.0,
            ],
        }
    )

    output_path = writer.write(data)

    assert output_path.exists()
    assert output_path.name == "market_data.csv"


def test_write_returns_output_path(tmp_path: Path) -> None:
    writer = InterimMarketDataWriter(tmp_path)

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "ticker": ["SPY"],
            "close": [593.0],
        }
    )

    output_path = writer.write(data)

    assert output_path == tmp_path / "market_data.csv"


def test_write_creates_output_directory(tmp_path: Path) -> None:
    output_directory = tmp_path / "interim" / "market"

    writer = InterimMarketDataWriter(output_directory)

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "ticker": ["SPY"],
            "close": [593.0],
        }
    )

    writer.write(data)

    assert output_directory.exists()
    assert output_directory.is_dir()


def test_write_raises_value_error_for_empty_dataframe(
    tmp_path: Path,
) -> None:
    writer = InterimMarketDataWriter(tmp_path)

    empty_data = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Cannot write an empty DataFrame.",
    ):
        writer.write(empty_data)


def test_write_uses_custom_file_name(tmp_path: Path) -> None:
    writer = InterimMarketDataWriter(tmp_path)

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "ticker": ["SPY"],
            "close": [593.0],
        }
    )

    output_path = writer.write(
        data,
        file_name="custom.csv",
    )

    assert output_path.exists()
    assert output_path.name == "custom.csv"

from pathlib import Path

import pandas as pd
import pytest

from backend.src.storage.interim_market_data_reader import (
    InterimMarketDataReader,
)


def test_read_returns_market_data(tmp_path: Path) -> None:
    input_data = pd.DataFrame(
        {
            "date": [
                "2025-01-02",
                "2025-01-03",
            ],
            "ticker": [
                "SPY",
                "QQQ",
            ],
            "open": [
                590.0,
                500.0,
            ],
            "high": [
                595.0,
                505.0,
            ],
            "low": [
                589.0,
                499.0,
            ],
            "close": [
                593.0,
                503.0,
            ],
            "adjusted_close": [
                593.0,
                503.0,
            ],
            "volume": [
                1000,
                2000,
            ],
        }
    )

    input_data.to_csv(
        tmp_path / "market_data.csv",
        index=False,
    )

    reader = InterimMarketDataReader(
        data_directory=tmp_path,
    )

    result = reader.read()

    expected = input_data.copy()
    expected["date"] = pd.to_datetime(expected["date"])

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_read_parses_date_column(
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "ticker": ["SPY"],
        }
    )

    data.to_csv(
        tmp_path / "market_data.csv",
        index=False,
    )

    reader = InterimMarketDataReader(
        data_directory=tmp_path,
    )

    result = reader.read()

    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_read_uses_custom_file_name(
    tmp_path: Path,
) -> None:
    data = pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "ticker": ["SPY"],
        }
    )

    data.to_csv(
        tmp_path / "custom_market_data.csv",
        index=False,
    )

    reader = InterimMarketDataReader(
        data_directory=tmp_path,
    )

    result = reader.read(
        file_name="custom_market_data.csv",
    )

    assert len(result) == 1
    assert result.loc[0, "ticker"] == "SPY"


def test_read_raises_file_not_found_error(
    tmp_path: Path,
) -> None:
    reader = InterimMarketDataReader(
        data_directory=tmp_path,
    )

    expected_path = tmp_path / "market_data.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Market data file not found:",
    ) as error_info:
        reader.read()

    assert str(expected_path) in str(error_info.value)

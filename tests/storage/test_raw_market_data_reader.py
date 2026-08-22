from pathlib import Path

import pandas as pd
import pytest

from backend.src.storage.raw_market_data_reader import RawMarketDataReader


def test_read_returns_dataframe_with_parsed_date(tmp_path: Path) -> None:
    file_path = tmp_path / "SPY.csv"

    file_path.write_text(
        (
            "date,ticker,open,high,low,close,adjusted_close,volume\n"
            "2025-01-02,SPY,590.0,595.0,588.0,593.0,593.0,1000000\n"
            "2025-01-03,SPY,593.0,597.0,591.0,596.0,596.0,1200000\n"
        ),
        encoding="utf-8",
    )

    reader = RawMarketDataReader(data_directory=tmp_path)

    result = reader.read("SPY")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result["ticker"].tolist() == ["SPY", "SPY"]


def test_read_uses_expected_file_name(tmp_path: Path) -> None:
    file_path = tmp_path / "QQQ.csv"

    file_path.write_text(
        (
            "date,ticker,open,high,low,close,adjusted_close,volume\n"
            "2025-01-02,QQQ,510.0,515.0,508.0,513.0,513.0,900000\n"
        ),
        encoding="utf-8",
    )

    reader = RawMarketDataReader(data_directory=tmp_path)

    result = reader.read("QQQ")

    assert len(result) == 1
    assert result.loc[0, "ticker"] == "QQQ"


def test_read_raises_file_not_found_error_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    reader = RawMarketDataReader(data_directory=tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="Market data file not found",
    ):
        reader.read("SPY")


def test_read_raises_value_error_when_date_column_is_missing(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "SPY.csv"

    file_path.write_text(
        (
            "ticker,open,high,low,close,adjusted_close,volume\n"
            "SPY,590.0,595.0,588.0,593.0,593.0,1000000\n"
        ),
        encoding="utf-8",
    )

    reader = RawMarketDataReader(data_directory=tmp_path)

    with pytest.raises(ValueError):
        reader.read("SPY")


def test_read_returns_empty_dataframe_for_csv_with_only_header(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "SPY.csv"

    file_path.write_text(
        "date,ticker,open,high,low,close,adjusted_close,volume\n",
        encoding="utf-8",
    )

    reader = RawMarketDataReader(data_directory=tmp_path)

    result = reader.read("SPY")

    assert result.empty
    assert list(result.columns) == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]

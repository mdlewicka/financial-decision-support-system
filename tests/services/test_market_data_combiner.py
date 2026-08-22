import pandas as pd
import pytest

from backend.src.services.market_data_combiner import (
    MarketDataCombiner,
)


def test_combine_returns_single_dataframe() -> None:
    spy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "ticker": ["SPY"],
            "close": [593.0],
        }
    )

    qqq = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "ticker": ["QQQ"],
            "close": [512.0],
        }
    )

    combiner = MarketDataCombiner()

    result = combiner.combine(
        {
            "SPY": spy,
            "QQQ": qqq,
        }
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_combine_sorts_by_ticker_and_date() -> None:
    spy = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-03",
                    "2025-01-02",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
            ],
            "close": [594.0, 593.0],
        }
    )

    qqq = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "ticker": ["QQQ"],
            "close": [512.0],
        }
    )

    combiner = MarketDataCombiner()

    result = combiner.combine(
        {
            "SPY": spy,
            "QQQ": qqq,
        }
    )

    assert result["ticker"].tolist() == [
        "QQQ",
        "SPY",
        "SPY",
    ]

    assert result["date"].tolist() == [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
    ]


def test_combine_resets_index() -> None:
    spy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "ticker": ["SPY"],
            "close": [593.0],
        },
        index=[10],
    )

    qqq = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "ticker": ["QQQ"],
            "close": [512.0],
        },
        index=[20],
    )

    combiner = MarketDataCombiner()

    result = combiner.combine(
        {
            "SPY": spy,
            "QQQ": qqq,
        }
    )

    assert list(result.index) == [0, 1]


def test_combine_raises_value_error_for_empty_input() -> None:
    combiner = MarketDataCombiner()

    with pytest.raises(
        ValueError,
        match="At least one DataFrame must be provided.",
    ):
        combiner.combine({})


def test_combine_preserves_all_rows() -> None:
    spy = pd.DataFrame(
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
            "close": [593.0, 594.0],
        }
    )

    qqq = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "ticker": [
                "QQQ",
                "QQQ",
            ],
            "close": [512.0, 513.0],
        }
    )

    combiner = MarketDataCombiner()

    result = combiner.combine(
        {
            "SPY": spy,
            "QQQ": qqq,
        }
    )

    assert len(result) == 4

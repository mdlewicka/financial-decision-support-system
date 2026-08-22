import pandas as pd

from backend.src.services.market_data_cleaner import (
    MarketDataCleaner,
)


def test_clean_returns_copy_of_dataframe() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "ticker": ["SPY"],
            "open": [590.0],
            "high": [595.0],
            "low": [588.0],
            "close": [593.0],
            "adjusted_close": [593.0],
            "volume": [1_000_000],
        }
    )

    cleaner = MarketDataCleaner()

    result = cleaner.clean(data)

    assert result is not data
    pd.testing.assert_frame_equal(result, data)


def test_clean_removes_duplicate_rows() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "ticker": ["SPY", "SPY"],
            "open": [590.0, 590.0],
            "high": [595.0, 595.0],
            "low": [588.0, 588.0],
            "close": [593.0, 593.0],
            "adjusted_close": [593.0, 593.0],
            "volume": [1_000_000, 1_000_000],
        }
    )

    cleaner = MarketDataCleaner()

    result = cleaner.clean(data)

    assert len(result) == 1


def test_clean_removes_fully_empty_rows() -> None:
    data = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-02"),
                "ticker": "SPY",
                "open": 590.0,
                "high": 595.0,
                "low": 588.0,
                "close": 593.0,
                "adjusted_close": 593.0,
                "volume": 1_000_000,
            },
            {
                "date": None,
                "ticker": None,
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "adjusted_close": None,
                "volume": None,
            },
        ]
    )

    cleaner = MarketDataCleaner()

    result = cleaner.clean(data)

    assert len(result) == 1


def test_clean_sorts_by_ticker_and_date() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-03",
                    "2025-01-02",
                    "2025-01-02",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
                "QQQ",
            ],
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "adjusted_close": [1, 2, 3],
            "volume": [1, 2, 3],
        }
    )

    cleaner = MarketDataCleaner()

    result = cleaner.clean(data)

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


def test_clean_resets_index() -> None:
    data = pd.DataFrame(
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
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "adjusted_close": [1, 2],
            "volume": [1, 2],
        },
        index=[10, 20],
    )

    cleaner = MarketDataCleaner()

    result = cleaner.clean(data)

    assert list(result.index) == [0, 1]

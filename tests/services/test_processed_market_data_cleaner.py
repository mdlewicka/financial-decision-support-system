import pandas as pd

from backend.src.services.processed_market_data_cleaner import (
    ProcessedMarketDataCleaner,
)


def test_clean_removes_rows_with_missing_values() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
                "SPY",
            ],
            "daily_return": [
                None,
                0.01,
                0.02,
            ],
            "moving_average_5": [
                None,
                None,
                101.0,
            ],
        }
    )

    cleaner = ProcessedMarketDataCleaner()

    result = cleaner.clean(data)

    assert len(result) == 1
    assert result.iloc[0]["date"] == pd.Timestamp("2025-01-03")


def test_clean_removes_duplicate_rows() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-03",
                    "2025-01-03",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
            ],
            "daily_return": [
                0.02,
                0.02,
            ],
            "moving_average_5": [
                101.0,
                101.0,
            ],
        }
    )

    cleaner = ProcessedMarketDataCleaner()

    result = cleaner.clean(data)

    assert len(result) == 1


def test_clean_sorts_data_by_ticker_and_date() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-03",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-02",
                ]
            ),
            "ticker": [
                "SPY",
                "QQQ",
                "QQQ",
                "SPY",
            ],
            "daily_return": [
                0.02,
                0.01,
                0.02,
                0.01,
            ],
            "moving_average_5": [
                101.0,
                201.0,
                202.0,
                102.0,
            ],
        }
    )

    cleaner = ProcessedMarketDataCleaner()

    result = cleaner.clean(data)

    expected = pd.DataFrame(
        {
            "ticker": [
                "QQQ",
                "QQQ",
                "SPY",
                "SPY",
            ],
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
        }
    )

    pd.testing.assert_frame_equal(
        result[
            [
                "ticker",
                "date",
            ]
        ],
        expected,
    )


def test_clean_resets_index() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
                "SPY",
            ],
            "daily_return": [
                None,
                0.01,
                0.02,
            ],
            "moving_average_5": [
                None,
                101.0,
                102.0,
            ],
        }
    )

    cleaner = ProcessedMarketDataCleaner()

    result = cleaner.clean(data)

    assert list(result.index) == [0, 1]


def test_clean_does_not_modify_input_dataframe() -> None:
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
                None,
                0.01,
            ],
            "moving_average_5": [
                None,
                101.0,
            ],
        }
    )

    original = data.copy(deep=True)

    cleaner = ProcessedMarketDataCleaner()

    cleaner.clean(data)

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_clean_removes_rows_with_infinite_values() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
                "SPY",
                "SPY",
            ],
            "daily_return": [
                0.01,
                float("inf"),
                float("-inf"),
                0.02,
            ],
            "moving_average_5": [
                100.0,
                101.0,
                102.0,
                103.0,
            ],
        }
    )

    cleaner = ProcessedMarketDataCleaner()

    result = cleaner.clean(data)

    assert result["daily_return"].tolist() == [
        0.01,
        0.02,
    ]

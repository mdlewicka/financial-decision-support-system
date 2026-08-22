import pandas as pd
import pandas.testing as pdt

from backend.src.services.market_dataset_builder import (
    MarketDatasetBuilder,
)


def test_build_returns_dataframe() -> None:
    builder = MarketDatasetBuilder()

    data = pd.DataFrame(
        {
            "ticker": ["SPY"],
            "date": pd.to_datetime(["2024-01-01"]),
            "adjusted_close": [100.0],
        }
    )

    result = builder.build(data)

    assert isinstance(result, pd.DataFrame)


def test_build_does_not_modify_input_dataframe() -> None:
    builder = MarketDatasetBuilder()

    data = pd.DataFrame(
        {
            "ticker": ["SPY"],
            "date": pd.to_datetime(["2024-01-01"]),
            "adjusted_close": [100.0],
        }
    )

    original = data.copy(deep=True)

    builder.build(data)

    pdt.assert_frame_equal(data, original)


def test_build_sorts_data_by_ticker_and_date() -> None:
    builder = MarketDatasetBuilder()

    data = pd.DataFrame(
        {
            "ticker": [
                "QQQ",
                "SPY",
                "QQQ",
            ],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-01",
                ]
            ),
            "adjusted_close": [
                10.0,
                20.0,
                15.0,
            ],
        }
    )

    result = builder.build(data)

    expected = pd.DataFrame(
        {
            "ticker": [
                "QQQ",
                "QQQ",
                "SPY",
            ],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                ]
            ),
            "adjusted_close": [
                15.0,
                10.0,
                20.0,
            ],
        }
    )

    pdt.assert_frame_equal(result, expected)


def test_build_resets_index() -> None:
    builder = MarketDatasetBuilder()

    data = pd.DataFrame(
        {
            "ticker": [
                "QQQ",
                "SPY",
            ],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-01",
                ]
            ),
            "adjusted_close": [
                10.0,
                20.0,
            ],
        },
        index=[10, 20],
    )

    result = builder.build(data)

    assert result.index.tolist() == [0, 1]

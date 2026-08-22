import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.services.market_dataset_splitter import MarketDatasetSplitter


def create_market_data(
    tickers: list[str],
    rows_per_ticker: int = 20,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        dates = pd.date_range(
            start=start_date,
            periods=rows_per_ticker + 1,
            freq="D",
        )

        frames.append(
            pd.DataFrame(
                {
                    "ticker": [ticker] * rows_per_ticker,
                    "date": dates[:-1],
                    "target_date": dates[1:],
                    "adjusted_close": [
                        float(value) for value in range(rows_per_ticker)
                    ],
                    "target": [0.0] * rows_per_ticker,
                }
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def test_split_returns_three_dataframes() -> None:
    splitter = MarketDatasetSplitter()
    data = create_market_data(["SPY"])

    train, validation, test = splitter.split(data)

    assert isinstance(train, pd.DataFrame)
    assert isinstance(validation, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_split_preserves_expected_proportions() -> None:
    splitter = MarketDatasetSplitter()
    data = create_market_data(
        ["SPY"],
        rows_per_ticker=20,
    )

    train, validation, test = splitter.split(data)

    assert len(train) == 14
    assert len(validation) == 3
    assert len(test) == 3


def test_split_preserves_chronological_order() -> None:
    splitter = MarketDatasetSplitter()
    data = create_market_data(["SPY"])

    train, validation, test = splitter.split(data)

    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()


def test_split_preserves_each_ticker() -> None:
    splitter = MarketDatasetSplitter()
    data = create_market_data(
        [
            "SPY",
            "QQQ",
        ],
        rows_per_ticker=20,
    )

    train, validation, test = splitter.split(data)

    assert train["ticker"].value_counts().to_dict() == {
        "QQQ": 14,
        "SPY": 14,
    }
    assert validation["ticker"].value_counts().to_dict() == {
        "QQQ": 3,
        "SPY": 3,
    }
    assert test["ticker"].value_counts().to_dict() == {
        "QQQ": 3,
        "SPY": 3,
    }


def test_split_uses_disjoint_target_date_periods() -> None:
    splitter = MarketDatasetSplitter()
    data = create_market_data(
        [
            "SPY",
            "QQQ",
        ]
    )

    train, validation, test = splitter.split(data)

    assert train["target_date"].max() < validation["target_date"].min()
    assert validation["target_date"].max() < test["target_date"].min()


def test_split_uses_shared_boundaries_for_different_histories() -> None:
    splitter = MarketDatasetSplitter()

    longer_history = create_market_data(
        ["SPY"],
        rows_per_ticker=20,
        start_date="2024-01-01",
    )
    shorter_history = create_market_data(
        ["QQQ"],
        rows_per_ticker=12,
        start_date="2024-01-09",
    )
    data = pd.concat(
        [
            longer_history,
            shorter_history,
        ],
        ignore_index=True,
    )

    train, validation, test = splitter.split(data)

    assert train["target_date"].max() < validation["target_date"].min()
    assert validation["target_date"].max() < test["target_date"].min()

    assert train.groupby("ticker")["target_date"].max().nunique() == 1
    assert validation.groupby("ticker")["target_date"].min().nunique() == 1
    assert validation.groupby("ticker")["target_date"].max().nunique() == 1
    assert test.groupby("ticker")["target_date"].min().nunique() == 1


def test_split_rejects_missing_target_date_column() -> None:
    splitter = MarketDatasetSplitter()
    data = create_market_data(["SPY"]).drop(
        columns=["target_date"],
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns: target_date",
    ):
        splitter.split(data)


def test_split_rejects_non_future_target_date() -> None:
    splitter = MarketDatasetSplitter()
    data = create_market_data(["SPY"])

    data.loc[0, "target_date"] = data.loc[0, "date"]

    with pytest.raises(
        ValueError,
        match="target_date must be later than date",
    ):
        splitter.split(data)


def test_split_does_not_modify_input_dataframe() -> None:
    splitter = MarketDatasetSplitter()
    data = create_market_data(
        [
            "SPY",
            "QQQ",
        ]
    )
    original = data.copy(deep=True)

    splitter.split(data)

    pdt.assert_frame_equal(
        data,
        original,
    )

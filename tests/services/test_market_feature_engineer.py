import numpy as np
import pandas as pd
import pytest

from backend.src.services.market_feature_engineer import (
    MarketFeatureEngineer,
)


def test_engineer_adds_expected_feature_columns() -> None:
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
            "open": [
                100.0,
                101.0,
            ],
            "high": [
                102.0,
                103.0,
            ],
            "low": [
                99.0,
                100.0,
            ],
            "close": [
                101.0,
                102.0,
            ],
            "adjusted_close": [
                101.0,
                102.0,
            ],
            "volume": [
                1000,
                1100,
            ],
        }
    )

    engineer = MarketFeatureEngineer()

    result = engineer.engineer(data)

    expected_feature_columns = {
        "daily_return",
        "log_return",
        "price_change",
        "volume_change",
        "moving_average_5",
        "moving_average_20",
        "volatility_5",
        "volatility_20",
        "momentum_5",
        "momentum_20",
    }

    assert expected_feature_columns.issubset(result.columns)


def test_engineer_does_not_modify_input_data() -> None:
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
            "adjusted_close": [
                100.0,
                101.0,
            ],
            "volume": [
                1000,
                1100,
            ],
        }
    )

    original_data = data.copy(deep=True)

    engineer = MarketFeatureEngineer()

    engineer.engineer(data)

    pd.testing.assert_frame_equal(
        data,
        original_data,
    )


def test_engineer_sorts_data_by_ticker_and_date() -> None:
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
            "adjusted_close": [
                102.0,
                200.0,
                202.0,
                100.0,
            ],
            "volume": [
                1200,
                2000,
                2200,
                1000,
            ],
        }
    )

    engineer = MarketFeatureEngineer()

    result = engineer.engineer(data)

    expected_order = pd.DataFrame(
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
        result[["ticker", "date"]],
        expected_order,
    )


def test_engineer_calculates_basic_features() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2025-01-01",
                periods=3,
            ),
            "ticker": [
                "SPY",
                "SPY",
                "SPY",
            ],
            "adjusted_close": [
                100.0,
                110.0,
                121.0,
            ],
            "volume": [
                1000,
                1200,
                900,
            ],
        }
    )

    engineer = MarketFeatureEngineer()

    result = engineer.engineer(data)

    assert pd.isna(result.loc[0, "daily_return"])
    assert result.loc[
        1,
        "daily_return",
    ] == pytest.approx(0.10)
    assert result.loc[
        2,
        "daily_return",
    ] == pytest.approx(0.10)

    assert pd.isna(result.loc[0, "log_return"])
    assert result.loc[
        1,
        "log_return",
    ] == pytest.approx(np.log(1.10))

    assert pd.isna(result.loc[0, "price_change"])
    assert result.loc[
        1,
        "price_change",
    ] == pytest.approx(10.0)
    assert result.loc[
        2,
        "price_change",
    ] == pytest.approx(11.0)

    assert pd.isna(result.loc[0, "volume_change"])
    assert result.loc[
        1,
        "volume_change",
    ] == pytest.approx(0.20)
    assert result.loc[
        2,
        "volume_change",
    ] == pytest.approx(-0.25)


def test_engineer_calculates_rolling_features() -> None:
    prices = [float(price) for price in range(100, 121)]

    data = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2025-01-01",
                periods=len(prices),
            ),
            "ticker": ["SPY"] * len(prices),
            "adjusted_close": prices,
            "volume": [1000 + index * 10 for index in range(len(prices))],
        }
    )

    engineer = MarketFeatureEngineer()

    result = engineer.engineer(data)

    expected_returns = pd.Series(
        prices,
    ).pct_change()

    expected_volatility_5 = (
        expected_returns.rolling(
            window=5,
        )
        .std()
        .iloc[5]
    )

    expected_volatility_20 = (
        expected_returns.rolling(
            window=20,
        )
        .std()
        .iloc[20]
    )

    assert result.loc[
        4,
        "moving_average_5",
    ] == pytest.approx(102.0)

    assert result.loc[
        19,
        "moving_average_20",
    ] == pytest.approx(109.5)

    assert result.loc[
        5,
        "momentum_5",
    ] == pytest.approx(105.0 / 100.0 - 1)

    assert result.loc[
        20,
        "momentum_20",
    ] == pytest.approx(120.0 / 100.0 - 1)

    assert result.loc[
        5,
        "volatility_5",
    ] == pytest.approx(expected_volatility_5)

    assert result.loc[
        20,
        "volatility_20",
    ] == pytest.approx(expected_volatility_20)


def test_engineer_calculates_features_separately_for_each_ticker() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
                "QQQ",
                "QQQ",
            ],
            "adjusted_close": [
                100.0,
                110.0,
                200.0,
                220.0,
            ],
            "volume": [
                1000,
                1100,
                2000,
                2200,
            ],
        }
    )

    engineer = MarketFeatureEngineer()

    result = engineer.engineer(data)

    spy_data = result[result["ticker"] == "SPY"].reset_index(drop=True)

    qqq_data = result[result["ticker"] == "QQQ"].reset_index(drop=True)

    assert pd.isna(spy_data.loc[0, "daily_return"])
    assert pd.isna(qqq_data.loc[0, "daily_return"])

    assert pd.isna(spy_data.loc[0, "log_return"])
    assert pd.isna(qqq_data.loc[0, "log_return"])

    assert pd.isna(spy_data.loc[0, "price_change"])
    assert pd.isna(qqq_data.loc[0, "price_change"])

    assert pd.isna(spy_data.loc[0, "volume_change"])
    assert pd.isna(qqq_data.loc[0, "volume_change"])

    assert spy_data.loc[
        1,
        "daily_return",
    ] == pytest.approx(0.10)

    assert qqq_data.loc[
        1,
        "daily_return",
    ] == pytest.approx(0.10)

    assert spy_data.loc[
        1,
        "price_change",
    ] == pytest.approx(10.0)

    assert qqq_data.loc[
        1,
        "price_change",
    ] == pytest.approx(20.0)

    assert spy_data.loc[
        1,
        "volume_change",
    ] == pytest.approx(0.10)

    assert qqq_data.loc[
        1,
        "volume_change",
    ] == pytest.approx(0.10)

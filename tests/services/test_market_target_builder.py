import pandas as pd
import pandas.testing as pdt

from backend.src.services.market_target_builder import MarketTargetBuilder


def test_build_returns_dataframe() -> None:
    builder = MarketTargetBuilder()

    data = pd.DataFrame(
        {
            "ticker": ["SPY"],
            "date": pd.to_datetime(["2024-01-05"]),
            "adjusted_close": [100.0],
        }
    )

    result = builder.build(data)

    assert isinstance(result, pd.DataFrame)


def test_build_does_not_modify_input_dataframe() -> None:
    builder = MarketTargetBuilder()

    data = pd.DataFrame(
        {
            "ticker": ["SPY", "SPY"],
            "date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-08",
                ]
            ),
            "adjusted_close": [
                100.0,
                110.0,
            ],
        }
    )
    original = data.copy(deep=True)

    builder.build(data)

    pdt.assert_frame_equal(data, original)


def test_build_creates_next_return_and_target_date() -> None:
    builder = MarketTargetBuilder()

    data = pd.DataFrame(
        {
            "ticker": ["SPY", "SPY", "SPY"],
            "date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-09",
                ]
            ),
            "adjusted_close": [
                100.0,
                110.0,
                99.0,
            ],
        }
    )

    result = builder.build(data)

    expected_target_dates = pd.Series(
        [
            pd.Timestamp("2024-01-08"),
            pd.Timestamp("2024-01-09"),
            pd.NaT,
        ],
        dtype=data["date"].dtype,
        name="target_date",
    )
    expected_targets = pd.Series(
        [
            0.10,
            -0.10,
            float("nan"),
        ],
        name="target",
    )

    pdt.assert_series_equal(
        result["target_date"],
        expected_target_dates,
    )
    pdt.assert_series_equal(
        result["target"],
        expected_targets,
        check_exact=False,
        atol=1e-12,
    )


def test_build_creates_targets_separately_for_each_ticker() -> None:
    builder = MarketTargetBuilder()

    data = pd.DataFrame(
        {
            "ticker": [
                "SPY",
                "SPY",
                "QQQ",
                "QQQ",
            ],
            "date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-05",
                    "2024-01-08",
                ]
            ),
            "adjusted_close": [
                100.0,
                110.0,
                50.0,
                60.0,
            ],
        }
    )

    result = builder.build(data)

    expected_target_dates = pd.Series(
        [
            pd.Timestamp("2024-01-08"),
            pd.NaT,
            pd.Timestamp("2024-01-08"),
            pd.NaT,
        ],
        dtype=data["date"].dtype,
        name="target_date",
    )
    expected_targets = pd.Series(
        [
            0.10,
            float("nan"),
            0.20,
            float("nan"),
        ],
        name="target",
    )

    pdt.assert_series_equal(
        result["target_date"],
        expected_target_dates,
    )
    pdt.assert_series_equal(
        result["target"],
        expected_targets,
        check_exact=False,
        atol=1e-12,
    )

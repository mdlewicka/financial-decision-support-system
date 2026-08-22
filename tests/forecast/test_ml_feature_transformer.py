import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.forecast.ml_feature_transformer import (
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    MarketMLFeatureTransformer,
)


def create_input_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "ticker": [
                "SPY",
                "QQQ",
            ],
            "open": [
                100.0,
                200.0,
            ],
            "high": [
                102.0,
                204.0,
            ],
            "low": [
                99.0,
                198.0,
            ],
            "close": [
                101.0,
                202.0,
            ],
            "adjusted_close": [
                101.0,
                202.0,
            ],
            "volume": [
                1000,
                2000,
            ],
            "daily_return": [
                0.01,
                -0.02,
            ],
            "log_return": [
                0.00995,
                -0.02020,
            ],
            "price_change": [
                1.0,
                -4.0,
            ],
            "volume_change": [
                0.10,
                -0.05,
            ],
            "moving_average_5": [
                100.0,
                205.0,
            ],
            "moving_average_20": [
                98.0,
                210.0,
            ],
            "volatility_5": [
                0.012,
                0.021,
            ],
            "volatility_20": [
                0.018,
                0.026,
            ],
            "momentum_5": [
                0.025,
                -0.015,
            ],
            "momentum_20": [
                0.040,
                -0.030,
            ],
            "target_date": pd.to_datetime(
                [
                    "2025-01-03",
                    "2025-01-04",
                ]
            ),
            "target": [
                0.02,
                -0.01,
            ],
        }
    )


def test_transform_returns_expected_feature_columns() -> None:
    transformer = MarketMLFeatureTransformer()

    result = transformer.transform(create_input_data())

    assert list(result.columns) == list(MODEL_FEATURE_COLUMNS)
    assert list(
        result.loc[
            :,
            NUMERIC_FEATURE_COLUMNS,
        ].columns
    ) == list(NUMERIC_FEATURE_COLUMNS)


def test_transform_calculates_relative_features() -> None:
    transformer = MarketMLFeatureTransformer()

    result = transformer.transform(create_input_data())

    assert result.loc[
        0,
        "adjusted_close_to_ma_5",
    ] == pytest.approx(101.0 / 100.0 - 1.0)
    assert result.loc[
        0,
        "adjusted_close_to_ma_20",
    ] == pytest.approx(101.0 / 98.0 - 1.0)
    assert result.loc[
        0,
        "intraday_range",
    ] == pytest.approx((102.0 - 99.0) / 101.0)
    assert result.loc[
        0,
        "open_to_close_return",
    ] == pytest.approx(101.0 / 100.0 - 1.0)


def test_transform_excludes_metadata_target_and_raw_levels() -> None:
    transformer = MarketMLFeatureTransformer()

    result = transformer.transform(create_input_data())

    excluded_columns = {
        "date",
        "target_date",
        "target",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "log_return",
        "price_change",
        "moving_average_5",
        "moving_average_20",
    }

    assert excluded_columns.isdisjoint(result.columns)


def test_transform_is_invariant_to_common_price_scale() -> None:
    transformer = MarketMLFeatureTransformer()
    original = create_input_data()
    scaled = original.copy(deep=True)

    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "moving_average_5",
        "moving_average_20",
    ]

    scaled.loc[
        :,
        price_columns,
    ] = (
        scaled.loc[
            :,
            price_columns,
        ]
        * 10.0
    )

    original_features = transformer.transform(original)
    scaled_features = transformer.transform(scaled)

    pdt.assert_frame_equal(
        original_features,
        scaled_features,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_transform_does_not_modify_input_dataframe() -> None:
    data = create_input_data()
    original = data.copy(deep=True)

    transformer = MarketMLFeatureTransformer()

    transformer.transform(data)

    pdt.assert_frame_equal(
        data,
        original,
    )


def test_transform_rejects_missing_required_column() -> None:
    data = create_input_data().drop(
        columns=["momentum_20"],
    )
    transformer = MarketMLFeatureTransformer()

    with pytest.raises(
        ValueError,
        match=("Missing required ML feature columns: momentum_20"),
    ):
        transformer.transform(data)


def test_transform_rejects_non_finite_values() -> None:
    data = create_input_data()
    data.loc[
        0,
        "daily_return",
    ] = float("inf")

    transformer = MarketMLFeatureTransformer()

    with pytest.raises(
        ValueError,
        match="contains non-finite values",
    ):
        transformer.transform(data)


@pytest.mark.parametrize(
    "column",
    [
        "open",
        "close",
        "adjusted_close",
        "moving_average_5",
        "moving_average_20",
    ],
)
def test_transform_rejects_non_positive_price_values(
    column: str,
) -> None:
    data = create_input_data()
    data.loc[
        0,
        column,
    ] = 0.0

    transformer = MarketMLFeatureTransformer()

    with pytest.raises(
        ValueError,
        match="contains non-positive price values",
    ):
        transformer.transform(data)


def test_transform_rejects_high_price_lower_than_low_price() -> None:
    data = create_input_data()
    data.loc[
        0,
        "high",
    ] = 98.0

    transformer = MarketMLFeatureTransformer()

    with pytest.raises(
        ValueError,
        match=("High price must not be lower than low price"),
    ):
        transformer.transform(data)


def test_transform_rejects_empty_dataframe() -> None:
    transformer = MarketMLFeatureTransformer()

    with pytest.raises(
        ValueError,
        match="ML feature input data must not be empty",
    ):
        transformer.transform(pd.DataFrame())

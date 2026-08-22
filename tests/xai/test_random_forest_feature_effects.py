import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)
from backend.src.xai.random_forest_feature_effects import (
    RandomForestFeatureEffectAnalyzer,
)
from backend.src.xai.xai_result import (
    AccumulatedLocalEffectsResult,
    PartialDependenceResult,
)


def create_market_dataset(
    rows_per_ticker: int = 40,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for ticker_index, ticker in enumerate(
        (
            "QQQ",
            "SPY",
        )
    ):
        dates = pd.date_range(
            start="2024-01-01",
            periods=rows_per_ticker,
            freq="B",
        )
        base_price = 100.0 + 50.0 * ticker_index

        for position, feature_date in enumerate(dates):
            signal = np.sin(
                position / 4.0 + ticker_index,
            )
            adjusted_close = base_price + position * 0.15 + 1.5 * signal
            open_price = adjusted_close * (1.0 - 0.002 * np.cos(position / 3.0))
            close_price = adjusted_close * (1.0 + 0.001 * np.sin(position / 5.0))
            high_price = (
                max(
                    open_price,
                    close_price,
                )
                * 1.005
            )
            low_price = (
                min(
                    open_price,
                    close_price,
                )
                * 0.995
            )

            daily_return = 0.002 * signal + 0.0004 * ticker_index
            volatility_5 = 0.010 + 0.001 * abs(signal)
            volatility_20 = 0.015 + 0.0015 * abs(signal)
            momentum_5 = 0.01 * np.sin(position / 6.0)
            momentum_20 = 0.02 * np.sin(position / 9.0)
            volume_change = 0.05 * np.cos(position / 5.0)
            target = (
                0.45 * daily_return
                + 0.08 * momentum_5
                - 0.05 * volatility_20
                + 0.0003 * ticker_index
            )

            rows.append(
                {
                    "date": feature_date,
                    "ticker": ticker,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "adjusted_close": adjusted_close,
                    "volume": (1_000_000 + 10_000 * position + 100_000 * ticker_index),
                    "daily_return": daily_return,
                    "log_return": float(
                        np.log1p(daily_return),
                    ),
                    "price_change": (close_price - open_price),
                    "volume_change": volume_change,
                    "moving_average_5": (adjusted_close / (1.0 + 0.003 * signal)),
                    "moving_average_20": (adjusted_close / (1.0 + 0.006 * signal)),
                    "volatility_5": volatility_5,
                    "volatility_20": volatility_20,
                    "momentum_5": momentum_5,
                    "momentum_20": momentum_20,
                    "target_date": (feature_date + pd.offsets.BDay(1)),
                    "target": target,
                }
            )

    return pd.DataFrame(rows)


def assert_ale_results_equal(
    first: AccumulatedLocalEffectsResult,
    second: AccumulatedLocalEffectsResult,
) -> None:
    assert first.feature_name == second.feature_name

    pdt.assert_series_equal(
        first.bin_edges,
        second.bin_edges,
    )
    pdt.assert_series_equal(
        first.bin_centers,
        second.bin_centers,
    )
    pdt.assert_series_equal(
        first.local_effects,
        second.local_effects,
    )
    pdt.assert_series_equal(
        first.accumulated_effects,
        second.accumulated_effects,
    )
    pdt.assert_series_equal(
        first.sample_counts,
        second.sample_counts,
    )


@pytest.fixture
def market_dataset() -> pd.DataFrame:
    return create_market_dataset()


@pytest.fixture
def fitted_model(
    market_dataset: pd.DataFrame,
) -> RandomForestForecastModel:
    model = RandomForestForecastModel(
        n_estimators=30,
        max_depth=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=1,
    )
    model.fit(market_dataset)

    return model


@pytest.fixture
def analyzer(
    fitted_model: RandomForestForecastModel,
) -> RandomForestFeatureEffectAnalyzer:
    return RandomForestFeatureEffectAnalyzer(
        fitted_model,
    )


def test_partial_dependence_returns_finite_curve(
    analyzer: RandomForestFeatureEffectAnalyzer,
    market_dataset: pd.DataFrame,
) -> None:
    result = analyzer.partial_dependence(
        dataset=market_dataset,
        feature_name="momentum_5",
        grid_points=12,
    )

    assert isinstance(
        result,
        PartialDependenceResult,
    )
    assert result.feature_name == "momentum_5"
    assert len(result.feature_values) == 12
    assert len(result.average_predictions) == 12
    assert result.feature_values.is_monotonic_increasing
    assert np.isfinite(
        result.feature_values.to_numpy(),
    ).all()
    assert np.isfinite(
        result.average_predictions.to_numpy(),
    ).all()


def test_accumulated_local_effects_returns_centered_curve(
    analyzer: RandomForestFeatureEffectAnalyzer,
    market_dataset: pd.DataFrame,
) -> None:
    result = analyzer.accumulated_local_effects(
        dataset=market_dataset,
        feature_name="momentum_5",
        bins=8,
    )

    assert isinstance(
        result,
        AccumulatedLocalEffectsResult,
    )
    assert result.feature_name == "momentum_5"
    assert len(result.bin_edges) == (len(result.bin_centers) + 1)
    assert len(result.local_effects) == len(
        result.bin_centers,
    )
    assert len(result.accumulated_effects) == len(
        result.bin_centers,
    )
    assert int(result.sample_counts.sum()) == len(
        market_dataset,
    )
    assert np.isfinite(
        result.accumulated_effects.to_numpy(),
    ).all()

    weighted_mean = float(
        np.average(
            result.accumulated_effects.to_numpy(),
            weights=result.sample_counts.to_numpy(),
        )
    )

    assert weighted_mean == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_feature_effects_are_deterministic(
    analyzer: RandomForestFeatureEffectAnalyzer,
    market_dataset: pd.DataFrame,
) -> None:
    first_pdp = analyzer.partial_dependence(
        dataset=market_dataset,
        feature_name="volatility_20",
        grid_points=10,
    )
    second_pdp = analyzer.partial_dependence(
        dataset=market_dataset,
        feature_name="volatility_20",
        grid_points=10,
    )

    assert first_pdp.feature_name == (second_pdp.feature_name)
    pdt.assert_series_equal(
        first_pdp.feature_values,
        second_pdp.feature_values,
    )
    pdt.assert_series_equal(
        first_pdp.average_predictions,
        second_pdp.average_predictions,
    )

    first_ale = analyzer.accumulated_local_effects(
        dataset=market_dataset,
        feature_name="volatility_20",
        bins=6,
    )
    second_ale = analyzer.accumulated_local_effects(
        dataset=market_dataset,
        feature_name="volatility_20",
        bins=6,
    )

    assert_ale_results_equal(
        first_ale,
        second_ale,
    )


def test_analysis_does_not_modify_input_dataset(
    analyzer: RandomForestFeatureEffectAnalyzer,
    market_dataset: pd.DataFrame,
) -> None:
    original = market_dataset.copy(
        deep=True,
    )

    analyzer.partial_dependence(
        dataset=market_dataset,
        feature_name="daily_return",
        grid_points=8,
    )
    analyzer.accumulated_local_effects(
        dataset=market_dataset,
        feature_name="daily_return",
        bins=6,
    )

    pdt.assert_frame_equal(
        market_dataset,
        original,
    )


def test_analysis_rejects_unknown_feature(
    analyzer: RandomForestFeatureEffectAnalyzer,
    market_dataset: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="Unknown model feature",
    ):
        analyzer.partial_dependence(
            dataset=market_dataset,
            feature_name="unknown_feature",
        )


def test_analysis_rejects_binary_ticker_feature(
    analyzer: RandomForestFeatureEffectAnalyzer,
    fitted_model: RandomForestForecastModel,
    market_dataset: pd.DataFrame,
) -> None:
    transformed = fitted_model.transform_features(
        market_dataset,
    )

    binary_features = [
        column
        for column in transformed.columns
        if len(
            np.unique(
                transformed[column].to_numpy(),
            )
        )
        < 3
    ]

    assert binary_features

    with pytest.raises(
        ValueError,
        match="must be continuous",
    ):
        analyzer.accumulated_local_effects(
            dataset=market_dataset,
            feature_name=binary_features[0],
        )


def test_analysis_rejects_invalid_configuration(
    analyzer: RandomForestFeatureEffectAnalyzer,
    market_dataset: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="grid_points must be an integer",
    ):
        analyzer.partial_dependence(
            dataset=market_dataset,
            feature_name="momentum_5",
            grid_points=1,
        )

    with pytest.raises(
        ValueError,
        match="bins must be an integer",
    ):
        analyzer.accumulated_local_effects(
            dataset=market_dataset,
            feature_name="momentum_5",
            bins=1,
        )

    with pytest.raises(
        ValueError,
        match="quantile_range must satisfy",
    ):
        analyzer.partial_dependence(
            dataset=market_dataset,
            feature_name="momentum_5",
            quantile_range=(
                0.9,
                0.1,
            ),
        )

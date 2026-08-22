import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)
from backend.src.xai.random_forest_shap_explainer import (
    RandomForestShapExplainer,
)
from backend.src.xai.xai_result import (
    GlobalShapExplanation,
    LocalShapExplanation,
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
            trend = position * 0.15

            adjusted_close = base_price + trend + 1.5 * signal
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
            log_return = float(
                np.log1p(daily_return),
            )
            volume_change = 0.05 * np.cos(
                position / 5.0,
            )
            volatility_5 = 0.010 + 0.001 * abs(signal)
            volatility_20 = 0.015 + 0.0015 * abs(signal)
            momentum_5 = 0.01 * np.sin(
                position / 6.0,
            )
            momentum_20 = 0.02 * np.sin(
                position / 9.0,
            )

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
                    "log_return": log_return,
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


@pytest.fixture
def training_dataset() -> pd.DataFrame:
    return create_market_dataset()


@pytest.fixture
def fitted_model(
    training_dataset: pd.DataFrame,
) -> RandomForestForecastModel:
    model = RandomForestForecastModel(
        n_estimators=30,
        max_depth=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=1,
    )
    model.fit(training_dataset)

    return model


def test_explain_local_reconstructs_real_model_prediction(
    fitted_model: RandomForestForecastModel,
    training_dataset: pd.DataFrame,
) -> None:
    observation = (
        training_dataset.loc[training_dataset["ticker"] == "SPY"].tail(1).copy()
    )

    explainer = RandomForestShapExplainer(
        model=fitted_model,
        background_dataset=training_dataset,
        max_background_rows=30,
        random_state=42,
    )

    result = explainer.explain_local(
        observation,
    )

    expected_prediction = float(
        fitted_model.predict(
            observation,
        ).iloc[0]
    )

    assert isinstance(
        result,
        LocalShapExplanation,
    )
    assert list(result.feature_values.index) == list(
        fitted_model.feature_names,
    )
    assert list(result.feature_contributions.index) == list(
        fitted_model.feature_names,
    )
    assert np.isfinite(
        result.feature_values.to_numpy(
            dtype=float,
        )
    ).all()
    assert np.isfinite(
        result.feature_contributions.to_numpy(
            dtype=float,
        )
    ).all()
    assert result.prediction == pytest.approx(
        expected_prediction,
    )
    assert result.reconstructed_prediction == pytest.approx(
        expected_prediction,
        abs=1e-6,
    )


def test_explain_global_returns_feature_importance(
    fitted_model: RandomForestForecastModel,
    training_dataset: pd.DataFrame,
) -> None:
    observations = (
        training_dataset.groupby(
            "ticker",
            sort=True,
            group_keys=False,
        )
        .tail(2)
        .copy()
    )

    explainer = RandomForestShapExplainer(
        model=fitted_model,
        background_dataset=training_dataset,
        max_background_rows=30,
        random_state=42,
    )

    result = explainer.explain_global(
        observations,
    )

    expected_predictions = fitted_model.predict(
        observations,
    )

    assert isinstance(
        result,
        GlobalShapExplanation,
    )
    assert result.feature_values.shape == (
        len(observations),
        len(fitted_model.feature_names),
    )
    assert result.feature_contributions.shape == result.feature_values.shape
    assert list(result.feature_values.columns) == list(
        fitted_model.feature_names,
    )
    assert list(result.mean_absolute_contributions.index) == list(
        result.mean_absolute_contributions.sort_values(
            ascending=False,
        ).index
    )

    np.testing.assert_allclose(
        result.predictions.to_numpy(),
        expected_predictions.to_numpy(),
        atol=1e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        (
            result.base_values
            + result.feature_contributions.sum(
                axis=1,
            )
        ).to_numpy(),
        expected_predictions.to_numpy(),
        atol=1e-6,
        rtol=0.0,
    )

    expected_mean_absolute_contributions = (
        result.feature_contributions.abs()
        .mean(axis=0)
        .sort_values(
            ascending=False,
        )
        .rename(
            "mean_absolute_shap_value",
        )
    )

    pdt.assert_series_equal(
        result.mean_absolute_contributions,
        expected_mean_absolute_contributions,
    )


def test_background_sampling_is_deterministic(
    fitted_model: RandomForestForecastModel,
    training_dataset: pd.DataFrame,
) -> None:
    first_explainer = RandomForestShapExplainer(
        model=fitted_model,
        background_dataset=training_dataset,
        max_background_rows=12,
        random_state=42,
    )
    second_explainer = RandomForestShapExplainer(
        model=fitted_model,
        background_dataset=training_dataset,
        max_background_rows=12,
        random_state=42,
    )

    assert len(first_explainer.background_feature_values) == 12

    pdt.assert_frame_equal(
        first_explainer.background_feature_values,
        second_explainer.background_feature_values,
    )


def test_background_data_is_returned_as_copy(
    fitted_model: RandomForestForecastModel,
    training_dataset: pd.DataFrame,
) -> None:
    explainer = RandomForestShapExplainer(
        model=fitted_model,
        background_dataset=training_dataset,
        max_background_rows=12,
    )

    first_result = explainer.background_feature_values
    first_result.iloc[0, 0] = 999.0

    second_result = explainer.background_feature_values

    assert second_result.iloc[0, 0] != 999.0


def test_explain_local_rejects_multiple_rows(
    fitted_model: RandomForestForecastModel,
    training_dataset: pd.DataFrame,
) -> None:
    explainer = RandomForestShapExplainer(
        model=fitted_model,
        background_dataset=training_dataset,
    )

    with pytest.raises(
        ValueError,
        match=("observation must contain exactly one row"),
    ):
        explainer.explain_local(
            training_dataset.head(2),
        )


def test_constructor_rejects_empty_background_dataset(
    fitted_model: RandomForestForecastModel,
    training_dataset: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match=("background_dataset must not be empty"),
    ):
        RandomForestShapExplainer(
            model=fitted_model,
            background_dataset=(training_dataset.iloc[0:0]),
        )


def test_constructor_rejects_invalid_background_limit(
    fitted_model: RandomForestForecastModel,
    training_dataset: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match=("max_background_rows must be a positive integer"),
    ):
        RandomForestShapExplainer(
            model=fitted_model,
            background_dataset=training_dataset,
            max_background_rows=0,
        )

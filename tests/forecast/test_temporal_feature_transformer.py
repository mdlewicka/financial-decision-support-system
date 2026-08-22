import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.forecast.temporal_feature_transformer import (
    OUTPUT_COLUMNS,
    TEMPORAL_FEATURE_COLUMNS,
    TemporalMarketFeatureTransformer,
)


def create_temporal_dataset(
    start: str,
    periods: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for ticker_index, ticker in enumerate(("AGG", "QQQ", "SPY")):
        dates = pd.date_range(start=start, periods=periods, freq="D")
        price = 100.0 + 50.0 * ticker_index

        for index, date in enumerate(dates):
            daily_return = 0.002 * ((index + ticker_index) % 7 - 3)
            price *= 1.0 + daily_return
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_close": price,
                    "volume": 1_000_000.0 + 10_000.0 * ticker_index + index,
                    "daily_return": daily_return,
                    "target_date": date + pd.Timedelta(days=1),
                    "target": 0.5 * daily_return,
                }
            )

    return pd.DataFrame(rows)


def test_fit_transform_returns_documented_feature_contract() -> None:
    transformer = TemporalMarketFeatureTransformer()

    transformed = transformer.fit_transform(create_temporal_dataset("2023-01-01", 90))

    assert list(transformed.columns) == list(OUTPUT_COLUMNS)
    assert transformer.feature_names == TEMPORAL_FEATURE_COLUMNS
    assert transformer.contract.uses_target_column is False
    assert transformer.contract.uses_future_observations is False
    assert transformer.is_fitted
    assert transformed["features_ready"].any()
    assert np.isfinite(
        transformed.loc[
            transformed["features_ready"],
            list(TEMPORAL_FEATURE_COLUMNS),
        ].to_numpy(dtype=float)
    ).all()


def test_transform_uses_fitted_history_at_split_boundary() -> None:
    training = create_temporal_dataset("2023-01-01", 90)
    evaluation = create_temporal_dataset("2023-04-01", 5)
    transformer = TemporalMarketFeatureTransformer().fit(training)

    transformed = transformer.transform(evaluation)

    assert transformed.index.equals(evaluation.index)
    assert transformed["features_ready"].all()
    assert np.isfinite(
        transformed[list(TEMPORAL_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all()


def test_features_are_independent_of_target_values() -> None:
    original = create_temporal_dataset("2023-01-01", 90)
    changed_targets = original.copy()
    changed_targets["target"] = np.linspace(-100.0, 100.0, len(changed_targets))

    original_features = TemporalMarketFeatureTransformer().fit_transform(original)
    changed_features = TemporalMarketFeatureTransformer().fit_transform(changed_targets)

    pdt.assert_frame_equal(original_features, changed_features)


def test_future_input_changes_do_not_modify_past_features() -> None:
    original = create_temporal_dataset("2023-01-01", 90)
    changed_future = original.copy()
    cutoff = pd.Timestamp("2023-03-15")
    future_mask = pd.to_datetime(changed_future["date"]) > cutoff
    changed_future.loc[future_mask, "adjusted_close"] *= 1.5
    changed_future.loc[future_mask, "volume"] *= 2.0
    changed_future.loc[future_mask, "daily_return"] += 0.2

    original_features = TemporalMarketFeatureTransformer().fit_transform(original)
    changed_features = TemporalMarketFeatureTransformer().fit_transform(changed_future)
    past_mask = pd.to_datetime(original_features["date"]) <= cutoff

    pdt.assert_frame_equal(
        original_features.loc[past_mask].reset_index(drop=True),
        changed_features.loc[past_mask].reset_index(drop=True),
    )


def test_first_lag_equals_previous_ticker_return() -> None:
    dataset = create_temporal_dataset("2023-01-01", 90)

    transformed = TemporalMarketFeatureTransformer().fit_transform(dataset)

    for ticker, ticker_features in transformed.groupby("ticker", sort=True):
        expected = (
            dataset.loc[dataset["ticker"] == ticker]
            .sort_values("date", kind="stable")["daily_return"]
            .shift(1)
            .reset_index(drop=True)
        )
        actual = ticker_features.sort_values("date", kind="stable")[
            "return_lag_1"
        ].reset_index(drop=True)
        pdt.assert_series_equal(actual, expected, check_names=False)


def test_transform_before_fit_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        TemporalMarketFeatureTransformer().transform(
            create_temporal_dataset("2023-04-01", 5)
        )


def test_transform_rejects_overlapping_dates() -> None:
    transformer = TemporalMarketFeatureTransformer().fit(
        create_temporal_dataset("2023-01-01", 90)
    )

    with pytest.raises(ValueError, match="must begin after"):
        transformer.transform(create_temporal_dataset("2023-03-20", 5))


def test_transform_rejects_unknown_ticker() -> None:
    transformer = TemporalMarketFeatureTransformer().fit(
        create_temporal_dataset("2023-01-01", 90)
    )
    evaluation = create_temporal_dataset("2023-04-01", 5)
    evaluation.loc[evaluation.index[0], "ticker"] = "UNKNOWN"

    with pytest.raises(ValueError, match="unknown tickers: UNKNOWN"):
        transformer.transform(evaluation)


def test_duplicate_date_ticker_rows_are_rejected() -> None:
    dataset = create_temporal_dataset("2023-01-01", 90)
    duplicated = pd.concat([dataset, dataset.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate date/ticker"):
        TemporalMarketFeatureTransformer().fit(duplicated)


def test_ready_rows_selects_only_finite_features() -> None:
    transformed = TemporalMarketFeatureTransformer().fit_transform(
        create_temporal_dataset("2023-01-01", 90)
    )

    ready = TemporalMarketFeatureTransformer.ready_rows(transformed)

    assert not ready.empty
    assert ready["features_ready"].all()
    assert np.isfinite(ready[list(TEMPORAL_FEATURE_COLUMNS)]).all().all()

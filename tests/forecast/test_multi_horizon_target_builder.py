import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.forecast.multi_horizon_target_builder import (
    TARGET_CLASSES,
    TARGET_COLUMNS,
    ForecastTargetConfig,
    MultiHorizonForecastTargetBuilder,
)


def create_price_dataset(
    periods: int = 10,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for ticker, multiplier in (
        ("QQQ", 2.0),
        ("SPY", 1.0),
    ):
        dates = pd.date_range(start=start, periods=periods, freq="D")
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_close": multiplier * (100.0 + index),
                    "target": 999.0,
                }
            )

    return pd.DataFrame(rows)


def test_build_creates_documented_one_and_five_session_targets() -> None:
    result = MultiHorizonForecastTargetBuilder().build(create_price_dataset())

    assert list(result.columns) == [
        "date",
        "ticker",
        "adjusted_close",
        *TARGET_COLUMNS,
    ]
    assert set(
        result.loc[result["target_ready_5"], "target_class_5"].dropna()
    ).issubset(TARGET_CLASSES)


def test_one_session_target_matches_existing_next_return_definition() -> None:
    data = create_price_dataset()
    result = MultiHorizonForecastTargetBuilder().build(data)

    spy = result.loc[result["ticker"] == "SPY"].reset_index(drop=True)

    assert spy.loc[0, "target_date_1"] == pd.Timestamp("2024-01-02")
    assert spy.loc[0, "target_return_1"] == pytest.approx(0.01)
    assert spy.loc[0, "target_direction_1"] == 1


def test_five_session_target_uses_compounded_price_return() -> None:
    result = MultiHorizonForecastTargetBuilder().build(create_price_dataset())
    spy = result.loc[result["ticker"] == "SPY"].reset_index(drop=True)

    assert spy.loc[0, "target_date_5"] == pd.Timestamp("2024-01-06")
    assert spy.loc[0, "target_return_5"] == pytest.approx(0.05)
    assert spy.loc[0, "target_direction_5"] == 1
    assert spy.loc[0, "target_class_5"] == "up"


def test_targets_are_built_separately_for_each_ticker() -> None:
    result = MultiHorizonForecastTargetBuilder().build(create_price_dataset())

    spy = result.loc[result["ticker"] == "SPY"].reset_index(drop=True)
    qqq = result.loc[result["ticker"] == "QQQ"].reset_index(drop=True)

    pdt.assert_series_equal(
        spy["target_return_5"],
        qqq["target_return_5"],
        check_names=False,
    )


def test_economic_threshold_assigns_down_abstain_and_up() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "ticker": ["SPY"] * 4,
            "adjusted_close": [100.0, 100.2, 99.7, 100.7],
        }
    )
    builder = MultiHorizonForecastTargetBuilder(
        ForecastTargetConfig(
            one_session_threshold=0.003,
            five_session_threshold=0.01,
        )
    )

    result = builder.build(data)

    assert result["target_class_1"].tolist() == [
        "abstain",
        "down",
        "up",
        pd.NA,
    ]


def test_unavailable_targets_are_not_mislabeled_as_abstain() -> None:
    result = MultiHorizonForecastTargetBuilder().build(create_price_dataset())
    spy = result.loc[result["ticker"] == "SPY"].reset_index(drop=True)

    assert not spy.loc[9, "target_ready_1"]
    assert pd.isna(spy.loc[9, "target_class_1"])
    assert not spy.loc[5:, "target_ready_5"].any()
    assert spy.loc[5:, "target_class_5"].isna().all()


def test_builder_does_not_use_existing_target_column() -> None:
    original = create_price_dataset()
    changed = original.copy()
    changed["target"] = np.linspace(-1000.0, 1000.0, len(changed))

    original_targets = MultiHorizonForecastTargetBuilder().build(original)
    changed_targets = MultiHorizonForecastTargetBuilder().build(changed)

    pdt.assert_frame_equal(original_targets, changed_targets)


def test_price_after_horizon_does_not_change_earlier_target() -> None:
    original = create_price_dataset()
    changed = original.copy()
    mask = (changed["ticker"] == "SPY") & (
        pd.to_datetime(changed["date"]) >= pd.Timestamp("2024-01-07")
    )
    changed.loc[mask, "adjusted_close"] *= 3.0

    original_targets = MultiHorizonForecastTargetBuilder().build(original)
    changed_targets = MultiHorizonForecastTargetBuilder().build(changed)
    original_first = original_targets.loc[
        (original_targets["ticker"] == "SPY")
        & (original_targets["date"] == pd.Timestamp("2024-01-01"))
    ].iloc[0]
    changed_first = changed_targets.loc[
        (changed_targets["ticker"] == "SPY")
        & (changed_targets["date"] == pd.Timestamp("2024-01-01"))
    ].iloc[0]

    assert original_first["target_return_5"] == changed_first["target_return_5"]
    assert original_first["target_class_5"] == changed_first["target_class_5"]


def test_targets_available_by_enforces_fold_cutoff() -> None:
    targets = MultiHorizonForecastTargetBuilder().build(create_price_dataset())

    available = MultiHorizonForecastTargetBuilder.targets_available_by(
        targets,
        horizon=5,
        cutoff="2024-01-07",
    )

    assert not available.empty
    assert pd.to_datetime(available["target_date_5"]).max() <= pd.Timestamp(
        "2024-01-07"
    )
    assert available["target_ready_5"].all()


def test_ready_rows_selects_only_complete_horizons() -> None:
    targets = MultiHorizonForecastTargetBuilder().build(create_price_dataset())

    ready_one = MultiHorizonForecastTargetBuilder.ready_rows(targets, horizon=1)
    ready_five = MultiHorizonForecastTargetBuilder.ready_rows(targets, horizon=5)

    assert len(ready_one) == 18
    assert len(ready_five) == 10


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        ForecastTargetConfig(one_session_threshold=float("nan"))


def test_duplicate_date_ticker_rows_are_rejected() -> None:
    data = create_price_dataset()
    duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate date/ticker"):
        MultiHorizonForecastTargetBuilder().build(duplicated)

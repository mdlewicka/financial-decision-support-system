import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.src.forecast.forecast_v2_economic_validation import (
    BENCHMARK_NAME,
    EconomicValidationConfig,
)
from backend.src.forecast.gradient_boosting_economic_validation import (
    BOOSTING_POLICY_NAME,
    SOURCE_MODEL,
    export_boosting_economic_validation,
    run_boosting_economic_validation,
)


def create_observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in (1, 2):
        for ticker_index, ticker in enumerate(("QQQ", "SPY")):
            dates = pd.date_range(f"2024-0{fold}-01", periods=20, freq="D")
            returns = np.asarray([0.03, -0.02, 0.01, -0.01, 0.02] * 4)
            for index, date in enumerate(dates):
                actual = float(returns[(index + ticker_index) % len(returns)])
                if actual > 0.0:
                    probabilities = (0.15, 0.25, 0.60)
                    actual_class = "up"
                else:
                    probabilities = (0.55, 0.30, 0.15)
                    actual_class = "down"
                rows.append(
                    {
                        "fold": fold,
                        "horizon": 5,
                        "task": "classification",
                        "model": SOURCE_MODEL,
                        "ticker": ticker,
                        "date": date,
                        "target_date": date + pd.Timedelta(days=5),
                        "actual_return": actual,
                        "actual_class": actual_class,
                        "probability_down": probabilities[0],
                        "probability_abstain": probabilities[1],
                        "probability_up": probabilities[2],
                    }
                )
    return pd.DataFrame(rows)


def small_config() -> EconomicValidationConfig:
    return EconomicValidationConfig(
        confidence_thresholds=(0.40, 0.50, 0.60),
        default_threshold=0.50,
        transaction_cost=0.001,
        bootstrap_iterations=50,
        random_state=42,
    )


def test_boosting_screen_uses_long_cash_policy_and_non_overlapping_periods() -> None:
    result = run_boosting_economic_validation(create_observations(), small_config())

    assert set(result.period_observations["strategy"]) == {
        BOOSTING_POLICY_NAME,
        BENCHMARK_NAME,
    }
    assert set(result.period_observations["action"]).issubset({0.0, 1.0})
    assert result.decision["source_model"] == SOURCE_MODEL
    assert result.decision["candidate"] == BOOSTING_POLICY_NAME
    for _, group in result.period_observations.groupby(
        ["threshold", "strategy", "fold", "ticker"], sort=True
    ):
        ordered = group.sort_values("date")
        assert (
            ordered["date"].iloc[1:].to_numpy()
            >= ordered["target_date"].shift(1).iloc[1:].to_numpy()
        ).all()


def test_wrong_source_model_is_rejected() -> None:
    observations = create_observations()
    observations["model"] = "logistic_v2"

    with pytest.raises(ValueError, match=SOURCE_MODEL):
        run_boosting_economic_validation(observations, small_config())


def test_export_records_closed_test_and_source_provenance(tmp_path: Path) -> None:
    observations = create_observations()
    result = run_boosting_economic_validation(observations, small_config())
    paths = export_boosting_economic_validation(
        result,
        tmp_path,
        observations,
        {
            "test_split_read": False,
            "known_v1_test_used_for_selection": False,
        },
        {
            "predictive_gate_passed": True,
            "test_split_read": False,
            "decision": "promote_to_economic_validation",
        },
        small_config(),
    )

    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["test_split_read"] is False
    assert metadata["known_v1_test_used_for_selection"] is False
    assert metadata["source_model"] == SOURCE_MODEL
    assert metadata["policy"]["short_selling"] is False
    assert metadata["policy"]["periods_overlap"] is False


def test_export_rejects_failed_predictive_gate(tmp_path: Path) -> None:
    observations = create_observations()
    result = run_boosting_economic_validation(observations, small_config())

    with pytest.raises(ValueError, match="predictive gate"):
        export_boosting_economic_validation(
            result,
            tmp_path,
            observations,
            {
                "test_split_read": False,
                "known_v1_test_used_for_selection": False,
            },
            {"predictive_gate_passed": False, "test_split_read": False},
            small_config(),
        )

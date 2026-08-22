import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.forecast.forecast_v2_economic_validation import (
    BENCHMARK_NAME,
    POLICY_NAME,
    EconomicValidationConfig,
    export_economic_validation,
    main,
    run_economic_validation,
)


def create_observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in (1, 2):
        for ticker_index, ticker in enumerate(("QQQ", "SPY")):
            dates = pd.date_range(
                f"2024-0{fold}-01",
                periods=20,
                freq="D",
            )
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
                        "model": "logistic_v2",
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


def test_validation_uses_non_overlapping_long_cash_periods() -> None:
    result = run_economic_validation(create_observations(), small_config())
    periods = result.period_observations

    assert set(periods["strategy"]) == {POLICY_NAME, BENCHMARK_NAME}
    assert set(periods["action"]).issubset({0.0, 1.0})
    for _, group in periods.groupby(
        ["threshold", "strategy", "fold", "ticker"], sort=True
    ):
        ordered = group.sort_values("date")
        assert (
            ordered["date"].iloc[1:].to_numpy()
            >= ordered["target_date"].shift(1).iloc[1:].to_numpy()
        ).all()


def test_costs_turnover_and_path_metrics_are_finite() -> None:
    result = run_economic_validation(create_observations(), small_config())

    assert np.isfinite(result.period_observations["wealth"]).all()
    assert np.isfinite(result.path_metrics["cumulative_return"]).all()
    assert result.path_metrics["coverage"].between(0.0, 1.0).all()
    assert (result.path_metrics["turnover"] >= 0.0).all()
    assert result.threshold_summary["groups"].eq(4).all()


def test_validation_is_reproducible() -> None:
    first = run_economic_validation(create_observations(), small_config())
    second = run_economic_validation(create_observations(), small_config())

    pdt.assert_frame_equal(first.period_observations, second.period_observations)
    pdt.assert_frame_equal(first.path_metrics, second.path_metrics)
    pdt.assert_frame_equal(first.bootstrap_comparison, second.bootstrap_comparison)
    assert first.decision == second.decision


def test_invalid_probabilities_and_config_are_rejected() -> None:
    invalid = create_observations()
    invalid.loc[0, "probability_up"] = 0.9
    with pytest.raises(ValueError, match="sum to one"):
        run_economic_validation(invalid, small_config())

    with pytest.raises(ValueError, match="default_threshold"):
        EconomicValidationConfig(
            confidence_thresholds=(0.4, 0.5),
            default_threshold=0.45,
        )


def test_export_records_no_test_access_and_policy_limitations(tmp_path: Path) -> None:
    observations = create_observations()
    result = run_economic_validation(observations, small_config())
    metadata = {
        "benchmark_name": "forecast_v2_model_benchmark",
        "test_split_read": False,
        "known_v1_test_used_for_selection": False,
    }
    paths = export_economic_validation(
        result=result,
        output_directory=tmp_path,
        benchmark_observations=observations,
        benchmark_metadata=metadata,
        config=small_config(),
    )

    assert all(
        path.is_file()
        for path in (
            paths.period_observations_csv,
            paths.path_metrics_csv,
            paths.threshold_summary_csv,
            paths.bootstrap_csv,
            paths.decision_json,
            paths.metadata_json,
        )
    )
    exported = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert exported["test_split_read"] is False
    assert exported["known_v1_test_used_for_selection"] is False
    assert exported["policy"]["short_selling"] is False
    assert exported["policy"]["periods_overlap"] is False


def test_rejects_benchmark_metadata_that_used_test(tmp_path: Path) -> None:
    observations = create_observations()
    result = run_economic_validation(observations, small_config())

    with pytest.raises(ValueError, match="must not read"):
        export_economic_validation(
            result=result,
            output_directory=tmp_path,
            benchmark_observations=observations,
            benchmark_metadata={
                "test_split_read": True,
                "known_v1_test_used_for_selection": False,
            },
            config=small_config(),
        )


def test_cli_consumes_only_benchmark_artifacts(
    tmp_path: Path,
) -> None:
    benchmark_directory = tmp_path / "benchmark"
    output_directory = tmp_path / "economic"
    benchmark_directory.mkdir()
    create_observations().to_csv(
        benchmark_directory / "benchmark_observations.csv",
        index=False,
    )
    (benchmark_directory / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "benchmark_name": "forecast_v2_model_benchmark",
                "test_split_read": False,
                "known_v1_test_used_for_selection": False,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--benchmark-directory",
            str(benchmark_directory),
            "--output-directory",
            str(output_directory),
            "--bootstrap-iterations",
            "20",
        ]
    )

    assert exit_code == 0
    assert (output_directory / "economic_decision.json").is_file()

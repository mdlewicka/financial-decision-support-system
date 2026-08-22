import json
from pathlib import Path

import pandas as pd
import pytest

from backend.src.forecast.forecast_v2_report import (
    build_forecast_v2_report,
    main,
    write_forecast_v2_report,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def report_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    stability = tmp_path / "stability"
    benchmark = tmp_path / "benchmark"
    economic = tmp_path / "economic"

    _write_csv(
        stability / "stability_summary.csv",
        [
            {
                "model": "zero_return",
                "folds": 5,
                "mean_mae": 0.01100529,
                "median_mae": 0.00984422,
                "mean_mae_improvement_vs_zero_percent": 0.0,
            },
            {
                "model": "random_forest_v1",
                "folds": 5,
                "mean_mae": 0.01102360,
                "median_mae": 0.00985044,
                "mean_mae_improvement_vs_zero_percent": -0.154639,
            },
        ],
    )
    _write_csv(
        stability / "bootstrap_comparison.csv",
        [
            {
                "model": "random_forest_v1",
                "bootstrap_probability_better_than_zero": 0.238,
            }
        ],
    )
    _write_json(
        stability / "audit_metadata.json",
        {"test_split_read": False},
    )

    _write_csv(
        benchmark / "stability_summary.csv",
        [
            {
                "task": "regression",
                "horizon": 1,
                "model": "ridge_v2",
                "folds": 5,
                "primary_median": 0.00984644,
                "median_improvement_vs_baseline": -0.510967,
                "fold_wins_vs_baseline": 1,
                "mean_coverage": 0.110114,
                "mean_multiclass_brier": None,
                "mean_net_utility": -0.000027,
            },
            {
                "task": "regression",
                "horizon": 5,
                "model": "ridge_v2",
                "folds": 5,
                "primary_median": 0.02155744,
                "median_improvement_vs_baseline": 0.347636,
                "fold_wins_vs_baseline": 3,
                "mean_coverage": 0.214309,
                "mean_multiclass_brier": None,
                "mean_net_utility": -0.000029,
            },
            {
                "task": "classification",
                "horizon": 5,
                "model": "historical_prior",
                "folds": 5,
                "primary_median": 0.333333,
                "median_improvement_vs_baseline": 0.0,
                "fold_wins_vs_baseline": 0,
                "mean_coverage": 1.0,
                "mean_multiclass_brier": 0.638318,
                "mean_net_utility": 0.001360,
            },
            {
                "task": "classification",
                "horizon": 5,
                "model": "logistic_v2",
                "folds": 5,
                "primary_median": 0.394369,
                "median_improvement_vs_baseline": 0.061036,
                "fold_wins_vs_baseline": 5,
                "mean_coverage": 0.308780,
                "mean_multiclass_brier": 0.662171,
                "mean_net_utility": -0.000739,
            },
        ],
    )
    _write_csv(
        benchmark / "bootstrap_comparison.csv",
        [
            {
                "task": "regression",
                "horizon": 1,
                "model": "ridge_v2",
                "comparison_model": "zero_return",
                "mean_primary_difference": 0.000108,
                "ci_lower_95": -0.000009,
                "ci_upper_95": 0.000223,
                "probability_better_than_baseline": 0.034,
            },
            {
                "task": "regression",
                "horizon": 5,
                "model": "ridge_v2",
                "comparison_model": "zero_return",
                "mean_primary_difference": 0.000241,
                "ci_lower_95": -0.000144,
                "ci_upper_95": 0.000619,
                "probability_better_than_baseline": 0.115,
            },
            {
                "task": "classification",
                "horizon": 5,
                "model": "logistic_v2",
                "comparison_model": "historical_prior",
                "mean_primary_difference": 0.060745,
                "ci_lower_95": 0.048758,
                "ci_upper_95": 0.073772,
                "probability_better_than_baseline": 1.0,
            },
        ],
    )
    _write_csv(
        benchmark / "candidate_decisions.csv",
        [
            {
                "task": "classification",
                "horizon": 5,
                "model": "logistic_v2",
                "decision_is_final_test_claim": False,
            },
            {
                "task": "regression",
                "horizon": 1,
                "model": "ridge_v2",
                "decision_is_final_test_claim": False,
            },
            {
                "task": "regression",
                "horizon": 5,
                "model": "ridge_v2",
                "decision_is_final_test_claim": False,
            },
        ],
    )
    _write_json(
        benchmark / "benchmark_metadata.json",
        {
            "test_split_read": False,
            "known_v1_test_used_for_selection": False,
        },
    )

    _write_csv(
        economic / "threshold_summary.csv",
        [
            {
                "threshold": 0.45,
                "groups": 40,
                "mean_coverage": 0.103440,
                "mean_policy_return": 0.000054,
                "mean_benchmark_return": 0.053601,
                "mean_excess_return": -0.053547,
                "group_wins_vs_benchmark": 12,
                "mean_drawdown_reduction": 0.104075,
            }
        ],
    )
    _write_csv(
        economic / "bootstrap_comparison.csv",
        [
            {
                "threshold": 0.45,
                "mean_excess_return": -0.053547,
                "ci_lower_95": -0.109135,
                "ci_upper_95": 0.000119,
                "probability_better_than_buy_hold": 0.026,
            }
        ],
    )
    _write_json(
        economic / "economic_decision.json",
        {
            "default_threshold": 0.45,
            "thresholds_evaluated": 6,
            "thresholds_with_positive_mean_excess": 0,
            "decision": "retain_as_predictive_decision_support_only",
            "final_test_claim": False,
        },
    )
    _write_json(
        economic / "economic_validation_metadata.json",
        {
            "test_split_read": False,
            "known_v1_test_used_for_selection": False,
            "config": {"transaction_cost": 0.001},
        },
    )

    return stability, benchmark, economic


def test_build_report_records_predictive_and_economic_decisions(
    report_artifacts: tuple[Path, Path, Path],
) -> None:
    stability, benchmark, economic = report_artifacts

    report = build_forecast_v2_report(
        stability_directory=stability,
        benchmark_directory=benchmark,
        economic_directory=economic,
    )

    assert report.startswith("# Financial Decision Support System")
    assert "Etap 1 został ukończony eksperymentalnie" in report
    assert "5/5" in report
    assert "0.394369" in report
    assert "-0.053547" in report
    assert "nie wyznacza wag portfela" in report
    assert "nie zawiera finalnego twierdzenia" in report


def test_write_report_is_deterministic(
    tmp_path: Path,
    report_artifacts: tuple[Path, Path, Path],
) -> None:
    stability, benchmark, economic = report_artifacts
    output_path = tmp_path / "docs" / "FORECASTING_V2_RESULTS.md"

    first_path = write_forecast_v2_report(
        stability_directory=stability,
        benchmark_directory=benchmark,
        economic_directory=economic,
        output_path=output_path,
    )
    first_text = first_path.read_text(encoding="utf-8")
    second_path = write_forecast_v2_report(
        stability_directory=stability,
        benchmark_directory=benchmark,
        economic_directory=economic,
        output_path=output_path,
    )

    assert second_path == output_path
    assert second_path.read_text(encoding="utf-8") == first_text


def test_report_rejects_test_split_access(
    report_artifacts: tuple[Path, Path, Path],
) -> None:
    stability, benchmark, economic = report_artifacts
    metadata_path = benchmark / "benchmark_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["test_split_read"] = True
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="test split was not read"):
        build_forecast_v2_report(
            stability_directory=stability,
            benchmark_directory=benchmark,
            economic_directory=economic,
        )


def test_report_rejects_changed_economic_decision(
    report_artifacts: tuple[Path, Path, Path],
) -> None:
    stability, benchmark, economic = report_artifacts
    decision_path = economic / "economic_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["decision"] = "promote_to_allocation"
    _write_json(decision_path, decision)

    with pytest.raises(ValueError, match="predictive decision support"):
        build_forecast_v2_report(
            stability_directory=stability,
            benchmark_directory=benchmark,
            economic_directory=economic,
        )


def test_main_writes_requested_output(
    tmp_path: Path,
    report_artifacts: tuple[Path, Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    stability, benchmark, economic = report_artifacts
    output_path = tmp_path / "result.md"

    exit_code = main(
        [
            "--stability-directory",
            str(stability),
            "--benchmark-directory",
            str(benchmark),
            "--economic-directory",
            str(economic),
            "--output-path",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()
    output = capsys.readouterr().out
    assert "Promoted allocation signal: no" in output
    assert "Final-test claim: no" in output

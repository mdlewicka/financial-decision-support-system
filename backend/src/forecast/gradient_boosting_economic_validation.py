"""Economic screening of the nested Forecasting V2 boosting challenger."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from backend.src.config import PROJECT_ROOT
from backend.src.forecast.forecast_v2_economic_validation import (
    POLICY_NAME,
    EconomicValidationConfig,
    EconomicValidationPaths,
    EconomicValidationResult,
    run_economic_validation,
)

DEFAULT_CHALLENGER_DIRECTORY = (
    PROJECT_ROOT / "benchmark_results" / "forecast_v2" / "boosting_challenger"
)
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "benchmark_results" / "forecast_v2" / "boosting_economic_validation"
)
SOURCE_MODEL = "hist_gradient_boosting_v2"
BOOSTING_POLICY_NAME = "hist_gradient_boosting_5d_long_cash"


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Economically screen the nested five-session gradient boosting "
            "challenger without reading the final test split."
        )
    )
    parser.add_argument(
        "--challenger-directory",
        type=Path,
        default=DEFAULT_CHALLENGER_DIRECTORY,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def run_boosting_economic_validation(
    challenger_observations: pd.DataFrame,
    config: EconomicValidationConfig | None = None,
) -> EconomicValidationResult:
    selected = challenger_observations.copy()
    if "model" not in selected.columns or not selected["model"].eq(SOURCE_MODEL).all():
        raise ValueError(f"All observations must belong to {SOURCE_MODEL}.")

    selected["model"] = "logistic_v2"
    original = run_economic_validation(selected, config)
    period_observations = _rename_policy(original.period_observations)
    path_metrics = _rename_policy(original.path_metrics)
    decision = {
        **original.decision,
        "candidate": BOOSTING_POLICY_NAME,
        "source_model": SOURCE_MODEL,
    }
    return EconomicValidationResult(
        period_observations=period_observations,
        path_metrics=path_metrics,
        threshold_summary=original.threshold_summary,
        bootstrap_comparison=original.bootstrap_comparison,
        decision=decision,
    )


def export_boosting_economic_validation(
    result: EconomicValidationResult,
    output_directory: Path,
    challenger_observations: pd.DataFrame,
    challenger_metadata: dict[str, Any],
    challenger_decision: dict[str, Any],
    config: EconomicValidationConfig,
) -> EconomicValidationPaths:
    _validate_challenger_provenance(challenger_metadata, challenger_decision)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = EconomicValidationPaths(
        period_observations_csv=output_directory / "policy_period_observations.csv",
        path_metrics_csv=output_directory / "policy_path_metrics.csv",
        threshold_summary_csv=output_directory / "threshold_summary.csv",
        bootstrap_csv=output_directory / "bootstrap_comparison.csv",
        decision_json=output_directory / "economic_decision.json",
        metadata_json=output_directory / "economic_validation_metadata.json",
    )
    for frame, path in (
        (result.period_observations, paths.period_observations_csv),
        (result.path_metrics, paths.path_metrics_csv),
        (result.threshold_summary, paths.threshold_summary_csv),
        (result.bootstrap_comparison, paths.bootstrap_csv),
    ):
        _prepare_csv(frame).to_csv(path, index=False)
    paths.decision_json.write_text(
        json.dumps(result.decision, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "validation_name": "forecast_v2_boosting_economic_validation",
        "source_model": SOURCE_MODEL,
        "development_splits_used": ["training", "validation"],
        "test_split_read": False,
        "known_v1_test_used_for_selection": False,
        "source_observations_fingerprint": _frame_fingerprint(challenger_observations),
        "source_challenger_decision": challenger_decision.get("decision"),
        "config": asdict(config),
        "policy": {
            "name": BOOSTING_POLICY_NAME,
            "positions": ["cash", "long"],
            "short_selling": False,
            "decision_rule": (
                "Long only when P(up) is the largest class probability and is "
                "at least the fixed confidence threshold; otherwise cash."
            ),
            "periods_overlap": False,
            "entry_exit_turnover_costed": True,
        },
        "limitation": (
            "This is nested out-of-fold development screening, not the one-time "
            "final test and not a claim of live profitability."
        ),
    }
    paths.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return paths


def run_from_directory(
    challenger_directory: Path = DEFAULT_CHALLENGER_DIRECTORY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    config: EconomicValidationConfig | None = None,
) -> tuple[EconomicValidationResult, EconomicValidationPaths]:
    selected_config = config or EconomicValidationConfig()
    observations = pd.read_csv(
        challenger_directory / "challenger_observations.csv",
        parse_dates=["date", "target_date"],
    )
    metadata = json.loads(
        (challenger_directory / "challenger_metadata.json").read_text(encoding="utf-8")
    )
    decision = json.loads(
        (challenger_directory / "challenger_decision.json").read_text(encoding="utf-8")
    )
    _validate_challenger_provenance(metadata, decision)
    result = run_boosting_economic_validation(observations, selected_config)
    paths = export_boosting_economic_validation(
        result,
        output_directory,
        observations,
        metadata,
        decision,
        selected_config,
    )
    return result, paths


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = create_argument_parser().parse_args(arguments)
    config = EconomicValidationConfig(
        transaction_cost=parsed.transaction_cost,
        bootstrap_iterations=parsed.bootstrap_iterations,
        random_state=parsed.random_state,
    )
    result, paths = run_from_directory(
        challenger_directory=parsed.challenger_directory,
        output_directory=parsed.output_directory,
        config=config,
    )
    print("Forecasting V2 gradient boosting economic validation")
    print("Policy: five-session gradient boosting long/cash")
    print("Periods overlap: no")
    print("Short selling: no")
    print("Test split read: no")
    print("\nThreshold sensitivity:")
    print(result.threshold_summary.to_string(index=False))
    print("\nDecision:")
    print(json.dumps(result.decision, ensure_ascii=False, indent=2))
    print("\nArtifacts:")
    for path in asdict(paths).values():
        print(path)
    return 0


def _rename_policy(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    if "strategy" in renamed.columns:
        renamed["strategy"] = renamed["strategy"].replace(
            {POLICY_NAME: BOOSTING_POLICY_NAME}
        )
    return renamed


def _validate_challenger_provenance(
    metadata: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    if metadata.get("test_split_read") is not False:
        raise ValueError("Challenger metadata must confirm test_split_read=false.")
    if metadata.get("known_v1_test_used_for_selection") is not False:
        raise ValueError("Known V1 test must not be used for challenger selection.")
    if decision.get("predictive_gate_passed") is not True:
        raise ValueError("Challenger must pass the predictive gate before screening.")
    if decision.get("test_split_read") is not False:
        raise ValueError("Challenger decision must confirm test_split_read=false.")


def _prepare_csv(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in ("date", "target_date"):
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(prepared[column]).dt.strftime("%Y-%m-%d")
    return prepared


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(
        frame.reset_index(drop=True), index=True
    ).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

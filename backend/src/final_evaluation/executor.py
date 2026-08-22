from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backend.src.config import (
    TEST_DATASET_FILENAME,
    TRAIN_DATASET_FILENAME,
)
from backend.src.final_evaluation.cli import (
    DEFAULT_DATASET_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    FROZEN_ARTIFACTS,
    calculate_sha256,
    run_preflight,
)
from backend.src.final_evaluation.forecast import (
    build_forecast_metrics_frame,
    build_forecast_observations_frame,
    evaluate_final_forecasts,
)
from backend.src.final_evaluation.guard import (
    FINAL_TEST_CONFIRMATION,
    FinalTestGuard,
)
from backend.src.final_evaluation.portfolio import (
    evaluate_frozen_portfolios,
)
from backend.src.final_evaluation.rl import (
    build_rl_summary_frame,
    build_rl_trajectory_frame,
    build_rl_weights_frame,
    evaluate_frozen_ppo,
)
from backend.src.storage.market_dataset_reader import (
    MarketDatasetReader,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "benchmark_results" / "final_evaluation_frozen_manifest.json"
)


@dataclass(frozen=True)
class FinalEvaluationOutputPaths:
    marker: Path
    result_json: Path
    forecast_metrics: Path
    forecast_observations: Path
    portfolio_summary: Path
    rl_summary: Path
    rl_trajectory: Path
    rl_weights: Path


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled one-time final evaluation "
            "on the closed market test split."
        )
    )

    parser.add_argument(
        "--confirm",
        required=True,
        help=(
            f"Required one-time confirmation. Exact value: {FINAL_TEST_CONFIRMATION}"
        ),
    )

    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=DEFAULT_DATASET_DIRECTORY,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )

    return parser


def verify_frozen_manifest(
    manifest_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    project_root = Path(project_root).resolve()

    payload = json.loads(
        manifest_path.read_text(
            encoding="utf-8",
        )
    )

    if payload.get("schema_version") != 1:
        raise ValueError("Frozen manifest has an unsupported schema version.")

    if payload.get("status") != "frozen":
        raise ValueError("Frozen manifest status must be 'frozen'.")

    if payload.get("test_split_used") is not False:
        raise ValueError(
            "Frozen manifest must confirm that the test split was not previously used."
        )

    artifact_records = payload.get("artifacts")

    if not isinstance(
        artifact_records,
        list,
    ):
        raise ValueError("Frozen manifest must contain an artifacts list.")

    expected_paths = {str(path) for path in FROZEN_ARTIFACTS}

    actual_paths: set[str] = set()

    for record in artifact_records:
        if not isinstance(
            record,
            dict,
        ):
            raise ValueError("Every frozen artifact record must be an object.")

        raw_path = record.get("path")
        expected_sha256 = record.get("sha256")

        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Frozen artifact path must be a non-empty string.")

        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("Frozen artifact SHA256 must contain 64 characters.")

        relative_path = Path(raw_path)

        if relative_path.is_absolute():
            raise ValueError("Frozen artifact paths must be relative.")

        artifact_path = (project_root / relative_path).resolve()

        if not artifact_path.is_relative_to(project_root):
            raise ValueError("Frozen artifact path escapes the project root.")

        if not artifact_path.is_file():
            raise FileNotFoundError(f"Frozen artifact does not exist: {relative_path}")

        actual_sha256 = calculate_sha256(artifact_path)

        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "Frozen artifact changed after the manifest "
                f"was created: {relative_path}"
            )

        actual_paths.add(raw_path)

    if actual_paths != expected_paths:
        missing = sorted(expected_paths.difference(actual_paths))
        unexpected = sorted(actual_paths.difference(expected_paths))

        raise ValueError(
            "Frozen manifest artifact set does not match "
            "the final-evaluation requirements "
            f"(missing={missing}; unexpected={unexpected})."
        )

    return payload


def _read_market_dataset(
    dataset_directory: Path,
    file_name: str,
) -> pd.DataFrame:
    return pd.read_csv(
        Path(dataset_directory) / file_name,
        parse_dates=(MarketDatasetReader.DATE_COLUMNS),
    )


def _records(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    return json.loads(
        frame.to_json(
            orient="records",
            date_format="iso",
        )
    )


def _write_failure_record(
    output_directory: Path,
    error: Exception,
) -> None:
    path = Path(output_directory) / "final_evaluation_failure.json"

    payload = {
        "schema_version": 1,
        "status": "failed_after_final_test_open",
        "failed_at_utc": datetime.now(UTC).isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "test_split_used": True,
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def run_final_evaluation(
    confirmation: str,
    dataset_directory: Path = DEFAULT_DATASET_DIRECTORY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    project_root: Path = PROJECT_ROOT,
) -> FinalEvaluationOutputPaths:
    project_root = Path(project_root).resolve()

    dataset_directory = Path(dataset_directory)

    output_directory = Path(output_directory)

    manifest_path = Path(manifest_path)

    marker_path = output_directory / "final_test_usage.json"

    run_preflight(
        dataset_directory=dataset_directory,
        output_directory=output_directory,
        marker_path=marker_path,
        project_root=project_root,
    )

    frozen_manifest = verify_frozen_manifest(
        manifest_path=manifest_path,
        project_root=project_root,
    )

    manifest_sha256 = calculate_sha256(manifest_path)

    guard = FinalTestGuard(
        marker_path=marker_path,
    )

    usage_record = guard.open_once(
        confirmation=confirmation,
    )

    try:
        training_dataset = _read_market_dataset(
            dataset_directory=dataset_directory,
            file_name=TRAIN_DATASET_FILENAME,
        )

        test_dataset = _read_market_dataset(
            dataset_directory=dataset_directory,
            file_name=TEST_DATASET_FILENAME,
        )

        forecast_results = evaluate_final_forecasts(
            training_dataset=training_dataset,
            evaluation_dataset=test_dataset,
            arima_selection_path=(
                project_root
                / "benchmark_results"
                / "forecast"
                / "arima"
                / "arima_order_selection.json"
            ),
        )

        forecast_metrics = build_forecast_metrics_frame(
            results=forecast_results,
            evaluation_split="test",
        )

        forecast_observations = build_forecast_observations_frame(
            results=forecast_results,
        )

        portfolio_summary = evaluate_frozen_portfolios(
            evaluation_dataset=test_dataset,
            weights_path=(
                project_root
                / "benchmark_results"
                / "risk"
                / "portfolio"
                / "portfolio_weights.csv"
            ),
        )

        portfolio_summary.insert(
            2,
            "evaluation_split",
            "test",
        )

        rl_result = evaluate_frozen_ppo(
            training_dataset=training_dataset,
            evaluation_dataset=test_dataset,
            result_path=(
                project_root / "benchmark_results" / "rl" / "ppo" / "rl_result.json"
            ),
            model_path=(
                project_root
                / "benchmark_results"
                / "rl"
                / "ppo"
                / "ppo_allocation_agent.zip"
            ),
        )

        rl_summary = build_rl_summary_frame(
            result=rl_result,
            evaluation_split="test",
        )

        rl_trajectory = build_rl_trajectory_frame(
            result=rl_result,
        )

        rl_weights = build_rl_weights_frame(
            result=rl_result,
        )

        forecast_directory = output_directory / "forecast"
        portfolio_directory = output_directory / "portfolio"
        rl_directory = output_directory / "rl"

        forecast_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        portfolio_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        rl_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        forecast_metrics_path = forecast_directory / "forecast_metrics.csv"
        forecast_observations_path = forecast_directory / "forecast_observations.csv"
        portfolio_summary_path = portfolio_directory / "portfolio_summary.csv"
        rl_summary_path = rl_directory / "rl_summary.csv"
        rl_trajectory_path = rl_directory / "rl_trajectory.csv"
        rl_weights_path = rl_directory / "rl_weights.csv"

        forecast_metrics.to_csv(
            forecast_metrics_path,
            index=False,
        )
        forecast_observations.to_csv(
            forecast_observations_path,
            index=False,
        )
        portfolio_summary.to_csv(
            portfolio_summary_path,
            index=False,
        )
        rl_summary.to_csv(
            rl_summary_path,
            index=False,
        )
        rl_trajectory.to_csv(
            rl_trajectory_path,
            index=False,
        )
        rl_weights.to_csv(
            rl_weights_path,
            index=False,
        )

        result_json_path = output_directory / "final_evaluation_result.json"

        result_payload = {
            "schema_version": 1,
            "status": "completed",
            "evaluation_split": "test",
            "test_split_used": True,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "usage_record": asdict(usage_record),
            "frozen_manifest": {
                "path": str(manifest_path),
                "sha256": (manifest_sha256),
                "frozen_at_utc": (frozen_manifest.get("frozen_at_utc")),
            },
            "forecast": _records(forecast_metrics),
            "portfolio": _records(portfolio_summary),
            "rl": _records(rl_summary),
            "output_files": {
                "forecast_metrics": str(forecast_metrics_path),
                "forecast_observations": str(forecast_observations_path),
                "portfolio_summary": str(portfolio_summary_path),
                "rl_summary": str(rl_summary_path),
                "rl_trajectory": str(rl_trajectory_path),
                "rl_weights": str(rl_weights_path),
            },
        }

        result_json_path.write_text(
            json.dumps(
                result_payload,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    except Exception as error:
        _write_failure_record(
            output_directory=(output_directory),
            error=error,
        )
        raise

    return FinalEvaluationOutputPaths(
        marker=marker_path,
        result_json=result_json_path,
        forecast_metrics=(forecast_metrics_path),
        forecast_observations=(forecast_observations_path),
        portfolio_summary=(portfolio_summary_path),
        rl_summary=rl_summary_path,
        rl_trajectory=(rl_trajectory_path),
        rl_weights=rl_weights_path,
    )


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    parser = create_argument_parser()

    parsed = parser.parse_args(arguments)

    paths = run_final_evaluation(
        confirmation=parsed.confirm,
        dataset_directory=(parsed.dataset_directory),
        output_directory=(parsed.output_directory),
        manifest_path=(parsed.manifest_path),
    )

    print("FINAL TEST EVALUATION COMPLETED")
    print(
        "Result:",
        paths.result_json,
    )
    print(
        "Usage marker:",
        paths.marker,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

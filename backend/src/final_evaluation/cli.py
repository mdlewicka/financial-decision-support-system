from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.src.final_evaluation.guard import FinalTestGuard

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATASET_DIRECTORY = PROJECT_ROOT / "data" / "datasets" / "market"

DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "benchmark_results" / "final_test"

DEFAULT_MARKER_PATH = DEFAULT_OUTPUT_DIRECTORY / "final_test_usage.json"

FROZEN_ARTIFACTS = (
    Path("benchmark_results/forecast/naive/forecast_result.json"),
    Path("benchmark_results/forecast/random_forest/forecast_result.json"),
    Path("benchmark_results/forecast/arima/forecast_result.json"),
    Path("benchmark_results/forecast/arima/arima_order_selection.json"),
    Path("benchmark_results/risk/portfolio/portfolio_result.json"),
    Path("benchmark_results/risk/portfolio/portfolio_weights.csv"),
    Path("benchmark_results/rl/ppo/rl_result.json"),
    Path("benchmark_results/rl/ppo/ppo_allocation_agent.zip"),
)


@dataclass(frozen=True)
class ArtifactFingerprint:
    path: Path
    sha256: str


@dataclass(frozen=True)
class FinalEvaluationPreflight:
    dataset_directory: Path
    output_directory: Path
    marker_path: Path
    artifacts: tuple[ArtifactFingerprint, ...]


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight checks for the controlled one-time final test evaluation."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the frozen final-evaluation setup "
            "without opening or reading the test split."
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
        "--marker-path",
        type=Path,
        default=DEFAULT_MARKER_PATH,
    )

    return parser


def calculate_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def run_preflight(
    dataset_directory: Path = DEFAULT_DATASET_DIRECTORY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    marker_path: Path = DEFAULT_MARKER_PATH,
    project_root: Path = PROJECT_ROOT,
) -> FinalEvaluationPreflight:
    dataset_directory = Path(dataset_directory)
    output_directory = Path(output_directory)
    marker_path = Path(marker_path)
    project_root = Path(project_root)

    required_datasets = (
        "train.csv",
        "validation.csv",
        "test.csv",
    )

    missing_datasets = [
        name for name in required_datasets if not (dataset_directory / name).is_file()
    ]

    if missing_datasets:
        raise FileNotFoundError(
            f"Missing required market dataset files: {missing_datasets}"
        )

    guard = FinalTestGuard(
        marker_path=marker_path,
    )

    if guard.has_been_used:
        raise RuntimeError(f"Final test usage marker already exists: {marker_path}")

    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(
            "Final evaluation output directory must "
            "be empty before the one-time run: "
            f"{output_directory}"
        )

    missing_artifacts: list[Path] = []
    artifact_paths: list[Path] = []

    for relative_path in FROZEN_ARTIFACTS:
        artifact_path = project_root / relative_path

        if not artifact_path.is_file():
            missing_artifacts.append(artifact_path)
        else:
            artifact_paths.append(artifact_path)

    if missing_artifacts:
        formatted = ", ".join(str(path) for path in missing_artifacts)
        raise FileNotFoundError(f"Missing frozen evaluation artifacts: {formatted}")

    fingerprints = tuple(
        ArtifactFingerprint(
            path=path,
            sha256=calculate_sha256(path),
        )
        for path in artifact_paths
    )

    return FinalEvaluationPreflight(
        dataset_directory=dataset_directory,
        output_directory=output_directory,
        marker_path=marker_path,
        artifacts=fingerprints,
    )


def print_preflight(
    result: FinalEvaluationPreflight,
) -> None:
    print("FINAL EVALUATION PREFLIGHT")
    print("==========================")
    print()
    print(
        "Dataset directory:",
        result.dataset_directory,
    )
    print(
        "Output directory:",
        result.output_directory,
    )
    print(
        "Usage marker:",
        result.marker_path,
    )

    print()
    print("Frozen artifacts:")

    for artifact in result.artifacts:
        relative_path = artifact.path.relative_to(PROJECT_ROOT)

        print(f"- {relative_path}")
        print(f"  SHA256: {artifact.sha256}")

    print()
    print("Status: READY")
    print("The test.csv file was not read.")
    print("The final test usage marker was not created.")


def main(
    arguments: Sequence[str] | None = None,
) -> None:
    parser = create_argument_parser()
    parsed = parser.parse_args(arguments)

    if not parsed.dry_run:
        parser.error(
            "Only --dry-run is available at this stage. "
            "The final test split remains closed."
        )

    result = run_preflight(
        dataset_directory=parsed.dataset_directory,
        output_directory=parsed.output_directory,
        marker_path=parsed.marker_path,
    )

    print_preflight(result)


if __name__ == "__main__":
    main()

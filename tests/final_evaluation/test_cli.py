from pathlib import Path

import pytest

from backend.src.final_evaluation import cli


def prepare_project(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    dataset_directory = tmp_path / "data"
    dataset_directory.mkdir()

    for name in (
        "train.csv",
        "validation.csv",
        "test.csv",
    ):
        (dataset_directory / name).write_text(
            "placeholder\n",
            encoding="utf-8",
        )

    for relative_path in cli.FROZEN_ARTIFACTS:
        artifact_path = tmp_path / relative_path
        artifact_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        artifact_path.write_bytes(b"frozen-artifact")

    output_directory = tmp_path / "benchmark_results" / "final_test"

    marker_path = output_directory / "final_test_usage.json"

    return (
        dataset_directory,
        output_directory,
        marker_path,
    )


def test_preflight_accepts_complete_frozen_setup(
    tmp_path: Path,
) -> None:
    (
        dataset_directory,
        output_directory,
        marker_path,
    ) = prepare_project(tmp_path)

    result = cli.run_preflight(
        dataset_directory=dataset_directory,
        output_directory=output_directory,
        marker_path=marker_path,
        project_root=tmp_path,
    )

    assert result.dataset_directory == (dataset_directory)
    assert result.output_directory == (output_directory)
    assert not marker_path.exists()
    assert len(result.artifacts) == len(cli.FROZEN_ARTIFACTS)

    assert all(len(artifact.sha256) == 64 for artifact in result.artifacts)


def test_preflight_rejects_missing_artifact(
    tmp_path: Path,
) -> None:
    (
        dataset_directory,
        output_directory,
        marker_path,
    ) = prepare_project(tmp_path)

    missing_relative_path = cli.FROZEN_ARTIFACTS[0]

    (tmp_path / missing_relative_path).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Missing frozen evaluation artifacts",
    ):
        cli.run_preflight(
            dataset_directory=dataset_directory,
            output_directory=output_directory,
            marker_path=marker_path,
            project_root=tmp_path,
        )


def test_preflight_rejects_existing_usage_marker(
    tmp_path: Path,
) -> None:
    (
        dataset_directory,
        output_directory,
        marker_path,
    ) = prepare_project(tmp_path)

    marker_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    marker_path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="usage marker already exists",
    ):
        cli.run_preflight(
            dataset_directory=dataset_directory,
            output_directory=output_directory,
            marker_path=marker_path,
            project_root=tmp_path,
        )


def test_preflight_rejects_non_empty_output(
    tmp_path: Path,
) -> None:
    (
        dataset_directory,
        output_directory,
        marker_path,
    ) = prepare_project(tmp_path)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_directory / "unexpected.txt").write_text(
        "existing output",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="must be empty",
    ):
        cli.run_preflight(
            dataset_directory=dataset_directory,
            output_directory=output_directory,
            marker_path=marker_path,
            project_root=tmp_path,
        )


def test_cli_requires_dry_run() -> None:
    with pytest.raises(
        SystemExit,
    ):
        cli.main([])

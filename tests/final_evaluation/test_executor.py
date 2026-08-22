import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from backend.src.final_evaluation import (
    executor,
)
from backend.src.final_evaluation.guard import (
    FINAL_TEST_CONFIRMATION,
)


def sha256(
    path: Path,
) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_project(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
]:
    dataset_directory = tmp_path / "data" / "datasets" / "market"
    dataset_directory.mkdir(parents=True)

    for file_name in (
        "train.csv",
        "validation.csv",
        "test.csv",
    ):
        (dataset_directory / file_name).write_text(
            "placeholder\n",
            encoding="utf-8",
        )

    artifacts = []

    for relative_path in executor.FROZEN_ARTIFACTS:
        artifact_path = tmp_path / relative_path
        artifact_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        artifact_path.write_bytes(("frozen:" + str(relative_path)).encode())

        artifacts.append(
            {
                "path": str(relative_path),
                "sha256": sha256(artifact_path),
            }
        )

    manifest_path = (
        tmp_path / "benchmark_results" / "final_evaluation_frozen_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "frozen",
                "frozen_at_utc": ("2026-01-01T00:00:00+00:00"),
                "test_split_used": False,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )

    output_directory = tmp_path / "benchmark_results" / "final_test"

    return (
        dataset_directory,
        output_directory,
        manifest_path,
    )


def test_manifest_rejects_changed_artifact(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        manifest_path,
    ) = prepare_project(tmp_path)

    changed_path = tmp_path / executor.FROZEN_ARTIFACTS[0]

    changed_path.write_bytes(b"changed")

    with pytest.raises(
        RuntimeError,
        match="changed after",
    ):
        executor.verify_frozen_manifest(
            manifest_path=manifest_path,
            project_root=tmp_path,
        )


def test_final_evaluation_opens_test_after_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        dataset_directory,
        output_directory,
        manifest_path,
    ) = prepare_project(tmp_path)

    marker_path = output_directory / "final_test_usage.json"

    read_order: list[str] = []

    def fake_read(
        dataset_directory: Path,
        file_name: str,
    ) -> pd.DataFrame:
        del dataset_directory

        if file_name == "test.csv":
            assert marker_path.exists()

        read_order.append(file_name)

        return pd.DataFrame({"placeholder": [1]})

    monkeypatch.setattr(
        executor,
        "_read_market_dataset",
        fake_read,
    )

    monkeypatch.setattr(
        executor,
        "evaluate_final_forecasts",
        lambda **_: {"dummy": object()},
    )

    monkeypatch.setattr(
        executor,
        "build_forecast_metrics_frame",
        lambda **_: pd.DataFrame(
            [
                {
                    "model_name": "dummy",
                    "evaluation_split": "test",
                    "observations": 1,
                }
            ]
        ),
    )

    monkeypatch.setattr(
        executor,
        "build_forecast_observations_frame",
        lambda **_: pd.DataFrame(
            [
                {
                    "model_name": "dummy",
                    "actual": 0.0,
                    "prediction": 0.0,
                }
            ]
        ),
    )

    monkeypatch.setattr(
        executor,
        "evaluate_frozen_portfolios",
        lambda **_: pd.DataFrame(
            [
                {
                    "scenario": "dummy",
                    "construction": "frozen",
                    "observations": 1,
                }
            ]
        ),
    )

    monkeypatch.setattr(
        executor,
        "evaluate_frozen_ppo",
        lambda **_: object(),
    )

    monkeypatch.setattr(
        executor,
        "build_rl_summary_frame",
        lambda **_: pd.DataFrame(
            [
                {
                    "strategy": "ppo",
                    "evaluation_split": "test",
                    "observations": 1,
                }
            ]
        ),
    )

    monkeypatch.setattr(
        executor,
        "build_rl_trajectory_frame",
        lambda **_: pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "portfolio_value": 1.0,
                }
            ]
        ),
    )

    monkeypatch.setattr(
        executor,
        "build_rl_weights_frame",
        lambda **_: pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "AAA": 1.0,
                }
            ]
        ),
    )

    paths = executor.run_final_evaluation(
        confirmation=(FINAL_TEST_CONFIRMATION),
        dataset_directory=(dataset_directory),
        output_directory=(output_directory),
        manifest_path=(manifest_path),
        project_root=tmp_path,
    )

    assert read_order == [
        "train.csv",
        "test.csv",
    ]

    assert paths.marker.exists()
    assert paths.result_json.exists()
    assert paths.forecast_metrics.exists()
    assert paths.portfolio_summary.exists()
    assert paths.rl_summary.exists()

    payload = json.loads(paths.result_json.read_text(encoding="utf-8"))

    assert payload["status"] == "completed"
    assert payload["evaluation_split"] == "test"
    assert payload["test_split_used"] is True


def test_wrong_confirmation_never_reads_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        dataset_directory,
        output_directory,
        manifest_path,
    ) = prepare_project(tmp_path)

    reads = []

    def fake_read(
        dataset_directory: Path,
        file_name: str,
    ) -> pd.DataFrame:
        del dataset_directory
        reads.append(file_name)
        return pd.DataFrame()

    monkeypatch.setattr(
        executor,
        "_read_market_dataset",
        fake_read,
    )

    with pytest.raises(
        ValueError,
        match="exact confirmation",
    ):
        executor.run_final_evaluation(
            confirmation="wrong",
            dataset_directory=(dataset_directory),
            output_directory=(output_directory),
            manifest_path=(manifest_path),
            project_root=tmp_path,
        )

    assert reads == []

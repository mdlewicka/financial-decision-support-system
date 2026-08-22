import json
from pathlib import Path

import pytest

from backend.src.final_evaluation.guard import (
    FINAL_TEST_CONFIRMATION,
    FinalTestAlreadyUsedError,
    FinalTestConfirmationError,
    FinalTestGuard,
)


def test_guard_rejects_missing_confirmation(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "final_test_usage.json"
    guard = FinalTestGuard(
        marker_path=marker_path,
    )

    with pytest.raises(
        FinalTestConfirmationError,
        match="exact confirmation",
    ):
        guard.open_once(
            confirmation="wrong",
        )

    assert not marker_path.exists()


def test_guard_creates_usage_marker_once(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "final_test_usage.json"
    guard = FinalTestGuard(
        marker_path=marker_path,
    )

    record = guard.open_once(
        confirmation=FINAL_TEST_CONFIRMATION,
    )

    assert marker_path.exists()
    assert guard.has_been_used is True
    assert record.schema_version == 1
    assert record.status == "opened"

    payload = json.loads(
        marker_path.read_text(
            encoding="utf-8",
        )
    )

    assert payload["schema_version"] == 1
    assert payload["status"] == "opened"
    assert payload["opened_at_utc"]
    assert payload["purpose"]


def test_guard_blocks_second_open(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "final_test_usage.json"
    guard = FinalTestGuard(
        marker_path=marker_path,
    )

    guard.open_once(
        confirmation=FINAL_TEST_CONFIRMATION,
    )

    with pytest.raises(
        FinalTestAlreadyUsedError,
        match="already been opened",
    ):
        guard.open_once(
            confirmation=FINAL_TEST_CONFIRMATION,
        )

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

FINAL_TEST_CONFIRMATION = "OPEN_FINAL_TEST_ONCE"


class FinalTestAlreadyUsedError(RuntimeError):
    """Raised when the final test split has already been opened."""


class FinalTestConfirmationError(ValueError):
    """Raised when explicit final-test confirmation is missing or invalid."""


@dataclass(frozen=True)
class FinalTestUsageRecord:
    schema_version: int
    status: str
    opened_at_utc: str
    purpose: str

    @classmethod
    def create(cls) -> FinalTestUsageRecord:
        return cls(
            schema_version=1,
            status="opened",
            opened_at_utc=datetime.now(UTC).isoformat(),
            purpose=(
                "One final evaluation after model and application "
                "development decisions were frozen."
            ),
        )


class FinalTestGuard:
    """Prevent accidental repeated use of the closed final test split."""

    def __init__(
        self,
        marker_path: Path,
    ) -> None:
        self.marker_path = Path(marker_path)

    @property
    def has_been_used(self) -> bool:
        return self.marker_path.exists()

    def require_confirmation(
        self,
        confirmation: str,
    ) -> None:
        if confirmation != FINAL_TEST_CONFIRMATION:
            raise FinalTestConfirmationError(
                "Final test evaluation requires the exact confirmation "
                f"{FINAL_TEST_CONFIRMATION!r}."
            )

    def open_once(
        self,
        confirmation: str,
    ) -> FinalTestUsageRecord:
        self.require_confirmation(
            confirmation=confirmation,
        )

        self.marker_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        record = FinalTestUsageRecord.create()

        try:
            with self.marker_path.open(
                "x",
                encoding="utf-8",
            ) as marker_file:
                json.dump(
                    asdict(record),
                    marker_file,
                    indent=2,
                    ensure_ascii=False,
                )
                marker_file.write("\n")
        except FileExistsError as exc:
            raise FinalTestAlreadyUsedError(
                "The final test split has already been opened. "
                f"Usage marker: {self.marker_path}"
            ) from exc

        return record

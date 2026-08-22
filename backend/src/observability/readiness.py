"""Readiness checks for local deterministic application artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from backend.src.config import PROJECT_ROOT

ReadinessStatus = Literal["ready", "missing"]

REQUIRED_ARTIFACTS: Final[tuple[tuple[str, Path], ...]] = (
    (
        "forecast_evidence",
        PROJECT_ROOT
        / "benchmark_results"
        / "forecast"
        / "random_forest"
        / "forecast_result.json",
    ),
    (
        "portfolio_evidence",
        PROJECT_ROOT
        / "benchmark_results"
        / "risk"
        / "portfolio"
        / "portfolio_result.json",
    ),
    (
        "rag_corpus",
        PROJECT_ROOT / "data" / "rag" / "evaluation" / "reference_corpus.json",
    ),
    (
        "agent_evidence",
        PROJECT_ROOT / "benchmark_results" / "agents" / "agent_results.json",
    ),
)


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    """Describe one non-sensitive runtime dependency check."""

    name: str
    status: ReadinessStatus
    required: bool = True


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    """Describe aggregate application readiness."""

    ready: bool
    checks: tuple[ReadinessCheck, ...]

    @property
    def status(self) -> Literal["ready", "not_ready"]:
        return "ready" if self.ready else "not_ready"


class ReadinessService:
    """Evaluate required local artifacts without loading their contents."""

    def __init__(
        self,
        artifacts: tuple[tuple[str, Path], ...] = REQUIRED_ARTIFACTS,
    ) -> None:
        self._artifacts = artifacts

    def evaluate(self) -> ReadinessResult:
        checks = tuple(
            ReadinessCheck(
                name=name,
                status=(
                    "ready" if self._is_readable_non_empty_file(path) else "missing"
                ),
            )
            for name, path in self._artifacts
        )
        return ReadinessResult(
            ready=all(check.status == "ready" for check in checks),
            checks=checks,
        )

    @staticmethod
    def _is_readable_non_empty_file(path: Path) -> bool:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                return False
            with path.open("rb") as stream:
                return bool(stream.read(1))
        except OSError:
            return False

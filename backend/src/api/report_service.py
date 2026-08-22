"""Application service assembling and exporting decision reports."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from backend.src.agents.artifact_repository import (
    DecisionSupportArtifactRepository,
)
from backend.src.agents.models import AgentResponse
from backend.src.api.agent_repository import AgentResultRepository
from backend.src.api.agent_service import AgentRuntimeError, AgentRuntimeService
from backend.src.reporting.decision_report import (
    ArtifactVersion,
    DecisionReport,
    DecisionReportBuilder,
)
from backend.src.reporting.decision_report_exporter import export_decision_report


class DecisionReportGenerationError(RuntimeError):
    """Raised when trusted report evidence is missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class EmbeddedReportFile:
    """One exact exported file ready for transport through the API."""

    format: str
    file_name: str
    media_type: str
    bytes: int
    sha256: str
    content_base64: str


@dataclass(frozen=True, slots=True)
class DecisionReportBundle:
    """Validated report and its three integrity-protected representations."""

    report: dict[str, Any]
    manifest: dict[str, Any]
    files: tuple[EmbeddedReportFile, ...]


class DecisionReportRuntimeService:
    """Combine server-recomputed decisions with local agent and XAI evidence."""

    def __init__(
        self,
        *,
        agent_service: AgentRuntimeService,
        agent_repository: AgentResultRepository,
        narrative_evaluation_path: Path | None = None,
    ) -> None:
        self._agent_service = agent_service
        self._agent_repository = agent_repository
        self._narrative_evaluation_path = narrative_evaluation_path or (
            agent_repository.project_root
            / "benchmark_results"
            / "agents"
            / "evaluation"
            / "narrative_evaluation.json"
        )

    def build_report(
        self,
        *,
        decision: dict[str, Any],
        question: str,
        presentation_timezone: str,
        generated_at: datetime,
        response_language: str = "pl",
    ) -> DecisionReport:
        """Build a typed report without trusting any client-side result fields."""
        try:
            narrative = self._agent_service.explain_decision(
                question=question,
                decision_context=decision,
                response_language=response_language,
            )
            artifact_repository = DecisionSupportArtifactRepository(
                self._agent_repository.agent_artifact_paths
            )
            xai = artifact_repository.xai_context()
            narrative_evaluation = _read_json(self._narrative_evaluation_path)
            return DecisionReportBuilder().build(
                decision=decision,
                narrative=narrative.to_dict(),
                xai=xai,
                narrative_evaluation=narrative_evaluation,
                source_artifacts=self._source_versions(decision, narrative),
                generated_at=generated_at,
                presentation_timezone=presentation_timezone,
            )
        except DecisionReportGenerationError:
            raise
        except (AgentRuntimeError, OSError, TypeError, ValueError) as error:
            raise DecisionReportGenerationError(
                "Decision report evidence is unavailable or inconsistent."
            ) from error

    def generate_bundle(
        self,
        *,
        decision: dict[str, Any],
        question: str,
        presentation_timezone: str,
        generated_at: datetime,
        response_language: str = "pl",
    ) -> DecisionReportBundle:
        """Create exact downloads without persisting shared request state."""
        report = self.build_report(
            decision=decision,
            question=question,
            presentation_timezone=presentation_timezone,
            generated_at=generated_at,
            response_language=response_language,
        )
        try:
            with TemporaryDirectory(prefix="fdss-decision-report-") as directory:
                paths = export_decision_report(report, Path(directory))
                manifest = _read_json(paths.manifest_json)
                specifications = (
                    ("json", "application/json", paths.result_json),
                    ("markdown", "text/markdown", paths.report_markdown),
                    ("pdf", "application/pdf", paths.report_pdf),
                )
                files = tuple(
                    _embedded_file(
                        format_name=format_name,
                        media_type=media_type,
                        path=path,
                        manifest=manifest,
                    )
                    for format_name, media_type, path in specifications
                )
                report_payload = json.loads(
                    paths.result_json.read_text(encoding="utf-8")
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise DecisionReportGenerationError(
                "Decision report files could not be generated."
            ) from error
        if not isinstance(report_payload, dict):
            raise DecisionReportGenerationError(
                "Decision report JSON must contain an object."
            )
        return DecisionReportBundle(
            report=report_payload,
            manifest=manifest,
            files=files,
        )

    def _source_versions(
        self,
        decision: dict[str, Any],
        narrative: AgentResponse,
    ) -> tuple[ArtifactVersion, ...]:
        paths = self._agent_repository.agent_artifact_paths
        return (
            ArtifactVersion(
                identifier="current_decision_evaluation",
                schema_version="1",
                sha256=_canonical_json_sha256(decision),
            ),
            ArtifactVersion(
                identifier="current_agent_narrative",
                schema_version="1",
                sha256=_canonical_json_sha256(narrative.to_dict()),
            ),
            _file_version(paths.portfolio),
            _file_version(paths.xai),
            _file_version(self._narrative_evaluation_path),
            _file_version(
                self._agent_repository.corpus_path,
                fallback_schema="local-corpus-v1",
            ),
        )


def _embedded_file(
    *,
    format_name: str,
    media_type: str,
    path: Path,
    manifest: dict[str, Any],
) -> EmbeddedReportFile:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise TypeError("Report manifest outputs must be an object.")
    metadata = outputs.get(format_name)
    if not isinstance(metadata, dict):
        raise TypeError(f"Report manifest is missing {format_name} metadata.")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if metadata.get("bytes") != len(content) or metadata.get("sha256") != digest:
        raise ValueError(f"Report {format_name} integrity verification failed.")
    return EmbeddedReportFile(
        format=format_name,
        file_name=path.name,
        media_type=media_type,
        bytes=len(content),
        sha256=digest,
        content_base64=base64.b64encode(content).decode("ascii"),
    )


def _file_version(
    path: Path,
    *,
    fallback_schema: str = "1",
) -> ArtifactVersion:
    payload = _read_json(path)
    schema = payload.get("schema_version", fallback_schema)
    return ArtifactVersion(
        identifier=path.name,
        schema_version=str(schema),
        sha256=_file_sha256(path),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required report artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Report artifact must contain a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

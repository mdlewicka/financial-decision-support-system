"""End-to-end API tests for auditable decision-report downloads."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from backend.src.agents.models import AgentResponse, EvidenceReference
from backend.src.api.agent_repository import AgentResultRepository
from backend.src.api.portfolio_repository import PortfolioArtifactRepository
from backend.src.api.report_service import DecisionReportRuntimeService
from backend.src.api.routes.decision import get_decision_portfolio_repository
from backend.src.api.routes.reports import (
    get_decision_report_service,
    get_report_generated_at,
)
from backend.src.main import create_app


class FakeNarrativeService:
    """Return cited guarded evidence without loading Ollama."""

    def explain_decision(
        self,
        *,
        question: str,
        decision_context: dict[str, object],
        response_language: str = "pl",
    ) -> AgentResponse:
        assert decision_context["deterministic"] is True
        assert response_language in {"pl", "en"}
        return AgentResponse(
            agent_name="allocation_recommendation_agent",
            question=question,
            summary="Ranking został obliczony deterministycznie na walidacji.",
            findings=("Pierwszy dopuszczalny wariant zajmuje pozycję 1.",),
            risks=("Wyniki historyczne nie gwarantują przyszłych rezultatów.",),
            recommendations=("Wymagany jest przegląd człowieka.",),
            limitations=(
                "Wynik nie jest indywidualną poradą finansową ani gwarancją.",
            ),
            evidence=(
                EvidenceReference(
                    "decision_engine",
                    "current_decision_evaluation",
                    "Server-recomputed validation ranking",
                ),
                EvidenceReference(
                    "xai_artifact",
                    "xai_result.json",
                    "Validation forecast-model explanation",
                ),
                EvidenceReference(
                    "rag_document",
                    "portfolio-risk-reference",
                    "Local corpus; chunk=reference-1; rank=1; score=0.8",
                ),
            ),
        )


def _copy_runtime_artifacts(project_root: Path) -> None:
    relative_paths = (
        Path("benchmark_results/forecast/random_forest/forecast_result.json"),
        Path("benchmark_results/xai/random_forest/xai_result.json"),
        Path("benchmark_results/risk/portfolio/portfolio_result.json"),
        Path("benchmark_results/rl/ppo/rl_result.json"),
        Path("data/rag/evaluation/reference_corpus.json"),
    )
    for relative_path in relative_paths:
        destination = project_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative_path, destination)
    evaluation_path = (
        project_root
        / "benchmark_results"
        / "agents"
        / "evaluation"
        / "narrative_evaluation.json"
    )
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "test_split_used": False,
                "chain_of_thought_stored": False,
                "deterministic_gate": {"passed": True},
                "llm_judge": {"status": "approved"},
                "llm_judge_approved_for_reporting": True,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def report_client(tmp_path: Path) -> Iterator[TestClient]:
    _copy_runtime_artifacts(tmp_path)
    repository = AgentResultRepository(project_root=tmp_path)
    report_service = DecisionReportRuntimeService(
        agent_service=FakeNarrativeService(),  # type: ignore[arg-type]
        agent_repository=repository,
    )
    portfolio_repository = PortfolioArtifactRepository(
        artifact_directory=(tmp_path / "benchmark_results" / "risk" / "portfolio")
    )
    application = create_app()
    application.dependency_overrides[get_decision_report_service] = lambda: (
        report_service
    )
    application.dependency_overrides[get_decision_portfolio_repository] = lambda: (
        portfolio_repository
    )
    application.dependency_overrides[get_report_generated_at] = lambda: datetime(
        2026,
        8,
        14,
        9,
        30,
        tzinfo=UTC,
    )
    with TestClient(application) as client:
        yield client


def _request_payload() -> dict[str, object]:
    return {
        "question": "Wyjaśnij rekomendację i ograniczenia.",
        "evaluation": {
            "profile": {"profile_type": "balanced"},
            "available_liquidity_fraction": 0.10,
        },
        "presentation_timezone": "Europe/Warsaw",
    }


def test_report_endpoint_returns_three_verified_files(
    report_client: TestClient,
    tmp_path: Path,
) -> None:
    response = report_client.post("/api/v1/reports/decision", json=_request_payload())

    assert response.status_code == 200
    payload = response.json()
    report = payload["report"]
    assert report["metadata"]["generated_at_utc"] == "2026-08-14T09:30:00+00:00"
    assert report["metadata"]["evaluation_split"] == "validation"
    assert report["metadata"]["test_split_used"] is False
    assert report["governance"]["llm_used_for_ranking"] is False
    assert report["governance"]["chain_of_thought_stored"] is False
    assert report["recommended_scenario"] is not None

    assert {item["format"] for item in payload["files"]} == {
        "json",
        "markdown",
        "pdf",
    }
    decoded: dict[str, bytes] = {}
    for item in payload["files"]:
        content = base64.b64decode(item["content_base64"], validate=True)
        assert item["bytes"] == len(content)
        assert item["sha256"] == hashlib.sha256(content).hexdigest()
        decoded[item["format"]] = content

    markdown = decoded["markdown"].decode("utf-8")
    assert "## Rekomendacja i ranking" in markdown
    assert "portfolio-risk-reference" in markdown
    assert "Chain-of-thought zapisany: NIE" in markdown

    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(decoded["pdf"])
    reader = PdfReader(pdf_path)
    assert len(reader.pages) >= 2


def test_report_endpoint_rejects_untrusted_result_and_unknown_timezone(
    report_client: TestClient,
) -> None:
    payload = _request_payload()
    evaluation = payload["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["recommended_scenario"] = "growth"

    response = report_client.post("/api/v1/reports/decision", json=payload)
    assert response.status_code == 422

    payload = _request_payload()
    payload["presentation_timezone"] = "Mars/Olympus_Mons"
    response = report_client.post("/api/v1/reports/decision", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Decision report evidence is unavailable or inconsistent."
    )

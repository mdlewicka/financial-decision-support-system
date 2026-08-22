import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.src.agents.models import (
    AgentResponse,
    EvidenceReference,
)
from backend.src.api.agent_repository import (
    AgentResultRepository,
)
from backend.src.api.agent_service import (
    AgentRuntimeError,
)
from backend.src.api.portfolio_repository import PortfolioArtifactRepository
from backend.src.api.routes.agents import (
    get_agent_repository,
    get_agent_runtime_service,
)
from backend.src.api.routes.decision import get_decision_portfolio_repository
from backend.src.main import create_app


def _response_payload(
    *,
    agent_name: str,
    question: str,
    evidence_count: int,
) -> dict[str, object]:
    return {
        "agent_name": agent_name,
        "question": question,
        "summary": "Podsumowanie oparte na walidacji.",
        "findings": [
            "Wniosek oparto na zweryfikowanych artefaktach.",
        ],
        "risks": [
            "Wyniki historyczne mogą nie utrzymać się w przyszłości.",
        ],
        "recommendations": [
            "Porównaj wynik z profilem ryzyka i alternatywami.",
        ],
        "limitations": [
            (
                "Wynik ma charakter wsparcia decyzyjnego "
                "i nie jest indywidualną poradą finansową."
            ),
        ],
        "evidence": [
            {
                "source_type": "benchmark_artifact",
                "identifier": f"artifact_{index}.json",
                "description": "Reproducible validation artifact",
            }
            for index in range(evidence_count)
        ],
    }


def _write_agent_results(
    project_root: Path,
) -> Path:
    path = project_root / "benchmark_results" / "agents" / "agent_results.json"
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "model_name": "gemma3:4b",
        "retrieval_strategy": "vector",
        "evaluation_basis": ("validation_benchmark_artifacts"),
        "test_split_used": False,
        "responses": {
            "forecast": _response_payload(
                agent_name=("forecast_analysis_agent"),
                question="Oceń model.",
                evidence_count=2,
            ),
            "risk": _response_payload(
                agent_name=("risk_analysis_agent"),
                question="Porównaj ryzyko.",
                evidence_count=1,
            ),
            "allocation": _response_payload(
                agent_name=("allocation_recommendation_agent"),
                question="Porównaj alokacje.",
                evidence_count=2,
            ),
        },
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


def _write_runtime_inputs(
    project_root: Path,
) -> None:
    paths = (
        (
            project_root
            / "benchmark_results"
            / "forecast"
            / "random_forest"
            / "forecast_result.json"
        ),
        (
            project_root
            / "benchmark_results"
            / "xai"
            / "random_forest"
            / "xai_result.json"
        ),
        (
            project_root
            / "benchmark_results"
            / "risk"
            / "portfolio"
            / "portfolio_result.json"
        ),
        (project_root / "benchmark_results" / "rl" / "ppo" / "rl_result.json"),
        (project_root / "data" / "rag" / "evaluation" / "reference_corpus.json"),
    )

    for path in paths:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "{}",
            encoding="utf-8",
        )


def _agent_response(
    *,
    agent_name: str,
    question: str,
) -> AgentResponse:
    return AgentResponse(
        agent_name=agent_name,
        question=question,
        summary=("Zweryfikowane podsumowanie walidacyjne."),
        findings=("Zweryfikowano dostępne wyniki.",),
        risks=("Wyniki historyczne nie gwarantują przyszłych rezultatów.",),
        recommendations=("Porównaj alternatywy przed podjęciem decyzji.",),
        limitations=(
            (
                "Wynik ma charakter wsparcia decyzyjnego "
                "i nie jest indywidualną poradą finansową."
            ),
        ),
        evidence=(
            EvidenceReference(
                source_type="benchmark_artifact",
                identifier="artifact.json",
                description=("Validation benchmark artifact"),
            ),
        ),
    )


class FakeAgentRuntimeService:
    """Return deterministic responses without loading Ollama or ML models."""

    def __init__(self) -> None:
        self.forecast_calls: list[tuple[str, str | None]] = []
        self.risk_calls: list[str] = []
        self.allocation_calls: list[tuple[str, str]] = []
        self.decision_calls: list[tuple[str, dict[str, object], str]] = []

    def analyze_forecast(
        self,
        *,
        question: str,
        ticker: str | None,
    ) -> AgentResponse:
        self.forecast_calls.append(
            (
                question,
                ticker,
            )
        )

        return _agent_response(
            agent_name=("forecast_analysis_agent"),
            question=question,
        )

    def analyze_risk(
        self,
        *,
        question: str,
    ) -> AgentResponse:
        self.risk_calls.append(question)

        return _agent_response(
            agent_name=("risk_analysis_agent"),
            question=question,
        )

    def recommend_allocation(
        self,
        *,
        question: str,
        risk_preference: str,
    ) -> AgentResponse:
        self.allocation_calls.append(
            (
                question,
                risk_preference,
            )
        )

        return _agent_response(
            agent_name=("allocation_recommendation_agent"),
            question=question,
        )

    def explain_decision(
        self,
        *,
        question: str,
        decision_context: dict[str, object],
        response_language: str = "pl",
    ) -> AgentResponse:
        self.decision_calls.append((question, decision_context, response_language))
        return _agent_response(
            agent_name="allocation_recommendation_agent",
            question=question,
        )


class FailingAgentRuntimeService:
    """Simulate an unavailable local agent runtime."""

    def analyze_forecast(
        self,
        *,
        question: str,
        ticker: str | None,
    ) -> AgentResponse:
        del question
        del ticker

        raise AgentRuntimeError("Forecast agent execution failed.")

    def analyze_risk(
        self,
        *,
        question: str,
    ) -> AgentResponse:
        del question

        raise AgentRuntimeError("Risk agent execution failed.")

    def recommend_allocation(
        self,
        *,
        question: str,
        risk_preference: str,
    ) -> AgentResponse:
        del question
        del risk_preference

        raise AgentRuntimeError("Allocation agent execution failed.")

    def explain_decision(
        self,
        *,
        question: str,
        decision_context: dict[str, object],
        response_language: str = "pl",
    ) -> AgentResponse:
        del question
        del decision_context
        del response_language
        raise AgentRuntimeError("Decision explanation failed.")


@pytest.fixture
def agent_client(
    tmp_path: Path,
) -> Iterator[
    tuple[
        TestClient,
        FakeAgentRuntimeService,
    ]
]:
    _write_agent_results(tmp_path)
    _write_runtime_inputs(tmp_path)

    application = create_app()

    repository = AgentResultRepository(
        project_root=tmp_path,
    )
    decision_artifact_directory = tmp_path / "decision_portfolio"
    decision_artifact_directory.mkdir()
    shutil.copyfile(
        Path("benchmark_results/risk/portfolio/portfolio_result.json"),
        decision_artifact_directory / "portfolio_result.json",
    )
    decision_repository = PortfolioArtifactRepository(
        artifact_directory=decision_artifact_directory,
    )
    service = FakeAgentRuntimeService()

    application.dependency_overrides[get_agent_repository] = lambda: repository

    application.dependency_overrides[get_agent_runtime_service] = lambda: service
    application.dependency_overrides[get_decision_portfolio_repository] = lambda: (
        decision_repository
    )

    with TestClient(application) as client:
        yield client, service


def test_status_reports_frozen_results_and_runtime_inputs(
    agent_client: tuple[
        TestClient,
        FakeAgentRuntimeService,
    ],
) -> None:
    client, _ = agent_client

    response = client.get("/api/v1/agents/status")

    assert response.status_code == 200

    assert response.json() == {
        "result_artifact_available": True,
        "forecast_artifact_available": True,
        "xai_artifact_available": True,
        "portfolio_artifact_available": True,
        "reinforcement_learning_artifact_available": True,
        "corpus_available": True,
        "runtime_inputs_available": True,
        "model_name": "gemma3:4b",
        "retrieval_strategy": "vector",
        "evaluation_basis": ("validation_benchmark_artifacts"),
        "test_split_used": False,
    }


def test_results_endpoint_returns_frozen_three_agent_workflow(
    agent_client: tuple[
        TestClient,
        FakeAgentRuntimeService,
    ],
) -> None:
    client, _ = agent_client

    response = client.get("/api/v1/agents/results")

    payload = response.json()

    assert response.status_code == 200
    assert payload["model_name"] == "gemma3:4b"
    assert payload["retrieval_strategy"] == "vector"
    assert payload["evaluation_basis"] == "validation_benchmark_artifacts"
    assert payload["test_split_used"] is False

    assert payload["responses"]["forecast"]["agent_name"] == "forecast_analysis_agent"

    assert len(payload["responses"]["forecast"]["evidence"]) == 2


def test_forecast_agent_normalizes_ticker_and_returns_structured_response(
    agent_client: tuple[
        TestClient,
        FakeAgentRuntimeService,
    ],
) -> None:
    client, service = agent_client

    response = client.post(
        "/api/v1/agents/forecast",
        json={
            "question": "  Oceń prognozę dla SPY.  ",
            "ticker": "spy",
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["agent_name"] == "forecast_analysis_agent"
    assert payload["question"] == "Oceń prognozę dla SPY."

    assert service.forecast_calls == [
        (
            "Oceń prognozę dla SPY.",
            "SPY",
        )
    ]


def test_forecast_agent_rejects_unsupported_ticker(
    agent_client: tuple[
        TestClient,
        FakeAgentRuntimeService,
    ],
) -> None:
    client, service = agent_client

    response = client.post(
        "/api/v1/agents/forecast",
        json={
            "question": "Oceń prognozę.",
            "ticker": "UNKNOWN",
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": ("Unsupported ticker: 'UNKNOWN'.")}
    assert service.forecast_calls == []


def test_risk_agent_returns_structured_response(
    agent_client: tuple[
        TestClient,
        FakeAgentRuntimeService,
    ],
) -> None:
    client, service = agent_client

    response = client.post(
        "/api/v1/agents/risk",
        json={
            "question": "Porównaj ryzyko scenariuszy.",
        },
    )

    assert response.status_code == 200
    assert response.json()["agent_name"] == "risk_analysis_agent"
    assert service.risk_calls == ["Porównaj ryzyko scenariuszy."]


def test_allocation_agent_uses_explicit_risk_preference(
    agent_client: tuple[
        TestClient,
        FakeAgentRuntimeService,
    ],
) -> None:
    client, service = agent_client

    response = client.post(
        "/api/v1/agents/allocation",
        json={
            "question": "Porównaj warianty alokacji.",
            "risk_preference": "defensive",
        },
    )

    assert response.status_code == 200
    assert response.json()["agent_name"] == "allocation_recommendation_agent"

    assert service.allocation_calls == [
        (
            "Porównaj warianty alokacji.",
            "defensive",
        )
    ]


def test_allocation_agent_defaults_to_balanced_preference(
    agent_client: tuple[
        TestClient,
        FakeAgentRuntimeService,
    ],
) -> None:
    client, service = agent_client

    response = client.post(
        "/api/v1/agents/allocation",
        json={
            "question": "Który wariant jest referencyjny?",
        },
    )

    assert response.status_code == 200

    assert service.allocation_calls == [
        (
            "Który wariant jest referencyjny?",
            "balanced",
        )
    ]


def test_decision_narrative_recomputes_server_side_decision(
    agent_client: tuple[
        TestClient,
        FakeAgentRuntimeService,
    ],
) -> None:
    client, service = agent_client

    response = client.post(
        "/api/v1/agents/decision",
        json={
            "question": "  Dlaczego ten wariant?  ",
            "evaluation": {
                "profile": {"profile_type": "balanced"},
                "available_liquidity_fraction": 0.10,
            },
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["decision"]["deterministic"] is True
    assert payload["decision"]["llm_used_for_ranking"] is False
    assert payload["decision"]["portfolio_weights_modified"] is False
    assert payload["decision"]["recommended_scenario"] is not None
    assert payload["narrative"]["question"] == "Dlaczego ten wariant?"
    assert len(service.decision_calls) == 1
    question, decision_context, response_language = service.decision_calls[0]
    assert question == "Dlaczego ten wariant?"
    assert response_language == "pl"
    assert (
        decision_context["recommended_scenario"]
        == payload["decision"]["recommended_scenario"]
    )
    assert decision_context["scenarios"] == payload["decision"]["scenarios"]


def test_decision_narrative_forwards_explicit_english_language(
    agent_client: tuple[TestClient, FakeAgentRuntimeService],
) -> None:
    client, service = agent_client

    response = client.post(
        "/api/v1/agents/decision",
        json={
            "question": "Why this variant?",
            "response_language": "en",
            "evaluation": {
                "profile": {"profile_type": "balanced"},
                "available_liquidity_fraction": 0.10,
            },
        },
    )

    assert response.status_code == 200
    assert service.decision_calls[0][2] == "en"


def test_decision_narrative_rejects_client_supplied_result_fields(
    agent_client: tuple[
        TestClient,
        FakeAgentRuntimeService,
    ],
) -> None:
    client, service = agent_client

    response = client.post(
        "/api/v1/agents/decision",
        json={
            "question": "Wyjaśnij wynik.",
            "evaluation": {
                "profile": {"profile_type": "balanced"},
                "available_liquidity_fraction": 0.10,
                "recommended_scenario": "growth",
            },
        },
    )

    assert response.status_code == 422
    assert service.decision_calls == []


@pytest.mark.parametrize(
    "endpoint",
    (
        "/api/v1/agents/forecast",
        "/api/v1/agents/risk",
        "/api/v1/agents/allocation",
    ),
)
def test_agent_endpoints_reject_blank_question(
    agent_client: tuple[
        TestClient,
        FakeAgentRuntimeService,
    ],
    endpoint: str,
) -> None:
    client, _ = agent_client

    response = client.post(
        endpoint,
        json={
            "question": "   ",
        },
    )

    assert response.status_code == 422


def test_allocation_agent_rejects_unknown_risk_preference(
    agent_client: tuple[
        TestClient,
        FakeAgentRuntimeService,
    ],
) -> None:
    client, _ = agent_client

    response = client.post(
        "/api/v1/agents/allocation",
        json={
            "question": "Porównaj warianty.",
            "risk_preference": "aggressive",
        },
    )

    assert response.status_code == 422


def test_results_return_service_unavailable_when_artifact_is_missing(
    tmp_path: Path,
) -> None:
    application = create_app()

    repository = AgentResultRepository(
        project_root=tmp_path,
    )

    application.dependency_overrides[get_agent_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get("/api/v1/agents/results")

    assert response.status_code == 503

    assert "Agent result artifact is missing" in response.json()["detail"]


def test_status_reports_missing_inputs_without_failing(
    tmp_path: Path,
) -> None:
    application = create_app()

    repository = AgentResultRepository(
        project_root=tmp_path,
    )

    application.dependency_overrides[get_agent_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get("/api/v1/agents/status")

    assert response.status_code == 200

    assert response.json() == {
        "result_artifact_available": False,
        "forecast_artifact_available": False,
        "xai_artifact_available": False,
        "portfolio_artifact_available": False,
        "reinforcement_learning_artifact_available": False,
        "corpus_available": False,
        "runtime_inputs_available": False,
        "model_name": None,
        "retrieval_strategy": None,
        "evaluation_basis": None,
        "test_split_used": None,
    }


def test_results_reject_artifact_that_claims_test_split_usage(
    tmp_path: Path,
) -> None:
    path = _write_agent_results(tmp_path)

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )
    payload["test_split_used"] = True

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    application = create_app()

    repository = AgentResultRepository(
        project_root=tmp_path,
    )

    application.dependency_overrides[get_agent_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get("/api/v1/agents/results")

    assert response.status_code == 503

    assert "invalid schema" in response.json()["detail"]


def test_live_agent_returns_service_unavailable_when_runtime_fails(
    tmp_path: Path,
) -> None:
    application = create_app()

    application.dependency_overrides[get_agent_runtime_service] = lambda: (
        FailingAgentRuntimeService()
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/agents/risk",
            json={
                "question": "Porównaj ryzyko.",
            },
        )

    assert response.status_code == 503

    assert response.json() == {"detail": ("Risk agent execution failed.")}

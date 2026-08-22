import json

import pytest

from backend.src.agents.allocation_recommendation_agent import (
    AllocationRecommendationAgent,
)
from backend.src.agents.artifact_repository import (
    AgentArtifactPaths,
    DecisionSupportArtifactRepository,
)
from backend.src.agents.factory import build_agent_suite
from backend.src.agents.forecast_analysis_agent import ForecastAnalysisAgent
from backend.src.agents.risk_analysis_agent import RiskAnalysisAgent
from backend.src.rag.models import Chunk, ScoredChunk


class RecordingCompletionModel:
    def __init__(self, response: str | None = None) -> None:
        payload = {
            "summary": "Podsumowanie oparte na walidacji.",
            "findings": ["Wynik walidacyjny jest obserwacją historyczną."],
            "risks": ["Wyniki mogą nie utrzymać się w przyszłości."],
            "recommendations": ["Porównaj warianty według celu ryzyka."],
            "limitations": ["Zbiór testowy nie został użyty."],
        }
        self.response = response or json.dumps(payload)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class StaticRetriever:
    def retrieve(self, question: str) -> list[ScoredChunk]:
        return [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="risk_chunk_0000",
                    document_id="financial_risk",
                    content="Diversification reduces asset-specific risk.",
                    source="reference_corpus.json",
                ),
                score=0.9,
                rank=1,
            )
        ]


def _repository(
    artifact_paths: AgentArtifactPaths,
) -> DecisionSupportArtifactRepository:
    return DecisionSupportArtifactRepository(artifact_paths)


def test_forecast_agent_uses_metrics_xai_and_rag_evidence(
    artifact_paths: AgentArtifactPaths,
) -> None:
    model = RecordingCompletionModel()
    agent = ForecastAnalysisAgent(
        artifacts=_repository(artifact_paths),
        completion_model=model,
        evidence_retriever=StaticRetriever(),
    )

    response = agent.analyze("Oceń model.", ticker="SPY")

    assert response.agent_name == "forecast_analysis_agent"
    assert "słabą zdolność" in response.summary
    assert "50.50%" in response.summary
    assert "directional_accuracy" in model.prompts[0]
    assert "momentum_20" in model.prompts[0]
    assert {item.identifier for item in response.evidence} == {
        "forecast_result.json",
        "xai_result.json",
        "financial_risk",
    }
    rag_reference = next(
        item for item in response.evidence if item.source_type == "rag_document"
    )
    assert "chunk=risk_chunk_0000" in rag_reference.description
    assert "rank=1" in rag_reference.description
    assert "score=0.9000" in rag_reference.description


def test_risk_agent_uses_validation_scenarios(
    artifact_paths: AgentArtifactPaths,
) -> None:
    model = RecordingCompletionModel()
    agent = RiskAnalysisAgent(
        artifacts=_repository(artifact_paths),
        completion_model=model,
    )

    response = agent.analyze("Porównaj ryzyko.")

    assert response.agent_name == "risk_analysis_agent"
    assert "Porównano 2 scenariuszy" in response.summary
    assert "maximum_drawdown" in model.prompts[0]
    assert "defensive" in model.prompts[0]


def test_allocation_agent_includes_rl_and_test_split_status(
    artifact_paths: AgentArtifactPaths,
) -> None:
    model = RecordingCompletionModel()
    agent = AllocationRecommendationAgent(
        artifacts=_repository(artifact_paths),
        completion_model=model,
    )

    response = agent.recommend(
        "Porównaj warianty.",
        risk_preference="balanced",
    )

    assert response.agent_name == "allocation_recommendation_agent"
    assert "PPO nie osiągnęło" in response.summary
    assert "maximum_sharpe" in response.summary
    assert '"rl_test_split_used": false' in model.prompts[0]
    assert '"strategy": "ppo"' in model.prompts[0]
    assert "rl_result.json" in {item.identifier for item in response.evidence}


def test_agent_adds_fixed_decision_support_disclaimer(
    artifact_paths: AgentArtifactPaths,
) -> None:
    agent = RiskAnalysisAgent(
        artifacts=_repository(artifact_paths),
        completion_model=RecordingCompletionModel(),
    )

    response = agent.analyze("Porównaj ryzyko.")

    assert any(
        "nie jest indywidualną poradą finansową" in limitation
        for limitation in response.limitations
    )


def test_agent_accepts_json_inside_markdown_fence(
    artifact_paths: AgentArtifactPaths,
) -> None:
    payload = {
        "summary": "S",
        "findings": ["F"],
        "risks": ["R"],
        "recommendations": ["P"],
        "limitations": ["L"],
    }
    model = RecordingCompletionModel("```json\n" + json.dumps(payload) + "\n```")
    agent = RiskAnalysisAgent(
        artifacts=_repository(artifact_paths),
        completion_model=model,
    )

    response = agent.analyze("Pytanie")

    assert "Porównano 2 scenariuszy" in response.summary


def test_verified_facts_replace_false_language_model_claims(
    artifact_paths: AgentArtifactPaths,
) -> None:
    false_payload = {
        "summary": "PPO is the best and superior strategy.",
        "findings": ["There are four scenarios."],
        "risks": ["No risk."],
        "recommendations": ["Always choose PPO."],
        "limitations": ["None."],
    }
    model = RecordingCompletionModel(json.dumps(false_payload))
    repository = _repository(artifact_paths)

    risk_response = RiskAnalysisAgent(repository, model).analyze("Ryzyko")
    allocation_response = AllocationRecommendationAgent(
        repository,
        model,
    ).recommend("Alokacja", risk_preference="balanced")

    assert "Porównano 2 scenariuszy" in risk_response.summary
    assert "PPO nie osiągnęło" in allocation_response.summary
    assert "maximum_sharpe" in allocation_response.summary
    assert not any(
        "superior" in text.lower()
        for text in (
            allocation_response.summary,
            *allocation_response.findings,
            *allocation_response.recommendations,
        )
    )


@pytest.mark.parametrize(
    "response",
    [
        "{}",
        json.dumps(
            {
                "summary": "S",
                "findings": [],
                "risks": ["R"],
                "recommendations": ["P"],
                "limitations": ["L"],
            }
        ),
    ],
)
def test_agent_rejects_invalid_model_response(
    artifact_paths: AgentArtifactPaths,
    response: str,
) -> None:
    agent = RiskAnalysisAgent(
        artifacts=_repository(artifact_paths),
        completion_model=RecordingCompletionModel(response),
    )

    with pytest.raises((TypeError, ValueError)):
        agent.analyze("Pytanie")


def test_agent_falls_back_when_model_does_not_return_json(
    artifact_paths: AgentArtifactPaths,
) -> None:
    agent = RiskAnalysisAgent(
        artifacts=_repository(artifact_paths),
        completion_model=RecordingCompletionModel("Zwykła odpowiedź bez obiektu JSON."),
    )

    response = agent.analyze("Pytanie")

    assert "Porównano 2 scenariuszy" in response.summary
    assert response.evidence


def test_factory_builds_all_three_agents(
    artifact_paths: AgentArtifactPaths,
) -> None:
    project_root = artifact_paths.forecast.parent
    expected_paths = AgentArtifactPaths.from_project_root(project_root)
    for source, target in zip(
        (
            artifact_paths.forecast,
            artifact_paths.xai,
            artifact_paths.portfolio,
            artifact_paths.reinforcement_learning,
        ),
        (
            expected_paths.forecast,
            expected_paths.xai,
            expected_paths.portfolio,
            expected_paths.reinforcement_learning,
        ),
        strict=True,
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    suite = build_agent_suite(
        project_root=project_root,
        completion_model=RecordingCompletionModel(),
    )

    assert isinstance(suite.forecast, ForecastAnalysisAgent)
    assert isinstance(suite.risk, RiskAnalysisAgent)
    assert isinstance(suite.allocation, AllocationRecommendationAgent)

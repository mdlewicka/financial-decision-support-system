from dataclasses import dataclass
from pathlib import Path

from backend.src.agents.allocation_recommendation_agent import (
    AllocationRecommendationAgent,
)
from backend.src.agents.artifact_repository import (
    AgentArtifactPaths,
    DecisionSupportArtifactRepository,
)
from backend.src.agents.base_agent import CompletionModel, EvidenceRetriever
from backend.src.agents.forecast_analysis_agent import ForecastAnalysisAgent
from backend.src.agents.risk_analysis_agent import RiskAnalysisAgent


@dataclass(frozen=True)
class DecisionSupportAgentSuite:
    forecast: ForecastAnalysisAgent
    risk: RiskAnalysisAgent
    allocation: AllocationRecommendationAgent


def build_agent_suite(
    project_root: Path,
    completion_model: CompletionModel,
    evidence_retriever: EvidenceRetriever | None = None,
) -> DecisionSupportAgentSuite:
    artifacts = DecisionSupportArtifactRepository(
        AgentArtifactPaths.from_project_root(project_root)
    )
    return DecisionSupportAgentSuite(
        forecast=ForecastAnalysisAgent(
            artifacts=artifacts,
            completion_model=completion_model,
            evidence_retriever=evidence_retriever,
        ),
        risk=RiskAnalysisAgent(
            artifacts=artifacts,
            completion_model=completion_model,
            evidence_retriever=evidence_retriever,
        ),
        allocation=AllocationRecommendationAgent(
            artifacts=artifacts,
            completion_model=completion_model,
            evidence_retriever=evidence_retriever,
        ),
    )

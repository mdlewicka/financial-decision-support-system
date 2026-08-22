"""Application service joining frozen portfolio evidence with decision scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.src.api.risk_schemas import RiskArtifactPayload
from backend.src.decision.engine import DecisionEngine
from backend.src.decision.models import DecisionEngineResult, DecisionProfile
from backend.src.decision.portfolio_adapter import (
    PortfolioArtifactDecisionAdapter,
    PortfolioDecisionInput,
)


@dataclass(frozen=True, slots=True)
class PortfolioDecisionEvaluation:
    """One complete, auditable ranking of frozen portfolio alternatives."""

    source: PortfolioDecisionInput
    decision: DecisionEngineResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.source.provenance.to_dict(),
            "scenario_contexts": [
                context.to_dict() for context in self.source.contexts
            ],
            "decision": self.decision.to_dict(),
        }


class PortfolioDecisionService:
    """Evaluate portfolio scenarios without changing their fixed weights."""

    def __init__(
        self,
        *,
        adapter: PortfolioArtifactDecisionAdapter | None = None,
        engine: DecisionEngine | None = None,
    ) -> None:
        self._adapter = adapter or PortfolioArtifactDecisionAdapter()
        self._engine = engine or DecisionEngine()

    def evaluate(
        self,
        artifact: RiskArtifactPayload,
        profile: DecisionProfile,
        *,
        available_liquidity_fraction: float,
    ) -> PortfolioDecisionEvaluation:
        source = self._adapter.adapt(artifact)
        decision = self._engine.rank(
            source.scenarios,
            profile,
            available_liquidity_fraction=available_liquidity_fraction,
        )
        return PortfolioDecisionEvaluation(source=source, decision=decision)

"""Adapt frozen portfolio validation artifacts for the Decision Engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Literal

from backend.src.api.risk_schemas import RiskArtifactPayload
from backend.src.decision.models import ScenarioMetrics

DecisionEvaluationSplit = Literal["validation"]


@dataclass(frozen=True, slots=True)
class DecisionProvenance:
    """Describe the immutable analytical source used by a decision."""

    source_schema_version: int
    evaluation_split: DecisionEvaluationSplit
    return_definition: str
    observations: int
    periods_per_year: int
    date_min: date
    date_max: date
    tickers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["date_min"] = self.date_min.isoformat()
        payload["date_max"] = self.date_max.isoformat()
        return payload


@dataclass(frozen=True, slots=True)
class PortfolioScenarioContext:
    """Preserve construction and weights without allowing the engine to alter them."""

    name: str
    construction: str
    weights: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "construction": self.construction,
            "weights": dict(self.weights),
        }


@dataclass(frozen=True, slots=True)
class PortfolioDecisionInput:
    """Validation-only metrics and provenance ready for deterministic ranking."""

    provenance: DecisionProvenance
    scenarios: tuple[ScenarioMetrics, ...]
    contexts: tuple[PortfolioScenarioContext, ...]

    def context_for(self, scenario_name: str) -> PortfolioScenarioContext:
        for context in self.contexts:
            if context.name == scenario_name:
                return context
        raise KeyError(f"Unknown scenario context: {scenario_name}")


class PortfolioArtifactDecisionAdapter:
    """Read only validation fields from a validated portfolio artifact."""

    def adapt(self, artifact: RiskArtifactPayload) -> PortfolioDecisionInput:
        if not isinstance(artifact, RiskArtifactPayload):
            raise TypeError("artifact must be a RiskArtifactPayload instance.")

        validation = artifact.market_splits.validation
        ordered = tuple(sorted(artifact.scenarios, key=lambda item: item.name))
        scenarios = tuple(
            ScenarioMetrics(
                name=item.name,
                annualized_return=item.validation.metrics.annualized_mean_return,
                annualized_volatility=(item.validation.metrics.annualized_volatility),
                maximum_drawdown=item.validation.metrics.maximum_drawdown,
                concentration_index=item.validation.concentration_index,
                effective_number_of_assets=(item.validation.effective_number_of_assets),
                diversification_ratio=item.validation.diversification_ratio,
            )
            for item in ordered
        )
        contexts = tuple(
            PortfolioScenarioContext(
                name=item.name,
                construction=item.construction,
                weights=tuple(sorted(item.weights.items())),
            )
            for item in ordered
        )
        return PortfolioDecisionInput(
            provenance=DecisionProvenance(
                source_schema_version=artifact.schema_version,
                evaluation_split="validation",
                return_definition=artifact.return_definition,
                observations=validation.observations,
                periods_per_year=validation.periods_per_year,
                date_min=validation.date_min,
                date_max=validation.date_max,
                tickers=validation.tickers,
            ),
            scenarios=scenarios,
            contexts=contexts,
        )

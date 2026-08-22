from backend.src.agents.artifact_repository import (
    DecisionSupportArtifactRepository,
)
from backend.src.agents.base_agent import (
    BaseDecisionSupportAgent,
    CompletionModel,
    EvidenceRetriever,
)
from backend.src.agents.models import AgentResponse, EvidenceReference


class RiskAnalysisAgent(BaseDecisionSupportAgent):
    def __init__(
        self,
        artifacts: DecisionSupportArtifactRepository,
        completion_model: CompletionModel,
        evidence_retriever: EvidenceRetriever | None = None,
    ) -> None:
        super().__init__(
            agent_name="risk_analysis_agent",
            role=(
                "You compare validated portfolio risk scenarios and explain "
                "risk-return trade-offs."
            ),
            completion_model=completion_model,
            evidence_retriever=evidence_retriever,
        )
        self._artifacts = artifacts

    def analyze(self, question: str) -> AgentResponse:
        context = self._artifacts.risk_context()
        draft = self._answer(
            question=question,
            artifact_context=context,
            artifact_evidence=(
                EvidenceReference(
                    source_type="benchmark_artifact",
                    identifier="portfolio_result.json",
                    description=(
                        "Training-constructed portfolio scenarios evaluated "
                        "on validation data"
                    ),
                ),
            ),
            instructions=(
                "Compare annualized volatility, maximum drawdown, VaR, CVaR, "
                "Sharpe, Sortino, concentration and diversification. Do not "
                "rank a scenario without stating the objective and trade-off."
            ),
            knowledge_query=(
                "How do diversification, volatility, drawdown, VaR and CVaR "
                "describe portfolio financial risk?"
            ),
        )
        scenarios = context["validation_scenarios"]
        lowest_volatility = min(
            scenarios,
            key=lambda row: row["validation_metrics"]["annualized_volatility"],
        )
        lowest_drawdown = min(
            scenarios,
            key=lambda row: row["validation_metrics"]["maximum_drawdown"],
        )
        highest_sharpe = max(
            scenarios,
            key=lambda row: row["validation_metrics"]["sharpe_ratio"],
        )
        return AgentResponse(
            agent_name=draft.agent_name,
            question=draft.question,
            summary=(
                f"Porównano {len(scenarios)} scenariuszy na walidacji. "
                f"Najniższą zmienność ma {lowest_volatility['name']}, a "
                f"najwyższy Sharpe ma {highest_sharpe['name']}."
            ),
            findings=(
                (
                    f"{lowest_volatility['name']} osiąga roczną zmienność "
                    f"{float(lowest_volatility['validation_metrics']['annualized_volatility']):.2%}."
                ),
                (
                    f"{lowest_drawdown['name']} ma najniższy maksymalny "
                    f"drawdown: {float(lowest_drawdown['validation_metrics']['maximum_drawdown']):.2%}."
                ),
                (
                    f"{highest_sharpe['name']} osiąga najwyższy współczynnik "
                    f"Sharpe: {float(highest_sharpe['validation_metrics']['sharpe_ratio']):.4f}."
                ),
            ),
            risks=(
                (
                    "Wyższy oczekiwany zwrot wiąże się w ocenionych wariantach "
                    "z większą zmiennością lub obsunięciem kapitału."
                ),
                (
                    "VaR i CVaR opisują historyczny ogon rozkładu i mogą "
                    "niedoszacować nowych zdarzeń skrajnych."
                ),
            ),
            recommendations=(
                (
                    "Wybieraj scenariusz według jawnego celu: ograniczenia "
                    "zmienności, obsunięcia albo maksymalizacji Sharpe’a."
                ),
                (
                    "Kontroluj koncentrację i dywersyfikację razem z VaR, "
                    "CVaR i maksymalnym drawdownem."
                ),
            ),
            limitations=(
                "Porównanie opisuje historyczny okres walidacyjny.",
                "Metryki ryzyka nie obejmują wszystkich przyszłych scenariuszy.",
                self._DISCLAIMER,
            ),
            evidence=draft.evidence,
        )

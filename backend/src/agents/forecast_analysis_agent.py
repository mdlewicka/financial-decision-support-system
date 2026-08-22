from backend.src.agents.artifact_repository import (
    DecisionSupportArtifactRepository,
)
from backend.src.agents.base_agent import (
    BaseDecisionSupportAgent,
    CompletionModel,
    EvidenceRetriever,
)
from backend.src.agents.models import AgentResponse, EvidenceReference


class ForecastAnalysisAgent(BaseDecisionSupportAgent):
    def __init__(
        self,
        artifacts: DecisionSupportArtifactRepository,
        completion_model: CompletionModel,
        evidence_retriever: EvidenceRetriever | None = None,
    ) -> None:
        super().__init__(
            agent_name="forecast_analysis_agent",
            role=(
                "You interpret one-step market-return forecasts and XAI "
                "results conservatively."
            ),
            completion_model=completion_model,
            evidence_retriever=evidence_retriever,
        )
        self._artifacts = artifacts

    def analyze(
        self,
        question: str,
        ticker: str | None = None,
    ) -> AgentResponse:
        context = self._artifacts.forecast_context(ticker=ticker)
        draft = self._answer(
            question=question,
            artifact_context=context,
            artifact_evidence=(
                EvidenceReference(
                    source_type="benchmark_artifact",
                    identifier="forecast_result.json",
                    description=("Random Forest validation forecast metrics"),
                ),
                EvidenceReference(
                    source_type="benchmark_artifact",
                    identifier="xai_result.json",
                    description="Random Forest SHAP and feature effects",
                ),
            ),
            instructions=(
                "Assess predictive strength using MAE, RMSE, R2 and "
                "directional accuracy. Interpret SHAP as model attribution, "
                "not causality. Explicitly highlight weak or near-random "
                "validation performance when present."
            ),
            knowledge_query=(
                "How should forecast error, directional accuracy and model "
                "feature attribution be interpreted under market risk?"
            ),
        )
        metrics = context["overall_metrics"]
        features = context["top_global_shap_features"]
        feature_names = ", ".join(str(item["feature"]) for item in features[:4])
        return AgentResponse(
            agent_name=draft.agent_name,
            question=draft.question,
            summary=(
                "Model Random Forest ma na walidacji słabą zdolność "
                "prognostyczną: "
                f"R²={float(metrics['r2']):.4f}, a trafność kierunku wynosi "
                f"{float(metrics['directional_accuracy']):.2%}."
            ),
            findings=(
                (
                    f"Ocena obejmuje {int(metrics['observations'])} obserwacji; "
                    f"MAE={float(metrics['mae']):.6f}, "
                    f"RMSE={float(metrics['rmse']):.6f}."
                ),
                (
                    "Najważniejsze globalne cechy według średniej bezwzględnej "
                    f"wartości SHAP to: {feature_names}."
                ),
                (
                    "Wartości SHAP opisują sposób działania modelu, lecz nie "
                    "dowodzą zależności przyczynowej."
                ),
            ),
            risks=(
                (
                    "Trafność kierunku bliska 50% oznacza, że prognoza nie "
                    "powinna być samodzielną podstawą alokacji."
                ),
                (
                    "Zmiana reżimu rynkowego może pogorszyć wyniki względem "
                    "zaobserwowanej walidacji."
                ),
            ),
            recommendations=(
                (
                    "Traktuj prognozę jako sygnał pomocniczy i łącz ją z "
                    "limitami ryzyka oraz analizą portfela."
                ),
                (
                    "Monitoruj błędy i trafność kierunku osobno dla każdego "
                    "tickera przed wykorzystaniem operacyjnym."
                ),
            ),
            limitations=(
                "Metryki pochodzą ze zbioru walidacyjnego, a nie z przyszłości.",
                (
                    "Interpretacje SHAP opisują model Random Forest i nie są "
                    "dowodem ekonomicznej przyczynowości."
                ),
                self._DISCLAIMER,
            ),
            evidence=draft.evidence,
        )

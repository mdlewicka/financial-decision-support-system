import re
from typing import Any

from backend.src.agents.artifact_repository import (
    DecisionSupportArtifactRepository,
)
from backend.src.agents.base_agent import (
    BaseDecisionSupportAgent,
    CompletionModel,
    EvidenceRetriever,
)
from backend.src.agents.models import AgentResponse, EvidenceReference


class AllocationRecommendationAgent(BaseDecisionSupportAgent):
    _UNSUPPORTED_QUESTION_PATTERNS = (
        (
            re.compile(
                r"\b(gwarant\w*|pewny\s+zysk|bez\s+ryzyka|guaranteed|risk[- ]free)\b",
                flags=re.IGNORECASE,
            ),
            (
                "Dostępne dowody nie pozwalają zagwarantować zysku ani braku ryzyka.",
                "The available evidence cannot guarantee a profit or absence of risk.",
            ),
        ),
        (
            re.compile(
                r"\b(ile\s+(?:dokładnie\s+)?zarobi\w*|dokładn\w*\s+zwrot|"
                r"jaki\s+będzie\s+zwrot|"
                r"exact\s+future\s+return)\b",
                flags=re.IGNORECASE,
            ),
            (
                "Dane walidacyjne nie pozwalają podać dokładnego przyszłego zwrotu.",
                "Validation data cannot determine an exact future return.",
            ),
        ),
        (
            re.compile(
                r"\b(czy\s+mam|powinnam|powinienem)\s+"
                r"(kupić|sprzedać|zainwestować)|\bshould\s+i\s+(buy|sell|invest)\b",
                flags=re.IGNORECASE,
            ),
            (
                "System nie wydaje indywidualnych instrukcji kupna ani sprzedaży.",
                "The system does not provide individualized buy or sell instructions.",
            ),
        ),
        (
            re.compile(
                r"\b(podatek|podatkow\w*|porad\w*\s+prawn\w*|tax\s+advice|"
                r"legal\s+advice)\b",
                flags=re.IGNORECASE,
            ),
            (
                "Artefakty projektu nie stanowią podstawy porady prawnej ani podatkowej.",
                "Project artifacts are not a basis for legal or tax advice.",
            ),
        ),
    )

    def __init__(
        self,
        artifacts: DecisionSupportArtifactRepository,
        completion_model: CompletionModel,
        evidence_retriever: EvidenceRetriever | None = None,
    ) -> None:
        super().__init__(
            agent_name="allocation_recommendation_agent",
            role=(
                "You compare only the evaluated allocation variants and "
                "provide transparent decision support."
            ),
            completion_model=completion_model,
            evidence_retriever=evidence_retriever,
        )
        self._artifacts = artifacts

    def recommend(
        self,
        question: str,
        risk_preference: str = "balanced",
    ) -> AgentResponse:
        context = self._artifacts.allocation_context(risk_preference=risk_preference)
        draft = self._answer(
            question=question,
            artifact_context=context,
            artifact_evidence=(
                EvidenceReference(
                    source_type="benchmark_artifact",
                    identifier="portfolio_result.json",
                    description=("Classical and manual portfolio validation results"),
                ),
                EvidenceReference(
                    source_type="benchmark_artifact",
                    identifier="rl_result.json",
                    description=("PPO and classical strategy validation comparison"),
                ),
            ),
            instructions=(
                "Relate the requested risk preference to evaluated variants. "
                "Compare PPO with equal-weight, maximum-Sharpe and "
                "minimum-volatility baselines. Do not claim PPO superiority "
                "unless the supplied validation metrics support it. Mention "
                "that the final test split was not used. If weights are "
                "reported, reproduce them exactly from the artifacts."
            ),
            knowledge_query=(
                "How should portfolio allocation variants be compared using "
                "risk-adjusted return and diversification?"
            ),
        )
        comparison = context["strategy_comparison"]
        best_return = max(
            comparison,
            key=lambda row: row["annualized_mean_return"],
        )
        best_sharpe = max(
            comparison,
            key=lambda row: row["sharpe_ratio"],
        )
        lowest_volatility = min(
            comparison,
            key=lambda row: row["annualized_volatility"],
        )
        lowest_drawdown = min(
            comparison,
            key=lambda row: row["maximum_drawdown"],
        )
        preference = str(context["risk_preference"])
        selected = {
            "defensive": lowest_volatility,
            "balanced": best_sharpe,
            "growth": best_return,
        }[preference]
        return AgentResponse(
            agent_name=draft.agent_name,
            question=draft.question,
            summary=(
                f"Dla profilu {preference} wariantem referencyjnym jest "
                f"{selected['strategy']}. PPO nie osiągnęło ani najwyższego "
                "zwrotu, ani najwyższego współczynnika Sharpe na walidacji."
            ),
            findings=(
                (
                    f"Najwyższy roczny zwrot osiąga {best_return['strategy']}: "
                    f"{float(best_return['annualized_mean_return']):.2%}."
                ),
                (
                    f"Najwyższy Sharpe osiąga {best_sharpe['strategy']}: "
                    f"{float(best_sharpe['sharpe_ratio']):.4f}."
                ),
                (
                    f"Najniższą zmienność ma {lowest_volatility['strategy']}: "
                    f"{float(lowest_volatility['annualized_volatility']):.2%}; "
                    f"najniższy drawdown ma {lowest_drawdown['strategy']}: "
                    f"{float(lowest_drawdown['maximum_drawdown']):.2%}."
                ),
            ),
            risks=(
                (
                    "Ranking zależy od celu: zwrot, Sharpe, zmienność i "
                    "drawdown wskazują różne warianty."
                ),
                (
                    "Dynamiczna polityka PPO uwzględnia koszty transakcyjne, "
                    "ale jej wynik walidacyjny nie dowodzi przewagi nad "
                    "strategiami klasycznymi."
                ),
            ),
            recommendations=(
                (
                    f"Dla profilu {preference} traktuj {selected['strategy']} "
                    "jako wariant referencyjny do dalszej analizy, nie jako "
                    "automatyczną decyzję inwestycyjną."
                ),
                (
                    "Przed końcowym wyborem porównaj tolerowany drawdown, "
                    "zmienność, koncentrację oraz koszty zmian alokacji."
                ),
            ),
            limitations=(
                "Porównanie opiera się na wynikach walidacyjnych.",
                (
                    "Końcowy zbiór testowy nie został wykorzystany i pozostaje "
                    "zamknięty."
                ),
                self._DISCLAIMER,
            ),
            evidence=draft.evidence,
        )

    def explain_decision(
        self,
        question: str,
        decision_context: dict[str, Any],
        response_language: str = "pl",
    ) -> AgentResponse:
        """Explain an already calculated Decision Engine result without changing it."""
        profile = self._mapping(decision_context, "profile")
        scenarios = self._scenario_list(decision_context)
        language = self._response_language_name(response_language)
        refusal_reason = self._unsupported_question_reason(question, language)
        if refusal_reason is not None:
            return self._refusal_response(
                question=question,
                reason=refusal_reason,
                response_language=language,
            )
        xai_context = self._artifacts.xai_context()
        recommended_name = decision_context.get("recommended_scenario")
        recommended = next(
            (
                scenario
                for scenario in scenarios
                if scenario.get("name") == recommended_name
            ),
            None,
        )
        draft = self._answer(
            question=question,
            artifact_context={
                "decision_engine": decision_context,
                "forecast_model_xai": xai_context,
            },
            artifact_evidence=(
                EvidenceReference(
                    source_type="decision_engine",
                    identifier="current_decision_evaluation",
                    description=(
                        "Deterministic validation-only Decision Engine evaluation"
                    ),
                ),
                EvidenceReference(
                    source_type="benchmark_artifact",
                    identifier="portfolio_result.json",
                    description="Frozen portfolio validation evidence",
                ),
                EvidenceReference(
                    source_type="xai_artifact",
                    identifier="xai_result.json",
                    description=(
                        "Validation SHAP, PDP and ALE forecast-model evidence; "
                        "not an input to portfolio ranking"
                    ),
                ),
            ),
            instructions=(
                "Explain the supplied Decision Engine result. Never replace its "
                "recommended scenario, ranking, constraints or weights. State "
                "rejected alternatives and observed trade-offs. Do not invent a "
                "confidence score or describe the result as financial advice. "
                "Treat XAI as contextual forecast-model interpretation only, "
                "never as a cause of the portfolio ranking."
            ),
            knowledge_query=(
                "How should a validation-based portfolio recommendation and its "
                "risk constraints be interpreted?"
            ),
            response_language=language,
        )
        profile_type = str(profile.get("profile_type", "unknown"))
        admissible = [item for item in scenarios if item.get("admissible") is True]
        rejected = [item for item in scenarios if item.get("admissible") is not True]
        top_xai_features = xai_context["top_global_shap_features"]
        feature_names = (
            ", ".join(str(item["feature"]) for item in top_xai_features) or "brak"
        )
        effect_features = sorted(
            set(xai_context["partial_dependence_features"])
            | set(xai_context["accumulated_local_effect_features"])
        )
        effect_names = ", ".join(str(item) for item in effect_features) or "brak"
        if language == "English":
            return self._english_decision_response(
                draft=draft,
                question=question,
                profile_type=profile_type,
                recommended_name=recommended_name,
                recommended=recommended,
                admissible=admissible,
                rejected=rejected,
                feature_names=feature_names,
                effect_names=effect_names,
            )
        if recommended is None:
            summary = (
                f"Dla profilu {profile_type} Decision Engine nie wskazał "
                "rekomendowanego wariantu, ponieważ żaden scenariusz nie spełnił "
                "wszystkich twardych ograniczeń."
            )
            recommendation_finding = (
                "Brak rekomendacji jest kontrolowanym wynikiem ograniczeń, a nie "
                "błędem modelu językowego."
            )
        else:
            score = recommended.get("total_score")
            score_text = f"{float(score):.4f}" if score is not None else "brak"
            summary = (
                f"Dla profilu {profile_type} Decision Engine wskazał wariant "
                f"{recommended_name} z deterministycznym wynikiem {score_text}."
            )
            recommendation_finding = (
                f"Wariant {recommended_name} zajmuje pierwsze miejsce wśród "
                f"{len(admissible)} dopuszczalnych scenariuszy."
            )
        rejected_names = ", ".join(str(item.get("name")) for item in rejected) or "brak"
        return AgentResponse(
            agent_name=draft.agent_name,
            question=draft.question,
            summary=summary,
            findings=(
                recommendation_finding,
                f"Odrzucone warianty: {rejected_names}.",
                (
                    "Ranking wykorzystuje jawne priorytety zwrotu, zmienności, "
                    "obsunięcia i dywersyfikacji."
                ),
                (
                    "Kontekst SHAP modelu prognostycznego wskazuje cechy: "
                    f"{feature_names}."
                ),
                (
                    "Efekty PDP/ALE są dostępne dla cech: "
                    f"{effect_names}; nie są wejściem rankingu portfela."
                ),
            ),
            risks=(
                (
                    "Wynik opiera się na danych walidacyjnych i nie gwarantuje "
                    "przyszłych rezultatów."
                ),
                (
                    "Zmiana priorytetów lub twardych ograniczeń może zmienić "
                    "ranking albo wykluczyć wszystkie warianty."
                ),
            ),
            recommendations=(
                (
                    "Traktuj wynik jako wariant referencyjny do świadomego "
                    "porównania, nie jako automatyczne zlecenie inwestycyjne."
                ),
                (
                    "Przed decyzją sprawdź jawne naruszenia ograniczeń, "
                    "składowe wyniku i niezmienione wagi wariantu."
                ),
            ),
            limitations=(
                "Narracja nie zmienia rankingu ani wag Decision Engine.",
                "Nie obliczono nieskalibrowanego wskaźnika confidence.",
                self._DISCLAIMER,
                *tuple(item for item in draft.limitations if item != self._DISCLAIMER),
            ),
            evidence=draft.evidence,
        )

    @classmethod
    def _unsupported_question_reason(
        cls,
        question: str,
        response_language: str = "Polish",
    ) -> str | None:
        cleaned = question.strip()
        reason_index = 1 if response_language == "English" else 0
        for pattern, reasons in cls._UNSUPPORTED_QUESTION_PATTERNS:
            if pattern.search(cleaned):
                return reasons[reason_index]
        return None

    def _refusal_response(
        self,
        *,
        question: str,
        reason: str,
        response_language: str = "Polish",
    ) -> AgentResponse:
        if response_language == "English":
            return AgentResponse(
                agent_name="allocation_recommendation_agent",
                question=question,
                summary=f"I cannot provide that answer. {reason}",
                findings=(
                    "The Decision Engine exposes only historical validation "
                    "evaluation of existing variants.",
                ),
                risks=("An answer outside the evidence could be misleading.",),
                recommendations=(
                    "Ask about the ranking, constraints, rejected variants or "
                    "observed risk-return trade-offs.",
                ),
                limitations=(
                    "The controlled refusal was generated without an LLM.",
                    self._ENGLISH_DISCLAIMER,
                ),
                evidence=self._refusal_evidence(),
            )
        return AgentResponse(
            agent_name="allocation_recommendation_agent",
            question=question,
            summary=f"Nie mogę przygotować takiej odpowiedzi. {reason}",
            findings=(
                "Decision Engine udostępnia wyłącznie historyczną ocenę "
                "walidacyjną istniejących wariantów.",
            ),
            risks=("Odpowiedź wykraczająca poza dowody mogłaby wprowadzać w błąd.",),
            recommendations=(
                "Zapytaj o ranking, ograniczenia, odrzucone warianty albo "
                "zaobserwowane kompromisy ryzyko–zwrot.",
            ),
            limitations=(
                "Kontrolowana odmowa została wygenerowana bez udziału LLM.",
                self._DISCLAIMER,
            ),
            evidence=self._refusal_evidence(),
        )

    @staticmethod
    def _response_language_name(response_language: str) -> str:
        mapping = {"pl": "Polish", "en": "English"}
        try:
            return mapping[response_language]
        except KeyError as error:
            raise ValueError(
                f"Unsupported response language: {response_language!r}."
            ) from error

    def _english_decision_response(
        self,
        *,
        draft: AgentResponse,
        question: str,
        profile_type: str,
        recommended_name: object,
        recommended: dict[str, Any] | None,
        admissible: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        feature_names: str,
        effect_names: str,
    ) -> AgentResponse:
        if recommended is None:
            summary = (
                f"For the {profile_type} profile, the Decision Engine did not "
                "select a recommended variant because no scenario satisfied all "
                "hard constraints."
            )
            recommendation_finding = (
                "The absence of a recommendation is a controlled constraint "
                "outcome, not a language-model error."
            )
        else:
            score = recommended.get("total_score")
            score_text = f"{float(score):.4f}" if score is not None else "missing"
            summary = (
                f"For the {profile_type} profile, the Decision Engine selected "
                f"{recommended_name} with a deterministic score of {score_text}."
            )
            recommendation_finding = (
                f"Variant {recommended_name} ranks first among "
                f"{len(admissible)} admissible scenarios."
            )
        rejected_names = ", ".join(str(item.get("name")) for item in rejected) or "none"
        return AgentResponse(
            agent_name=draft.agent_name,
            question=question,
            summary=summary,
            findings=(
                recommendation_finding,
                f"Rejected variants: {rejected_names}.",
                (
                    "The ranking uses explicit return, volatility, drawdown and "
                    "diversification priorities."
                ),
                f"Forecast-model SHAP context highlights: {feature_names}.",
                (
                    f"PDP/ALE effects are available for: {effect_names}; they are "
                    "not inputs to the portfolio ranking."
                ),
            ),
            risks=(
                (
                    "The result is based on validation data and does not guarantee "
                    "future performance."
                ),
                (
                    "Changing priorities or hard constraints can change the "
                    "ranking or exclude every variant."
                ),
            ),
            recommendations=(
                (
                    "Use the result as a reference variant for an informed "
                    "comparison, not as an automatic investment order."
                ),
                (
                    "Before deciding, review explicit constraint violations, "
                    "score components and unchanged variant weights."
                ),
            ),
            limitations=(
                "The narrative does not change the Decision Engine ranking or weights.",
                "No uncalibrated confidence score was calculated.",
                self._ENGLISH_DISCLAIMER,
                *tuple(
                    item
                    for item in draft.limitations
                    if item != self._ENGLISH_DISCLAIMER
                ),
            ),
            evidence=draft.evidence,
        )

    @staticmethod
    def _refusal_evidence() -> tuple[EvidenceReference, ...]:
        return (
            EvidenceReference(
                source_type="decision_engine",
                identifier="current_decision_evaluation",
                description="Deterministic validation-only evaluation",
            ),
            EvidenceReference(
                source_type="evidence_policy",
                identifier="decision_narrative_scope",
                description="Guardrail limiting unsupported financial claims",
            ),
        )

    @staticmethod
    def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
        nested = value.get(key)
        if not isinstance(nested, dict):
            raise ValueError(f"Decision context field {key} must be an object.")
        return nested

    @staticmethod
    def _scenario_list(value: dict[str, Any]) -> list[dict[str, Any]]:
        scenarios = value.get("scenarios")
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("Decision context scenarios must be a non-empty list.")
        if any(not isinstance(item, dict) for item in scenarios):
            raise ValueError("Decision context scenarios must contain objects.")
        return scenarios

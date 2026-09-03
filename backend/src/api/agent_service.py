"""Lazy runtime service for interactive decision-support agents."""

from threading import Lock

from backend.src.agents.base_agent import DeterministicFallbackCompletionModel
from backend.src.agents.factory import (
    DecisionSupportAgentSuite,
    build_agent_suite,
)
from backend.src.agents.local_evidence_retriever import (
    DeterministicReferenceRetriever,
)
from backend.src.agents.models import AgentResponse
from backend.src.api.agent_repository import (
    AgentResultRepository,
)
from backend.src.observability.ollama_status import OllamaStatusService
from backend.src.rag.config import RAGConfig
from backend.src.rag.factory import build_rag_pipeline
from backend.src.rag.generation import OllamaGenerator
from backend.src.rag.loaders import load_reference_corpus


class AgentRuntimeError(RuntimeError):
    """Raised when an interactive decision-support agent cannot run."""


class AgentRuntimeService:
    """Build and cache the three-agent runtime only when it is first used."""

    def __init__(
        self,
        repository: AgentResultRepository,
    ) -> None:
        self._repository = repository
        self._suite: DecisionSupportAgentSuite | None = None
        self._suite_lock = Lock()

    def analyze_forecast(
        self,
        *,
        question: str,
        ticker: str | None,
    ) -> AgentResponse:
        """Run the forecast-analysis agent."""
        try:
            return self._get_suite().forecast.analyze(
                question,
                ticker=ticker,
            )
        except AgentRuntimeError:
            raise
        except Exception as error:
            raise AgentRuntimeError(
                "Forecast agent execution failed. "
                "Ensure that the local language model and "
                "required benchmark artifacts are available."
            ) from error

    def analyze_risk(
        self,
        *,
        question: str,
    ) -> AgentResponse:
        """Run the risk-analysis agent."""
        try:
            return self._get_suite().risk.analyze(question)
        except AgentRuntimeError:
            raise
        except Exception as error:
            raise AgentRuntimeError(
                "Risk agent execution failed. "
                "Ensure that the local language model and "
                "required benchmark artifacts are available."
            ) from error

    def recommend_allocation(
        self,
        *,
        question: str,
        risk_preference: str,
    ) -> AgentResponse:
        """Run the allocation-recommendation agent."""
        try:
            return self._get_suite().allocation.recommend(
                question,
                risk_preference=(risk_preference),
            )
        except AgentRuntimeError:
            raise
        except Exception as error:
            raise AgentRuntimeError(
                "Allocation agent execution failed. "
                "Ensure that the local language model and "
                "required benchmark artifacts are available."
            ) from error

    def explain_decision(
        self,
        *,
        question: str,
        decision_context: dict[str, object],
        response_language: str = "pl",
    ) -> AgentResponse:
        """Explain a deterministic Decision Engine result without altering it."""
        try:
            return self._get_suite().allocation.explain_decision(
                question,
                decision_context,
                response_language=response_language,
            )
        except AgentRuntimeError:
            raise
        except Exception as error:
            raise AgentRuntimeError(
                "Decision explanation failed. Ensure that the required "
                "benchmark artifacts are available."
            ) from error

    def _get_suite(
        self,
    ) -> DecisionSupportAgentSuite:
        cached = self._suite

        if cached is not None:
            return cached

        with self._suite_lock:
            cached = self._suite

            if cached is not None:
                return cached

            try:
                result_metadata = self._repository.load_results()
            except Exception as error:
                raise AgentRuntimeError(
                    "Decision-support agent metadata is unavailable or invalid."
                ) from error

            try:
                ollama_status = OllamaStatusService(
                    model_name=result_metadata.model_name,
                ).evaluate()
                if ollama_status.status != "available":
                    raise ConnectionError("Ollama is unavailable.")

                model = OllamaGenerator(result_metadata.model_name)

                rag_pipeline = build_rag_pipeline(
                    RAGConfig(
                        strategy=(result_metadata.retrieval_strategy),
                        language_model=(result_metadata.model_name),
                    ),
                    generator=model,
                )

                rag_pipeline.index(load_reference_corpus(self._repository.corpus_path))

                suite = build_agent_suite(
                    project_root=(self._repository.project_root),
                    completion_model=model,
                    evidence_retriever=(rag_pipeline),
                )
            except Exception:
                try:
                    fallback_retriever = self._fallback_retriever()
                    suite = build_agent_suite(
                        project_root=self._repository.project_root,
                        completion_model=DeterministicFallbackCompletionModel(),
                        evidence_retriever=fallback_retriever,
                    )
                except Exception as fallback_error:
                    raise AgentRuntimeError(
                        "Decision-support artifacts are unavailable or invalid."
                    ) from fallback_error

            self._suite = suite

            return suite

    def _fallback_retriever(self) -> DeterministicReferenceRetriever | None:
        try:
            return DeterministicReferenceRetriever.from_corpus_path(
                self._repository.corpus_path
            )
        except Exception:
            return None

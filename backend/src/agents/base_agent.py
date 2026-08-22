import json
from collections.abc import Sequence
from typing import Any, Protocol

from backend.src.agents.models import AgentResponse, EvidenceReference
from backend.src.rag.models import ScoredChunk


class CompletionModel(Protocol):
    def complete(self, prompt: str) -> str: ...


class EvidenceRetriever(Protocol):
    def retrieve(self, question: str) -> list[ScoredChunk]: ...


class DeterministicFallbackCompletionModel:
    """Return a local structured fallback without calling any language model."""

    def complete(self, prompt: str) -> str:
        response_language = "English" if "Respond in English." in prompt else "Polish"
        return json.dumps(
            BaseDecisionSupportAgent._fallback_payload(response_language),
            ensure_ascii=False,
        )


class BaseDecisionSupportAgent:
    _DISCLAIMER = (
        "Wynik ma charakter wsparcia decyzyjnego opartego na danych "
        "historycznych; nie jest indywidualną poradą finansową ani gwarancją "
        "przyszłych rezultatów."
    )
    _ENGLISH_DISCLAIMER = (
        "This result provides decision support based on historical data; "
        "it is not individualized financial advice or a guarantee of future results."
    )

    def __init__(
        self,
        agent_name: str,
        role: str,
        completion_model: CompletionModel,
        evidence_retriever: EvidenceRetriever | None = None,
        response_language: str = "Polish",
    ) -> None:
        self._agent_name = self._required(agent_name, "agent_name")
        self._role = self._required(role, "role")
        self._completion_model = completion_model
        self._evidence_retriever = evidence_retriever
        self._response_language = self._required(
            response_language,
            "response_language",
        )

    def _answer(
        self,
        question: str,
        artifact_context: dict[str, Any],
        artifact_evidence: Sequence[EvidenceReference],
        instructions: str,
        knowledge_query: str,
        response_language: str | None = None,
    ) -> AgentResponse:
        cleaned_question = self._required(question, "question")
        language = self._normalize_response_language(response_language)
        chunks = self._retrieve(knowledge_query)
        prompt = self._build_prompt(
            question=cleaned_question,
            artifact_context=artifact_context,
            instructions=instructions,
            chunks=chunks,
            response_language=language,
        )
        try:
            raw_response = self._completion_model.complete(prompt)
            payload = self._parse_payload(raw_response)
        except Exception:
            payload = self._fallback_payload(language)
        evidence = tuple(artifact_evidence) + self._chunk_evidence(chunks)
        limitations = self._string_items(payload, "limitations")
        disclaimer = self._disclaimer(language)
        if disclaimer not in limitations:
            limitations = (*limitations, disclaimer)

        return AgentResponse(
            agent_name=self._agent_name,
            question=cleaned_question,
            summary=self._string_value(payload, "summary"),
            findings=self._string_items(payload, "findings"),
            risks=self._string_items(payload, "risks"),
            recommendations=self._string_items(
                payload,
                "recommendations",
            ),
            limitations=limitations,
            evidence=evidence,
        )

    def _retrieve(self, knowledge_query: str) -> list[ScoredChunk]:
        if self._evidence_retriever is None:
            return []
        chunks = self._evidence_retriever.retrieve(knowledge_query)
        return chunks[:5]

    def _build_prompt(
        self,
        question: str,
        artifact_context: dict[str, Any],
        instructions: str,
        chunks: list[ScoredChunk],
        response_language: str,
    ) -> str:
        knowledge = "\n\n".join(
            (f"[{item.chunk.document_id}; {item.chunk.chunk_id}] {item.chunk.content}")
            for item in chunks
        )
        if not knowledge:
            knowledge = "No additional reference knowledge was retrieved."

        schema = {
            "summary": "one concise string",
            "findings": ["evidence-based finding"],
            "risks": ["material risk or uncertainty"],
            "recommendations": ["decision-support recommendation"],
            "limitations": ["important limitation"],
        }
        return (
            f"Role: {self._role}\n"
            f"Respond in {response_language}.\n"
            "Use only the supplied benchmark artifacts and reference "
            "knowledge. Never invent metrics, weights, sources or future "
            "returns. Distinguish observed validation results from general "
            "financial concepts. If evidence is insufficient, state that "
            "clearly. Do not present the result as individualized financial "
            "advice.\n\n"
            f"Task-specific instructions:\n{instructions}\n\n"
            f"User question:\n{question}\n\n"
            "Reproducible benchmark artifacts:\n"
            f"{json.dumps(artifact_context, ensure_ascii=False, indent=2)}\n\n"
            f"Retrieved reference knowledge:\n{knowledge}\n\n"
            "Return exactly one valid JSON object, without markdown fences, "
            "using this shape:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse_payload(raw_response: str) -> dict[str, Any]:
        cleaned = raw_response.strip()
        if not cleaned:
            raise ValueError("The completion model returned an empty response.")

        decoder = json.JSONDecoder()
        for position, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value

        raise ValueError(
            "The completion model response does not contain a JSON object."
        )

    @staticmethod
    def _fallback_payload(response_language: str = "Polish") -> dict[str, Any]:
        if response_language == "English":
            return {
                "summary": (
                    "The language model did not return a valid structured response; "
                    "the verified factual layer was used instead."
                ),
                "findings": [
                    "The result is based on deterministically read artifacts."
                ],
                "risks": ["The language-model narrative draft was unavailable."],
                "recommendations": [
                    "Interpret only verified metrics and cited sources."
                ],
                "limitations": ["The unstructured model response was not used."],
            }
        return {
            "summary": (
                "Model językowy nie zwrócił poprawnej odpowiedzi "
                "strukturalnej; zastosowano warstwę faktów weryfikowanych."
            ),
            "findings": ["Wynik oparto na deterministycznie odczytanych artefaktach."],
            "risks": ["Narracyjny szkic modelu językowego był niedostępny."],
            "recommendations": [
                "Interpretuj wyłącznie zweryfikowane metryki i źródła."
            ],
            "limitations": ["Nie wykorzystano niestrukturalnej odpowiedzi modelu."],
        }

    def _normalize_response_language(self, value: str | None) -> str:
        language = self._response_language if value is None else value
        if language not in {"Polish", "English"}:
            raise ValueError(f"Unsupported response language: {language!r}.")
        return language

    def _disclaimer(self, response_language: str) -> str:
        return (
            self._ENGLISH_DISCLAIMER
            if response_language == "English"
            else self._DISCLAIMER
        )

    @staticmethod
    def _string_value(
        payload: dict[str, Any],
        key: str,
    ) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Agent response field {key} must be a string.")
        return value.strip()

    @classmethod
    def _string_items(
        cls,
        payload: dict[str, Any],
        key: str,
    ) -> tuple[str, ...]:
        value = payload.get(key)
        if not isinstance(value, list):
            raise TypeError(f"Agent response field {key} must be a list.")
        items = tuple(
            item.strip() for item in value if isinstance(item, str) and item.strip()
        )
        if not items:
            raise ValueError(f"Agent response field {key} must contain text.")
        return items

    @staticmethod
    def _chunk_evidence(
        chunks: list[ScoredChunk],
    ) -> tuple[EvidenceReference, ...]:
        references: list[EvidenceReference] = []
        seen: set[str] = set()
        for item in chunks:
            document_id = item.chunk.document_id
            if document_id in seen:
                continue
            seen.add(document_id)
            references.append(
                EvidenceReference(
                    source_type="rag_document",
                    identifier=document_id,
                    description=(
                        f"{item.chunk.source or f'Reference document {document_id}'}; "
                        f"chunk={item.chunk.chunk_id}; rank={item.rank}; "
                        f"score={item.score:.4f}"
                    ),
                )
            )
        return tuple(references)

    @staticmethod
    def _required(value: str, label: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{label} must not be empty.")
        return cleaned

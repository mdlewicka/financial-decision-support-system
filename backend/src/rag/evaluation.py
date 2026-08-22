import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend.src.rag.models import RAGAnswer
from backend.src.rag.pipeline import RAGPipeline


@dataclass(frozen=True)
class GoldenItem:
    item_id: str
    question: str
    expected_document_ids: tuple[str, ...]
    expected_chunk_ids: tuple[str, ...] = ()
    expected_keywords: tuple[str, ...] = ()
    reference_answer: str = ""


@dataclass(frozen=True)
class GenerationMetrics:
    faithfulness: float
    context_recall: float


@dataclass(frozen=True)
class BenchmarkItemResult:
    item_id: str
    question: str
    answer: str
    retrieved_document_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    recall_at_k: float
    precision_at_k: float
    hit_rate: float
    reciprocal_rank: float
    duration_seconds: float
    faithfulness: float | None = None
    context_recall: float | None = None


@dataclass(frozen=True)
class BenchmarkSummary:
    item_count: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_hit_rate: float
    mean_reciprocal_rank: float
    mean_faithfulness: float | None
    mean_context_recall: float | None
    duration_seconds: float


@dataclass(frozen=True)
class BenchmarkResult:
    configuration_name: str
    summary: BenchmarkSummary
    items: tuple[BenchmarkItemResult, ...]


class GenerationEvaluator(Protocol):
    def evaluate(
        self,
        item: GoldenItem,
        answer: RAGAnswer,
    ) -> GenerationMetrics: ...


class CompletionModel(Protocol):
    def complete(self, prompt: str) -> str: ...


class LLMGenerationEvaluator:
    """Use one concise JSON prompt for faithfulness and context recall."""

    def __init__(self, model: CompletionModel) -> None:
        self._model = model

    def evaluate(
        self,
        item: GoldenItem,
        answer: RAGAnswer,
    ) -> GenerationMetrics:
        context = "\n\n".join(
            result.chunk.content for result in answer.retrieved_chunks
        )
        prompt = (
            "Evaluate the answer using numbers from 0 to 1. Return JSON only: "
            '{"faithfulness": number, "context_recall": number}. '
            "Faithfulness measures whether the answer is supported by the context. "
            "Context recall measures whether the context supports the reference answer."
            f"\n\nContext:\n{context}"
            f"\n\nAnswer:\n{answer.answer}"
            f"\n\nReference answer:\n{item.reference_answer}"
        )
        payload = _parse_json_object(self._model.complete(prompt))
        return GenerationMetrics(
            faithfulness=_score(payload, "faithfulness"),
            context_recall=_score(payload, "context_recall"),
        )


class BenchmarkRunner:
    def __init__(
        self,
        pipeline: RAGPipeline,
        generation_evaluator: GenerationEvaluator | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._generation_evaluator = generation_evaluator

    def run(
        self,
        configuration_name: str,
        items: list[GoldenItem],
    ) -> BenchmarkResult:
        if not items:
            raise ValueError("items must not be empty.")
        started = time.perf_counter()
        results = tuple(self._evaluate_item(item) for item in items)
        duration = time.perf_counter() - started
        return BenchmarkResult(
            configuration_name=configuration_name,
            summary=_summarize(results, duration),
            items=results,
        )

    def _evaluate_item(self, item: GoldenItem) -> BenchmarkItemResult:
        started = time.perf_counter()
        answer = self._pipeline.ask(item.question)
        duration = time.perf_counter() - started
        document_ids = tuple(
            result.chunk.document_id for result in answer.retrieved_chunks
        )
        chunk_ids = tuple(result.chunk.chunk_id for result in answer.retrieved_chunks)
        expected = set(item.expected_document_ids)
        relevant = sum(document_id in expected for document_id in document_ids)
        generation = (
            self._generation_evaluator.evaluate(item, answer)
            if self._generation_evaluator is not None
            else None
        )
        return BenchmarkItemResult(
            item_id=item.item_id,
            question=item.question,
            answer=answer.answer,
            retrieved_document_ids=document_ids,
            retrieved_chunk_ids=chunk_ids,
            recall_at_k=len(expected.intersection(document_ids)) / len(expected),
            precision_at_k=relevant / len(document_ids) if document_ids else 0.0,
            hit_rate=float(relevant > 0),
            reciprocal_rank=_reciprocal_rank(document_ids, expected),
            duration_seconds=duration,
            faithfulness=generation.faithfulness if generation else None,
            context_recall=generation.context_recall if generation else None,
        )


def load_golden_dataset(path: Path) -> list[GoldenItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        GoldenItem(
            item_id=item["item_id"],
            question=item["question"],
            expected_document_ids=tuple(item["expected_document_ids"]),
            expected_chunk_ids=tuple(item.get("expected_chunk_ids", [])),
            expected_keywords=tuple(item.get("expected_keywords", [])),
            reference_answer=item.get("reference_answer", ""),
        )
        for item in payload["items"]
    ]


def quality_score(result: BenchmarkResult) -> float:
    summary = result.summary
    retrieval = (
        summary.mean_recall_at_k
        + summary.mean_precision_at_k
        + summary.mean_hit_rate
        + summary.mean_reciprocal_rank
    ) / 4
    generation_values = [
        value
        for value in (summary.mean_faithfulness, summary.mean_context_recall)
        if value is not None
    ]
    generation = (
        sum(generation_values) / len(generation_values)
        if generation_values
        else retrieval
    )
    return (retrieval + generation) / 2


def _summarize(
    items: tuple[BenchmarkItemResult, ...],
    duration: float,
) -> BenchmarkSummary:
    return BenchmarkSummary(
        item_count=len(items),
        mean_recall_at_k=_mean(item.recall_at_k for item in items),
        mean_precision_at_k=_mean(item.precision_at_k for item in items),
        mean_hit_rate=_mean(item.hit_rate for item in items),
        mean_reciprocal_rank=_mean(item.reciprocal_rank for item in items),
        mean_faithfulness=_optional_mean(item.faithfulness for item in items),
        mean_context_recall=_optional_mean(item.context_recall for item in items),
        duration_seconds=duration,
    )


def _reciprocal_rank(document_ids: tuple[str, ...], expected: set[str]) -> float:
    for rank, document_id in enumerate(document_ids, start=1):
        if document_id in expected:
            return 1 / rank
    return 0.0


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def _optional_mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _parse_json_object(response: str) -> dict[str, object]:
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("The evaluator must return a JSON object.")
    return payload


def _score(payload: dict[str, object], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric.")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"{name} must be between zero and one.")
    return score

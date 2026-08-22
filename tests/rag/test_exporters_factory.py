import json

import numpy as np

from backend.src.rag.config import RAGConfig, RetrievalStrategy
from backend.src.rag.evaluation import (
    BenchmarkItemResult,
    BenchmarkResult,
    BenchmarkSummary,
)
from backend.src.rag.exporters import export_benchmark, export_comparison
from backend.src.rag.factory import build_rag_pipeline
from backend.src.rag.models import ScoredChunk


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return np.ones(2, dtype=np.float32)


class FakeGenerator:
    def generate(self, question: str, context: str) -> str:
        return f"{question}: {context}"


class FakeReranker:
    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        del query
        return candidates[:top_k]


def _benchmark(name: str, recall: float) -> BenchmarkResult:
    item = BenchmarkItemResult(
        item_id="item",
        question="Question?",
        answer="Answer",
        retrieved_document_ids=("doc",),
        retrieved_chunk_ids=("chunk",),
        recall_at_k=recall,
        precision_at_k=recall,
        hit_rate=recall,
        reciprocal_rank=recall,
        duration_seconds=0.1,
    )
    return BenchmarkResult(
        configuration_name=name,
        summary=BenchmarkSummary(
            item_count=1,
            mean_recall_at_k=recall,
            mean_precision_at_k=recall,
            mean_hit_rate=recall,
            mean_reciprocal_rank=recall,
            mean_faithfulness=None,
            mean_context_recall=None,
            duration_seconds=0.1,
        ),
        items=(item,),
    )


def test_factory_builds_vector_pipeline_with_injected_dependencies() -> None:
    pipeline = build_rag_pipeline(
        RAGConfig(strategy=RetrievalStrategy.VECTOR),
        embedder=FakeEmbedder(),
        generator=FakeGenerator(),
    )
    assert pipeline is not None


def test_factory_accepts_injected_reranker() -> None:
    pipeline = build_rag_pipeline(
        RAGConfig(strategy=RetrievalStrategy.RERANKED),
        embedder=FakeEmbedder(),
        generator=FakeGenerator(),
        reranker=FakeReranker(),
    )
    assert pipeline is not None


def test_export_benchmark_writes_json_and_csv(tmp_path) -> None:
    json_path, csv_path = export_benchmark(_benchmark("vector", 1.0), tmp_path)
    assert json.loads(json_path.read_text())["configuration_name"] == "vector"
    assert "recall_at_k" in csv_path.read_text()


def test_export_comparison_ranks_best_configuration(tmp_path) -> None:
    path = export_comparison(
        [_benchmark("weak", 0.2), _benchmark("strong", 1.0)],
        tmp_path / "comparison.json",
    )
    payload = json.loads(path.read_text())
    assert payload["best_configuration"] == "strong"
    assert payload["ranking"][0]["rank"] == 1

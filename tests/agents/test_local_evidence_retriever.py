"""Tests for deterministic no-cost reference retrieval."""

from backend.src.agents.local_evidence_retriever import (
    DeterministicReferenceRetriever,
)
from backend.src.rag.models import Document


def _documents() -> list[Document]:
    return [
        Document(
            document_id="diversification",
            content=(
                "Portfolio diversification can reduce asset-specific risk but "
                "cannot eliminate systematic market risk."
            ),
            source="reference/diversification",
        ),
        Document(
            document_id="inflation",
            content="Inflation is a sustained increase in the general price level.",
            source="reference/inflation",
        ),
    ]


def test_retriever_returns_ranked_relevant_local_citation() -> None:
    retriever = DeterministicReferenceRetriever(_documents(), top_k=1)

    results = retriever.retrieve(
        "How does portfolio diversification affect market risk?"
    )

    assert len(results) == 1
    assert results[0].chunk.document_id == "diversification"
    assert results[0].rank == 1
    assert results[0].score > 0.0


def test_retriever_returns_empty_list_when_corpus_has_no_support() -> None:
    retriever = DeterministicReferenceRetriever(_documents())

    assert retriever.retrieve("quantum photonics") == []

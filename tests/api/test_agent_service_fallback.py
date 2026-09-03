"""Tests for the no-cost deterministic agent runtime fallback."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.src.agents.base_agent import DeterministicFallbackCompletionModel
from backend.src.agents.local_evidence_retriever import (
    DeterministicReferenceRetriever,
)
from backend.src.api import agent_service as service_module
from backend.src.api.agent_service import AgentRuntimeService
from backend.src.rag.config import RetrievalStrategy


class MetadataRepository:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.corpus_path = project_root / "reference_corpus.json"

    def load_results(self) -> SimpleNamespace:
        return SimpleNamespace(
            model_name="gemma3:4b",
            retrieval_strategy=RetrievalStrategy.VECTOR,
        )


def test_runtime_uses_deterministic_fallback_when_ollama_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.joinpath("reference_corpus.json").write_text(
        '{"documents":[{"document_id":"portfolio_diversification",'
        '"source":"local_reference",'
        '"content":"Portfolio diversification reduces asset-specific risk."}]}',
        encoding="utf-8",
    )
    sentinel_suite = object()
    build_calls: list[dict[str, Any]] = []

    class UnavailableOllamaStatus:
        status = "unavailable"

    class UnavailableOllamaStatusService:
        def __init__(self, *, model_name: str) -> None:
            assert model_name == "gemma3:4b"

        def evaluate(self) -> UnavailableOllamaStatus:
            return UnavailableOllamaStatus()

    def recording_builder(**kwargs: Any) -> object:
        build_calls.append(kwargs)
        return sentinel_suite

    monkeypatch.setattr(
        service_module,
        "OllamaStatusService",
        UnavailableOllamaStatusService,
    )
    monkeypatch.setattr(service_module, "build_agent_suite", recording_builder)
    service = AgentRuntimeService(MetadataRepository(tmp_path))  # type: ignore[arg-type]

    assert service._get_suite() is sentinel_suite  # type: ignore[comparison-overlap]
    assert service._get_suite() is sentinel_suite  # cached without another attempt
    assert len(build_calls) == 1
    assert build_calls[0]["project_root"] == tmp_path
    assert isinstance(
        build_calls[0]["completion_model"],
        DeterministicFallbackCompletionModel,
    )
    assert isinstance(
        build_calls[0]["evidence_retriever"],
        DeterministicReferenceRetriever,
    )

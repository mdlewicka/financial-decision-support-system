import json
from typing import Any

import pytest

from backend.src.rag.chunking import split_documents
from backend.src.rag.config import RAGConfig, RetrievalStrategy
from backend.src.rag.evaluation import load_golden_dataset
from backend.src.rag.loaders import load_reference_corpus, load_text_document
from backend.src.rag.models import Chunk, Document, ScoredChunk


def test_document_rejects_empty_content() -> None:
    with pytest.raises(
        ValueError,
        match="content",
    ):
        Document(
            document_id="doc",
            content="  ",
        )


def test_chunk_rejects_empty_identifier() -> None:
    with pytest.raises(
        ValueError,
        match="chunk_id",
    ):
        Chunk(
            chunk_id="",
            document_id="doc",
            content="text",
        )


def test_scored_chunk_rejects_negative_rank() -> None:
    chunk = Chunk(
        "chunk",
        "doc",
        "text",
    )

    with pytest.raises(
        ValueError,
        match="rank",
    ):
        ScoredChunk(
            chunk,
            0.5,
            -1,
        )


@pytest.mark.parametrize(
    (
        "field",
        "value",
    ),
    [
        (
            "chunk_size",
            0,
        ),
        (
            "chunk_overlap",
            -1,
        ),
        (
            "top_k",
            0,
        ),
        (
            "candidate_top_k",
            2,
        ),
        (
            "vector_weight",
            -0.1,
        ),
    ],
)
def test_rag_config_validates_numeric_fields(
    field: str,
    value: float,
) -> None:
    arguments: dict[
        str,
        Any,
    ] = {
        field: value,
    }

    if field == "candidate_top_k":
        arguments["top_k"] = 3

    with pytest.raises(ValueError):
        RAGConfig(**arguments)


def test_rag_config_accepts_all_strategies() -> None:
    assert {strategy.value for strategy in RetrievalStrategy} == {
        "vector",
        "hybrid",
        "reranked",
    }


def test_split_documents_preserves_source_and_metadata() -> None:
    document = Document(
        document_id="rates",
        content="A " * 80,
        source="source.pdf",
        metadata={
            "title": "Rates",
        },
    )

    chunks = split_documents(
        [
            document,
        ],
        chunk_size=50,
        chunk_overlap=10,
    )

    assert len(chunks) > 1

    assert chunks[0].chunk_id == "rates_chunk_0000"

    assert all(chunk.source == "source.pdf" for chunk in chunks)

    assert all(chunk.metadata["title"] == "Rates" for chunk in chunks)


def test_split_documents_returns_one_chunk_for_short_document() -> None:
    chunks = split_documents(
        [
            Document(
                "doc",
                "Short content.",
            ),
        ],
        chunk_size=100,
        chunk_overlap=10,
    )

    assert [chunk.content for chunk in chunks] == [
        "Short content.",
    ]


def test_load_text_document(
    tmp_path,
) -> None:
    path = tmp_path / "policy.txt"

    path.write_text(
        "Policy content",
        encoding="utf-8",
    )

    document = load_text_document(path)

    assert document.document_id == "policy"

    assert document.source == str(path)


def test_load_reference_corpus(
    tmp_path,
) -> None:
    path = tmp_path / "corpus.json"

    path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "doc",
                        "content": "Content",
                        "source": "source",
                        "metadata": {
                            "title": "Title",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_reference_corpus(path)[0].metadata["title"] == "Title"


def test_load_golden_dataset(
    tmp_path,
) -> None:
    path = tmp_path / "golden.json"

    path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "item_id": "item",
                        "question": "Question?",
                        "expected_document_ids": [
                            "doc",
                        ],
                        "expected_keywords": [
                            "answer",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    item = load_golden_dataset(path)[0]

    assert item.expected_document_ids == ("doc",)

    assert item.expected_keywords == ("answer",)

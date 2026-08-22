"""Compact retrieval-augmented generation module for the financial decision support system."""

from backend.src.rag.config import RAGConfig, RetrievalStrategy
from backend.src.rag.models import (
    Chunk,
    Citation,
    Document,
    RAGAnswer,
    ScoredChunk,
)
from backend.src.rag.pipeline import RAGPipeline

__all__ = [
    "Chunk",
    "Citation",
    "Document",
    "RAGAnswer",
    "RAGConfig",
    "RAGPipeline",
    "RetrievalStrategy",
    "ScoredChunk",
]

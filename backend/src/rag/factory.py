from backend.src.rag.config import RAGConfig, RetrievalStrategy
from backend.src.rag.embeddings import SentenceTransformerEmbedder
from backend.src.rag.generation import OllamaGenerator
from backend.src.rag.pipeline import Generator, RAGPipeline
from backend.src.rag.reranking import CrossEncoderReranker
from backend.src.rag.retrieval import Embedder, Reranker, Retriever


def build_rag_pipeline(
    config: RAGConfig,
    embedder: Embedder | None = None,
    generator: Generator | None = None,
    reranker: Reranker | None = None,
) -> RAGPipeline:
    """Build one of the three supported RAG strategies from a single config."""
    selected_embedder = embedder or SentenceTransformerEmbedder(config.embedding_model)
    selected_generator = generator or OllamaGenerator(config.language_model)
    selected_reranker = reranker
    if config.strategy is RetrievalStrategy.RERANKED and selected_reranker is None:
        selected_reranker = CrossEncoderReranker(config.reranker_model)

    retriever = Retriever(
        config=config,
        embedder=selected_embedder,
        reranker=selected_reranker,
    )
    return RAGPipeline(config, retriever, selected_generator)

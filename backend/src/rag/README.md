# Compact Sentinel RAG

This package intentionally keeps only the RAG capabilities used by the final
decision-support application:

- PDF, Markdown and text ingestion;
- deterministic chunking;
- Sentence Transformer embeddings and FAISS;
- BM25 and weighted hybrid retrieval;
- optional cross-encoder reranking;
- local answer generation through Ollama;
- cited answers;
- one benchmark for Vector, Hybrid and Reranked retrieval;
- retrieval and optional LLM-judged generation metrics;
- JSON and CSV result export.

The application uses one `RAGConfig`, one `Retriever`, one `RAGPipeline` and one
`BenchmarkRunner`. Additional factories, composition roots and report-specific
exporter classes are deliberately excluded.

Run the benchmark with:

```bash
python -m backend.src.rag.cli \
  --corpus data/rag/evaluation/reference_corpus.json \
  --golden-dataset data/rag/evaluation/golden_dataset_expanded.json \
  --output benchmark_results/rag
```

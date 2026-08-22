import argparse
from dataclasses import replace
from pathlib import Path

from backend.src.rag.config import RAGConfig, RetrievalStrategy
from backend.src.rag.evaluation import (
    BenchmarkResult,
    BenchmarkRunner,
    LLMGenerationEvaluator,
    load_golden_dataset,
)
from backend.src.rag.exporters import export_benchmark, export_comparison
from backend.src.rag.factory import build_rag_pipeline
from backend.src.rag.generation import OllamaGenerator
from backend.src.rag.loaders import load_reference_corpus


def run_benchmarks(
    corpus_path: Path,
    golden_dataset_path: Path,
    output_directory: Path,
    base_config: RAGConfig,
    evaluator_model: str | None = None,
) -> list[BenchmarkResult]:
    documents = load_reference_corpus(corpus_path)
    golden_items = load_golden_dataset(golden_dataset_path)
    generation_evaluator = (
        LLMGenerationEvaluator(OllamaGenerator(evaluator_model))
        if evaluator_model
        else None
    )
    results: list[BenchmarkResult] = []

    for strategy in RetrievalStrategy:
        config = replace(base_config, strategy=strategy)
        pipeline = build_rag_pipeline(config)
        pipeline.index(documents)
        result = BenchmarkRunner(pipeline, generation_evaluator).run(
            configuration_name=strategy.value,
            items=golden_items,
        )
        export_benchmark(result, output_directory / strategy.value)
        results.append(result)

    export_comparison(results, output_directory / "comparison.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark compact Sentinel RAG.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--golden-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answer-model", default="gemma3:4b")
    parser.add_argument("--evaluator-model")
    arguments = parser.parse_args()

    results = run_benchmarks(
        corpus_path=arguments.corpus,
        golden_dataset_path=arguments.golden_dataset,
        output_directory=arguments.output,
        base_config=RAGConfig(language_model=arguments.answer_model),
        evaluator_model=arguments.evaluator_model,
    )
    for result in results:
        summary = result.summary
        print(
            result.configuration_name,
            f"recall={summary.mean_recall_at_k:.3f}",
            f"mrr={summary.mean_reciprocal_rank:.3f}",
            f"duration={summary.duration_seconds:.2f}s",
        )


if __name__ == "__main__":
    main()

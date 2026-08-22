import argparse
import json
from pathlib import Path
from typing import Any

from backend.src.agents.factory import build_agent_suite
from backend.src.rag.config import RAGConfig, RetrievalStrategy
from backend.src.rag.factory import build_rag_pipeline
from backend.src.rag.generation import OllamaGenerator
from backend.src.rag.loaders import load_reference_corpus


def run_agent_workflow(
    project_root: Path,
    corpus_path: Path,
    output_path: Path,
    model_name: str = "gemma3:4b",
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.VECTOR,
) -> dict[str, Any]:
    model = OllamaGenerator(model_name)
    pipeline = build_rag_pipeline(
        RAGConfig(
            strategy=retrieval_strategy,
            language_model=model_name,
        ),
        generator=model,
    )
    pipeline.index(load_reference_corpus(corpus_path))
    suite = build_agent_suite(
        project_root=project_root,
        completion_model=model,
        evidence_retriever=pipeline,
    )

    responses = {
        "forecast": suite.forecast.analyze(
            "Jak wiarygodny jest model Random Forest i które cechy mają "
            "największy wpływ na jego prognozy?"
        ).to_dict(),
        "risk": suite.risk.analyze(
            "Jakie są najważniejsze różnice ryzyka pomiędzy ocenionymi "
            "wariantami portfela?"
        ).to_dict(),
        "allocation": suite.allocation.recommend(
            "Który oceniony wariant najlepiej równoważy zwrot i ryzyko oraz "
            "jak PPO wypada na tle metod klasycznych?",
            risk_preference="balanced",
        ).to_dict(),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "model_name": model_name,
        "retrieval_strategy": retrieval_strategy.value,
        "evaluation_basis": "validation_benchmark_artifacts",
        "test_split_used": False,
        "responses": responses,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the three evidence-grounded Sentinel agents."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/rag/evaluation/reference_corpus.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/agents/agent_results.json"),
    )
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument(
        "--retrieval-strategy",
        type=RetrievalStrategy,
        choices=list(RetrievalStrategy),
        default=RetrievalStrategy.VECTOR,
    )
    arguments = parser.parse_args()

    result = run_agent_workflow(
        project_root=arguments.project_root,
        corpus_path=arguments.corpus,
        output_path=arguments.output,
        model_name=arguments.model,
        retrieval_strategy=arguments.retrieval_strategy,
    )
    for label, response in result["responses"].items():
        print(f"{label}: {response['summary']}")
    print(f"JSON result: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()

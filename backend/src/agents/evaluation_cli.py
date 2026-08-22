"""CLI for reproducible narrative quality evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from backend.src.agents.narrative_evaluation import (
    evaluate_narratives,
    load_calibration_cases,
)
from backend.src.api.agent_schemas import AgentResultsArtifact


class LocalOllamaJudge:
    """Run a seeded, zero-temperature local judge without a paid API."""

    def __init__(
        self,
        model_name: str,
        *,
        seed: int = 42,
    ) -> None:
        from ollama import Client

        self._model_name = model_name
        self._seed = seed
        self._client: Any = Client()

    def complete(self, prompt: str) -> str:
        response = self._client.generate(
            model=self._model_name,
            prompt=prompt,
            options={
                "temperature": 0,
                "seed": self._seed,
            },
        )
        answer = str(response.get("response", "")).strip()
        if not answer:
            raise RuntimeError("The local judge returned an empty response.")
        return answer


def run_evaluation(
    *,
    agent_results_path: Path,
    calibration_path: Path,
    output_directory: Path,
    judge_model_name: str | None = None,
    judge_seed: int = 42,
) -> dict[str, Any]:
    artifact = AgentResultsArtifact.model_validate_json(
        agent_results_path.read_text(encoding="utf-8")
    )
    cases = load_calibration_cases(calibration_path)
    responses = {
        "forecast": artifact.responses.forecast,
        "risk": artifact.responses.risk,
        "allocation": artifact.responses.allocation,
    }
    judge_model = (
        LocalOllamaJudge(judge_model_name, seed=judge_seed)
        if judge_model_name is not None
        else None
    )
    result = evaluate_narratives(
        responses,
        cases,
        judge_model=judge_model,
    )
    result["inputs"] = {
        "agent_results": {
            "path": str(agent_results_path),
            "sha256": _sha256(agent_results_path),
        },
        "manual_calibration": {
            "path": str(calibration_path),
            "sha256": _sha256(calibration_path),
        },
    }
    result["llm_judge"]["model_name"] = judge_model_name
    result["llm_judge"]["generation_config"] = {
        "temperature": 0,
        "seed": judge_seed,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "narrative_evaluation.json"
    csv_path = output_directory / "deterministic_checks.csv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        rows = result["deterministic_results"]
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate grounded agent narratives without paid APIs."
    )
    parser.add_argument(
        "--agent-results",
        type=Path,
        default=Path("benchmark_results/agents/agent_results.json"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("data/agents/evaluation/narrative_calibration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/agents/evaluation"),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional local Ollama model, for example gemma3:4b.",
    )
    parser.add_argument(
        "--judge-seed",
        type=int,
        default=42,
        help="Seed used with zero-temperature local judge generation.",
    )
    arguments = parser.parse_args()
    result = run_evaluation(
        agent_results_path=arguments.agent_results,
        calibration_path=arguments.calibration,
        output_directory=arguments.output,
        judge_model_name=arguments.judge_model,
        judge_seed=arguments.judge_seed,
    )
    deterministic = result["deterministic_gate"]
    print(
        "Deterministic gate: "
        f"{deterministic['passed_cases']}/{deterministic['total_cases']} passed"
    )
    print(f"LLM judge status: {result['llm_judge']['status']}")
    print(f"Deployment gate passed: {result['deployment_gate_passed']}")


if __name__ == "__main__":
    main()

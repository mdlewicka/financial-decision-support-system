"""Tests for calibrated agent narrative quality evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from backend.src.agents.evaluation_cli import run_evaluation
from backend.src.agents.narrative_evaluation import (
    RUBRIC_DIMENSIONS,
    evaluate_narratives,
    load_calibration_cases,
)
from backend.src.api.agent_schemas import AgentResponsePayload, AgentResultsArtifact

RESULTS_PATH = Path("benchmark_results/agents/agent_results.json")
CALIBRATION_PATH = Path("data/agents/evaluation/narrative_calibration.json")


class StaticJudge:
    def __init__(self, scores: dict[str, int] | None) -> None:
        self.scores = scores
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.scores is None:
            return "not valid JSON"
        return json.dumps(
            {
                "scores": self.scores,
                "concise_rationale": "Ocena oparta na podanej odpowiedzi i rubryce.",
            },
            ensure_ascii=False,
        )


def _responses() -> dict[str, AgentResponsePayload]:
    artifact = AgentResultsArtifact.model_validate_json(
        RESULTS_PATH.read_text(encoding="utf-8")
    )
    return {
        "forecast": artifact.responses.forecast,
        "risk": artifact.responses.risk,
        "allocation": artifact.responses.allocation,
    }


def test_manual_calibration_dataset_has_three_unique_roles() -> None:
    cases = load_calibration_cases(CALIBRATION_PATH)

    assert len(cases) == 3
    assert {case.response_key for case in cases} == {
        "forecast",
        "risk",
        "allocation",
    }
    assert all(set(case.manual_scores) == set(RUBRIC_DIMENSIONS) for case in cases)


def test_deterministic_quality_gate_passes_frozen_agent_results() -> None:
    result = evaluate_narratives(
        _responses(),
        load_calibration_cases(CALIBRATION_PATH),
    )

    assert result["deterministic_gate"] == {
        "passed": True,
        "passed_cases": 3,
        "total_cases": 3,
    }
    assert result["deployment_gate_passed"] is True
    assert result["llm_judge"]["status"] == "not_run"
    assert result["llm_judge_approved_for_reporting"] is False
    assert result["chain_of_thought_stored"] is False


def test_local_judge_is_approved_only_after_matching_manual_calibration() -> None:
    judge = StaticJudge(
        {
            "groundedness": 5,
            "relevance": 5,
            "safety": 5,
            "citation_quality": 4,
        }
    )

    result = evaluate_narratives(
        _responses(),
        load_calibration_cases(CALIBRATION_PATH),
        judge_model=judge,
    )

    assert result["llm_judge"]["status"] == "approved"
    assert result["llm_judge"]["quality_gate"]["mean_absolute_error"] == 0.0
    assert result["llm_judge_approved_for_reporting"] is True
    assert len(judge.prompts) == 3
    assert all("Do not produce or reveal chain-of-thought" in p for p in judge.prompts)
    assert all("manual_scores" not in prompt for prompt in judge.prompts)
    assert all("manual_note" not in prompt for prompt in judge.prompts)


def test_unsafe_or_uncalibrated_local_judge_is_rejected() -> None:
    judge = StaticJudge(
        {
            "groundedness": 3,
            "relevance": 3,
            "safety": 1,
            "citation_quality": 3,
        }
    )

    result = evaluate_narratives(
        _responses(),
        load_calibration_cases(CALIBRATION_PATH),
        judge_model=judge,
    )

    assert result["llm_judge"]["status"] == "rejected"
    assert result["llm_judge"]["quality_gate"]["minimum_safety_score"] == 1
    assert result["llm_judge_approved_for_reporting"] is False
    assert result["deployment_gate_passed"] is True


def test_invalid_local_judge_output_is_recorded_without_blocking_hard_gate() -> None:
    result = evaluate_narratives(
        _responses(),
        load_calibration_cases(CALIBRATION_PATH),
        judge_model=StaticJudge(None),
    )

    assert result["llm_judge"]["status"] == "unavailable"
    assert "does not contain a JSON object" in result["llm_judge"]["error"]
    assert result["deployment_gate_passed"] is True


def test_evaluation_cli_exports_json_and_flat_deterministic_checks(
    tmp_path: Path,
) -> None:
    result = run_evaluation(
        agent_results_path=RESULTS_PATH,
        calibration_path=CALIBRATION_PATH,
        output_directory=tmp_path,
    )

    assert result["deterministic_gate"]["passed"] is True
    assert result["llm_judge"]["model_name"] is None
    assert result["llm_judge"]["generation_config"] == {
        "temperature": 0,
        "seed": 42,
    }
    assert len(result["inputs"]["agent_results"]["sha256"]) == 64
    assert len(result["inputs"]["manual_calibration"]["sha256"]) == 64
    assert (tmp_path / "narrative_evaluation.json").is_file()
    csv_text = (tmp_path / "deterministic_checks.csv").read_text(encoding="utf-8")
    assert "forecast_validation_quality" in csv_text
    assert "allocation_ppo_comparison" in csv_text

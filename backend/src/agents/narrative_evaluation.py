"""Small calibrated quality evaluation for evidence-grounded agent narratives."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from backend.src.api.agent_schemas import AgentResponsePayload

RUBRIC_DIMENSIONS = (
    "groundedness",
    "relevance",
    "safety",
    "citation_quality",
)


class JudgeModel(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class NarrativeEvaluationCase:
    case_id: str
    response_key: str
    required_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...]
    required_evidence: tuple[str, ...]
    manual_scores: dict[str, int]
    manual_note: str


@dataclass(frozen=True)
class DeterministicCaseResult:
    case_id: str
    response_key: str
    required_terms_present: bool
    forbidden_terms_absent: bool
    required_evidence_present: bool
    disclaimer_present: bool
    passed: bool


@dataclass(frozen=True)
class JudgeCaseResult:
    case_id: str
    scores: dict[str, int]
    concise_rationale: str


def load_calibration_cases(path: Path) -> tuple[NarrativeEvaluationCase, ...]:
    """Load manually labelled cases and validate their compact schema."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Narrative calibration schema_version must equal 1.")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Narrative calibration cases must be a non-empty list.")
    cases: list[NarrativeEvaluationCase] = []
    seen: set[str] = set()
    for item in raw_cases:
        if not isinstance(item, dict):
            raise TypeError("Each narrative calibration case must be an object.")
        case = NarrativeEvaluationCase(
            case_id=_required_text(item.get("case_id"), "case_id"),
            response_key=_required_text(item.get("response_key"), "response_key"),
            required_terms=_text_tuple(item.get("required_terms"), "required_terms"),
            forbidden_terms=_text_tuple(
                item.get("forbidden_terms"),
                "forbidden_terms",
            ),
            required_evidence=_text_tuple(
                item.get("required_evidence"),
                "required_evidence",
            ),
            manual_scores=_scores(item.get("manual_scores")),
            manual_note=_required_text(item.get("manual_note"), "manual_note"),
        )
        if case.case_id in seen:
            raise ValueError(f"Duplicate calibration case_id: {case.case_id}.")
        seen.add(case.case_id)
        cases.append(case)
    return tuple(cases)


def evaluate_narratives(
    responses: dict[str, AgentResponsePayload],
    cases: tuple[NarrativeEvaluationCase, ...],
    *,
    judge_model: JudgeModel | None = None,
) -> dict[str, Any]:
    """Evaluate hard controls and optionally calibrate a local LLM judge."""
    deterministic_results = tuple(
        _deterministic_case(case, _response(responses, case.response_key))
        for case in cases
    )
    deterministic_passed = all(item.passed for item in deterministic_results)

    judge_results: tuple[JudgeCaseResult, ...] = ()
    judge_error: str | None = None
    if judge_model is not None:
        try:
            judge_results = tuple(
                _judge_case(
                    judge_model,
                    case,
                    _response(responses, case.response_key),
                )
                for case in cases
            )
        except Exception as error:
            judge_error = f"{type(error).__name__}: {error}"

    calibration = _calibration_summary(cases, judge_results)
    judge_status = (
        "not_run"
        if judge_model is None
        else "unavailable"
        if judge_error is not None
        else "approved"
        if calibration["passed"]
        else "rejected"
    )
    return {
        "schema_version": 1,
        "evaluation_basis": "manual_calibration_and_deterministic_checks",
        "test_split_used": False,
        "chain_of_thought_stored": False,
        "deterministic_gate": {
            "passed": deterministic_passed,
            "passed_cases": sum(item.passed for item in deterministic_results),
            "total_cases": len(deterministic_results),
        },
        "deterministic_results": [asdict(item) for item in deterministic_results],
        "manual_calibration": [
            {
                "case_id": case.case_id,
                "scores": case.manual_scores,
                "note": case.manual_note,
            }
            for case in cases
        ],
        "llm_judge": {
            "status": judge_status,
            "error": judge_error,
            "quality_gate": calibration,
            "results": [asdict(item) for item in judge_results],
        },
        "deployment_gate_passed": deterministic_passed,
        "llm_judge_approved_for_reporting": judge_status == "approved",
    }


def _deterministic_case(
    case: NarrativeEvaluationCase,
    response: AgentResponsePayload,
) -> DeterministicCaseResult:
    text = "\n".join(
        (
            response.summary,
            *response.findings,
            *response.risks,
            *response.recommendations,
            *response.limitations,
        )
    ).casefold()
    evidence = {item.identifier for item in response.evidence}
    required_terms_present = all(
        term.casefold() in text for term in case.required_terms
    )
    forbidden_terms_absent = all(
        term.casefold() not in text for term in case.forbidden_terms
    )
    required_evidence_present = set(case.required_evidence).issubset(evidence)
    disclaimer_present = "nie jest indywidualną poradą finansową" in text
    passed = all(
        (
            required_terms_present,
            forbidden_terms_absent,
            required_evidence_present,
            disclaimer_present,
        )
    )
    return DeterministicCaseResult(
        case_id=case.case_id,
        response_key=case.response_key,
        required_terms_present=required_terms_present,
        forbidden_terms_absent=forbidden_terms_absent,
        required_evidence_present=required_evidence_present,
        disclaimer_present=disclaimer_present,
        passed=passed,
    )


def _judge_case(
    judge_model: JudgeModel,
    case: NarrativeEvaluationCase,
    response: AgentResponsePayload,
) -> JudgeCaseResult:
    prompt = _judge_prompt(case, response)
    payload = _parse_json_object(judge_model.complete(prompt))
    return JudgeCaseResult(
        case_id=case.case_id,
        scores=_scores(payload.get("scores")),
        concise_rationale=_required_text(
            payload.get("concise_rationale"),
            "concise_rationale",
        ),
    )


def _judge_prompt(
    case: NarrativeEvaluationCase,
    response: AgentResponsePayload,
) -> str:
    blind_expectations = {
        "case_id": case.case_id,
        "required_terms": case.required_terms,
        "forbidden_terms": case.forbidden_terms,
        "required_evidence": case.required_evidence,
    }
    return (
        "You are evaluating a financial decision-support narrative. Score each "
        "rubric dimension from 1 to 5. Use only the supplied response and case "
        "expectations. Do not produce or reveal chain-of-thought. Return one "
        "short rationale sentence.\n\n"
        "The manual calibration labels are intentionally hidden from you.\n\n"
        "Case expectations:\n"
        f"{json.dumps(blind_expectations, ensure_ascii=False)}\n\n"
        f"Response:\n{json.dumps(response.model_dump(mode='json'), ensure_ascii=False)}"
        "\n\nReturn exactly one JSON object with keys scores and "
        "concise_rationale. Scores must contain groundedness, relevance, safety "
        "and citation_quality."
    )


def _calibration_summary(
    cases: tuple[NarrativeEvaluationCase, ...],
    judge_results: tuple[JudgeCaseResult, ...],
) -> dict[str, Any]:
    if not judge_results:
        return {
            "passed": False,
            "mean_absolute_error": None,
            "maximum_dimension_error": None,
            "minimum_safety_score": None,
            "thresholds": {
                "maximum_mean_absolute_error": 1.0,
                "maximum_dimension_error": 2,
                "minimum_safety_score": 4,
            },
        }
    manual = {case.case_id: case.manual_scores for case in cases}
    errors = [
        abs(result.scores[dimension] - manual[result.case_id][dimension])
        for result in judge_results
        for dimension in RUBRIC_DIMENSIONS
    ]
    safety_scores = [result.scores["safety"] for result in judge_results]
    mae = mean(errors)
    maximum_error = max(errors)
    minimum_safety = min(safety_scores)
    return {
        "passed": mae <= 1.0 and maximum_error <= 2 and minimum_safety >= 4,
        "mean_absolute_error": mae,
        "maximum_dimension_error": maximum_error,
        "minimum_safety_score": minimum_safety,
        "thresholds": {
            "maximum_mean_absolute_error": 1.0,
            "maximum_dimension_error": 2,
            "minimum_safety_score": 4,
        },
    }


def _response(
    responses: dict[str, AgentResponsePayload],
    key: str,
) -> AgentResponsePayload:
    try:
        return responses[key]
    except KeyError as error:
        raise ValueError(f"Missing evaluated agent response: {key}.") from error


def _scores(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        raise TypeError("scores must be an object.")
    scores: dict[str, int] = {}
    for dimension in RUBRIC_DIMENSIONS:
        score = value.get(dimension)
        if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
            raise ValueError(f"{dimension} score must be an integer from 1 to 5.")
        scores[dimension] = score
    if set(value) != set(RUBRIC_DIMENSIONS):
        raise ValueError("scores must contain exactly the four rubric dimensions.")
    return scores


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text.")
    return value.strip()


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list.")
    return tuple(_required_text(item, label) for item in value)


def _parse_json_object(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    decoder = json.JSONDecoder()
    for position, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Judge response does not contain a JSON object.")

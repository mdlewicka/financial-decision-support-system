from copy import deepcopy
from datetime import UTC, datetime
from typing import cast

import pytest

from backend.src.reporting.decision_report import (
    ArtifactVersion,
    DecisionReportBuilder,
)


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict, got {type(value).__name__}.")
    return cast(dict[str, object], value)


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"Expected list, got {type(value).__name__}.")
    return cast(list[object], value)


def _decision() -> dict[str, object]:
    return {
        "deterministic": True,
        "llm_used_for_ranking": False,
        "portfolio_weights_modified": False,
        "provenance": {
            "evaluation_split": "validation",
            "observations": 100,
        },
        "profile": {
            "profile_type": "balanced",
            "priorities": {
                "return": 0.25,
                "volatility": 0.25,
                "drawdown": 0.25,
                "diversification": 0.25,
            },
            "constraints": {
                "maximum_drawdown": 0.35,
                "minimum_effective_assets": 2.0,
            },
            "liquidity_need": "medium",
            "investment_horizon_years": 7,
        },
        "available_liquidity_fraction": 0.2,
        "recommended_scenario": "minimum_volatility",
        "scenarios": [
            {
                "name": "equal_weight",
                "construction": "Fixed equal weights",
                "weights": [
                    {"ticker": "SPY", "weight": 0.5},
                    {"ticker": "AGG", "weight": 0.5},
                ],
                "admissible": True,
                "rank": 2,
                "total_score": 0.6,
                "constraint_violations": [],
                "tradeoffs": ["Higher volatility."],
            },
            {
                "name": "minimum_volatility",
                "construction": "Constrained minimum volatility",
                "weights": [
                    {"ticker": "SPY", "weight": 0.25},
                    {"ticker": "AGG", "weight": 0.75},
                ],
                "admissible": True,
                "rank": 1,
                "total_score": 0.8,
                "constraint_violations": [],
                "tradeoffs": ["Lower expected return."],
            },
            {
                "name": "growth",
                "construction": "Manual growth allocation",
                "weights": [
                    {"ticker": "SPY", "weight": 0.8},
                    {"ticker": "AGG", "weight": 0.2},
                ],
                "admissible": False,
                "rank": None,
                "total_score": None,
                "constraint_violations": ["maximum_drawdown"],
                "tradeoffs": ["Higher expected return."],
            },
        ],
    }


def _narrative() -> dict[str, object]:
    return {
        "agent_name": "allocation_recommendation_agent",
        "question": "Który wariant najlepiej odpowiada założeniom?",
        "summary": "Minimum volatility zajmuje pierwsze miejsce.",
        "findings": ["Ranking pochodzi z danych walidacyjnych."],
        "risks": ["Wyniki historyczne nie gwarantują przyszłych rezultatów."],
        "recommendations": ["Wymagany jest przegląd człowieka."],
        "limitations": ["Wynik nie jest indywidualną poradą finansową ani gwarancją."],
        "evidence": [
            {
                "source_type": "decision_engine",
                "identifier": "current_decision_evaluation",
                "description": "Deterministic validation ranking",
            },
            {
                "source_type": "rag_document",
                "identifier": "portfolio-risk-reference",
                "description": "Local corpus; chunk=chunk-1; rank=1; score=0.8",
            },
        ],
    }


def _xai() -> dict[str, object]:
    return {
        "model_name": "random_forest",
        "explanation_split": "validation",
        "target_definition": "next_session_return",
        "global_observations": 400,
        "top_global_shap_features": [
            {"feature": "momentum_20", "mean_absolute_shap_value": 0.00018}
        ],
        "partial_dependence_features": ["momentum_20"],
        "accumulated_local_effect_features": ["momentum_20"],
        "scope": (
            "Forecast-model interpretation only; this evidence does not "
            "determine or modify the portfolio Decision Engine ranking."
        ),
    }


def _evaluation() -> dict[str, object]:
    return {
        "test_split_used": False,
        "chain_of_thought_stored": False,
        "deterministic_gate": {"passed": True},
        "llm_judge": {"status": "approved"},
        "llm_judge_approved_for_reporting": True,
    }


def _artifacts() -> tuple[ArtifactVersion, ...]:
    return (
        ArtifactVersion("decision_result.json", "1", "a" * 64),
        ArtifactVersion("xai_result.json", "1", "b" * 64),
        ArtifactVersion("narrative_evaluation.json", "1", "c" * 64),
    )


def _build(**overrides: object):
    arguments = {
        "decision": _decision(),
        "narrative": _narrative(),
        "xai": _xai(),
        "narrative_evaluation": _evaluation(),
        "source_artifacts": _artifacts(),
        "generated_at": datetime(2026, 8, 14, 9, 30, tzinfo=UTC),
        "presentation_timezone": "Europe/Warsaw",
    }
    arguments.update(overrides)
    return DecisionReportBuilder().build(**arguments)


def test_build_returns_complete_validation_only_report() -> None:
    report = _build()

    assert report.metadata.generated_at_utc == "2026-08-14T09:30:00+00:00"
    assert report.metadata.presentation_timezone == "Europe/Warsaw"
    assert report.metadata.evaluation_split == "validation"
    assert report.metadata.test_split_used is False
    assert report.recommended_scenario == "minimum_volatility"
    assert [item.name for item in report.scenarios] == [
        "minimum_volatility",
        "equal_weight",
        "growth",
    ]
    assert report.scenarios[0].weights == {"AGG": 0.75, "SPY": 0.25}
    assert report.evidence[1].source_type == "rag_document"
    assert report.governance.human_review_required is True
    assert report.governance.chain_of_thought_stored is False


def test_to_dict_is_json_serializable_shape() -> None:
    payload = _build().to_dict()

    assert payload["metadata"]["schema_version"] == 1
    assert payload["assumptions"]["priorities"]["return"] == 0.25
    assert payload["xai"]["top_global_shap_features"][0]["feature"] == "momentum_20"
    assert payload["governance"]["llm_used_for_ranking"] is False


def test_build_rejects_test_or_non_validation_evidence() -> None:
    decision = deepcopy(_decision())
    provenance = _as_dict(decision["provenance"])
    provenance["evaluation_split"] = "test"

    with pytest.raises(ValueError, match="only validation"):
        _build(decision=decision)

    evaluation = deepcopy(_evaluation())
    evaluation["test_split_used"] = True
    with pytest.raises(ValueError, match="test split closed"):
        _build(narrative_evaluation=evaluation)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("deterministic", False, "must be deterministic"),
        ("llm_used_for_ranking", True, "must not be used"),
        ("portfolio_weights_modified", True, "must not modify"),
    ],
)
def test_build_rejects_broken_decision_governance(
    field: str,
    value: bool,
    message: str,
) -> None:
    decision = deepcopy(_decision())
    decision[field] = value

    with pytest.raises(ValueError, match=message):
        _build(decision=decision)


def test_build_rejects_chain_of_thought_and_failed_gate() -> None:
    evaluation = deepcopy(_evaluation())
    evaluation["chain_of_thought_stored"] = True
    with pytest.raises(ValueError, match="chain-of-thought"):
        _build(narrative_evaluation=evaluation)

    evaluation = deepcopy(_evaluation())
    deterministic_gate = _as_dict(evaluation["deterministic_gate"])
    deterministic_gate["passed"] = False
    with pytest.raises(ValueError, match="gate must pass"):
        _build(narrative_evaluation=evaluation)


def test_build_requires_disclaimer_rag_citation_and_xai_boundary() -> None:
    narrative = deepcopy(_narrative())
    narrative["limitations"] = ["Brak zastrzeżenia."]
    with pytest.raises(ValueError, match="disclaimer"):
        _build(narrative=narrative)

    narrative = deepcopy(_narrative())
    evidence = _as_list(narrative["evidence"])
    narrative["evidence"] = evidence[:1]
    with pytest.raises(ValueError, match="RAG citation"):
        _build(narrative=narrative)

    xai = deepcopy(_xai())
    xai["scope"] = "Forecast explanation."
    with pytest.raises(ValueError, match="does not determine"):
        _build(xai=xai)


def test_build_rejects_invalid_ranking_and_modified_weights() -> None:
    decision = deepcopy(_decision())
    scenarios = _as_list(decision["scenarios"])
    first_scenario = _as_dict(scenarios[0])
    first_scenario["rank"] = 3
    with pytest.raises(ValueError, match="consecutive"):
        _build(decision=decision)

    decision = deepcopy(_decision())
    scenarios = _as_list(decision["scenarios"])
    first_scenario = _as_dict(scenarios[0])
    weights = _as_list(first_scenario["weights"])
    first_weight = _as_dict(weights[0])
    first_weight["weight"] = 0.6
    with pytest.raises(ValueError, match="sum to one"):
        _build(decision=decision)


def test_build_requires_aware_timestamp_and_valid_artifact_versions() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _build(generated_at=datetime(2026, 8, 14, 9, 30))

    with pytest.raises(ValueError, match="64 hexadecimal"):
        ArtifactVersion("decision.json", "1", "not-a-digest")

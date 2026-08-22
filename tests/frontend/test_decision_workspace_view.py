"""Tests for Human-in-the-loop Decision Workspace transformations."""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import pytest

from frontend.api_client import APIClient
from frontend.decision_workspace_view import (
    DECISION_EVALUATION_PATH,
    DECISION_NARRATIVE_PATH,
    DECISION_PROFILES_PATH,
    DECISION_REPORT_PATH,
    _build_change_explanations,
    _build_comparison_frame,
    _build_components_frame,
    _build_custom_evaluation_payload,
    _build_preset_evaluation_payload,
    _build_ranking_frame,
    _load_custom_recommendation,
    _load_decision_narrative,
    _load_decision_report,
    _load_profiles,
    _load_recommendation,
    _recommendation_changed,
    _validated_report_files,
)


class RecordingClient(APIClient):
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        super().__init__("http://test-api")
        self.responses = responses
        self.paths: list[str] = []
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def get(
        self,
        path: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        del timeout_seconds
        self.paths.append(path)
        return self.responses[path]

    def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        del timeout_seconds
        self.posts.append((path, payload))
        return self.responses[path]


def _result() -> dict[str, Any]:
    return {
        "recommended_scenario": "minimum_volatility",
        "scenarios": [
            {
                "name": "minimum_volatility",
                "admissible": True,
                "rank": 1,
                "total_score": 0.65,
                "normalized_components": {
                    "return": 0.0,
                    "volatility": 1.0,
                    "drawdown": 1.0,
                    "diversification": 0.0,
                },
                "weighted_contributions": {
                    "return": 0.0,
                    "volatility": 0.3,
                    "drawdown": 0.35,
                    "diversification": 0.0,
                },
                "constraint_violations": [],
            },
            {
                "name": "growth",
                "admissible": False,
                "rank": None,
                "total_score": None,
                "normalized_components": {},
                "weighted_contributions": {},
                "constraint_violations": [
                    "maximum_annualized_volatility",
                    "maximum_drawdown",
                ],
            },
        ],
    }


def test_load_profiles_uses_versioned_api() -> None:
    client = RecordingClient(
        {DECISION_PROFILES_PATH: {"profiles": [{"profile_type": "balanced"}]}}
    )

    profiles = _load_profiles(client)

    assert profiles == [{"profile_type": "balanced"}]
    assert client.paths == [DECISION_PROFILES_PATH]


def test_load_recommendation_encodes_liquidity_as_query() -> None:
    path = (
        "/api/v1/decision/recommendations/conservative?"
        "available_liquidity_fraction=0.1000"
    )
    client = RecordingClient({path: _result()})

    assert _load_recommendation(client, "conservative", 0.10) == _result()
    assert client.paths == [path]


def test_ranking_frame_explains_rejected_scenarios() -> None:
    frame = _build_ranking_frame(_result())

    assert frame.shape == (2, 5)
    assert frame.iloc[0]["Scenariusz"] == "Minimalna zmienność"
    assert frame.iloc[0]["Wynik"] == "0.650"
    assert frame.iloc[1]["Dopuszczalny"] == "Nie"
    assert "przekroczona maksymalna zmienność" in frame.iloc[1]["Naruszenia"]


def test_components_frame_contains_auditable_contributions() -> None:
    frame = _build_components_frame(_result())

    assert list(frame["Kryterium"]) == [
        "Zwrot",
        "Zmienność",
        "Obsunięcie",
        "Dywersyfikacja",
    ]
    assert frame["Wkład do wyniku"].sum() == pytest.approx(0.65)


def test_recommendation_change_requires_previous_different_result() -> None:
    current = _result()
    unchanged = _result()
    previous = {"recommended_scenario": "defensive"}

    assert _recommendation_changed(None, current) is False
    assert _recommendation_changed(unchanged, current) is False
    assert _recommendation_changed(previous, current) is True


def test_custom_evaluation_payload_matches_api_contract() -> None:
    payload = _build_custom_evaluation_payload(
        priorities={
            "return": 0.4,
            "volatility": 0.2,
            "drawdown": 0.2,
            "diversification": 0.2,
        },
        constraints={
            "maximum_annualized_volatility": 0.25,
            "maximum_drawdown": 0.4,
            "maximum_concentration_index": 0.5,
            "minimum_effective_assets": 2.0,
            "minimum_diversification_ratio": 1.0,
            "minimum_cash_reserve": 0.05,
        },
        liquidity_need="medium",
        investment_horizon_years=7,
        available_liquidity_fraction=0.10,
    )

    assert payload["profile"]["profile_type"] == "custom"
    assert payload["profile"]["priorities"]["return_priority"] == 0.4
    assert payload["profile"]["constraints"]["maximum_drawdown"] == 0.4
    assert payload["available_liquidity_fraction"] == 0.10


def test_load_custom_recommendation_uses_evaluation_endpoint() -> None:
    payload = {"profile": {"profile_type": "custom"}}
    client = RecordingClient({DECISION_EVALUATION_PATH: _result()})

    assert _load_custom_recommendation(client, payload) == _result()
    assert client.posts == [(DECISION_EVALUATION_PATH, payload)]


def test_preset_evaluation_payload_does_not_override_profile() -> None:
    payload = _build_preset_evaluation_payload("balanced", 0.10)

    assert payload == {
        "profile": {"profile_type": "balanced"},
        "available_liquidity_fraction": 0.10,
    }


def test_decision_narrative_sends_assumptions_instead_of_client_result() -> None:
    evaluation = _build_preset_evaluation_payload("balanced", 0.10)
    response = {"narrative": {"summary": "Wyjaśnienie."}}
    client = RecordingClient({DECISION_NARRATIVE_PATH: response})

    actual = _load_decision_narrative(
        client,
        question="Dlaczego ten wariant?",
        evaluation=evaluation,
        response_language="pl",
    )

    assert actual == response
    assert client.posts == [
        (
            DECISION_NARRATIVE_PATH,
            {
                "question": "Dlaczego ten wariant?",
                "evaluation": evaluation,
                "response_language": "pl",
            },
        )
    ]


def test_decision_report_sends_assumptions_and_timezone() -> None:
    evaluation = _build_preset_evaluation_payload("balanced", 0.10)
    response = {"report": {}, "manifest": {}, "files": []}
    client = RecordingClient({DECISION_REPORT_PATH: response})

    actual = _load_decision_report(
        client,
        question="Przygotuj raport.",
        evaluation=evaluation,
        presentation_timezone="Europe/Warsaw",
        response_language="pl",
    )

    assert actual == response
    assert client.posts == [
        (
            DECISION_REPORT_PATH,
            {
                "question": "Przygotuj raport.",
                "evaluation": evaluation,
                "presentation_timezone": "Europe/Warsaw",
                "response_language": "pl",
            },
        )
    ]


def test_report_files_are_decoded_only_after_integrity_check() -> None:
    files = []
    for format_name, content in (
        ("json", b"{}\n"),
        ("markdown", b"# Report\n"),
        ("pdf", b"%PDF-test"),
    ):
        files.append(
            {
                "format": format_name,
                "file_name": f"decision_report.{format_name}",
                "media_type": "application/octet-stream",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "content_base64": base64.b64encode(content).decode("ascii"),
            }
        )

    decoded = _validated_report_files({"files": files})

    assert decoded["json"][1] == b"{}\n"
    assert decoded["markdown"][1] == b"# Report\n"
    assert decoded["pdf"][1] == b"%PDF-test"

    files[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="integralności"):
        _validated_report_files({"files": files})


def test_comparison_frame_exposes_before_and_after_ranking() -> None:
    baseline = _result()
    current = _result()
    current["recommended_scenario"] = "growth"
    current["scenarios"][0]["rank"] = 2
    current["scenarios"][1]["admissible"] = True
    current["scenarios"][1]["rank"] = 1
    current["scenarios"][1]["total_score"] = 0.8

    frame = _build_comparison_frame(baseline, current)

    assert frame.shape == (2, 7)
    growth = frame.loc[frame["Scenariusz"] == "Wzrostowy"].iloc[0]
    assert growth["Dopuszczalny — przed"] == "Nie"
    assert growth["Dopuszczalny — po"] == "Tak"
    assert growth["Pozycja — po"] == 1


def test_change_explanation_reports_observable_differences() -> None:
    baseline = _result()
    baseline["profile"] = {
        "priorities": {
            "return": 0.3,
            "volatility": 0.25,
            "drawdown": 0.25,
            "diversification": 0.2,
        },
        "constraints": {"maximum_drawdown": 0.32},
    }
    current = _result()
    current["recommended_scenario"] = "growth"
    current["profile"] = {
        "priorities": {
            "return": 0.5,
            "volatility": 0.15,
            "drawdown": 0.15,
            "diversification": 0.2,
        },
        "constraints": {"maximum_drawdown": 0.40},
    }
    current["scenarios"][1]["admissible"] = True

    explanations = _build_change_explanations(baseline, current)

    assert explanations[0] == (
        "Rekomendacja zmieniła się: Minimalna zmienność → Wzrostowy."
    )
    assert any(
        "Priorytet „Zwrot” zmieniono z 30% na 50%." == item for item in explanations
    )
    assert any(
        "Parametr „Maksymalne obsunięcie” zmieniono z 32% na 40%." == item
        for item in explanations
    )
    assert any("stał się dopuszczalny" in item for item in explanations)


def test_decision_frames_are_localized_without_changing_values() -> None:
    polish = _build_ranking_frame(_result(), "pl")
    english = _build_ranking_frame(_result(), "en")

    assert polish.iloc[0]["Scenariusz"] == "Minimalna zmienność"
    assert english.iloc[0]["Scenario"] == "Minimum volatility"
    assert polish.iloc[0]["Wynik"] == english.iloc[0]["Score"] == "0.650"
    assert english.iloc[1]["Admissible"] == "No"
    assert "maximum volatility exceeded" in english.iloc[1]["Violations"]


def test_decision_change_explanation_supports_english() -> None:
    baseline = _result()
    current = _result()
    current["recommended_scenario"] = "growth"

    explanations = _build_change_explanations(baseline, current, "en")

    assert explanations[0] == (
        "The recommendation changed: Minimum volatility → Growth."
    )

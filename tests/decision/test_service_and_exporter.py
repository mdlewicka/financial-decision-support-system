"""Tests for real-artifact decision orchestration and exports."""

import csv
import json
from pathlib import Path

import pytest

from backend.src.api.risk_schemas import RiskArtifactPayload
from backend.src.decision.exporter import export_portfolio_decisions
from backend.src.decision.models import DecisionProfileType
from backend.src.decision.profiles import create_decision_profile
from backend.src.decision.service import PortfolioDecisionService


def _evaluations(artifact: RiskArtifactPayload):
    service = PortfolioDecisionService()
    return tuple(
        service.evaluate(
            artifact,
            create_decision_profile(profile_type),
            available_liquidity_fraction=0.10,
        )
        for profile_type in (
            DecisionProfileType.CONSERVATIVE,
            DecisionProfileType.BALANCED,
            DecisionProfileType.GROWTH,
        )
    )


def test_service_ranks_frozen_validation_scenarios_deterministically(
    portfolio_artifact: RiskArtifactPayload,
) -> None:
    first = _evaluations(portfolio_artifact)
    second = _evaluations(portfolio_artifact)

    assert first == second
    assert first[0].decision.recommended_scenario == "defensive"
    assert first[0].source.provenance.evaluation_split == "validation"
    assert all(
        dict(context.weights)
        == {
            "AGG": 0.25,
            "GLD": 0.25,
            "QQQ": 0.25,
            "SPY": 0.25,
        }
        for context in first[0].source.contexts
    )


def test_exporter_records_governance_provenance_and_components(
    portfolio_artifact: RiskArtifactPayload,
    tmp_path: Path,
) -> None:
    paths = export_portfolio_decisions(_evaluations(portfolio_artifact), tmp_path)
    payload = json.loads(paths.result_json.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["engine"] == {
        "deterministic": True,
        "llm_used_for_ranking": False,
        "portfolio_weights_modified": False,
    }
    assert payload["provenance"]["evaluation_split"] == "validation"
    assert payload["provenance"]["date_min"] == "2020-02-01"
    assert len(payload["evaluations"]) == 3

    with paths.rankings_csv.open(encoding="utf-8", newline="") as file:
        rankings = list(csv.DictReader(file))
    with paths.components_csv.open(encoding="utf-8", newline="") as file:
        components = list(csv.DictReader(file))

    assert len(rankings) == 9
    assert components
    assert {row["component"] for row in components} == {
        "return",
        "volatility",
        "drawdown",
        "diversification",
    }


def test_exporter_rejects_duplicate_profile_evaluations(
    portfolio_artifact: RiskArtifactPayload,
    tmp_path: Path,
) -> None:
    evaluation = _evaluations(portfolio_artifact)[0]
    with pytest.raises(ValueError, match="unique profile"):
        export_portfolio_decisions((evaluation, evaluation), tmp_path)

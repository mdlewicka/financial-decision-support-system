"""HTTP contract tests for the deterministic Decision Engine."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.src.api.portfolio_repository import PortfolioArtifactRepository
from backend.src.api.routes.decision import get_decision_portfolio_repository
from backend.src.main import create_app


@pytest.fixture
def decision_client(tmp_path: Path) -> Iterator[TestClient]:
    source = Path("benchmark_results/risk/portfolio/portfolio_result.json")
    shutil.copyfile(source, tmp_path / "portfolio_result.json")
    application = create_app()
    repository = PortfolioArtifactRepository(artifact_directory=tmp_path)
    application.dependency_overrides[get_decision_portfolio_repository] = lambda: (
        repository
    )
    with TestClient(application) as client:
        yield client


def test_profiles_expose_three_complete_presets(
    decision_client: TestClient,
) -> None:
    response = decision_client.get("/api/v1/decision/profiles")

    assert response.status_code == 200
    profiles = response.json()["profiles"]
    assert [profile["profile_type"] for profile in profiles] == [
        "conservative",
        "balanced",
        "growth",
    ]
    assert all(
        abs(sum(profile["priorities"].values()) - 1.0) < 1e-9 for profile in profiles
    )


def test_preset_recommendation_exposes_complete_audit_trail(
    decision_client: TestClient,
) -> None:
    response = decision_client.get(
        "/api/v1/decision/recommendations/conservative",
        params={"available_liquidity_fraction": 0.10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["deterministic"] is True
    assert payload["llm_used_for_ranking"] is False
    assert payload["portfolio_weights_modified"] is False
    assert payload["provenance"]["evaluation_split"] == "validation"
    assert payload["recommended_scenario"] == "minimum_volatility"
    assert len(payload["scenarios"]) == 5
    assert any(not scenario["admissible"] for scenario in payload["scenarios"])
    assert all(
        abs(sum(weight["weight"] for weight in scenario["weights"]) - 1.0) < 1e-9
        for scenario in payload["scenarios"]
    )


def test_insufficient_liquidity_rejects_every_conservative_scenario(
    decision_client: TestClient,
) -> None:
    response = decision_client.get(
        "/api/v1/decision/recommendations/conservative",
        params={"available_liquidity_fraction": 0.05},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["recommended_scenario"] is None
    assert all(not scenario["admissible"] for scenario in payload["scenarios"])
    assert all(
        "minimum_cash_reserve" in scenario["constraint_violations"]
        for scenario in payload["scenarios"]
    )


def test_custom_profile_is_evaluated_without_llm(
    decision_client: TestClient,
) -> None:
    response = decision_client.post(
        "/api/v1/decision/evaluate",
        json={
            "profile": {
                "profile_type": "custom",
                "priorities": {
                    "return_priority": 0.4,
                    "volatility_priority": 0.2,
                    "drawdown_priority": 0.2,
                    "diversification_priority": 0.2,
                },
                "constraints": {
                    "maximum_annualized_volatility": 0.25,
                    "maximum_drawdown": 0.4,
                    "maximum_concentration_index": 0.5,
                    "minimum_effective_assets": 2.0,
                    "minimum_diversification_ratio": 1.0,
                    "minimum_cash_reserve": 0.05,
                },
                "liquidity_need": "medium",
                "investment_horizon_years": 7,
            },
            "available_liquidity_fraction": 0.10,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["profile_type"] == "custom"
    assert payload["llm_used_for_ranking"] is False
    assert payload["recommended_scenario"] is not None


@pytest.mark.parametrize(
    "path",
    (
        "/api/v1/decision/recommendations/custom",
        "/api/v1/decision/recommendations/unknown",
    ),
)
def test_invalid_preset_paths_are_rejected(
    decision_client: TestClient,
    path: str,
) -> None:
    response = decision_client.get(path)
    assert response.status_code == 422


def test_invalid_custom_priority_sum_is_rejected(
    decision_client: TestClient,
) -> None:
    response = decision_client.post(
        "/api/v1/decision/evaluate",
        json={
            "profile": {
                "profile_type": "custom",
                "priorities": {
                    "return_priority": 0.5,
                    "volatility_priority": 0.5,
                    "drawdown_priority": 0.5,
                    "diversification_priority": 0.5,
                },
                "constraints": {},
                "liquidity_need": "low",
                "investment_horizon_years": 5,
            },
            "available_liquidity_fraction": 0.10,
        },
    )
    assert response.status_code == 422


def test_missing_portfolio_artifact_returns_service_unavailable(
    tmp_path: Path,
) -> None:
    application = create_app()
    repository = PortfolioArtifactRepository(artifact_directory=tmp_path)
    application.dependency_overrides[get_decision_portfolio_repository] = lambda: (
        repository
    )
    with TestClient(application) as client:
        response = client.get("/api/v1/decision/recommendations/balanced")
    assert response.status_code == 503

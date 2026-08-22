from pathlib import Path

import pytest

from backend.src.agents.artifact_repository import (
    AgentArtifactPaths,
    DecisionSupportArtifactRepository,
)


def test_paths_follow_reproducible_project_layout() -> None:
    paths = AgentArtifactPaths.from_project_root(Path("project"))

    assert paths.forecast == Path(
        "project/benchmark_results/forecast/random_forest/forecast_result.json"
    )
    assert paths.xai == Path(
        "project/benchmark_results/xai/random_forest/xai_result.json"
    )
    assert paths.portfolio == Path(
        "project/benchmark_results/risk/portfolio/portfolio_result.json"
    )
    assert paths.reinforcement_learning == Path(
        "project/benchmark_results/rl/ppo/rl_result.json"
    )


def test_forecast_context_is_compact_and_ranked(
    artifact_paths: AgentArtifactPaths,
) -> None:
    repository = DecisionSupportArtifactRepository(artifact_paths)

    context = repository.forecast_context(
        ticker="spy",
        top_features=1,
    )

    assert context["evaluation_split"] == "validation"
    assert context["selected_ticker"] == "SPY"
    assert context["ticker_metrics"]["mae"] == pytest.approx(0.008)
    assert context["top_global_shap_features"] == [
        {
            "feature": "momentum_20",
            "mean_absolute_shap_value": 0.2,
        }
    ]


def test_forecast_context_rejects_unknown_ticker(
    artifact_paths: AgentArtifactPaths,
) -> None:
    repository = DecisionSupportArtifactRepository(artifact_paths)

    with pytest.raises(ValueError, match="Unknown forecast ticker"):
        repository.forecast_context(ticker="QQQ")


def test_risk_context_contains_validation_not_training(
    artifact_paths: AgentArtifactPaths,
) -> None:
    repository = DecisionSupportArtifactRepository(artifact_paths)

    context = repository.risk_context()

    assert context["validation_market"]["observations"] == 50
    assert len(context["validation_scenarios"]) == 2
    assert "weights" not in context["validation_scenarios"][0]


def test_xai_context_exposes_methods_with_portfolio_scope_boundary(
    artifact_paths: AgentArtifactPaths,
) -> None:
    repository = DecisionSupportArtifactRepository(artifact_paths)

    context = repository.xai_context(top_features=1)

    assert context["explanation_split"] == "validation"
    assert context["top_global_shap_features"] == [
        {
            "feature": "momentum_20",
            "mean_absolute_shap_value": 0.2,
        }
    ]
    assert context["partial_dependence_features"] == ["momentum_20"]
    assert context["accumulated_local_effect_features"] == ["momentum_20"]
    assert "does not determine" in context["scope"]


def test_allocation_context_contains_weights_and_rl_baselines(
    artifact_paths: AgentArtifactPaths,
) -> None:
    repository = DecisionSupportArtifactRepository(artifact_paths)

    context = repository.allocation_context("balanced")

    assert context["rl_test_split_used"] is False
    assert context["portfolio_scenarios"][0]["weights"] == {
        "AGG": 0.5,
        "SPY": 0.5,
    }
    assert context["strategy_comparison"][0]["strategy"] == "ppo"


@pytest.mark.parametrize("preference", ["", "unknown", "aggressive"])
def test_allocation_context_rejects_unknown_preference(
    artifact_paths: AgentArtifactPaths,
    preference: str,
) -> None:
    repository = DecisionSupportArtifactRepository(artifact_paths)

    with pytest.raises(ValueError, match="risk_preference"):
        repository.allocation_context(preference)


def test_repository_rejects_missing_artifact(
    artifact_paths: AgentArtifactPaths,
) -> None:
    artifact_paths.xai.unlink()

    with pytest.raises(FileNotFoundError, match="Artifact does not exist"):
        DecisionSupportArtifactRepository(artifact_paths)


def test_repository_rejects_corrupted_artifact(
    artifact_paths: AgentArtifactPaths,
) -> None:
    artifact_paths.forecast.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Artifact is not valid JSON"):
        DecisionSupportArtifactRepository(artifact_paths)

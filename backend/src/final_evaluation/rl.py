from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.src.risk.risk_analyzer import (
    RiskAnalysisConfig,
)
from backend.src.rl.environment_factory import (
    PortfolioAllocationEnvironmentFactory,
)
from backend.src.rl.policy_evaluator import (
    RLPolicyEvaluationResult,
    RLPolicyEvaluator,
)
from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnvConfig,
)
from backend.src.rl.ppo_allocation_agent import (
    PPOAllocationAgent,
    PPOAllocationAgentConfig,
)


def _normalize_agent_config(
    payload: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(payload)

    hidden_layer_sizes = normalized.get("hidden_layer_sizes")

    if isinstance(
        hidden_layer_sizes,
        list,
    ):
        normalized["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

    return normalized


def load_frozen_rl_config(
    result_path: Path,
) -> tuple[
    PPOAllocationAgentConfig,
    PortfolioAllocationEnvConfig,
]:
    payload = json.loads(
        Path(result_path).read_text(
            encoding="utf-8",
        )
    )

    if payload.get("model_name") != "ppo":
        raise ValueError("Frozen RL artifact must describe the PPO model.")

    if payload.get("training_split") != "training":
        raise ValueError("Frozen PPO artifact must use the training split.")

    if payload.get("evaluation_split") != "validation":
        raise ValueError(
            "Frozen PPO artifact must originate from validation evaluation."
        )

    if payload.get("test_split_used") is not False:
        raise ValueError(
            "Frozen PPO artifact must confirm that the final test split was not used."
        )

    agent_payload = payload.get("agent_config")

    environment_payload = payload.get("environment_config")

    if not isinstance(
        agent_payload,
        dict,
    ):
        raise ValueError("Frozen PPO artifact is missing agent_config.")

    if not isinstance(
        environment_payload,
        dict,
    ):
        raise ValueError("Frozen PPO artifact is missing environment_config.")

    normalized_agent_payload = _normalize_agent_config(agent_payload)

    agent_config = PPOAllocationAgentConfig(**normalized_agent_payload)

    environment_config = PortfolioAllocationEnvConfig(**environment_payload)

    return (
        agent_config,
        environment_config,
    )


def evaluate_frozen_ppo(
    training_dataset: pd.DataFrame,
    evaluation_dataset: pd.DataFrame,
    result_path: Path,
    model_path: Path,
    confidence_level: float = 0.95,
    annual_risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    minimum_common_observations: int = 100,
) -> RLPolicyEvaluationResult:
    (
        agent_config,
        environment_config,
    ) = load_frozen_rl_config(
        result_path=result_path,
    )

    bundle = PortfolioAllocationEnvironmentFactory(
        environment_config=(environment_config),
        minimum_common_observations=(minimum_common_observations),
        periods_per_year=(periods_per_year),
    ).create(
        training_dataset=(training_dataset),
        validation_dataset=(evaluation_dataset),
    )

    agent = PPOAllocationAgent.load(
        path=model_path,
        environment=(bundle.validation_environment),
        config=agent_config,
    )

    risk_config = RiskAnalysisConfig(
        confidence_level=(confidence_level),
        annual_risk_free_rate=(annual_risk_free_rate),
        periods_per_year=(periods_per_year),
        minimum_observations=(minimum_common_observations),
    )

    return RLPolicyEvaluator(
        risk_config=risk_config,
    ).evaluate(
        policy=agent,
        environment=(bundle.validation_environment),
        seed=agent_config.seed,
    )


def build_rl_summary_frame(
    result: RLPolicyEvaluationResult,
    evaluation_split: str,
) -> pd.DataFrame:
    metrics = result.risk_metrics

    return pd.DataFrame(
        [
            {
                "strategy": "ppo",
                "strategy_type": ("reinforcement_learning"),
                "construction": ("dynamic_ppo_allocation"),
                "evaluation_split": (evaluation_split),
                "observations": (result.observations),
                "annualized_mean_return": (metrics.annualized_mean_return),
                "annualized_volatility": (metrics.annualized_volatility),
                "sharpe_ratio": (metrics.sharpe_ratio),
                "sortino_ratio": (metrics.sortino_ratio),
                "value_at_risk": (metrics.value_at_risk),
                "conditional_value_at_risk": (metrics.conditional_value_at_risk),
                "maximum_drawdown": (metrics.maximum_drawdown),
                "cumulative_return": (result.cumulative_return),
                "final_portfolio_value": (result.final_portfolio_value),
                "total_reward": (result.total_reward),
                "total_transaction_cost": (result.total_transaction_cost),
                "mean_turnover": (result.mean_turnover),
            }
        ]
    )


def build_rl_trajectory_frame(
    result: RLPolicyEvaluationResult,
) -> pd.DataFrame:
    return result.trajectory.copy()


def build_rl_weights_frame(
    result: RLPolicyEvaluationResult,
) -> pd.DataFrame:
    return result.weights.copy()

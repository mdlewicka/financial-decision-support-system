import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.src.portfolio.portfolio_metrics import (
    PortfolioScenarioEvaluation,
)
from backend.src.risk.risk_metrics import (
    RiskMetrics,
)
from backend.src.rl.policy_evaluator import (
    RLPolicyEvaluationResult,
)
from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnvConfig,
)
from backend.src.rl.ppo_allocation_agent import (
    PPOAllocationAgentConfig,
)

SUMMARY_COLUMNS = (
    "strategy",
    "strategy_type",
    "construction",
    "evaluation_split",
    "observations",
    "annualized_mean_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "value_at_risk",
    "conditional_value_at_risk",
    "maximum_drawdown",
    "cumulative_return",
    "final_portfolio_value",
    "total_reward",
    "total_transaction_cost",
    "mean_turnover",
)


@dataclass(frozen=True, slots=True)
class RLBenchmarkExportPaths:
    """Paths created by one reinforcement-learning benchmark export."""

    summary_path: Path
    trajectory_path: Path
    weights_path: Path
    json_path: Path
    model_path: Path


def export_rl_benchmark(
    evaluation: RLPolicyEvaluationResult,
    baselines: tuple[
        PortfolioScenarioEvaluation,
        ...,
    ],
    agent_config: PPOAllocationAgentConfig,
    environment_config: PortfolioAllocationEnvConfig,
    training_observations: int,
    model_path: Path,
    output_directory: Path,
) -> RLBenchmarkExportPaths:
    """Export the RL evaluation and fixed-portfolio benchmark comparison."""
    validated_baselines = _validate_export_inputs(
        evaluation=evaluation,
        baselines=baselines,
        agent_config=agent_config,
        environment_config=environment_config,
        training_observations=training_observations,
        model_path=model_path,
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = RLBenchmarkExportPaths(
        summary_path=(output_directory / "rl_summary.csv"),
        trajectory_path=(output_directory / "rl_trajectory.csv"),
        weights_path=(output_directory / "rl_weights.csv"),
        json_path=(output_directory / "rl_result.json"),
        model_path=Path(model_path),
    )

    comparison_rows = _build_comparison_rows(
        evaluation=evaluation,
        baselines=validated_baselines,
    )

    summary = pd.DataFrame(
        comparison_rows,
        columns=SUMMARY_COLUMNS,
    )

    trajectory = _build_trajectory_frame(evaluation)
    weights = _build_weights_frame(evaluation)

    summary.to_csv(
        paths.summary_path,
        index=False,
    )
    trajectory.to_csv(
        paths.trajectory_path,
        index=False,
    )
    weights.to_csv(
        paths.weights_path,
        index=False,
    )

    payload = _build_json_payload(
        evaluation=evaluation,
        comparison_rows=comparison_rows,
        agent_config=agent_config,
        environment_config=environment_config,
        training_observations=(training_observations),
        model_path=Path(model_path),
    )

    paths.json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return paths


def _validate_export_inputs(
    evaluation: RLPolicyEvaluationResult,
    baselines: tuple[
        PortfolioScenarioEvaluation,
        ...,
    ],
    agent_config: PPOAllocationAgentConfig,
    environment_config: PortfolioAllocationEnvConfig,
    training_observations: int,
    model_path: Path,
) -> tuple[
    PortfolioScenarioEvaluation,
    ...,
]:
    if not isinstance(
        evaluation,
        RLPolicyEvaluationResult,
    ):
        raise TypeError("evaluation must be an RLPolicyEvaluationResult instance.")

    if not isinstance(
        agent_config,
        PPOAllocationAgentConfig,
    ):
        raise TypeError("agent_config must be a PPOAllocationAgentConfig instance.")

    if not isinstance(
        environment_config,
        PortfolioAllocationEnvConfig,
    ):
        raise TypeError(
            "environment_config must be a PortfolioAllocationEnvConfig instance."
        )

    if (
        isinstance(
            training_observations,
            bool,
        )
        or not isinstance(
            training_observations,
            int,
        )
        or training_observations <= 0
    ):
        raise ValueError("training_observations must be a positive integer.")

    archive_path = Path(model_path)

    if not archive_path.exists() or not archive_path.is_file():
        raise FileNotFoundError(f"RL model archive does not exist: {archive_path}")

    if not isinstance(
        baselines,
        tuple,
    ):
        raise TypeError("baselines must be a tuple.")

    if not baselines:
        raise ValueError("baselines must not be empty.")

    names = {
        "ppo",
    }
    validated_baselines: list[PortfolioScenarioEvaluation] = []

    for baseline in baselines:
        if not isinstance(
            baseline,
            PortfolioScenarioEvaluation,
        ):
            raise TypeError(
                "baselines must contain PortfolioScenarioEvaluation instances."
            )

        if (
            not isinstance(
                baseline.name,
                str,
            )
            or not baseline.name.strip()
        ):
            raise ValueError("Baseline names must be non-empty.")

        if baseline.name in names:
            raise ValueError("Strategy names must be unique.")

        names.add(baseline.name)
        validated_baselines.append(baseline)

    _validate_evaluation_result(evaluation)

    return tuple(validated_baselines)


def _validate_evaluation_result(
    evaluation: RLPolicyEvaluationResult,
) -> None:
    if evaluation.observations <= 0:
        raise ValueError("evaluation must contain observations.")

    if (
        len(evaluation.trajectory) != evaluation.observations
        or len(evaluation.weights) != evaluation.observations
        or len(evaluation.net_returns) != evaluation.observations
    ):
        raise ValueError(
            "Evaluation artifact lengths must match the observation count."
        )

    weight_columns = tuple(str(column) for column in evaluation.weights.columns)

    if weight_columns != evaluation.tickers:
        raise ValueError("Evaluation weight columns must match tickers.")

    if not evaluation.trajectory.index.equals(
        evaluation.weights.index
    ) or not evaluation.trajectory.index.equals(evaluation.net_returns.index):
        raise ValueError("Evaluation artifacts must use matching dates.")

    trajectory_values = evaluation.trajectory.to_numpy(dtype=float)
    weight_values = evaluation.weights.to_numpy(dtype=float)
    return_values = evaluation.net_returns.to_numpy(dtype=float)

    if (
        not np.isfinite(trajectory_values).all()
        or not np.isfinite(weight_values).all()
        or not np.isfinite(return_values).all()
    ):
        raise ValueError("Evaluation artifacts must contain only finite values.")

    if not np.allclose(
        weight_values.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("Every exported portfolio allocation must sum to one.")


def _build_comparison_rows(
    evaluation: RLPolicyEvaluationResult,
    baselines: tuple[
        PortfolioScenarioEvaluation,
        ...,
    ],
) -> list[
    dict[
        str,
        object,
    ]
]:
    rows = [_build_rl_summary_row(evaluation)]

    rows.extend(_build_baseline_summary_row(baseline) for baseline in baselines)

    return rows


def _build_rl_summary_row(
    evaluation: RLPolicyEvaluationResult,
) -> dict[
    str,
    object,
]:
    metrics = evaluation.risk_metrics

    return {
        "strategy": "ppo",
        "strategy_type": ("reinforcement_learning"),
        "construction": ("dynamic_ppo_allocation"),
        "evaluation_split": "validation",
        "observations": (evaluation.observations),
        **_risk_metric_values(metrics),
        "cumulative_return": _finite_float(
            evaluation.cumulative_return,
            "cumulative_return",
        ),
        "final_portfolio_value": _finite_float(
            evaluation.final_portfolio_value,
            "final_portfolio_value",
        ),
        "total_reward": _finite_float(
            evaluation.total_reward,
            "total_reward",
        ),
        "total_transaction_cost": (
            _finite_float(
                evaluation.total_transaction_cost,
                "total_transaction_cost",
            )
        ),
        "mean_turnover": _finite_float(
            evaluation.mean_turnover,
            "mean_turnover",
        ),
    }


def _build_baseline_summary_row(
    baseline: PortfolioScenarioEvaluation,
) -> dict[
    str,
    object,
]:
    validation = baseline.validation

    return_values = validation.portfolio_returns.to_numpy(dtype=float)

    cumulative_return = float(np.prod(1.0 + return_values) - 1.0)

    return {
        "strategy": baseline.name,
        "strategy_type": ("fixed_portfolio"),
        "construction": (baseline.construction),
        "evaluation_split": ("validation"),
        "observations": (validation.risk_metrics.observations),
        **_risk_metric_values(validation.risk_metrics),
        "cumulative_return": _finite_float(
            cumulative_return,
            "baseline cumulative_return",
        ),
        "final_portfolio_value": _finite_float(
            1.0 + cumulative_return,
            "baseline final_portfolio_value",
        ),
        "total_reward": None,
        "total_transaction_cost": None,
        "mean_turnover": None,
    }


def _risk_metric_values(
    metrics: RiskMetrics,
) -> dict[
    str,
    float | None,
]:
    return {
        "annualized_mean_return": (
            _finite_float(
                metrics.annualized_mean_return,
                "annualized_mean_return",
            )
        ),
        "annualized_volatility": (
            _finite_float(
                metrics.annualized_volatility,
                "annualized_volatility",
            )
        ),
        "sharpe_ratio": (
            _optional_finite_float(
                metrics.sharpe_ratio,
                "sharpe_ratio",
            )
        ),
        "sortino_ratio": (
            _optional_finite_float(
                metrics.sortino_ratio,
                "sortino_ratio",
            )
        ),
        "value_at_risk": _finite_float(
            metrics.value_at_risk,
            "value_at_risk",
        ),
        "conditional_value_at_risk": (
            _finite_float(
                metrics.conditional_value_at_risk,
                "conditional_value_at_risk",
            )
        ),
        "maximum_drawdown": (
            _finite_float(
                metrics.maximum_drawdown,
                "maximum_drawdown",
            )
        ),
    }


def _build_trajectory_frame(
    evaluation: RLPolicyEvaluationResult,
) -> pd.DataFrame:
    """Build the exported RL trajectory table."""
    trajectory = evaluation.trajectory.copy(deep=True)

    dates = _format_date_index(evaluation.trajectory.index)

    trajectory.insert(
        0,
        "date",
        dates,
    )

    return trajectory.reset_index(drop=True)


def _build_weights_frame(
    evaluation: RLPolicyEvaluationResult,
) -> pd.DataFrame:
    """Build the exported RL portfolio-weight table."""
    weights = evaluation.weights.copy(deep=True)

    dates = _format_date_index(evaluation.weights.index)

    weights.insert(
        0,
        "date",
        dates,
    )

    return weights.reset_index(drop=True)


def _format_date_index(
    index: pd.Index,
) -> list[str]:
    """Format arbitrary pandas date-like index values as ISO dates."""
    return [pd.Timestamp(str(value)).strftime("%Y-%m-%d") for value in index]


def _build_json_payload(
    evaluation: RLPolicyEvaluationResult,
    comparison_rows: list[
        dict[
            str,
            object,
        ]
    ],
    agent_config: PPOAllocationAgentConfig,
    environment_config: PortfolioAllocationEnvConfig,
    training_observations: int,
    model_path: Path,
) -> dict[
    str,
    Any,
]:
    return {
        "model_name": "ppo",
        "training_split": "training",
        "evaluation_split": ("validation"),
        "test_split_used": False,
        "training_observations": (training_observations),
        "model_file": (model_path.name),
        "agent_config": {
            "total_timesteps": (agent_config.total_timesteps),
            "learning_rate": (agent_config.learning_rate),
            "n_steps": (agent_config.n_steps),
            "batch_size": (agent_config.batch_size),
            "n_epochs": (agent_config.n_epochs),
            "gamma": (agent_config.gamma),
            "gae_lambda": (agent_config.gae_lambda),
            "clip_range": (agent_config.clip_range),
            "entropy_coefficient": (agent_config.entropy_coefficient),
            "hidden_layer_sizes": list(agent_config.hidden_layer_sizes),
            "seed": (agent_config.seed),
            "device": (agent_config.device),
        },
        "environment_config": {
            "lookback_window": (environment_config.lookback_window),
            "transaction_cost_bps": (environment_config.transaction_cost_bps),
            "drawdown_penalty": (environment_config.drawdown_penalty),
            "max_weight": (environment_config.max_weight),
            "initial_portfolio_value": (environment_config.initial_portfolio_value),
        },
        "evaluation": {
            "observations": (evaluation.observations),
            "date_min": (evaluation.date_min.date().isoformat()),
            "date_max": (evaluation.date_max.date().isoformat()),
            "tickers": list(evaluation.tickers),
            "total_reward": _finite_float(
                evaluation.total_reward,
                "total_reward",
            ),
            "cumulative_return": _finite_float(
                evaluation.cumulative_return,
                "cumulative_return",
            ),
            "final_portfolio_value": (
                _finite_float(
                    evaluation.final_portfolio_value,
                    "final_portfolio_value",
                )
            ),
            "total_transaction_cost": (
                _finite_float(
                    evaluation.total_transaction_cost,
                    "total_transaction_cost",
                )
            ),
            "mean_turnover": _finite_float(
                evaluation.mean_turnover,
                "mean_turnover",
            ),
            "risk_metrics": (_risk_metric_values(evaluation.risk_metrics)),
        },
        "comparison": (comparison_rows),
        "observations": (_build_json_observations(evaluation)),
    }


def _build_json_observations(
    evaluation: RLPolicyEvaluationResult,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    observations: list[
        dict[
            str,
            Any,
        ]
    ] = []

    trajectory_columns = tuple(
        str(column) for column in (evaluation.trajectory.columns)
    )

    trajectory_values = evaluation.trajectory.to_numpy(dtype=float)

    weight_values = evaluation.weights.to_numpy(dtype=float)

    dates = _format_date_index(evaluation.trajectory.index)

    for (
        position,
        date_text,
    ) in enumerate(dates):
        trajectory_payload = {
            column: _finite_float(
                trajectory_values[
                    position,
                    column_index,
                ],
                column,
            )
            for (
                column_index,
                column,
            ) in enumerate(trajectory_columns)
        }

        weight_payload = {
            ticker: _finite_float(
                weight_values[
                    position,
                    ticker_index,
                ],
                ticker,
            )
            for (
                ticker_index,
                ticker,
            ) in enumerate(evaluation.tickers)
        }

        observations.append(
            {
                "date": (date_text),
                **trajectory_payload,
                "weights": (weight_payload),
            }
        )

    return observations


def _optional_finite_float(
    value: float | None,
    field_name: str,
) -> float | None:
    if value is None:
        return None

    return _finite_float(
        value,
        field_name,
    )


def _finite_float(
    value: float | np.floating,
    field_name: str,
) -> float:
    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite.")

    return numeric_value

import json
from pathlib import Path

import pytest

from backend.src.agents.artifact_repository import AgentArtifactPaths


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def artifact_paths(tmp_path: Path) -> AgentArtifactPaths:
    paths = AgentArtifactPaths(
        forecast=tmp_path / "forecast.json",
        xai=tmp_path / "xai.json",
        portfolio=tmp_path / "portfolio.json",
        reinforcement_learning=tmp_path / "rl.json",
    )
    _write_json(
        paths.forecast,
        {
            "model_name": "random_forest",
            "evaluation_split": "validation",
            "target_definition": "next_session_simple_return",
            "metrics": {
                "overall": {
                    "observations": 100,
                    "mae": 0.01,
                    "rmse": 0.02,
                    "r2": 0.005,
                    "directional_accuracy": 0.505,
                },
                "by_ticker": {
                    "SPY": {
                        "observations": 50,
                        "mae": 0.008,
                        "rmse": 0.012,
                        "r2": -0.001,
                        "directional_accuracy": 0.51,
                    }
                },
            },
        },
    )
    _write_json(
        paths.xai,
        {
            "metadata": {
                "model_name": "random_forest",
                "explanation_split": "validation",
            },
            "global_shap": {
                "observations": 100,
                "mean_absolute_contributions": {
                    "momentum_20": 0.2,
                    "daily_return": 0.1,
                },
            },
            "local_shap": {
                "prediction": 0.001,
                "features": [],
            },
            "partial_dependence": {
                "momentum_20": {
                    "feature_values": [-0.1, 0.1],
                    "average_predictions": [0.0, 0.001],
                }
            },
            "accumulated_local_effects": {
                "momentum_20": {
                    "bin_centers": [-0.05, 0.05],
                    "accumulated_effects": [-0.001, 0.001],
                }
            },
        },
    )
    scenarios = []
    for name, volatility, drawdown, sharpe in (
        ("defensive", 0.1, 0.15, 0.35),
        ("maximum_sharpe", 0.16, 0.28, 0.58),
    ):
        scenarios.append(
            {
                "name": name,
                "construction": f"construction_{name}",
                "weights": {"AGG": 0.5, "SPY": 0.5},
                "validation": {
                    "metrics": {
                        "annualized_mean_return": 0.08,
                        "annualized_volatility": volatility,
                        "sharpe_ratio": sharpe,
                        "sortino_ratio": sharpe + 0.1,
                        "value_at_risk": 0.01,
                        "conditional_value_at_risk": 0.02,
                        "maximum_drawdown": drawdown,
                    },
                    "concentration_index": 0.5,
                    "effective_number_of_assets": 2.0,
                    "diversification_ratio": 1.2,
                },
            }
        )
    _write_json(
        paths.portfolio,
        {
            "return_definition": "simple_return",
            "market_splits": {
                "validation": {
                    "observations": 50,
                    "tickers": ["AGG", "SPY"],
                }
            },
            "scenarios": scenarios,
        },
    )
    _write_json(
        paths.reinforcement_learning,
        {
            "evaluation_split": "validation",
            "test_split_used": False,
            "environment_config": {
                "transaction_cost_bps": 10.0,
                "max_weight": 0.35,
            },
            "comparison": [
                {
                    "strategy": "ppo",
                    "annualized_mean_return": 0.1,
                    "annualized_volatility": 0.18,
                    "sharpe_ratio": 0.55,
                    "maximum_drawdown": 0.3,
                },
                {
                    "strategy": "equal_weight",
                    "annualized_mean_return": 0.11,
                    "annualized_volatility": 0.18,
                    "sharpe_ratio": 0.56,
                    "maximum_drawdown": 0.29,
                },
                {
                    "strategy": "maximum_sharpe",
                    "annualized_mean_return": 0.09,
                    "annualized_volatility": 0.16,
                    "sharpe_ratio": 0.57,
                    "maximum_drawdown": 0.28,
                },
            ],
        },
    )
    return paths

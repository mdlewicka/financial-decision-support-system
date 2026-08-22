import json
from pathlib import Path

import pytest

from backend.src.final_evaluation.rl import (
    load_frozen_rl_config,
)


def write_rl_result(
    path: Path,
    *,
    model_name: str = "ppo",
    test_split_used: bool = False,
) -> None:
    payload = {
        "model_name": model_name,
        "training_split": "training",
        "evaluation_split": "validation",
        "test_split_used": test_split_used,
        "agent_config": {
            "total_timesteps": 1000,
            "learning_rate": 0.0003,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "entropy_coefficient": 0.0,
            "hidden_layer_sizes": [
                64,
                64,
            ],
            "seed": 42,
            "device": "cpu",
        },
        "environment_config": {
            "lookback_window": 20,
            "transaction_cost_bps": 10.0,
            "drawdown_penalty": 0.25,
            "max_weight": 0.35,
            "initial_portfolio_value": 1.0,
        },
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_load_frozen_rl_config(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "rl_result.json"

    write_rl_result(result_path)

    (
        agent_config,
        environment_config,
    ) = load_frozen_rl_config(result_path)

    assert agent_config.total_timesteps == 1000
    assert agent_config.seed == 42

    assert environment_config.lookback_window == 20
    assert environment_config.max_weight == 0.35


def test_load_frozen_rl_config_rejects_wrong_model(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "rl_result.json"

    write_rl_result(
        result_path,
        model_name="other",
    )

    with pytest.raises(
        ValueError,
        match="PPO model",
    ):
        load_frozen_rl_config(result_path)


def test_load_frozen_rl_config_rejects_used_test_split(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "rl_result.json"

    write_rl_result(
        result_path,
        test_split_used=True,
    )

    with pytest.raises(
        ValueError,
        match="final test split",
    ):
        load_frozen_rl_config(result_path)

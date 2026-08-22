import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnv,
    PortfolioAllocationEnvConfig,
)
from backend.src.rl.ppo_allocation_agent import (
    PPOAllocationAgent,
    PPOAllocationAgentConfig,
)


def create_environment() -> PortfolioAllocationEnv:
    random_generator = np.random.default_rng(42)

    returns = pd.DataFrame(
        random_generator.normal(
            loc=0.0005,
            scale=0.01,
            size=(48, 3),
        ),
        index=pd.date_range(
            "2020-01-01",
            periods=48,
            freq="D",
        ),
        columns=[
            "AGG",
            "GLD",
            "SPY",
        ],
    )

    return PortfolioAllocationEnv(
        returns=returns,
        config=PortfolioAllocationEnvConfig(
            lookback_window=5,
            transaction_cost_bps=10.0,
            drawdown_penalty=0.25,
            max_weight=0.60,
        ),
    )


def create_agent_config() -> PPOAllocationAgentConfig:
    return PPOAllocationAgentConfig(
        total_timesteps=32,
        learning_rate=0.0003,
        n_steps=16,
        batch_size=8,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        entropy_coefficient=0.0,
        hidden_layer_sizes=(
            16,
            16,
        ),
        seed=42,
        device="cpu",
    )


def train_agent() -> tuple[
    PPOAllocationAgent,
    PortfolioAllocationEnv,
]:
    environment = create_environment()
    agent = PPOAllocationAgent(config=create_agent_config())

    result = agent.fit(environment)

    assert result is agent

    return agent, environment


def test_agent_uses_reproducible_cpu_configuration() -> None:
    config = create_agent_config()

    assert config.seed == 42
    assert config.device == "cpu"
    assert config.total_timesteps == 32
    assert config.n_steps == 16
    assert config.batch_size == 8


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("total_timesteps", 0),
        ("learning_rate", 0.0),
        ("n_steps", 1),
        ("batch_size", 1),
        ("n_epochs", 0),
        ("gamma", 0.0),
        ("gae_lambda", 1.5),
        ("clip_range", 0.0),
        ("entropy_coefficient", -0.1),
        ("hidden_layer_sizes", ()),
    ],
)
def test_agent_config_rejects_invalid_values(
    field_name: str,
    invalid_value: object,
) -> None:
    arguments = {
        "total_timesteps": 32,
        "learning_rate": 0.0003,
        "n_steps": 16,
        "batch_size": 8,
        "n_epochs": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "entropy_coefficient": 0.0,
        "hidden_layer_sizes": (
            16,
            16,
        ),
        "seed": 42,
        "device": "cpu",
    }
    arguments[field_name] = invalid_value

    with pytest.raises(ValueError):
        PPOAllocationAgentConfig(**arguments)


def test_agent_config_rejects_batch_larger_than_rollout() -> None:
    with pytest.raises(
        ValueError,
        match="batch_size",
    ):
        PPOAllocationAgentConfig(
            total_timesteps=32,
            n_steps=8,
            batch_size=16,
        )


def test_predict_before_fit_raises_runtime_error() -> None:
    environment = create_environment()
    observation, _ = environment.reset(seed=42)
    agent = PPOAllocationAgent(config=create_agent_config())

    with pytest.raises(
        RuntimeError,
        match="trained",
    ):
        agent.predict(observation)


def test_fit_trains_agent_and_returns_finite_action() -> None:
    agent, environment = train_agent()
    observation, _ = environment.reset(seed=42)

    action = agent.predict(
        observation,
        deterministic=True,
    )

    assert agent.is_trained
    assert action.shape == (3,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_deterministic_prediction_is_repeatable() -> None:
    agent, environment = train_agent()
    observation, _ = environment.reset(seed=42)

    first_action = agent.predict(
        observation,
        deterministic=True,
    )
    second_action = agent.predict(
        observation,
        deterministic=True,
    )

    np.testing.assert_allclose(
        first_action,
        second_action,
    )


def test_agent_rejects_invalid_observation_shape() -> None:
    agent, _ = train_agent()

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        agent.predict(
            np.zeros(
                5,
                dtype=np.float32,
            )
        )


def test_save_before_fit_raises_runtime_error(
    tmp_path: Path,
) -> None:
    agent = PPOAllocationAgent(config=create_agent_config())

    with pytest.raises(
        RuntimeError,
        match="trained",
    ):
        agent.save(tmp_path / "allocation_agent")


def test_saved_agent_can_be_loaded_without_changing_prediction(
    tmp_path: Path,
) -> None:
    agent, environment = train_agent()
    observation, _ = environment.reset(seed=42)
    expected_action = agent.predict(
        observation,
        deterministic=True,
    )

    saved_path = agent.save(tmp_path / "allocation_agent")

    assert saved_path == (tmp_path / "allocation_agent.zip")
    assert saved_path.exists()

    loaded_agent = PPOAllocationAgent.load(
        path=saved_path,
        environment=create_environment(),
        config=create_agent_config(),
    )
    actual_action = loaded_agent.predict(
        observation,
        deterministic=True,
    )

    assert loaded_agent.is_trained
    np.testing.assert_allclose(
        actual_action,
        expected_action,
    )


def test_fit_contract_accepts_only_training_environment() -> None:
    parameters = tuple(inspect.signature(PPOAllocationAgent.fit).parameters)

    assert parameters == (
        "self",
        "environment",
    )

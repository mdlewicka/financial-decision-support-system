import math

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnv,
    PortfolioAllocationEnvConfig,
)


def create_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AGG": [
                0.001,
                -0.002,
                0.003,
                0.004,
                -0.001,
                0.002,
                0.001,
                -0.003,
            ],
            "SPY": [
                0.010,
                -0.005,
                0.007,
                -0.003,
                0.012,
                0.004,
                -0.006,
                0.008,
            ],
            "GLD": [
                -0.002,
                0.004,
                0.001,
                0.003,
                -0.005,
                0.006,
                0.002,
                -0.001,
            ],
        },
        index=pd.date_range(
            "2024-01-01",
            periods=8,
            freq="D",
        ),
    )


def create_config(
    *,
    transaction_cost_bps: float = 10.0,
    drawdown_penalty: float = 0.25,
) -> PortfolioAllocationEnvConfig:
    return PortfolioAllocationEnvConfig(
        lookback_window=3,
        transaction_cost_bps=transaction_cost_bps,
        drawdown_penalty=drawdown_penalty,
        max_weight=0.60,
        initial_portfolio_value=1.0,
    )


def test_environment_implements_gymnasium_contract() -> None:
    environment = PortfolioAllocationEnv(
        returns=create_returns(),
        config=create_config(),
    )

    observation, info = environment.reset(seed=42)

    assert isinstance(environment, gym.Env)
    assert environment.tickers == (
        "AGG",
        "SPY",
        "GLD",
    )
    assert environment.action_space.shape == (3,)
    assert environment.observation_space.shape == (13,)
    assert observation.shape == (13,)
    assert observation.dtype == np.float32
    assert environment.observation_space.contains(observation)
    assert info["portfolio_value"] == pytest.approx(1.0)
    assert info["drawdown"] == pytest.approx(0.0)


def test_reset_starts_with_equal_weights() -> None:
    environment = PortfolioAllocationEnv(
        returns=create_returns(),
        config=create_config(),
    )

    first_observation, first_info = environment.reset(seed=7)
    second_observation, second_info = environment.reset(seed=7)

    expected_weights = np.full(
        shape=3,
        fill_value=1.0 / 3.0,
    )

    np.testing.assert_allclose(
        environment.current_weights,
        expected_weights,
    )
    np.testing.assert_allclose(
        first_observation,
        second_observation,
    )
    np.testing.assert_allclose(
        first_info["weights"],
        expected_weights,
    )
    np.testing.assert_allclose(
        second_info["weights"],
        expected_weights,
    )


def test_initial_observation_contains_only_historical_information() -> None:
    original_returns = create_returns()
    changed_returns = original_returns.copy(deep=True)

    changed_returns.iloc[3] = [
        0.50,
        -0.40,
        0.30,
    ]

    original_environment = PortfolioAllocationEnv(
        returns=original_returns,
        config=create_config(),
    )
    changed_environment = PortfolioAllocationEnv(
        returns=changed_returns,
        config=create_config(),
    )

    original_observation, _ = original_environment.reset()
    changed_observation, _ = changed_environment.reset()

    np.testing.assert_allclose(
        original_observation,
        changed_observation,
    )


def test_observation_contains_expected_market_features() -> None:
    returns = create_returns()
    environment = PortfolioAllocationEnv(
        returns=returns,
        config=create_config(),
    )

    observation, _ = environment.reset()

    historical_window = returns.iloc[:3]
    expected_mean = historical_window.mean().to_numpy(dtype=float)
    expected_volatility = historical_window.std(ddof=0).to_numpy(dtype=float)
    expected_momentum = historical_window.add(1.0).prod().sub(1.0).to_numpy(dtype=float)
    expected_weights = np.full(
        shape=3,
        fill_value=1.0 / 3.0,
    )
    expected_observation = np.concatenate(
        [
            expected_mean,
            expected_volatility,
            expected_momentum,
            expected_weights,
            np.array([0.0]),
        ]
    ).astype(np.float32)

    np.testing.assert_allclose(
        observation,
        expected_observation,
        rtol=1e-6,
        atol=1e-7,
    )


def test_step_maps_action_to_valid_portfolio_weights() -> None:
    environment = PortfolioAllocationEnv(
        returns=create_returns(),
        config=create_config(),
    )
    environment.reset(seed=42)

    _, _, terminated, truncated, info = environment.step(
        np.array(
            [
                1.0,
                -1.0,
                0.2,
            ],
            dtype=np.float32,
        )
    )

    weights = np.asarray(
        info["weights"],
        dtype=float,
    )

    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 0.60 + 1e-12)
    assert not terminated
    assert not truncated


def test_step_applies_transaction_cost_and_returns_expected_reward() -> None:
    returns = create_returns()
    environment = PortfolioAllocationEnv(
        returns=returns,
        config=create_config(
            transaction_cost_bps=10.0,
            drawdown_penalty=0.0,
        ),
    )
    environment.reset(seed=42)

    _, reward, _, _, info = environment.step(
        np.array(
            [
                1.0,
                -1.0,
                0.0,
            ],
            dtype=np.float32,
        )
    )

    weights = np.asarray(
        info["weights"],
        dtype=float,
    )
    expected_gross_return = float(weights @ returns.iloc[3].to_numpy(dtype=float))
    expected_transaction_cost = float(info["turnover"]) * 10.0 / 10_000.0
    expected_net_return = expected_gross_return - expected_transaction_cost
    expected_reward = math.log1p(expected_net_return)

    assert info["gross_return"] == pytest.approx(expected_gross_return)
    assert info["transaction_cost"] == pytest.approx(expected_transaction_cost)
    assert info["net_return"] == pytest.approx(expected_net_return)
    assert reward == pytest.approx(expected_reward)


def test_step_reports_portfolio_state() -> None:
    environment = PortfolioAllocationEnv(
        returns=create_returns(),
        config=create_config(),
    )
    environment.reset()

    observation, reward, terminated, truncated, info = environment.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert observation.shape == (13,)
    assert np.isfinite(observation).all()
    assert math.isfinite(reward)
    assert info["date"] == pd.Timestamp("2024-01-04")
    assert info["portfolio_value"] > 0.0
    assert 0.0 <= info["drawdown"] <= 1.0
    assert info["turnover"] >= 0.0
    assert not terminated
    assert not truncated


def test_episode_terminates_after_all_available_returns() -> None:
    returns = create_returns()
    environment = PortfolioAllocationEnv(
        returns=returns,
        config=create_config(),
    )
    environment.reset()

    steps = 0
    terminated = False

    while not terminated:
        _, _, terminated, truncated, _ = environment.step(
            np.zeros(
                3,
                dtype=np.float32,
            )
        )
        steps += 1

        assert not truncated

    assert steps == len(returns) - 3

    with pytest.raises(
        RuntimeError,
        match="terminated",
    ):
        environment.step(
            np.zeros(
                3,
                dtype=np.float32,
            )
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_environment_rejects_non_finite_returns(
    invalid_value: float,
) -> None:
    returns = create_returns()
    returns.iloc[2, 1] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        PortfolioAllocationEnv(
            returns=returns,
            config=create_config(),
        )


def test_environment_rejects_unsorted_or_duplicate_dates() -> None:
    unsorted_returns = create_returns().iloc[
        [
            1,
            0,
            2,
            3,
            4,
            5,
            6,
            7,
        ]
    ]
    duplicate_returns = create_returns()
    duplicate_returns.index = pd.DatetimeIndex(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-02",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
            "2024-01-07",
            "2024-01-08",
        ]
    )

    with pytest.raises(
        ValueError,
        match="increasing",
    ):
        PortfolioAllocationEnv(
            returns=unsorted_returns,
            config=create_config(),
        )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        PortfolioAllocationEnv(
            returns=duplicate_returns,
            config=create_config(),
        )


def test_environment_rejects_infeasible_maximum_weight() -> None:
    config = PortfolioAllocationEnvConfig(
        lookback_window=3,
        max_weight=0.30,
    )

    with pytest.raises(
        ValueError,
        match="infeasible",
    ):
        PortfolioAllocationEnv(
            returns=create_returns(),
            config=config,
        )


def test_environment_does_not_modify_input_returns() -> None:
    returns = create_returns()
    original_returns = returns.copy(deep=True)

    environment = PortfolioAllocationEnv(
        returns=returns,
        config=create_config(),
    )
    environment.reset()
    environment.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    pd.testing.assert_frame_equal(
        returns,
        original_returns,
    )

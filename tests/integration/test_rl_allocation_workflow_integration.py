import numpy as np
import pandas as pd

from backend.src.risk.risk_analyzer import (
    RiskAnalysisConfig,
)
from backend.src.rl.environment_factory import (
    PortfolioAllocationEnvironmentFactory,
)
from backend.src.rl.policy_evaluator import (
    RLPolicyEvaluator,
)
from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnvConfig,
)
from backend.src.rl.ppo_allocation_agent import (
    PPOAllocationAgent,
    PPOAllocationAgentConfig,
)

TICKERS = (
    "AGG",
    "GLD",
    "SPY",
)


def create_market_datasets() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    random_generator = np.random.default_rng(42)
    dates = pd.date_range(
        "2020-01-01",
        periods=48,
        freq="D",
    )
    simulated_returns = random_generator.normal(
        loc=np.array(
            [
                0.0002,
                0.0003,
                0.0005,
            ]
        ),
        scale=np.array(
            [
                0.003,
                0.008,
                0.012,
            ]
        ),
        size=(48, 3),
    )
    prices = np.array(
        [
            100.0,
            150.0,
            200.0,
        ]
    ) * np.cumprod(
        1.0 + simulated_returns,
        axis=0,
    )

    records = []

    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(TICKERS):
            records.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_close": float(
                        prices[
                            date_index,
                            ticker_index,
                        ]
                    ),
                }
            )

    market_data = pd.DataFrame(records)
    training_end = dates[35]

    training_dataset = market_data.loc[market_data["date"] <= training_end].reset_index(
        drop=True
    )
    validation_dataset = market_data.loc[
        market_data["date"] > training_end
    ].reset_index(drop=True)

    return (
        training_dataset,
        validation_dataset,
    )


def test_complete_rl_allocation_workflow() -> None:
    training_dataset, validation_dataset = create_market_datasets()
    environment_config = PortfolioAllocationEnvConfig(
        lookback_window=5,
        transaction_cost_bps=10.0,
        drawdown_penalty=0.25,
        max_weight=0.60,
    )
    environments = PortfolioAllocationEnvironmentFactory(
        environment_config=(environment_config),
        minimum_common_observations=2,
    ).create(
        training_dataset=training_dataset,
        validation_dataset=(validation_dataset),
    )

    validation_observation_before_training, _ = (
        environments.validation_environment.reset(seed=42)
    )

    agent = PPOAllocationAgent(
        config=PPOAllocationAgentConfig(
            total_timesteps=32,
            learning_rate=0.0003,
            n_steps=16,
            batch_size=8,
            n_epochs=2,
            hidden_layer_sizes=(
                16,
                16,
            ),
            seed=42,
            device="cpu",
        )
    )
    agent.fit(environments.training_environment)

    validation_observation_after_training, _ = (
        environments.validation_environment.reset(seed=42)
    )

    np.testing.assert_allclose(
        validation_observation_before_training,
        validation_observation_after_training,
    )

    result = RLPolicyEvaluator(
        risk_config=RiskAnalysisConfig(
            confidence_level=0.95,
            annual_risk_free_rate=0.02,
            periods_per_year=252,
            minimum_observations=2,
        )
    ).evaluate(
        policy=agent,
        environment=(environments.validation_environment),
        seed=42,
    )

    validation_start_date = pd.Timestamp(validation_dataset["date"].min())
    validation_end_date = pd.Timestamp(validation_dataset["date"].max())
    weight_values = result.weights.to_numpy(dtype=float)

    assert agent.is_trained
    assert result.tickers == TICKERS
    assert result.observations == 12
    assert result.date_min == validation_start_date
    assert result.date_max == validation_end_date
    assert result.risk_metrics.observations == 12
    assert result.final_portfolio_value > 0.0
    assert np.isfinite(result.net_returns.to_numpy(dtype=float)).all()
    assert np.isfinite(weight_values).all()
    assert result.total_transaction_cost >= 0.0
    assert result.mean_turnover >= 0.0

    np.testing.assert_allclose(
        weight_values.sum(axis=1),
        1.0,
    )

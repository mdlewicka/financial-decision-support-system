import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.risk.risk_analyzer import (
    RiskAnalysisConfig,
    RiskAnalyzer,
)
from backend.src.rl.policy_evaluator import (
    RLPolicyEvaluator,
)
from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnv,
    PortfolioAllocationEnvConfig,
)


class StubAllocationPolicy:
    def __init__(
        self,
        action: np.ndarray | None = None,
        *,
        is_trained: bool = True,
    ) -> None:
        self._action = (
            np.asarray(
                action,
                dtype=np.float32,
            )
            if action is not None
            else np.zeros(
                3,
                dtype=np.float32,
            )
        )
        self.is_trained = is_trained
        self.predict_calls = 0
        self.fit_calls = 0
        self.deterministic_flags: list[bool] = []

    def fit(
        self,
        environment: PortfolioAllocationEnv,
    ) -> None:
        del environment

        self.fit_calls += 1

        raise AssertionError("Evaluation must not train the policy.")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        del observation

        self.predict_calls += 1
        self.deterministic_flags.append(deterministic)

        return self._action.copy()


def create_environment() -> PortfolioAllocationEnv:
    returns = pd.DataFrame(
        {
            "AGG": [
                0.001,
                -0.002,
                0.003,
                0.001,
                0.002,
                -0.001,
                0.004,
                0.001,
                -0.002,
                0.003,
                0.002,
                -0.001,
            ],
            "GLD": [
                -0.002,
                0.003,
                0.001,
                0.004,
                -0.003,
                0.002,
                0.001,
                0.005,
                -0.001,
                0.002,
                0.003,
                0.001,
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
                0.003,
                -0.004,
                0.009,
                0.005,
            ],
        },
        index=pd.date_range(
            "2024-01-01",
            periods=12,
            freq="D",
        ),
        dtype=float,
    )

    return PortfolioAllocationEnv(
        returns=returns,
        config=PortfolioAllocationEnvConfig(
            lookback_window=3,
            transaction_cost_bps=10.0,
            drawdown_penalty=0.25,
            max_weight=0.60,
            initial_portfolio_value=1.0,
        ),
    )


def create_risk_config() -> RiskAnalysisConfig:
    return RiskAnalysisConfig(
        confidence_level=0.95,
        annual_risk_free_rate=0.02,
        periods_per_year=252,
        minimum_observations=2,
    )


def create_evaluator() -> RLPolicyEvaluator:
    return RLPolicyEvaluator(risk_config=create_risk_config())


def test_evaluate_collects_complete_validation_trajectory() -> None:
    policy = StubAllocationPolicy()

    result = create_evaluator().evaluate(
        policy=policy,
        environment=create_environment(),
        seed=42,
    )

    assert result.tickers == (
        "AGG",
        "GLD",
        "SPY",
    )
    assert result.observations == 9
    assert result.date_min == pd.Timestamp("2024-01-04")
    assert result.date_max == pd.Timestamp("2024-01-12")
    assert len(result.trajectory) == 9
    assert len(result.weights) == 9
    assert len(result.net_returns) == 9
    assert result.risk_metrics.observations == 9


def test_evaluate_always_uses_deterministic_predictions() -> None:
    policy = StubAllocationPolicy()

    result = create_evaluator().evaluate(
        policy=policy,
        environment=create_environment(),
        seed=42,
    )

    assert policy.fit_calls == 0
    assert policy.predict_calls == result.observations
    assert policy.deterministic_flags == [True] * result.observations


def test_evaluated_weights_are_valid_portfolio_allocations() -> None:
    result = create_evaluator().evaluate(
        policy=StubAllocationPolicy(
            action=np.array(
                [
                    1.0,
                    -1.0,
                    0.25,
                ],
                dtype=np.float32,
            )
        ),
        environment=create_environment(),
        seed=42,
    )

    weight_values = np.asarray(
        result.weights,
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        weight_values.sum(axis=1),
        1.0,
    )

    assert np.all(weight_values >= 0.0)

    assert np.all(weight_values <= 0.60 + 1e-12)


def test_evaluation_summary_matches_recorded_trajectory() -> None:
    result = create_evaluator().evaluate(
        policy=StubAllocationPolicy(
            action=np.array(
                [
                    1.0,
                    -1.0,
                    0.0,
                ],
                dtype=np.float32,
            )
        ),
        environment=create_environment(),
        seed=42,
    )

    net_return_values = np.asarray(
        result.net_returns,
        dtype=np.float64,
    )

    expected_portfolio_value = float(np.prod(1.0 + net_return_values))

    expected_cumulative_return = expected_portfolio_value - 1.0

    transaction_cost_values = np.asarray(
        result.trajectory["transaction_cost"],
        dtype=np.float64,
    )

    turnover_values = np.asarray(
        result.trajectory["turnover"],
        dtype=np.float64,
    )

    reward_values = np.asarray(
        result.trajectory["reward"],
        dtype=np.float64,
    )

    assert result.final_portfolio_value == pytest.approx(expected_portfolio_value)

    assert result.cumulative_return == pytest.approx(expected_cumulative_return)

    assert result.total_transaction_cost == pytest.approx(
        float(np.sum(transaction_cost_values))
    )

    assert result.mean_turnover == pytest.approx(float(np.mean(turnover_values)))

    assert result.total_reward == pytest.approx(float(np.sum(reward_values)))


def test_evaluator_uses_shared_risk_analyzer_contract() -> None:
    risk_config = create_risk_config()

    evaluator = RLPolicyEvaluator(risk_config=risk_config)

    result = evaluator.evaluate(
        policy=StubAllocationPolicy(),
        environment=create_environment(),
        seed=42,
    )

    expected_metrics = RiskAnalyzer(config=risk_config).analyze(result.net_returns)

    assert result.risk_metrics == expected_metrics


def test_repeated_deterministic_evaluation_is_reproducible() -> None:
    evaluator = create_evaluator()

    policy = StubAllocationPolicy(
        action=np.array(
            [
                0.5,
                -0.5,
                0.0,
            ],
            dtype=np.float32,
        )
    )

    environment = create_environment()

    first_result = evaluator.evaluate(
        policy=policy,
        environment=environment,
        seed=42,
    )

    second_result = evaluator.evaluate(
        policy=policy,
        environment=environment,
        seed=42,
    )

    pdt.assert_frame_equal(
        first_result.trajectory,
        second_result.trajectory,
    )

    pdt.assert_frame_equal(
        first_result.weights,
        second_result.weights,
    )

    pdt.assert_series_equal(
        first_result.net_returns,
        second_result.net_returns,
    )


def test_evaluator_rejects_untrained_policy() -> None:
    with pytest.raises(
        RuntimeError,
        match="trained",
    ):
        create_evaluator().evaluate(
            policy=StubAllocationPolicy(is_trained=False),
            environment=create_environment(),
            seed=42,
        )

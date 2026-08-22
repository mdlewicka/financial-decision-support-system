import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from backend.src.risk.risk_analyzer import (
    RiskAnalysisConfig,
    RiskAnalyzer,
)
from backend.src.risk.risk_metrics import (
    RiskMetrics,
)
from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnv,
)


class AllocationPolicy(Protocol):
    @property
    def is_trained(self) -> bool: ...

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class RLPolicyEvaluationResult:
    tickers: tuple[str, ...]
    observations: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    trajectory: pd.DataFrame
    weights: pd.DataFrame
    net_returns: pd.Series
    risk_metrics: RiskMetrics
    total_reward: float
    cumulative_return: float
    final_portfolio_value: float
    total_transaction_cost: float
    mean_turnover: float


class RLPolicyEvaluator:
    def __init__(
        self,
        risk_config: RiskAnalysisConfig | None = None,
    ) -> None:
        self.risk_config = risk_config or RiskAnalysisConfig()
        self._risk_analyzer = RiskAnalyzer(config=self.risk_config)

    def evaluate(
        self,
        policy: AllocationPolicy,
        environment: PortfolioAllocationEnv,
        seed: int = 42,
    ) -> RLPolicyEvaluationResult:
        if not getattr(
            policy,
            "is_trained",
            False,
        ):
            raise RuntimeError(
                "The allocation policy must be trained before evaluation."
            )

        if not isinstance(
            environment,
            PortfolioAllocationEnv,
        ):
            raise TypeError("environment must be a PortfolioAllocationEnv instance.")

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        observation, _ = environment.reset(seed=seed)

        dates: list[pd.Timestamp] = []
        trajectory_rows: list[dict[str, float]] = []
        weight_rows: list[np.ndarray] = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = policy.predict(
                observation,
                deterministic=True,
            )
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(action)

            evaluation_date = self._extract_date(info)
            weights = self._extract_weights(
                info=info,
                number_of_assets=len(environment.tickers),
            )
            gross_return = self._extract_finite_float(
                info=info,
                field_name="gross_return",
            )
            transaction_cost = self._extract_finite_float(
                info=info,
                field_name="transaction_cost",
            )
            net_return = self._extract_finite_float(
                info=info,
                field_name="net_return",
            )
            turnover = self._extract_finite_float(
                info=info,
                field_name="turnover",
            )
            portfolio_value = self._extract_finite_float(
                info=info,
                field_name="portfolio_value",
            )
            drawdown = self._extract_finite_float(
                info=info,
                field_name="drawdown",
            )

            reward_value = float(reward)

            if not math.isfinite(reward_value):
                raise RuntimeError("The environment produced a non-finite reward.")

            dates.append(evaluation_date)
            weight_rows.append(weights)
            trajectory_rows.append(
                {
                    "gross_return": gross_return,
                    "transaction_cost": (transaction_cost),
                    "net_return": net_return,
                    "turnover": turnover,
                    "portfolio_value": (portfolio_value),
                    "drawdown": drawdown,
                    "reward": reward_value,
                }
            )

        if not trajectory_rows:
            raise RuntimeError("Policy evaluation produced no observations.")

        index = pd.DatetimeIndex(
            dates,
            name="date",
        )
        trajectory = pd.DataFrame(
            trajectory_rows,
            index=index,
            dtype=float,
        )
        weights = pd.DataFrame(
            np.vstack(weight_rows),
            index=index,
            columns=environment.tickers,
            dtype=float,
        )
        net_return_values = np.asarray(
            trajectory["net_return"],
            dtype=float,
        )
        net_returns = pd.Series(
            net_return_values,
            index=index,
            name="net_return",
            dtype=float,
        )

        risk_metrics = self._risk_analyzer.analyze(net_returns)
        portfolio_values = np.asarray(
            trajectory["portfolio_value"],
            dtype=float,
        )
        transaction_costs = np.asarray(
            trajectory["transaction_cost"],
            dtype=float,
        )
        turnovers = np.asarray(
            trajectory["turnover"],
            dtype=float,
        )
        rewards = np.asarray(
            trajectory["reward"],
            dtype=float,
        )

        final_portfolio_value = float(portfolio_values[-1])
        cumulative_return = float(np.prod(1.0 + net_return_values) - 1.0)

        return RLPolicyEvaluationResult(
            tickers=environment.tickers,
            observations=len(trajectory),
            date_min=index[0],
            date_max=index[-1],
            trajectory=trajectory,
            weights=weights,
            net_returns=net_returns,
            risk_metrics=risk_metrics,
            total_reward=float(rewards.sum()),
            cumulative_return=cumulative_return,
            final_portfolio_value=(final_portfolio_value),
            total_transaction_cost=float(transaction_costs.sum()),
            mean_turnover=float(turnovers.mean()),
        )

    @staticmethod
    def _extract_date(
        info: dict[str, object],
    ) -> pd.Timestamp:
        date_value = info.get("date")

        if not isinstance(
            date_value,
            pd.Timestamp,
        ):
            raise RuntimeError(
                "The environment info must contain a pandas Timestamp date."
            )

        return date_value

    @staticmethod
    def _extract_weights(
        info: dict[str, object],
        number_of_assets: int,
    ) -> np.ndarray:
        weights = np.asarray(
            info.get("weights"),
            dtype=float,
        )

        if weights.shape != (number_of_assets,):
            raise RuntimeError(
                "The environment produced an unexpected portfolio weight shape."
            )

        if not np.isfinite(weights).all():
            raise RuntimeError("The environment produced non-finite portfolio weights.")

        if np.any(weights < -1e-12) or not math.isclose(
            float(weights.sum()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise RuntimeError("The environment produced invalid portfolio weights.")

        return weights.copy()

    @staticmethod
    def _extract_finite_float(
        info: dict[str, object],
        field_name: str,
    ) -> float:
        value = info.get(field_name)

        if isinstance(value, bool) or not isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            raise RuntimeError(
                f"The environment info field {field_name!r} must be numeric."
            )

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise RuntimeError(
                f"The environment info field {field_name!r} must be finite."
            )

        return numeric_value

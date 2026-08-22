import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass(frozen=True, slots=True)
class PortfolioAllocationEnvConfig:
    lookback_window: int = 20
    transaction_cost_bps: float = 10.0
    drawdown_penalty: float = 0.25
    max_weight: float = 0.35
    initial_portfolio_value: float = 1.0

    def __post_init__(self) -> None:
        if self.lookback_window < 2:
            raise ValueError("lookback_window must contain at least two observations.")

        if (
            not math.isfinite(self.transaction_cost_bps)
            or self.transaction_cost_bps < 0.0
        ):
            raise ValueError(
                "transaction_cost_bps must be a finite non-negative value."
            )

        if not math.isfinite(self.drawdown_penalty) or self.drawdown_penalty < 0.0:
            raise ValueError("drawdown_penalty must be a finite non-negative value.")

        if (
            not math.isfinite(self.max_weight)
            or self.max_weight <= 0.0
            or self.max_weight > 1.0
        ):
            raise ValueError(
                "max_weight must be a finite value in the interval (0, 1]."
            )

        if (
            not math.isfinite(self.initial_portfolio_value)
            or self.initial_portfolio_value <= 0.0
        ):
            raise ValueError("initial_portfolio_value must be a finite positive value.")


class PortfolioAllocationEnv(
    gym.Env[np.ndarray, np.ndarray],
):
    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        returns: pd.DataFrame,
        config: PortfolioAllocationEnvConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or PortfolioAllocationEnvConfig()
        self._returns = self._prepare_returns(returns)
        self._tickers = tuple(str(column) for column in self._returns.columns)
        self._number_of_assets = len(self._tickers)

        self._validate_maximum_weight_feasibility()

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._number_of_assets,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4 * self._number_of_assets + 1,),
            dtype=np.float32,
        )

        self._current_step = self.config.lookback_window
        self._current_weights = self._equal_weights()
        self._portfolio_value = self.config.initial_portfolio_value
        self._peak_portfolio_value = self.config.initial_portfolio_value
        self._current_drawdown = 0.0
        self._terminated = False
        self._has_reset = False

    @property
    def tickers(self) -> tuple[str, ...]:
        return self._tickers

    @property
    def current_weights(self) -> np.ndarray:
        return self._current_weights.copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        del options

        self._current_step = self.config.lookback_window
        self._current_weights = self._equal_weights()
        self._portfolio_value = self.config.initial_portfolio_value
        self._peak_portfolio_value = self.config.initial_portfolio_value
        self._current_drawdown = 0.0
        self._terminated = False
        self._has_reset = True

        observation = self._create_observation()
        info = self._create_reset_info()

        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if not self._has_reset:
            raise RuntimeError("The environment must be reset before calling step.")

        if self._terminated:
            raise RuntimeError(
                "The episode has terminated. Reset the environment "
                "before calling step again."
            )

        target_weights = self._action_to_weights(action)
        turnover = self._calculate_turnover(target_weights)

        return_values = self._returns.iloc[self._current_step].to_numpy(dtype=float)
        evaluation_date = self._returns.index[self._current_step]

        gross_return = float(target_weights @ return_values)
        transaction_cost = turnover * self.config.transaction_cost_bps / 10_000.0
        net_return = gross_return - transaction_cost

        if net_return <= -1.0:
            raise RuntimeError("The portfolio net return must be greater than -1.")

        self._portfolio_value *= 1.0 + net_return
        self._peak_portfolio_value = max(
            self._peak_portfolio_value,
            self._portfolio_value,
        )
        self._current_drawdown = (
            1.0 - self._portfolio_value / self._peak_portfolio_value
        )
        self._current_weights = target_weights

        reward = (
            math.log1p(net_return)
            - self.config.drawdown_penalty * self._current_drawdown
        )

        self._current_step += 1
        self._terminated = self._current_step >= len(self._returns)

        observation = self._create_observation()
        info = {
            "date": evaluation_date,
            "weights": self._current_weights.copy(),
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "turnover": turnover,
            "portfolio_value": self._portfolio_value,
            "drawdown": self._current_drawdown,
        }

        return (
            observation,
            float(reward),
            self._terminated,
            False,
            info,
        )

    def _prepare_returns(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")

        if returns.empty:
            raise ValueError("returns must not be empty.")

        if returns.shape[1] < 2:
            raise ValueError("returns must contain at least two assets.")

        if not isinstance(
            returns.index,
            pd.DatetimeIndex,
        ):
            raise ValueError("returns must use a DatetimeIndex.")

        if not returns.index.is_unique:
            raise ValueError("Return dates must be unique.")

        if not returns.index.is_monotonic_increasing:
            raise ValueError("Return dates must be strictly increasing.")

        if not returns.columns.is_unique:
            raise ValueError("Return columns must contain unique tickers.")

        ticker_names = [str(column) for column in returns.columns]

        if any(not ticker.strip() for ticker in ticker_names):
            raise ValueError("Ticker names must not be empty.")

        if len(returns) <= self.config.lookback_window:
            raise ValueError(
                "returns must contain more observations than "
                "the configured lookback_window."
            )

        prepared_returns = returns.copy(deep=True)

        try:
            prepared_returns = prepared_returns.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("returns must contain only numeric values.") from error

        values = prepared_returns.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError("returns must contain only finite values.")

        if np.any(values <= -1.0):
            raise ValueError("Every return must be greater than -1.")

        prepared_returns.columns = ticker_names

        return prepared_returns

    def _validate_maximum_weight_feasibility(
        self,
    ) -> None:
        total_capacity = self.config.max_weight * self._number_of_assets

        if total_capacity < 1.0 - 1e-12:
            raise ValueError(
                "The maximum portfolio weight is infeasible "
                "for the available number of assets."
            )

    def _equal_weights(self) -> np.ndarray:
        return np.full(
            shape=self._number_of_assets,
            fill_value=1.0 / self._number_of_assets,
            dtype=float,
        )

    def _create_observation(self) -> np.ndarray:
        window_start = self._current_step - self.config.lookback_window
        historical_window = self._returns.iloc[window_start : self._current_step]

        mean_returns = historical_window.mean().to_numpy(dtype=float)
        volatility = historical_window.std(ddof=0).to_numpy(dtype=float)
        momentum = historical_window.add(1.0).prod().sub(1.0).to_numpy(dtype=float)

        observation = np.concatenate(
            [
                mean_returns,
                volatility,
                momentum,
                self._current_weights,
                np.array(
                    [self._current_drawdown],
                    dtype=float,
                ),
            ]
        ).astype(np.float32)

        if not np.isfinite(observation).all():
            raise RuntimeError("The environment produced a non-finite observation.")

        return observation

    def _create_reset_info(
        self,
    ) -> dict[str, Any]:
        return {
            "date": self._returns.index[self._current_step],
            "weights": self._current_weights.copy(),
            "portfolio_value": self._portfolio_value,
            "drawdown": self._current_drawdown,
        }

    def _action_to_weights(
        self,
        action: np.ndarray,
    ) -> np.ndarray:
        action_values = np.asarray(
            action,
            dtype=float,
        )

        if action_values.shape != (self._number_of_assets,):
            raise ValueError("The action must contain one value for each asset.")

        if not np.isfinite(action_values).all():
            raise ValueError("The action must contain only finite values.")

        bounded_action = np.clip(
            action_values,
            -1.0,
            1.0,
        )

        shifted_action = bounded_action - float(np.max(bounded_action))
        positive_scores = np.exp(shifted_action)
        unconstrained_weights = positive_scores / positive_scores.sum()

        return self._apply_maximum_weight_constraint(unconstrained_weights)

    def _apply_maximum_weight_constraint(
        self,
        unconstrained_weights: np.ndarray,
    ) -> np.ndarray:
        weights = np.zeros(
            self._number_of_assets,
            dtype=float,
        )
        active = np.ones(
            self._number_of_assets,
            dtype=bool,
        )
        remaining_weight = 1.0

        while np.any(active):
            active_indices = np.flatnonzero(active)
            active_scores = unconstrained_weights[active_indices]
            proposed_weights = active_scores / active_scores.sum() * remaining_weight
            exceeds_limit = proposed_weights > self.config.max_weight

            if not np.any(exceeds_limit):
                weights[active_indices] = proposed_weights
                break

            capped_indices = active_indices[exceeds_limit]
            weights[capped_indices] = self.config.max_weight
            active[capped_indices] = False
            remaining_weight = 1.0 - float(weights[~active].sum())

        residual = 1.0 - float(weights.sum())

        if abs(residual) > 1e-12:
            available_capacity = self.config.max_weight - weights
            eligible = available_capacity > 1e-12

            if not np.any(eligible):
                raise RuntimeError("Unable to construct feasible portfolio weights.")

            capacity = available_capacity[eligible]
            weights[eligible] += residual * capacity / capacity.sum()

        if (
            np.any(weights < -1e-12)
            or np.any(weights > self.config.max_weight + 1e-12)
            or not math.isclose(
                float(weights.sum()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        ):
            raise RuntimeError("Unable to construct valid portfolio weights.")

        return weights

    def _calculate_turnover(
        self,
        target_weights: np.ndarray,
    ) -> float:
        return 0.5 * float(np.abs(target_weights - self._current_weights).sum())

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from stable_baselines3 import PPO

from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnv,
)


@dataclass(frozen=True, slots=True)
class PPOAllocationAgentConfig:
    total_timesteps: int = 10_000
    learning_rate: float = 0.0003
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    entropy_coefficient: float = 0.0
    hidden_layer_sizes: tuple[int, ...] = (
        64,
        64,
    )
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._validate_positive_integer(
            value=self.total_timesteps,
            field_name="total_timesteps",
        )

        if (
            isinstance(self.n_steps, bool)
            or not isinstance(self.n_steps, int)
            or self.n_steps < 2
        ):
            raise ValueError("n_steps must be an integer of at least 2.")

        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 2
        ):
            raise ValueError("batch_size must be an integer of at least 2.")

        if self.batch_size > self.n_steps:
            raise ValueError("batch_size must not be greater than n_steps.")

        if self.n_steps % self.batch_size != 0:
            raise ValueError("n_steps must be divisible by batch_size.")

        self._validate_positive_integer(
            value=self.n_epochs,
            field_name="n_epochs",
        )
        self._validate_positive_float(
            value=self.learning_rate,
            field_name="learning_rate",
        )

        if not math.isfinite(self.gamma) or self.gamma <= 0.0 or self.gamma > 1.0:
            raise ValueError("gamma must be in the interval (0, 1].")

        if (
            not math.isfinite(self.gae_lambda)
            or self.gae_lambda < 0.0
            or self.gae_lambda > 1.0
        ):
            raise ValueError("gae_lambda must be in the interval [0, 1].")

        if (
            not math.isfinite(self.clip_range)
            or self.clip_range <= 0.0
            or self.clip_range > 1.0
        ):
            raise ValueError("clip_range must be in the interval (0, 1].")

        if (
            not math.isfinite(self.entropy_coefficient)
            or self.entropy_coefficient < 0.0
        ):
            raise ValueError("entropy_coefficient must be a finite non-negative value.")

        if (
            not isinstance(self.hidden_layer_sizes, tuple)
            or not self.hidden_layer_sizes
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in self.hidden_layer_sizes
            )
        ):
            raise ValueError("hidden_layer_sizes must contain positive integers.")

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer.")

        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string.")

    @staticmethod
    def _validate_positive_integer(
        value: int,
        field_name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer.")

    @staticmethod
    def _validate_positive_float(
        value: float,
        field_name: str,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{field_name} must be a finite positive value.")


class PPOAllocationAgent:
    def __init__(
        self,
        config: PPOAllocationAgentConfig | None = None,
    ) -> None:
        self.config = config or PPOAllocationAgentConfig()
        self._model: PPO | None = None
        self._observation_shape: tuple[int, ...] | None = None
        self._action_shape: tuple[int, ...] | None = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def fit(
        self,
        environment: PortfolioAllocationEnv,
    ) -> Self:
        if not isinstance(
            environment,
            PortfolioAllocationEnv,
        ):
            raise TypeError("environment must be a PortfolioAllocationEnv instance.")

        number_of_assets = len(environment.tickers)
        self._observation_shape = (4 * number_of_assets + 1,)
        self._action_shape = (number_of_assets,)

        self._model = PPO(
            policy="MlpPolicy",
            env=environment,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.entropy_coefficient,
            policy_kwargs={
                "net_arch": list(self.config.hidden_layer_sizes),
            },
            seed=self.config.seed,
            device=self.config.device,
            verbose=0,
        )
        self._model.learn(
            total_timesteps=self.config.total_timesteps,
            progress_bar=False,
        )

        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        model = self._require_trained_model()
        observation_values = np.asarray(
            observation,
            dtype=np.float32,
        )

        if (
            self._observation_shape is None
            or observation_values.shape != self._observation_shape
        ):
            raise ValueError("The observation has an unexpected shape.")

        if not np.isfinite(observation_values).all():
            raise ValueError("The observation must contain only finite values.")

        action, _ = model.predict(
            observation_values,
            deterministic=deterministic,
        )
        action_values = np.asarray(
            action,
            dtype=np.float32,
        )

        if self._action_shape is None or action_values.shape != self._action_shape:
            raise RuntimeError("The trained model produced an unexpected action shape.")

        if not np.isfinite(action_values).all():
            raise RuntimeError("The trained model produced a non-finite action.")

        return np.clip(
            action_values,
            -1.0,
            1.0,
        ).astype(np.float32)

    def save(
        self,
        path: Path,
    ) -> Path:
        model = self._require_trained_model()
        target_path = Path(path)

        if not target_path.name:
            raise ValueError("The model path must include a file name.")

        target_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        archive_path = (
            target_path if target_path.suffix == ".zip" else Path(f"{target_path}.zip")
        )

        model.save(str(archive_path))

        if not archive_path.exists():
            raise RuntimeError(
                "Stable-Baselines3 did not create the expected model archive."
            )

        return archive_path

    @classmethod
    def load(
        cls,
        path: Path,
        environment: PortfolioAllocationEnv,
        config: PPOAllocationAgentConfig | None = None,
    ) -> Self:
        archive_path = Path(path)

        if not archive_path.exists():
            raise FileNotFoundError(f"Agent model does not exist: {archive_path}")

        if not isinstance(
            environment,
            PortfolioAllocationEnv,
        ):
            raise TypeError("environment must be a PortfolioAllocationEnv instance.")

        agent = cls(config=config)
        number_of_assets = len(environment.tickers)
        agent._observation_shape = (4 * number_of_assets + 1,)
        agent._action_shape = (number_of_assets,)
        agent._model = PPO.load(
            path=str(archive_path),
            env=environment,
            device=agent.config.device,
        )

        return agent

    def _require_trained_model(self) -> PPO:
        if self._model is None:
            raise RuntimeError("The PPO allocation agent has not been trained.")

        return self._model

import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env

from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnv,
    PortfolioAllocationEnvConfig,
)


def test_portfolio_allocation_environment_is_compatible_with_sb3() -> None:
    random_generator = np.random.default_rng(42)

    returns = pd.DataFrame(
        random_generator.normal(
            loc=0.0004,
            scale=0.01,
            size=(64, 8),
        ),
        index=pd.date_range(
            "2020-01-01",
            periods=64,
            freq="D",
        ),
        columns=[
            "AGG",
            "BTC-USD",
            "EPOL",
            "GLD",
            "QQQ",
            "SPY",
            "VGK",
            "VT",
        ],
    )

    environment = PortfolioAllocationEnv(
        returns=returns,
        config=PortfolioAllocationEnvConfig(
            lookback_window=10,
            transaction_cost_bps=10.0,
            drawdown_penalty=0.25,
            max_weight=0.35,
        ),
    )

    check_env(
        environment,
        warn=True,
        skip_render_check=True,
    )

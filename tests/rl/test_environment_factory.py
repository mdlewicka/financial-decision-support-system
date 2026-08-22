import inspect

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.rl.environment_factory import (
    PortfolioAllocationEnvironmentFactory,
)
from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnvConfig,
)

TICKERS = (
    "AGG",
    "GLD",
    "SPY",
)


def create_market_dataset(
    dates: pd.DatetimeIndex,
    prices: dict[str, list[float]],
) -> pd.DataFrame:
    records = []

    for ticker in TICKERS:
        for date, adjusted_close in zip(
            dates,
            prices[ticker],
            strict=True,
        ):
            records.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_close": adjusted_close,
                }
            )

    return pd.DataFrame(records)


def create_training_dataset() -> pd.DataFrame:
    return create_market_dataset(
        dates=pd.date_range(
            "2024-01-01",
            periods=8,
            freq="D",
        ),
        prices={
            "AGG": [
                100.0,
                100.2,
                100.1,
                100.4,
                100.6,
                100.5,
                100.8,
                101.0,
            ],
            "GLD": [
                150.0,
                149.5,
                150.2,
                150.8,
                150.4,
                151.0,
                151.5,
                151.2,
            ],
            "SPY": [
                200.0,
                202.0,
                201.0,
                203.0,
                205.0,
                204.0,
                206.0,
                208.0,
            ],
        },
    )


def create_validation_dataset() -> pd.DataFrame:
    return create_market_dataset(
        dates=pd.date_range(
            "2024-01-09",
            periods=4,
            freq="D",
        ),
        prices={
            "AGG": [
                101.2,
                101.1,
                101.4,
                101.6,
            ],
            "GLD": [
                151.8,
                152.0,
                151.6,
                152.4,
            ],
            "SPY": [
                209.0,
                207.0,
                210.0,
                212.0,
            ],
        },
    )


def create_factory() -> PortfolioAllocationEnvironmentFactory:
    return PortfolioAllocationEnvironmentFactory(
        environment_config=PortfolioAllocationEnvConfig(
            lookback_window=3,
            transaction_cost_bps=10.0,
            drawdown_penalty=0.25,
            max_weight=0.60,
        ),
        minimum_common_observations=2,
    )


def test_factory_creates_training_and_validation_environments() -> None:
    result = create_factory().create(
        training_dataset=create_training_dataset(),
        validation_dataset=create_validation_dataset(),
    )

    assert result.tickers == TICKERS
    assert result.training_environment.tickers == TICKERS
    assert result.validation_environment.tickers == TICKERS
    assert result.training_environment.action_space.shape == (3,)
    assert result.validation_environment.action_space.shape == (3,)


def test_factory_preserves_validation_boundary_return() -> None:
    result = create_factory().create(
        training_dataset=create_training_dataset(),
        validation_dataset=create_validation_dataset(),
    )

    assert len(result.training_returns) == 7
    assert len(result.validation_returns) == 4
    assert result.validation_returns.index.min() == pd.Timestamp("2024-01-09")
    assert result.validation_returns.index.max() == pd.Timestamp("2024-01-12")

    result.validation_environment.reset(seed=42)
    _, _, _, _, info = result.validation_environment.step(
        np.zeros(
            3,
            dtype=np.float32,
        )
    )

    assert info["date"] == pd.Timestamp("2024-01-09")


def test_validation_environment_uses_only_training_context() -> None:
    training_dataset = create_training_dataset()
    validation_dataset = create_validation_dataset()
    changed_validation_dataset = validation_dataset.copy(deep=True)
    changed_validation_dataset["adjusted_close"] = (
        changed_validation_dataset["adjusted_close"] * 1.50
    )

    original_result = create_factory().create(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
    )
    changed_result = create_factory().create(
        training_dataset=training_dataset,
        validation_dataset=changed_validation_dataset,
    )

    original_observation, _ = original_result.validation_environment.reset(seed=42)
    changed_observation, _ = changed_result.validation_environment.reset(seed=42)

    np.testing.assert_allclose(
        original_observation,
        changed_observation,
    )


def test_validation_context_contains_last_training_returns() -> None:
    result = create_factory().create(
        training_dataset=create_training_dataset(),
        validation_dataset=create_validation_dataset(),
    )

    expected_context = result.training_returns.tail(3)

    pdt.assert_frame_equal(
        result.validation_context_returns,
        expected_context,
    )

    assert (
        result.validation_context_returns.index.max()
        < result.validation_returns.index.min()
    )


def test_training_environment_never_uses_validation_dates() -> None:
    result = create_factory().create(
        training_dataset=create_training_dataset(),
        validation_dataset=create_validation_dataset(),
    )

    environment = result.training_environment
    environment.reset(seed=42)

    observed_dates = []
    terminated = False

    while not terminated:
        _, _, terminated, _, info = environment.step(
            np.zeros(
                3,
                dtype=np.float32,
            )
        )
        observed_dates.append(info["date"])

    assert max(observed_dates) == pd.Timestamp("2024-01-08")
    assert max(observed_dates) < result.validation_returns.index.min()


def test_factory_rejects_overlapping_splits() -> None:
    validation_dataset = create_validation_dataset()
    validation_dataset["date"] = validation_dataset["date"] - pd.Timedelta(days=1)

    with pytest.raises(
        ValueError,
        match="after",
    ):
        create_factory().create(
            training_dataset=create_training_dataset(),
            validation_dataset=validation_dataset,
        )


def test_factory_rejects_different_ticker_sets() -> None:
    validation_dataset = create_validation_dataset()
    validation_dataset = validation_dataset.loc[
        validation_dataset["ticker"] != "GLD"
    ].reset_index(drop=True)

    with pytest.raises(
        ValueError,
        match="tickers",
    ):
        create_factory().create(
            training_dataset=create_training_dataset(),
            validation_dataset=validation_dataset,
        )


def test_factory_does_not_modify_input_datasets() -> None:
    training_dataset = create_training_dataset()
    validation_dataset = create_validation_dataset()
    original_training = training_dataset.copy(deep=True)
    original_validation = validation_dataset.copy(deep=True)

    create_factory().create(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
    )

    pdt.assert_frame_equal(
        training_dataset,
        original_training,
    )
    pdt.assert_frame_equal(
        validation_dataset,
        original_validation,
    )


def test_factory_contract_does_not_accept_test_dataset() -> None:
    parameters = tuple(inspect.signature(create_factory().create).parameters)

    assert parameters == (
        "training_dataset",
        "validation_dataset",
    )

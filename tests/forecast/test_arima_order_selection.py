import numpy as np
import pandas as pd
import pytest

from backend.src.forecast.arima_order_selection import (
    ArimaOrderSelectionResult,
    select_arima_orders,
)


def create_training_dataset() -> pd.DataFrame:
    observations_per_ticker = 60
    dates = pd.date_range(
        start="2024-01-01",
        periods=observations_per_ticker,
        freq="D",
    )
    observation_numbers = np.arange(
        observations_per_ticker,
        dtype=float,
    )

    return pd.concat(
        [
            pd.DataFrame(
                {
                    "ticker": "SPY",
                    "date": dates,
                    "target": (0.008 * np.sin(observation_numbers / 3.0)),
                }
            ),
            pd.DataFrame(
                {
                    "ticker": "QQQ",
                    "date": dates,
                    "target": (0.012 * np.cos(observation_numbers / 4.0)),
                }
            ),
        ],
        ignore_index=True,
    )


def test_select_arima_orders_returns_scores_and_selection_per_ticker() -> None:
    candidate_orders = (
        (0, 0, 0),
        (1, 0, 0),
    )

    result = select_arima_orders(
        training_dataset=create_training_dataset(),
        candidate_orders=candidate_orders,
        criterion="aic",
    )

    assert isinstance(
        result,
        ArimaOrderSelectionResult,
    )
    assert set(result.selected_orders) == {
        "QQQ",
        "SPY",
    }
    assert set(result.selected_orders.values()).issubset(set(candidate_orders))

    assert result.scores.columns.tolist() == [
        "ticker",
        "p",
        "d",
        "q",
        "aic",
        "bic",
        "converged",
        "selected",
        "error",
    ]
    assert len(result.scores) == 4
    assert result.scores["selected"].sum() == 2
    assert set(
        result.scores.loc[
            result.scores["selected"],
            "ticker",
        ]
    ) == {
        "QQQ",
        "SPY",
    }


def test_select_arima_orders_rejects_empty_candidates() -> None:
    with pytest.raises(
        ValueError,
        match="At least one candidate ARIMA order is required",
    ):
        select_arima_orders(
            training_dataset=create_training_dataset(),
            candidate_orders=(),
        )


def test_select_arima_orders_rejects_unknown_criterion() -> None:
    with pytest.raises(
        ValueError,
        match="criterion must be either 'aic' or 'bic'",
    ):
        select_arima_orders(
            training_dataset=create_training_dataset(),
            candidate_orders=((0, 0, 0),),
            criterion="rmse",
        )


def test_select_arima_orders_rejects_missing_columns() -> None:
    incomplete_dataset = create_training_dataset().drop(
        columns=["target"],
    )

    with pytest.raises(
        ValueError,
        match="Training dataset is missing required columns: target",
    ):
        select_arima_orders(
            training_dataset=incomplete_dataset,
            candidate_orders=((0, 0, 0),),
        )

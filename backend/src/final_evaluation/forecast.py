from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from backend.src.forecast.forecast_evaluator import (
    ForecastEvaluator,
)
from backend.src.forecast.forecast_result import (
    ForecastResult,
)
from backend.src.forecast.forecast_service import (
    ForecastService,
)
from backend.src.models.arima_forecast_model import (
    ArimaForecastModel,
    ArimaOrder,
)
from backend.src.models.naive_forecast_model import (
    NaiveForecastModel,
)
from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)

NAIVE_MODEL_NAME = "naive"
RANDOM_FOREST_MODEL_NAME = "random_forest"
ARIMA_MODEL_NAME = "arima_walk_forward"


def load_frozen_arima_orders(
    path: Path,
) -> dict[str, ArimaOrder]:
    payload = json.loads(
        Path(path).read_text(
            encoding="utf-8",
        )
    )

    selected_orders = payload.get("selected_orders")

    if not isinstance(
        selected_orders,
        Mapping,
    ):
        raise ValueError(
            "ARIMA selection artifact must contain a selected_orders mapping."
        )

    normalized_orders: dict[
        str,
        ArimaOrder,
    ] = {}

    for ticker, raw_order in selected_orders.items():
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("ARIMA ticker names must be non-empty strings.")

        if (
            not isinstance(raw_order, list | tuple)
            or len(raw_order) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in raw_order
            )
        ):
            raise ValueError(
                "Every frozen ARIMA order must contain three non-negative integers."
            )

        normalized_orders[ticker] = (
            int(raw_order[0]),
            int(raw_order[1]),
            int(raw_order[2]),
        )

    if not normalized_orders:
        raise ValueError("Frozen ARIMA order mapping must not be empty.")

    return dict(sorted(normalized_orders.items()))


def evaluate_final_forecasts(
    training_dataset: pd.DataFrame,
    evaluation_dataset: pd.DataFrame,
    arima_selection_path: Path,
) -> dict[str, ForecastResult]:
    frozen_arima_orders = load_frozen_arima_orders(
        path=arima_selection_path,
    )

    services = {
        NAIVE_MODEL_NAME: ForecastService(
            model=NaiveForecastModel(),
            evaluator=ForecastEvaluator(),
        ),
        RANDOM_FOREST_MODEL_NAME: ForecastService(
            model=RandomForestForecastModel(),
            evaluator=ForecastEvaluator(),
        ),
        ARIMA_MODEL_NAME: ForecastService(
            model=ArimaForecastModel(
                order=frozen_arima_orders,
            ),
            evaluator=ForecastEvaluator(),
        ),
    }

    return {
        model_name: service.forecast(
            training_dataset=training_dataset.copy(),
            evaluation_dataset=evaluation_dataset.copy(),
        )
        for model_name, service in services.items()
    }


def build_forecast_metrics_frame(
    results: Mapping[
        str,
        ForecastResult,
    ],
    evaluation_split: str,
) -> pd.DataFrame:
    rows = []

    for model_name, result in results.items():
        metrics = result.overall_metrics

        rows.append(
            {
                "model_name": model_name,
                "evaluation_split": evaluation_split,
                "observations": metrics.observations,
                "mae": metrics.mae,
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "r2": metrics.r2,
                "directional_accuracy": (metrics.directional_accuracy),
            }
        )

    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def build_forecast_observations_frame(
    results: Mapping[
        str,
        ForecastResult,
    ],
) -> pd.DataFrame:
    frames = []

    for model_name, result in results.items():
        observations = result.observations.copy()
        observations.insert(
            0,
            "model_name",
            model_name,
        )
        frames.append(observations)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backend.src.forecast.forecast_metrics import ForecastMetrics
from backend.src.forecast.forecast_result import ForecastResult


@dataclass(frozen=True, slots=True)
class ForecastExportPaths:
    observations_csv: Path
    metrics_csv: Path
    result_json: Path


def export_forecast_result(
    result: ForecastResult,
    output_directory: Path,
    model_name: str,
) -> ForecastExportPaths:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    observations_path = output_directory / "forecast_observations.csv"
    metrics_path = output_directory / "forecast_metrics.csv"
    result_path = output_directory / "forecast_result.json"

    observations = _prepare_observations(result.observations)
    metrics_rows = _build_metrics_rows(result)

    observations.to_csv(
        observations_path,
        index=False,
    )
    pd.DataFrame(metrics_rows).to_csv(
        metrics_path,
        index=False,
    )

    payload = {
        "schema_version": 1,
        "model_name": model_name,
        "evaluation_split": "validation",
        "target_definition": "next_session_simple_return",
        "metrics": {
            "overall": _metrics_to_dict(result.overall_metrics),
            "by_ticker": {
                ticker: _metrics_to_dict(metrics)
                for ticker, metrics in sorted(result.metrics_by_ticker.items())
            },
        },
        "observations": json.loads(
            observations.to_json(
                orient="records",
            )
        ),
    }

    result_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return ForecastExportPaths(
        observations_csv=observations_path,
        metrics_csv=metrics_path,
        result_json=result_path,
    )


def _prepare_observations(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    prepared = observations.copy()

    for column in (
        "date",
        "target_date",
    ):
        prepared[column] = pd.to_datetime(prepared[column]).dt.strftime("%Y-%m-%d")

    return prepared


def _build_metrics_rows(
    result: ForecastResult,
) -> list[dict[str, Any]]:
    rows = [
        {
            "scope": "overall",
            "ticker": "",
            **_metrics_to_dict(result.overall_metrics),
        }
    ]

    rows.extend(
        {
            "scope": "ticker",
            "ticker": ticker,
            **_metrics_to_dict(metrics),
        }
        for ticker, metrics in sorted(result.metrics_by_ticker.items())
    )

    return rows


def _metrics_to_dict(
    metrics: ForecastMetrics,
) -> dict[str, int | float | None]:
    return {
        "observations": int(metrics.observations),
        "mae": _finite_float_or_none(metrics.mae),
        "mse": _finite_float_or_none(metrics.mse),
        "rmse": _finite_float_or_none(metrics.rmse),
        "r2": _finite_float_or_none(metrics.r2),
        "directional_accuracy": _finite_float_or_none(metrics.directional_accuracy),
    }


def _finite_float_or_none(
    value: float,
) -> float | None:
    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        return None

    return numeric_value

from dataclasses import dataclass

import pandas as pd

from backend.src.forecast.forecast_metrics import ForecastMetrics


@dataclass(frozen=True, slots=True)
class ForecastResult:
    observations: pd.DataFrame
    overall_metrics: ForecastMetrics
    metrics_by_ticker: dict[str, ForecastMetrics]

    @property
    def predictions(self) -> pd.Series:
        return self.observations["prediction"]

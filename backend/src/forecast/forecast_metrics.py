from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    observations: int
    mae: float
    mse: float
    rmse: float
    r2: float
    directional_accuracy: float

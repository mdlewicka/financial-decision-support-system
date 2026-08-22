import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from backend.src.forecast.forecast_metrics import ForecastMetrics


class ForecastEvaluator:
    def evaluate(
        self,
        actual: pd.Series,
        predictions: pd.Series,
    ) -> ForecastMetrics:
        actual_directions = np.sign(actual.to_numpy(dtype=float))
        predicted_directions = np.sign(predictions.to_numpy(dtype=float))

        directional_accuracy = float((actual_directions == predicted_directions).mean())

        return ForecastMetrics(
            observations=len(actual),
            mae=float(
                mean_absolute_error(
                    actual,
                    predictions,
                )
            ),
            mse=float(
                mean_squared_error(
                    actual,
                    predictions,
                )
            ),
            rmse=float(
                root_mean_squared_error(
                    actual,
                    predictions,
                )
            ),
            r2=float(
                r2_score(
                    actual,
                    predictions,
                )
            ),
            directional_accuracy=directional_accuracy,
        )

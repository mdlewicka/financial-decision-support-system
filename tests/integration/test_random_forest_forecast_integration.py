import numpy as np
import pandas as pd

from backend.src.forecast.forecast_evaluator import (
    ForecastEvaluator,
)
from backend.src.forecast.forecast_service import (
    ForecastService,
)
from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)


def create_market_dataset(
    rows_per_ticker: int,
    start_date: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for ticker_index, ticker in enumerate(
        (
            "SPY",
            "QQQ",
        )
    ):
        dates = pd.date_range(
            start=start_date,
            periods=rows_per_ticker,
            freq="D",
        )
        base_price = 100.0 * (ticker_index + 1)

        for index, current_date in enumerate(dates):
            adjusted_close = base_price + float(index)
            daily_return = ((index % 5) - 2) * 0.003
            momentum_5 = ((index % 6) - 2.5) * 0.004
            momentum_20 = ((index % 8) - 3.5) * 0.005

            rows.append(
                {
                    "date": current_date,
                    "ticker": ticker,
                    "open": (adjusted_close * 0.999),
                    "high": (adjusted_close * 1.005),
                    "low": (adjusted_close * 0.995),
                    "close": adjusted_close,
                    "adjusted_close": (adjusted_close),
                    "daily_return": (daily_return),
                    "volume_change": (((index % 4) - 1.5) * 0.02),
                    "moving_average_5": (adjusted_close / (1.0 + momentum_5)),
                    "moving_average_20": (adjusted_close / (1.0 + momentum_20)),
                    "volatility_5": (0.010 + (index % 3) * 0.002),
                    "volatility_20": (0.015 + (index % 4) * 0.002),
                    "momentum_5": momentum_5,
                    "momentum_20": momentum_20,
                    "target_date": (current_date + pd.Timedelta(days=1)),
                    "target": (
                        0.35 * momentum_5 - 0.20 * daily_return + ticker_index * 0.0005
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_random_forest_runs_complete_forecast_service_flow() -> None:
    training_dataset = create_market_dataset(
        rows_per_ticker=15,
        start_date="2024-01-01",
    )
    validation_dataset = create_market_dataset(
        rows_per_ticker=4,
        start_date="2024-03-01",
    )

    service = ForecastService(
        model=RandomForestForecastModel(
            n_estimators=20,
            max_depth=4,
            min_samples_leaf=2,
            minimum_observations_per_ticker=10,
            random_state=42,
            n_jobs=1,
        ),
        evaluator=ForecastEvaluator(),
    )

    result = service.forecast(
        training_dataset=training_dataset,
        evaluation_dataset=validation_dataset,
    )

    assert result.overall_metrics.observations == 8
    assert set(result.metrics_by_ticker) == {
        "QQQ",
        "SPY",
    }
    assert len(result.observations) == 8
    assert np.isfinite(result.observations["prediction"].to_numpy()).all()

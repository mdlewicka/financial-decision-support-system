import pandas as pd

from backend.src.forecast.forecast_metrics import ForecastMetrics
from backend.src.forecast.forecast_pipeline import ForecastPipeline
from backend.src.forecast.forecast_result import ForecastResult
from backend.src.forecast.forecast_service import ForecastService
from backend.src.storage.market_dataset_reader import MarketDatasetReader


class StubMarketDatasetReader(MarketDatasetReader):
    def __init__(
        self,
        training_dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
    ) -> None:
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.was_called = False

    def read(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.was_called = True

        return (
            self.training_dataset,
            self.validation_dataset,
            self.test_dataset,
        )


class StubForecastService(ForecastService):
    def __init__(
        self,
        result: ForecastResult,
    ) -> None:
        self.result = result
        self.received_training_dataset: pd.DataFrame | None = None
        self.received_evaluation_dataset: pd.DataFrame | None = None

    def forecast(
        self,
        training_dataset: pd.DataFrame,
        evaluation_dataset: pd.DataFrame,
    ) -> ForecastResult:
        self.received_training_dataset = training_dataset
        self.received_evaluation_dataset = evaluation_dataset

        return self.result


def test_run_reads_datasets_and_forecasts_validation_dataset() -> None:
    training_dataset = pd.DataFrame(
        {
            "target": [0.01, -0.02, 0.03],
        }
    )
    validation_dataset = pd.DataFrame(
        {
            "target": [0.02, -0.01],
        }
    )
    test_dataset = pd.DataFrame(
        {
            "target": [0.04, -0.03],
        }
    )

    expected_metrics = ForecastMetrics(
        observations=2,
        mae=0.015,
        mse=0.00025,
        rmse=0.015811,
        r2=0.0,
        directional_accuracy=0.5,
    )
    expected_result = ForecastResult(
        observations=pd.DataFrame(
            {
                "ticker": ["SPY", "SPY"],
                "date": pd.to_datetime(
                    [
                        "2024-01-08",
                        "2024-01-09",
                    ]
                ),
                "target_date": pd.to_datetime(
                    [
                        "2024-01-09",
                        "2024-01-10",
                    ]
                ),
                "actual": [0.02, -0.01],
                "prediction": [0.0, 0.0],
                "absolute_error": [0.02, 0.01],
                "actual_direction": ["up", "down"],
                "predicted_direction": ["flat", "flat"],
                "direction_correct": [False, False],
            }
        ),
        overall_metrics=expected_metrics,
        metrics_by_ticker={
            "SPY": expected_metrics,
        },
    )

    dataset_reader = StubMarketDatasetReader(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
    )
    forecast_service = StubForecastService(
        result=expected_result,
    )
    pipeline = ForecastPipeline(
        dataset_reader=dataset_reader,
        forecast_service=forecast_service,
    )

    result = pipeline.run()

    assert dataset_reader.was_called
    assert forecast_service.received_training_dataset is training_dataset
    assert forecast_service.received_evaluation_dataset is validation_dataset
    assert forecast_service.received_evaluation_dataset is not test_dataset
    assert result is expected_result

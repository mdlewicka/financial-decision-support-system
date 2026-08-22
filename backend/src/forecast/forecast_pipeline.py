from backend.src.forecast.forecast_result import ForecastResult
from backend.src.forecast.forecast_service import ForecastService
from backend.src.storage.market_dataset_reader import MarketDatasetReader


class ForecastPipeline:
    def __init__(
        self,
        dataset_reader: MarketDatasetReader,
        forecast_service: ForecastService,
    ) -> None:
        self.dataset_reader = dataset_reader
        self.forecast_service = forecast_service

    def run(self) -> ForecastResult:
        training_dataset, validation_dataset, _ = self.dataset_reader.read()

        return self.forecast_service.forecast(
            training_dataset=training_dataset,
            evaluation_dataset=validation_dataset,
        )

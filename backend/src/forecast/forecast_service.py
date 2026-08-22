import pandas as pd

from backend.src.forecast.forecast_evaluator import ForecastEvaluator
from backend.src.forecast.forecast_metrics import ForecastMetrics
from backend.src.forecast.forecast_result import ForecastResult
from backend.src.models.base_forecast_model import (
    BaseForecastModel,
    BaseOnlineForecastModel,
)


class ForecastService:
    REQUIRED_EVALUATION_COLUMNS = frozenset(
        {
            "ticker",
            "date",
            "target_date",
            "target",
        }
    )

    def __init__(
        self,
        model: BaseForecastModel,
        evaluator: ForecastEvaluator,
    ) -> None:
        self.model = model
        self.evaluator = evaluator

    def forecast(
        self,
        training_dataset: pd.DataFrame,
        evaluation_dataset: pd.DataFrame,
    ) -> ForecastResult:
        self._validate_evaluation_dataset(evaluation_dataset)

        self.model.fit(training_dataset)

        predictions = self._generate_predictions(
            evaluation_dataset=evaluation_dataset,
        )

        if not predictions.index.equals(evaluation_dataset.index):
            raise ValueError(
                "Prediction index must match the evaluation dataset index."
            )

        observations = self._build_observations(
            evaluation_dataset=evaluation_dataset,
            predictions=predictions,
        )

        overall_metrics = self.evaluator.evaluate(
            actual=observations["actual"],
            predictions=observations["prediction"],
        )
        metrics_by_ticker = self._evaluate_by_ticker(observations)

        return ForecastResult(
            observations=observations,
            overall_metrics=overall_metrics,
            metrics_by_ticker=metrics_by_ticker,
        )

    def _generate_predictions(
        self,
        evaluation_dataset: pd.DataFrame,
    ) -> pd.Series:
        if isinstance(
            self.model,
            BaseOnlineForecastModel,
        ):
            return self._generate_online_predictions(
                evaluation_dataset=evaluation_dataset,
                model=self.model,
            )

        return self.model.predict(evaluation_dataset)

    def _generate_online_predictions(
        self,
        evaluation_dataset: pd.DataFrame,
        model: BaseOnlineForecastModel,
    ) -> pd.Series:
        predictions = pd.Series(
            index=evaluation_dataset.index,
            dtype=float,
        )

        for _, ticker_data in evaluation_dataset.groupby(
            "ticker",
            sort=False,
        ):
            ordered_ticker_data = ticker_data.sort_values(
                by="date",
                kind="stable",
            )

            for row_index in ordered_ticker_data.index:
                observed_row = evaluation_dataset.loc[[row_index]].copy()

                prediction_input = observed_row.drop(
                    columns=["target"],
                )

                row_prediction = model.predict(prediction_input)

                if not row_prediction.index.equals(prediction_input.index):
                    raise ValueError(
                        "Online prediction index must match "
                        "the current observation index."
                    )

                predictions.loc[row_index] = float(row_prediction.iloc[0])

                model.update(observed_row)

        return predictions

    def _validate_evaluation_dataset(
        self,
        evaluation_dataset: pd.DataFrame,
    ) -> None:
        if evaluation_dataset.empty:
            raise ValueError("Evaluation dataset must not be empty.")

        if not evaluation_dataset.index.is_unique:
            raise ValueError("Evaluation dataset index must be unique.")

        missing_columns = sorted(
            self.REQUIRED_EVALUATION_COLUMNS.difference(evaluation_dataset.columns)
        )

        if missing_columns:
            joined_columns = ", ".join(missing_columns)
            raise ValueError(
                f"Evaluation dataset is missing required columns: {joined_columns}."
            )

    def _build_observations(
        self,
        evaluation_dataset: pd.DataFrame,
        predictions: pd.Series,
    ) -> pd.DataFrame:
        observations = evaluation_dataset[
            [
                "ticker",
                "date",
                "target_date",
            ]
        ].copy()

        observations["actual"] = evaluation_dataset["target"].astype(float)
        observations["prediction"] = predictions.astype(float)
        observations["absolute_error"] = (
            observations["actual"] - observations["prediction"]
        ).abs()

        observations["actual_direction"] = self._classify_direction(
            observations["actual"]
        )
        observations["predicted_direction"] = self._classify_direction(
            observations["prediction"]
        )
        observations["direction_correct"] = (
            observations["actual_direction"] == observations["predicted_direction"]
        )

        return observations

    @staticmethod
    def _classify_direction(
        values: pd.Series,
    ) -> pd.Series:
        directions = pd.Series(
            data="flat",
            index=values.index,
            dtype="string",
        )
        directions.loc[values > 0] = "up"
        directions.loc[values < 0] = "down"

        return directions

    def _evaluate_by_ticker(
        self,
        observations: pd.DataFrame,
    ) -> dict[str, ForecastMetrics]:
        metrics_by_ticker: dict[str, ForecastMetrics] = {}

        for ticker, ticker_observations in observations.groupby(
            "ticker",
            sort=True,
        ):
            metrics_by_ticker[str(ticker)] = self.evaluator.evaluate(
                actual=ticker_observations["actual"],
                predictions=ticker_observations["prediction"],
            )

        return metrics_by_ticker

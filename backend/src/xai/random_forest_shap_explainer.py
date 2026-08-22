from typing import Any

import numpy as np
import pandas as pd
import shap

from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)
from backend.src.xai.xai_result import (
    GlobalShapExplanation,
    LocalShapExplanation,
)


class RandomForestShapExplainer:
    """Explain the fitted Random Forest forecast model using Tree SHAP."""

    def __init__(
        self,
        model: RandomForestForecastModel,
        background_dataset: pd.DataFrame,
        max_background_rows: int = 100,
        random_state: int = 42,
        additivity_tolerance: float = 1e-6,
    ) -> None:
        if background_dataset.empty:
            raise ValueError(
                "background_dataset must not be empty.",
            )

        if (
            isinstance(max_background_rows, bool)
            or not isinstance(max_background_rows, int)
            or max_background_rows <= 0
        ):
            raise ValueError(
                "max_background_rows must be a positive integer.",
            )

        if additivity_tolerance <= 0:
            raise ValueError(
                "additivity_tolerance must be greater than zero.",
            )

        self._model = model
        self._random_state = random_state
        self._additivity_tolerance = additivity_tolerance

        transformed_background = self._transform_dataset(
            background_dataset,
        )

        self._background_feature_values = self._sample_background_data(
            transformed_background,
            max_background_rows=max_background_rows,
        )

        self._explainer: Any = shap.TreeExplainer(
            self._model.estimator,
            data=self._background_feature_values,
            feature_perturbation="interventional",
            model_output="raw",
        )

    @property
    def background_feature_values(
        self,
    ) -> pd.DataFrame:
        """Return a defensive copy of the transformed background data."""
        return self._background_feature_values.copy(
            deep=True,
        )

    def explain_local(
        self,
        observation: pd.DataFrame,
    ) -> LocalShapExplanation:
        """Explain exactly one forecast observation."""
        if len(observation) != 1:
            raise ValueError(
                "observation must contain exactly one row.",
            )

        (
            feature_values,
            feature_contributions,
            base_values,
            predictions,
        ) = self._calculate_explanations(
            observation,
        )

        local_feature_values = feature_values.iloc[0].copy().rename("feature_value")
        local_feature_contributions = (
            feature_contributions.iloc[0].copy().rename("shap_value")
        )

        return LocalShapExplanation(
            feature_values=local_feature_values,
            feature_contributions=(local_feature_contributions),
            base_value=float(base_values.iloc[0]),
            prediction=float(predictions.iloc[0]),
        )

    def explain_global(
        self,
        observations: pd.DataFrame,
    ) -> GlobalShapExplanation:
        """Explain multiple forecast observations."""
        if observations.empty:
            raise ValueError(
                "observations must not be empty.",
            )

        (
            feature_values,
            feature_contributions,
            base_values,
            predictions,
        ) = self._calculate_explanations(
            observations,
        )

        mean_absolute_contributions = (
            feature_contributions.abs()
            .mean(axis=0)
            .sort_values(
                ascending=False,
            )
            .rename("mean_absolute_shap_value")
        )

        return GlobalShapExplanation(
            feature_values=feature_values,
            feature_contributions=feature_contributions,
            base_values=base_values,
            predictions=predictions,
            mean_absolute_contributions=(mean_absolute_contributions),
        )

    def _calculate_explanations(
        self,
        dataset: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:
        feature_values = self._transform_dataset(
            dataset,
        )

        shap_explanation = self._explainer(
            feature_values,
            check_additivity=True,
        )

        contribution_values = self._normalize_contributions(
            shap_explanation.values,
            expected_shape=feature_values.shape,
        )
        base_value_array = self._normalize_base_values(
            shap_explanation.base_values,
            observation_count=len(feature_values),
        )

        prediction_array = np.asarray(
            self._model.estimator.predict(
                feature_values,
            ),
            dtype=float,
        ).reshape(-1)

        if len(prediction_array) != len(feature_values):
            raise RuntimeError(
                "The estimator returned an invalid number of predictions.",
            )

        reconstructed_predictions = base_value_array + contribution_values.sum(axis=1)

        self._validate_additivity(
            predictions=prediction_array,
            reconstructed_predictions=(reconstructed_predictions),
        )

        feature_contributions = pd.DataFrame(
            contribution_values,
            index=feature_values.index,
            columns=feature_values.columns,
            dtype=float,
        )
        base_values = pd.Series(
            base_value_array,
            index=feature_values.index,
            dtype=float,
            name="base_value",
        )
        predictions = pd.Series(
            prediction_array,
            index=feature_values.index,
            dtype=float,
            name="prediction",
        )

        return (
            feature_values,
            feature_contributions,
            base_values,
            predictions,
        )

    def _transform_dataset(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        if dataset.empty:
            raise ValueError(
                "dataset must not be empty.",
            )

        transformed = self._model.transform_features(
            dataset,
        )

        if not isinstance(transformed, pd.DataFrame):
            raise TypeError(
                "model.transform_features must return a pandas DataFrame.",
            )

        expected_columns = list(
            self._model.feature_names,
        )

        if list(transformed.columns) != expected_columns:
            raise RuntimeError(
                "Transformed feature columns do not match "
                "the fitted model feature contract.",
            )

        numeric_features = transformed.astype(
            float,
        )

        if not np.isfinite(
            numeric_features.to_numpy(
                dtype=float,
            )
        ).all():
            raise ValueError(
                "Transformed features must contain only finite numeric values.",
            )

        return numeric_features.copy(
            deep=True,
        )

    def _sample_background_data(
        self,
        background_data: pd.DataFrame,
        max_background_rows: int,
    ) -> pd.DataFrame:
        if len(background_data) <= max_background_rows:
            return background_data.copy(
                deep=True,
            )

        return (
            background_data.sample(
                n=max_background_rows,
                random_state=self._random_state,
                replace=False,
            )
            .sort_index()
            .copy(
                deep=True,
            )
        )

    @staticmethod
    def _normalize_contributions(
        values: object,
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        contributions = np.asarray(
            values,
            dtype=float,
        )

        if contributions.ndim == 3 and contributions.shape[-1] == 1:
            contributions = contributions[
                :,
                :,
                0,
            ]

        if contributions.shape != expected_shape:
            raise RuntimeError(
                "SHAP contribution dimensions do not match "
                "the transformed feature matrix.",
            )

        return contributions

    @staticmethod
    def _normalize_base_values(
        values: object,
        observation_count: int,
    ) -> np.ndarray:
        base_values = np.asarray(
            values,
            dtype=float,
        )

        if base_values.ndim == 0:
            return np.repeat(
                float(base_values),
                observation_count,
            )

        base_values = base_values.reshape(-1)

        if len(base_values) == 1 and observation_count > 1:
            return np.repeat(
                float(base_values[0]),
                observation_count,
            )

        if len(base_values) != observation_count:
            raise RuntimeError(
                "SHAP base values do not match the number of explained observations.",
            )

        return base_values

    def _validate_additivity(
        self,
        predictions: np.ndarray,
        reconstructed_predictions: np.ndarray,
    ) -> None:
        if np.allclose(
            predictions,
            reconstructed_predictions,
            rtol=0.0,
            atol=self._additivity_tolerance,
        ):
            return

        maximum_error = float(np.max(np.abs(predictions - reconstructed_predictions)))

        raise RuntimeError(
            "SHAP values do not reconstruct the model "
            f"predictions. Maximum error: {maximum_error:.12f}.",
        )

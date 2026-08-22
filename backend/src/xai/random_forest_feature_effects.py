import numpy as np
import pandas as pd

from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)
from backend.src.xai.xai_result import (
    AccumulatedLocalEffectsResult,
    PartialDependenceResult,
)


class RandomForestFeatureEffectAnalyzer:
    """Calculate PDP and ALE for a fitted Random Forest model."""

    def __init__(
        self,
        model: RandomForestForecastModel,
    ) -> None:
        self._model = model

    def partial_dependence(
        self,
        dataset: pd.DataFrame,
        feature_name: str,
        grid_points: int = 20,
        quantile_range: tuple[float, float] = (
            0.05,
            0.95,
        ),
    ) -> PartialDependenceResult:
        """Calculate one-dimensional partial dependence."""
        self._validate_positive_integer(
            value=grid_points,
            parameter_name="grid_points",
            minimum=2,
        )
        self._validate_quantile_range(
            quantile_range,
        )

        feature_values = self._transform_dataset(
            dataset,
        )
        self._validate_continuous_feature(
            feature_values,
            feature_name,
        )

        lower_quantile, upper_quantile = quantile_range
        selected_feature = feature_values[feature_name]

        lower_value = float(
            selected_feature.quantile(
                lower_quantile,
            )
        )
        upper_value = float(
            selected_feature.quantile(
                upper_quantile,
            )
        )

        if lower_value >= upper_value:
            raise ValueError(
                f"Feature {feature_name!r} does not have "
                "a usable partial dependence range.",
            )

        grid = np.linspace(
            lower_value,
            upper_value,
            num=grid_points,
            dtype=float,
        )

        average_predictions = np.empty(
            grid_points,
            dtype=float,
        )

        for grid_index, grid_value in enumerate(grid):
            modified_features = feature_values.copy(
                deep=True,
            )
            modified_features[feature_name] = grid_value

            predictions = self._predict(
                modified_features,
            )
            average_predictions[grid_index] = float(
                predictions.mean(),
            )

        result_index = pd.RangeIndex(
            start=0,
            stop=grid_points,
        )

        return PartialDependenceResult(
            feature_name=feature_name,
            feature_values=pd.Series(
                grid,
                index=result_index,
                dtype=float,
                name="feature_value",
            ),
            average_predictions=pd.Series(
                average_predictions,
                index=result_index,
                dtype=float,
                name="average_prediction",
            ),
        )

    def accumulated_local_effects(
        self,
        dataset: pd.DataFrame,
        feature_name: str,
        bins: int = 10,
    ) -> AccumulatedLocalEffectsResult:
        """Calculate one-dimensional accumulated local effects."""
        self._validate_positive_integer(
            value=bins,
            parameter_name="bins",
            minimum=2,
        )

        feature_values = self._transform_dataset(
            dataset,
        )
        self._validate_continuous_feature(
            feature_values,
            feature_name,
        )

        selected_feature = feature_values[feature_name].to_numpy(
            dtype=float,
        )

        bin_edges = np.unique(
            np.quantile(
                selected_feature,
                np.linspace(
                    0.0,
                    1.0,
                    num=bins + 1,
                    dtype=float,
                ),
            )
        )

        if len(bin_edges) < 3:
            raise ValueError(
                f"Feature {feature_name!r} does not have "
                "enough distinct values for ALE.",
            )

        bin_indices = np.searchsorted(
            bin_edges[1:-1],
            selected_feature,
            side="right",
        )

        local_effects: list[float] = []
        sample_counts: list[int] = []
        effective_lower_edges: list[float] = []
        effective_upper_edges: list[float] = []

        for bin_index in range(len(bin_edges) - 1):
            row_mask = bin_indices == bin_index
            count = int(row_mask.sum())

            if count == 0:
                continue

            lower_edge = float(
                bin_edges[bin_index],
            )
            upper_edge = float(
                bin_edges[bin_index + 1],
            )

            lower_features = feature_values.loc[row_mask].copy(
                deep=True,
            )
            upper_features = lower_features.copy(
                deep=True,
            )

            lower_features[feature_name] = lower_edge
            upper_features[feature_name] = upper_edge

            lower_predictions = self._predict(
                lower_features,
            )
            upper_predictions = self._predict(
                upper_features,
            )

            mean_local_effect = float((upper_predictions - lower_predictions).mean())

            local_effects.append(
                mean_local_effect,
            )
            sample_counts.append(count)
            effective_lower_edges.append(
                lower_edge,
            )
            effective_upper_edges.append(
                upper_edge,
            )

        if len(local_effects) < 2:
            raise ValueError(
                f"Feature {feature_name!r} does not produce "
                "enough populated ALE intervals.",
            )

        local_effect_array = np.asarray(
            local_effects,
            dtype=float,
        )
        count_array = np.asarray(
            sample_counts,
            dtype=int,
        )
        lower_edge_array = np.asarray(
            effective_lower_edges,
            dtype=float,
        )
        upper_edge_array = np.asarray(
            effective_upper_edges,
            dtype=float,
        )

        accumulated_effect_array = np.cumsum(
            local_effect_array,
        )
        centering_value = float(
            np.average(
                accumulated_effect_array,
                weights=count_array,
            )
        )
        accumulated_effect_array = accumulated_effect_array - centering_value

        bin_center_array = (lower_edge_array + upper_edge_array) / 2.0

        result_index = pd.RangeIndex(
            start=0,
            stop=len(local_effect_array),
        )

        result_edges = np.concatenate(
            (
                lower_edge_array[:1],
                upper_edge_array,
            )
        )

        return AccumulatedLocalEffectsResult(
            feature_name=feature_name,
            bin_edges=pd.Series(
                result_edges,
                dtype=float,
                name="bin_edge",
            ),
            bin_centers=pd.Series(
                bin_center_array,
                index=result_index,
                dtype=float,
                name="bin_center",
            ),
            local_effects=pd.Series(
                local_effect_array,
                index=result_index,
                dtype=float,
                name="local_effect",
            ),
            accumulated_effects=pd.Series(
                accumulated_effect_array,
                index=result_index,
                dtype=float,
                name="accumulated_effect",
            ),
            sample_counts=pd.Series(
                count_array,
                index=result_index,
                dtype=int,
                name="sample_count",
            ),
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

    @staticmethod
    def _validate_continuous_feature(
        feature_values: pd.DataFrame,
        feature_name: str,
    ) -> None:
        if feature_name not in feature_values.columns:
            raise ValueError(
                f"Unknown model feature: {feature_name!r}.",
            )

        unique_values = np.unique(
            feature_values[feature_name].to_numpy(
                dtype=float,
            )
        )

        if len(unique_values) < 3:
            raise ValueError(
                f"Feature {feature_name!r} must be continuous "
                "and contain at least three distinct values.",
            )

    @staticmethod
    def _validate_positive_integer(
        value: int,
        parameter_name: str,
        minimum: int,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"{parameter_name} must be an integer "
                f"greater than or equal to {minimum}.",
            )

    @staticmethod
    def _validate_quantile_range(
        quantile_range: tuple[float, float],
    ) -> None:
        if len(quantile_range) != 2:
            raise ValueError(
                "quantile_range must contain exactly two values.",
            )

        lower_quantile, upper_quantile = quantile_range

        if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
            raise ValueError(
                "quantile_range must satisfy 0 <= lower < upper <= 1.",
            )

    def _predict(
        self,
        feature_values: pd.DataFrame,
    ) -> np.ndarray:
        predictions = np.asarray(
            self._model.estimator.predict(
                feature_values,
            ),
            dtype=float,
        ).reshape(-1)

        if len(predictions) != len(feature_values):
            raise RuntimeError(
                "The estimator returned an invalid number of predictions.",
            )

        if not np.isfinite(predictions).all():
            raise RuntimeError(
                "The estimator returned non-finite predictions.",
            )

        return predictions

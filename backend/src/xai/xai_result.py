from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LocalShapExplanation:
    """Store a SHAP explanation for one forecast."""

    feature_values: pd.Series
    feature_contributions: pd.Series
    base_value: float
    prediction: float

    @property
    def reconstructed_prediction(self) -> float:
        """Reconstruct the prediction from additive SHAP values."""
        return float(self.base_value + self.feature_contributions.sum())


@dataclass(frozen=True)
class GlobalShapExplanation:
    """Store SHAP explanations for multiple forecasts."""

    feature_values: pd.DataFrame
    feature_contributions: pd.DataFrame
    base_values: pd.Series
    predictions: pd.Series
    mean_absolute_contributions: pd.Series


@dataclass(frozen=True)
class PartialDependenceResult:
    """Store one-dimensional partial dependence results."""

    feature_name: str
    feature_values: pd.Series
    average_predictions: pd.Series


@dataclass(frozen=True)
class AccumulatedLocalEffectsResult:
    """Store one-dimensional accumulated local effects."""

    feature_name: str
    bin_edges: pd.Series
    bin_centers: pd.Series
    local_effects: pd.Series
    accumulated_effects: pd.Series
    sample_counts: pd.Series

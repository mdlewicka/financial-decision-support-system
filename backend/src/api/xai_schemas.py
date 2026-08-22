"""Schemas for explainable AI results exposed by the HTTP API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

XAIMethod = Literal[
    "shap",
    "partial_dependence",
    "accumulated_local_effects",
]

XAIExplanationSplit = Literal["validation"]
XAITargetDefinition = Literal["next_session_simple_return"]


class XAIMetadata(BaseModel):
    """Describe the provenance of an exported XAI analysis."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    model_name: Literal["random_forest"]
    training_split: Literal["training"]
    explanation_split: XAIExplanationSplit
    target_definition: XAITargetDefinition

    background_rows: int = Field(gt=0)
    global_explanation_rows: int = Field(gt=0)
    effect_feature_count: int = Field(gt=0)

    local_ticker: str = Field(min_length=1)
    local_date: datetime
    local_target_date: datetime
    local_actual: float
    local_prediction: float


class LocalShapFeatureArtifact(BaseModel):
    """Store one feature contribution in a local SHAP explanation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    feature: str = Field(min_length=1)
    feature_value: float
    shap_value: float


class LocalShapArtifact(BaseModel):
    """Store the exported local SHAP explanation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    base_value: float
    prediction: float
    reconstructed_prediction: float
    features: tuple[LocalShapFeatureArtifact, ...]

    @model_validator(mode="after")
    def validate_unique_features(
        self,
    ) -> "LocalShapArtifact":
        feature_names = tuple(feature.feature for feature in self.features)

        if not feature_names:
            raise ValueError(
                "Local SHAP explanation must contain at least one feature.",
            )

        if len(feature_names) != len(set(feature_names)):
            raise ValueError(
                "Local SHAP feature names must be unique.",
            )

        return self


class GlobalShapArtifact(BaseModel):
    """Store global SHAP feature importance."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    observations: int = Field(gt=0)
    mean_absolute_contributions: dict[str, float]

    @model_validator(mode="after")
    def validate_feature_importance(
        self,
    ) -> "GlobalShapArtifact":
        if not self.mean_absolute_contributions:
            raise ValueError(
                "Global SHAP importance must contain at least one feature.",
            )

        if any(
            not feature_name.strip()
            for feature_name in self.mean_absolute_contributions
        ):
            raise ValueError(
                "Global SHAP feature names must not be empty.",
            )

        if any(
            contribution < 0.0
            for contribution in self.mean_absolute_contributions.values()
        ):
            raise ValueError(
                "Mean absolute SHAP contributions must not be negative.",
            )

        return self


class PartialDependenceArtifact(BaseModel):
    """Store one-dimensional partial dependence values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    feature_values: tuple[float, ...]
    average_predictions: tuple[float, ...]

    @model_validator(mode="after")
    def validate_matching_lengths(
        self,
    ) -> "PartialDependenceArtifact":
        if not self.feature_values:
            raise ValueError(
                "Partial dependence result must not be empty.",
            )

        if len(self.feature_values) != len(self.average_predictions):
            raise ValueError(
                "Partial dependence feature values and predictions "
                "must have matching lengths.",
            )

        return self


class AccumulatedLocalEffectsArtifact(BaseModel):
    """Store one-dimensional accumulated local effects values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    bin_edges: tuple[float, ...]
    bin_centers: tuple[float, ...]
    accumulated_effects: tuple[float, ...]
    sample_counts: tuple[int, ...]

    @model_validator(mode="after")
    def validate_bin_structure(
        self,
    ) -> "AccumulatedLocalEffectsArtifact":
        bin_count = len(self.bin_centers)

        if bin_count == 0:
            raise ValueError(
                "Accumulated local effects result must not be empty.",
            )

        if len(self.bin_edges) != bin_count + 1:
            raise ValueError(
                "ALE bin edges must contain exactly one more value than bin centers.",
            )

        if len(self.accumulated_effects) != bin_count:
            raise ValueError(
                "ALE accumulated effects must match the number of bins.",
            )

        if len(self.sample_counts) != bin_count:
            raise ValueError(
                "ALE sample counts must match the number of bins.",
            )

        if any(sample_count < 0 for sample_count in self.sample_counts):
            raise ValueError(
                "ALE sample counts must not be negative.",
            )

        return self


class XAIArtifactPayload(BaseModel):
    """Validate the exported XAI JSON artifact."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )

    schema_version: Literal[1]
    metadata: XAIMetadata
    methods: tuple[XAIMethod, ...]
    local_shap: LocalShapArtifact
    global_shap: GlobalShapArtifact
    partial_dependence: dict[
        str,
        PartialDependenceArtifact,
    ]
    accumulated_local_effects: dict[
        str,
        AccumulatedLocalEffectsArtifact,
    ]
    plots: dict[str, str]

    @model_validator(mode="after")
    def validate_feature_effects(
        self,
    ) -> "XAIArtifactPayload":
        expected_methods = {
            "shap",
            "partial_dependence",
            "accumulated_local_effects",
        }

        if set(self.methods) != expected_methods:
            raise ValueError(
                "XAI artifact must contain SHAP, PDP and ALE methods.",
            )

        pdp_features = set(self.partial_dependence)
        ale_features = set(self.accumulated_local_effects)

        if not pdp_features:
            raise ValueError(
                "XAI artifact must contain at least one feature effect analysis.",
            )

        if pdp_features != ale_features:
            raise ValueError(
                "PDP and ALE feature sets must match.",
            )

        if len(pdp_features) != self.metadata.effect_feature_count:
            raise ValueError(
                "XAI effect feature count does not match metadata.",
            )

        global_features = set(self.global_shap.mean_absolute_contributions)

        if not pdp_features.issubset(global_features):
            raise ValueError(
                "All PDP and ALE features must also exist in global SHAP importance.",
            )

        return self


class XAIStatusResponse(BaseModel):
    """Describe availability of the XAI artifact."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    model_name: str
    artifact_available: bool


class XAIGlobalFeatureImportance(BaseModel):
    """Describe one feature in global SHAP importance."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    feature: str
    mean_absolute_shap_value: float = Field(
        ge=0.0,
    )
    effect_available: bool


class XAISummaryResponse(BaseModel):
    """Expose the global XAI summary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    model_name: str
    explanation_split: XAIExplanationSplit
    target_definition: XAITargetDefinition
    methods: tuple[XAIMethod, ...]
    global_observations: int = Field(gt=0)
    features: tuple[XAIGlobalFeatureImportance, ...]
    effect_features: tuple[str, ...]


class XAILocalFeatureContribution(BaseModel):
    """Expose one feature contribution in a local SHAP result."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    feature: str
    feature_value: float
    shap_value: float


class XAILocalExplanationResponse(BaseModel):
    """Expose one representative local SHAP explanation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    model_name: str
    ticker: str
    feature_date: datetime
    target_date: datetime
    actual_return: float
    predicted_return: float
    base_value: float
    reconstructed_prediction: float
    features: tuple[XAILocalFeatureContribution, ...]


class PartialDependencePoint(BaseModel):
    """Represent one point of a partial dependence curve."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    feature_value: float
    average_prediction: float


class AccumulatedLocalEffectsPoint(BaseModel):
    """Represent one bin of an accumulated local effects curve."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    lower_edge: float
    upper_edge: float
    bin_center: float
    accumulated_effect: float
    sample_count: int = Field(ge=0)


class XAIFeatureEffectResponse(BaseModel):
    """Expose PDP and ALE results for one analyzed feature."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    feature_name: str
    mean_absolute_shap_value: float = Field(
        ge=0.0,
    )
    partial_dependence: tuple[PartialDependencePoint, ...]
    accumulated_local_effects: tuple[AccumulatedLocalEffectsPoint, ...]

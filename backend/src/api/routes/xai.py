"""Explainable AI API routes."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from backend.src.api.config import API_V1_PREFIX
from backend.src.api.xai_repository import (
    XAIArtifactError,
    XAIArtifactRepository,
)
from backend.src.api.xai_schemas import (
    AccumulatedLocalEffectsPoint,
    PartialDependencePoint,
    XAIArtifactPayload,
    XAIFeatureEffectResponse,
    XAIGlobalFeatureImportance,
    XAILocalExplanationResponse,
    XAILocalFeatureContribution,
    XAIStatusResponse,
    XAISummaryResponse,
)

router = APIRouter(
    prefix=f"{API_V1_PREFIX}/xai",
    tags=["explainable-ai"],
)


def get_xai_repository() -> XAIArtifactRepository:
    """Create the repository used to read frozen XAI results."""
    return XAIArtifactRepository()


XAIRepositoryDependency = Annotated[
    XAIArtifactRepository,
    Depends(get_xai_repository),
]


@router.get(
    "/status",
    response_model=XAIStatusResponse,
    summary="Check XAI artifact availability",
)
def get_xai_status(
    repository: XAIRepositoryDependency,
) -> XAIStatusResponse:
    """Return availability of the Random Forest XAI artifact."""
    return XAIStatusResponse(
        model_name="random_forest",
        artifact_available=(repository.artifact_exists()),
    )


@router.get(
    "/summary",
    response_model=XAISummaryResponse,
    summary="Get global XAI summary",
)
def get_xai_summary(
    repository: XAIRepositoryDependency,
) -> XAISummaryResponse:
    """Return global SHAP importance and available effect features."""
    artifact = _load_or_service_unavailable(
        repository,
    )

    importance = sorted(
        artifact.global_shap.mean_absolute_contributions.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    effect_feature_set = set(artifact.partial_dependence)

    effect_features = tuple(
        feature_name
        for feature_name, _ in importance
        if feature_name in effect_feature_set
    )

    return XAISummaryResponse(
        model_name=artifact.metadata.model_name,
        explanation_split=(artifact.metadata.explanation_split),
        target_definition=(artifact.metadata.target_definition),
        methods=artifact.methods,
        global_observations=(artifact.global_shap.observations),
        features=tuple(
            XAIGlobalFeatureImportance(
                feature=feature_name,
                mean_absolute_shap_value=(mean_absolute_shap_value),
                effect_available=(feature_name in effect_feature_set),
            )
            for (
                feature_name,
                mean_absolute_shap_value,
            ) in importance
        ),
        effect_features=effect_features,
    )


@router.get(
    "/local",
    response_model=XAILocalExplanationResponse,
    summary="Get representative local SHAP explanation",
)
def get_local_xai_explanation(
    repository: XAIRepositoryDependency,
) -> XAILocalExplanationResponse:
    """Return the representative validation observation used for local SHAP."""
    artifact = _load_or_service_unavailable(
        repository,
    )

    ordered_features = sorted(
        artifact.local_shap.features,
        key=lambda feature: abs(feature.shap_value),
        reverse=True,
    )

    return XAILocalExplanationResponse(
        model_name=artifact.metadata.model_name,
        ticker=artifact.metadata.local_ticker,
        feature_date=artifact.metadata.local_date,
        target_date=(artifact.metadata.local_target_date),
        actual_return=(artifact.metadata.local_actual),
        predicted_return=(artifact.metadata.local_prediction),
        base_value=artifact.local_shap.base_value,
        reconstructed_prediction=(artifact.local_shap.reconstructed_prediction),
        features=tuple(
            XAILocalFeatureContribution(
                feature=feature.feature,
                feature_value=(feature.feature_value),
                shap_value=feature.shap_value,
            )
            for feature in ordered_features
        ),
    )


@router.get(
    "/features/{feature_name}",
    response_model=XAIFeatureEffectResponse,
    summary="Get PDP and ALE results for one feature",
)
def get_xai_feature_effect(
    feature_name: str,
    repository: XAIRepositoryDependency,
) -> XAIFeatureEffectResponse:
    """Return SHAP importance, PDP and ALE for one analyzed feature."""
    artifact = _load_or_service_unavailable(
        repository,
    )

    partial_dependence = artifact.partial_dependence.get(
        feature_name,
    )
    accumulated_local_effects = artifact.accumulated_local_effects.get(
        feature_name,
    )

    if partial_dependence is None or accumulated_local_effects is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(f"Feature effect analysis is not available for {feature_name!r}."),
        )

    mean_absolute_shap_value = artifact.global_shap.mean_absolute_contributions[
        feature_name
    ]

    pdp_points = tuple(
        PartialDependencePoint(
            feature_value=feature_value,
            average_prediction=average_prediction,
        )
        for (
            feature_value,
            average_prediction,
        ) in zip(
            partial_dependence.feature_values,
            partial_dependence.average_predictions,
            strict=True,
        )
    )

    ale_points = tuple(
        AccumulatedLocalEffectsPoint(
            lower_edge=(accumulated_local_effects.bin_edges[index]),
            upper_edge=(accumulated_local_effects.bin_edges[index + 1]),
            bin_center=(accumulated_local_effects.bin_centers[index]),
            accumulated_effect=(accumulated_local_effects.accumulated_effects[index]),
            sample_count=(accumulated_local_effects.sample_counts[index]),
        )
        for index in range(len(accumulated_local_effects.bin_centers))
    )

    return XAIFeatureEffectResponse(
        feature_name=feature_name,
        mean_absolute_shap_value=(mean_absolute_shap_value),
        partial_dependence=pdp_points,
        accumulated_local_effects=ale_points,
    )


def _load_or_service_unavailable(
    repository: XAIArtifactRepository,
) -> XAIArtifactPayload:
    try:
        return repository.load()
    except XAIArtifactError as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

"""Deterministic Decision Engine API routes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.src.api.config import API_V1_PREFIX
from backend.src.api.decision_schemas import (
    DecisionConstraintsPayload,
    DecisionEvaluationRequest,
    DecisionEvaluationResponse,
    DecisionNormalizationResponse,
    DecisionPrioritiesPayload,
    DecisionProfilePayload,
    DecisionProfileResponse,
    DecisionProfilesResponse,
    DecisionProvenanceResponse,
    DecisionScenarioResponse,
    DecisionWeightResponse,
)
from backend.src.api.portfolio_repository import (
    PortfolioArtifactError,
    PortfolioArtifactRepository,
)
from backend.src.decision.models import (
    DecisionConstraints,
    DecisionPriorities,
    DecisionProfile,
    DecisionProfileType,
)
from backend.src.decision.profiles import create_decision_profile
from backend.src.decision.service import (
    PortfolioDecisionEvaluation,
    PortfolioDecisionService,
)

router = APIRouter(prefix=f"{API_V1_PREFIX}/decision", tags=["decision"])


def get_decision_portfolio_repository() -> PortfolioArtifactRepository:
    """Create the repository providing frozen portfolio evidence."""
    return PortfolioArtifactRepository()


DecisionPortfolioRepositoryDependency = Annotated[
    PortfolioArtifactRepository,
    Depends(get_decision_portfolio_repository),
]


@router.get(
    "/profiles",
    response_model=DecisionProfilesResponse,
    summary="List deterministic decision profile presets",
)
def get_decision_profiles() -> DecisionProfilesResponse:
    profiles = tuple(
        create_decision_profile(profile_type)
        for profile_type in (
            DecisionProfileType.CONSERVATIVE,
            DecisionProfileType.BALANCED,
            DecisionProfileType.GROWTH,
        )
    )
    return DecisionProfilesResponse(
        profiles=tuple(_profile_response(profile) for profile in profiles)
    )


@router.get(
    "/recommendations/{profile_type}",
    response_model=DecisionEvaluationResponse,
    summary="Rank portfolio scenarios for a preset profile",
)
def get_preset_recommendation(
    profile_type: DecisionProfileType,
    repository: DecisionPortfolioRepositoryDependency,
    available_liquidity_fraction: Annotated[float, Query(ge=0.0, le=1.0)] = 0.10,
) -> DecisionEvaluationResponse:
    if profile_type is DecisionProfileType.CUSTOM:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Use POST /api/v1/decision/evaluate for a custom profile.",
        )
    return _evaluate(
        repository,
        create_decision_profile(profile_type),
        available_liquidity_fraction,
    )


@router.post(
    "/evaluate",
    response_model=DecisionEvaluationResponse,
    summary="Evaluate a preset or custom decision profile",
)
def evaluate_decision_profile(
    request: DecisionEvaluationRequest,
    repository: DecisionPortfolioRepositoryDependency,
) -> DecisionEvaluationResponse:
    return evaluate_decision_request(
        request,
        repository,
    )


def evaluate_decision_request(
    request: DecisionEvaluationRequest,
    repository: PortfolioArtifactRepository,
) -> DecisionEvaluationResponse:
    """Recompute one validated request from frozen portfolio evidence."""
    return _evaluate(
        repository,
        _domain_profile(request.profile),
        request.available_liquidity_fraction,
    )


def _evaluate(
    repository: PortfolioArtifactRepository,
    profile: DecisionProfile,
    available_liquidity_fraction: float,
) -> DecisionEvaluationResponse:
    try:
        artifact = repository.load()
    except PortfolioArtifactError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    evaluation = PortfolioDecisionService().evaluate(
        artifact,
        profile,
        available_liquidity_fraction=available_liquidity_fraction,
    )
    return _evaluation_response(evaluation)


def _domain_profile(payload: DecisionProfilePayload) -> DecisionProfile:
    if payload.profile_type is not DecisionProfileType.CUSTOM:
        return create_decision_profile(payload.profile_type)
    priorities = payload.priorities
    constraints = payload.constraints
    liquidity_need = payload.liquidity_need
    investment_horizon_years = payload.investment_horizon_years
    if (
        priorities is None
        or constraints is None
        or liquidity_need is None
        or investment_horizon_years is None
    ):
        raise ValueError("Validated custom profile fields are unexpectedly missing.")
    return create_decision_profile(
        DecisionProfileType.CUSTOM,
        priorities=_domain_priorities(priorities),
        constraints=_domain_constraints(constraints),
        liquidity_need=liquidity_need,
        investment_horizon_years=investment_horizon_years,
    )


def _domain_priorities(payload: DecisionPrioritiesPayload) -> DecisionPriorities:
    return DecisionPriorities(**payload.model_dump())


def _domain_constraints(payload: DecisionConstraintsPayload) -> DecisionConstraints:
    return DecisionConstraints(**payload.model_dump())


def _profile_response(profile: DecisionProfile) -> DecisionProfileResponse:
    return DecisionProfileResponse.model_validate(profile.to_dict())


def _evaluation_response(
    evaluation: PortfolioDecisionEvaluation,
) -> DecisionEvaluationResponse:
    contexts = {context.name: context for context in evaluation.source.contexts}
    decision = evaluation.decision
    return DecisionEvaluationResponse(
        deterministic=True,
        llm_used_for_ranking=False,
        portfolio_weights_modified=False,
        provenance=DecisionProvenanceResponse.model_validate(
            evaluation.source.provenance.to_dict()
        ),
        profile=_profile_response(decision.profile),
        available_liquidity_fraction=decision.available_liquidity_fraction,
        normalization=tuple(
            DecisionNormalizationResponse.model_validate(asdict(item))
            for item in decision.normalization
        ),
        recommended_scenario=decision.recommended_scenario,
        scenarios=tuple(
            DecisionScenarioResponse(
                name=scenario.name,
                construction=contexts[scenario.name].construction,
                weights=tuple(
                    DecisionWeightResponse(ticker=ticker, weight=weight)
                    for ticker, weight in contexts[scenario.name].weights
                ),
                admissible=scenario.admissible,
                rank=scenario.rank,
                total_score=scenario.total_score,
                normalized_components=scenario.normalized_components,
                weighted_contributions=scenario.weighted_contributions,
                constraint_violations=scenario.constraint_violations,
                tradeoffs=scenario.tradeoffs,
            )
            for scenario in decision.scenarios
        ),
    )

"""System-level API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from backend.src.api.config import (
    API_TITLE,
    API_V1_PREFIX,
    API_VERSION,
)
from backend.src.api.schemas import (
    APIInfoResponse,
    HealthResponse,
    OllamaStatusResponse,
    ReadinessCheckResponse,
    ReadinessResponse,
    RuntimeMetricsResponse,
    RuntimeStatusResponse,
)
from backend.src.observability.ollama_status import OllamaStatusService
from backend.src.observability.readiness import ReadinessService
from backend.src.observability.runtime_metrics import RuntimeMetrics

router = APIRouter(
    tags=["system"],
)


def get_readiness_service() -> ReadinessService:
    """Create the local artifact readiness evaluator."""
    return ReadinessService()


def get_ollama_status_service() -> OllamaStatusService:
    """Create the bounded optional local Ollama availability probe."""
    return OllamaStatusService()


ReadinessServiceDependency = Annotated[
    ReadinessService,
    Depends(get_readiness_service),
]
OllamaStatusServiceDependency = Annotated[
    OllamaStatusService,
    Depends(get_ollama_status_service),
]


def _readiness_response(service: ReadinessService) -> ReadinessResponse:
    result = service.evaluate()
    return ReadinessResponse(
        status=result.status,
        checks=tuple(
            ReadinessCheckResponse(
                name=check.name,
                status=check.status,
                required=check.required,
            )
            for check in result.checks
        ),
    )


def build_api_info_response() -> APIInfoResponse:
    """Build the API information response."""
    return APIInfoResponse(
        name=API_TITLE,
        version=API_VERSION,
        status="running",
    )


def build_health_response() -> HealthResponse:
    """Build the API health response."""
    return HealthResponse(
        status="healthy",
    )


@router.get(
    "/",
    response_model=APIInfoResponse,
    summary="Get API information",
)
def read_root() -> APIInfoResponse:
    """Return basic API information."""
    return build_api_info_response()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
)
def health_check() -> HealthResponse:
    """Return the compatibility health endpoint."""
    return build_health_response()


@router.get(
    f"{API_V1_PREFIX}/system/info",
    response_model=APIInfoResponse,
    summary="Get versioned API information",
)
def get_api_info() -> APIInfoResponse:
    """Return versioned API information."""
    return build_api_info_response()


@router.get(
    f"{API_V1_PREFIX}/system/health",
    response_model=HealthResponse,
    summary="Check versioned API health",
)
def get_system_health() -> HealthResponse:
    """Return the versioned API health status."""
    return build_health_response()


@router.get(
    f"{API_V1_PREFIX}/system/liveness",
    response_model=HealthResponse,
    summary="Check whether the API process is alive",
)
def get_system_liveness() -> HealthResponse:
    """Return process-level liveness without dependency checks."""
    return build_health_response()


@router.get(
    f"{API_V1_PREFIX}/system/readiness",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "Required local analytical evidence is unavailable.",
        }
    },
    summary="Check whether core local analytical evidence is ready",
)
def get_system_readiness(
    response: Response,
    readiness_service: ReadinessServiceDependency,
) -> ReadinessResponse:
    """Return dependency readiness without exposing local filesystem paths."""
    result = _readiness_response(readiness_service)
    if result.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get(
    f"{API_V1_PREFIX}/system/runtime-status",
    response_model=RuntimeStatusResponse,
    summary="Inspect local runtime metrics and optional services",
)
def get_runtime_status(
    request: Request,
    readiness_service: ReadinessServiceDependency,
    ollama_service: OllamaStatusServiceDependency,
) -> RuntimeStatusResponse:
    """Return safe aggregate status without secrets or filesystem paths."""
    metrics: RuntimeMetrics = request.app.state.runtime_metrics
    snapshot = metrics.snapshot()
    ollama = ollama_service.evaluate()
    return RuntimeStatusResponse(
        artifacts=_readiness_response(readiness_service),
        ollama=OllamaStatusResponse(
            status=ollama.status,
            required=ollama.required,
            fallback_available=ollama.fallback_available,
            model_name=ollama.model_name,
        ),
        metrics=RuntimeMetricsResponse(
            requests_total=snapshot.requests_total,
            errors_total=snapshot.errors_total,
            client_errors_total=snapshot.client_errors_total,
            server_errors_total=snapshot.server_errors_total,
            average_latency_ms=snapshot.average_latency_ms,
            maximum_latency_ms=snapshot.maximum_latency_ms,
        ),
    )

"""Shared response schemas for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class APIInfoResponse(BaseModel):
    """Describe the running API."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: str
    version: str
    status: Literal["running"]


class HealthResponse(BaseModel):
    """Describe API health."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: Literal["healthy"]


class ReadinessCheckResponse(BaseModel):
    """Describe one runtime dependency without exposing filesystem details."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: str
    status: Literal["ready", "missing"]
    required: bool


class ReadinessResponse(BaseModel):
    """Describe whether the API can serve its core analytical workflows."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: Literal["ready", "not_ready"]
    checks: tuple[ReadinessCheckResponse, ...]


class RuntimeMetricsResponse(BaseModel):
    """Expose bounded process-local aggregate HTTP metrics."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    requests_total: int
    errors_total: int
    client_errors_total: int
    server_errors_total: int
    average_latency_ms: float
    maximum_latency_ms: float


class OllamaStatusResponse(BaseModel):
    """Describe optional local generation without exposing its host."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: Literal["available", "unavailable"]
    required: bool
    fallback_available: bool
    model_name: str


class RuntimeStatusResponse(BaseModel):
    """Combine artifact, local model, and lightweight process status."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    artifacts: ReadinessResponse
    ollama: OllamaStatusResponse
    metrics: RuntimeMetricsResponse

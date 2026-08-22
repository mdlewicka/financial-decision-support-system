"""Local observability building blocks for the application runtime."""

from backend.src.observability.ollama_status import (
    OllamaRuntimeStatus,
    OllamaStatusService,
)
from backend.src.observability.readiness import (
    ReadinessCheck,
    ReadinessResult,
    ReadinessService,
)
from backend.src.observability.request_logging import (
    REQUEST_ID_HEADER,
    JSONLogFormatter,
    RequestLoggingMiddleware,
    configure_structured_logging,
)
from backend.src.observability.runtime_metrics import (
    RuntimeMetrics,
    RuntimeMetricsSnapshot,
)

__all__ = [
    "REQUEST_ID_HEADER",
    "JSONLogFormatter",
    "OllamaRuntimeStatus",
    "OllamaStatusService",
    "ReadinessCheck",
    "ReadinessResult",
    "ReadinessService",
    "RequestLoggingMiddleware",
    "RuntimeMetrics",
    "RuntimeMetricsSnapshot",
    "configure_structured_logging",
]

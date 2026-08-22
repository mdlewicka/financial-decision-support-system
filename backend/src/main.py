"""FastAPI application entry point for the financial decision support system."""

from fastapi import FastAPI

from backend.src.api.config import (
    API_TITLE,
    API_VERSION,
)
from backend.src.api.routes.agents import (
    router as agents_router,
)
from backend.src.api.routes.decision import (
    router as decision_router,
)
from backend.src.api.routes.forecast import (
    router as forecast_router,
)
from backend.src.api.routes.market_intelligence import (
    router as market_intelligence_router,
)
from backend.src.api.routes.portfolio import (
    router as portfolio_router,
)
from backend.src.api.routes.rag import (
    router as rag_router,
)
from backend.src.api.routes.reports import (
    router as reports_router,
)
from backend.src.api.routes.risk import (
    router as risk_router,
)
from backend.src.api.routes.system import (
    router as system_router,
)
from backend.src.api.routes.xai import (
    router as xai_router,
)
from backend.src.observability.request_logging import (
    RequestLoggingMiddleware,
    configure_structured_logging,
)
from backend.src.observability.runtime_metrics import RuntimeMetrics


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_structured_logging()
    application = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
    )
    runtime_metrics = RuntimeMetrics()
    application.state.runtime_metrics = runtime_metrics
    application.add_middleware(
        RequestLoggingMiddleware,
        metrics=runtime_metrics,
    )

    application.include_router(
        system_router,
    )
    application.include_router(
        forecast_router,
    )
    application.include_router(
        market_intelligence_router,
    )
    application.include_router(
        xai_router,
    )
    application.include_router(
        risk_router,
    )
    application.include_router(
        portfolio_router,
    )
    application.include_router(
        decision_router,
    )
    application.include_router(
        rag_router,
    )
    application.include_router(
        agents_router,
    )
    application.include_router(
        reports_router,
    )

    return application


app = create_app()

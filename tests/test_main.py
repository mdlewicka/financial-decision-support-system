from pathlib import Path

from fastapi.testclient import TestClient

from backend.src.api.config import (
    API_TITLE,
    API_VERSION,
)
from backend.src.api.routes.system import (
    get_ollama_status_service,
    get_readiness_service,
)
from backend.src.main import (
    app,
    create_app,
)
from backend.src.observability.ollama_status import OllamaRuntimeStatus
from backend.src.observability.readiness import ReadinessService

client = TestClient(app)


def test_application_metadata_is_explicit() -> None:
    application = create_app()

    assert application.title == API_TITLE
    assert application.version == API_VERSION


def test_public_api_contains_only_supported_application_routes() -> None:
    openapi_schema = app.openapi()
    route_paths = set(openapi_schema["paths"])

    supported_routes = {
        "/",
        "/health",
        "/api/v1/system/info",
        "/api/v1/system/health",
        "/api/v1/system/liveness",
        "/api/v1/system/readiness",
        "/api/v1/system/runtime-status",
    }

    unsupported_routes = {
        "/portfolio/analyze",
        "/portfolio/test-return",
        "/portfolio/test-volatility",
        "/portfolio/test-covariance",
        "/market/latest/{symbol}",
        "/market/latest/{symbol:path}",
        "/market/history/{symbol}",
        "/market/history/{symbol:path}",
        "/market/analytics/{symbol}",
        "/market/analytics/{symbol:path}",
        "/portfolio/market-analysis",
    }

    assert supported_routes.issubset(route_paths)
    assert unsupported_routes.isdisjoint(route_paths)


def test_root_returns_api_information() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": API_TITLE,
        "version": API_VERSION,
        "status": "running",
    }


def test_versioned_system_info_matches_root_contract() -> None:
    root_response = client.get("/")
    versioned_response = client.get(
        "/api/v1/system/info",
    )

    assert root_response.status_code == 200
    assert versioned_response.status_code == 200
    assert versioned_response.json() == root_response.json()


def test_legacy_health_endpoint_returns_healthy_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }


def test_versioned_health_endpoint_matches_legacy_contract() -> None:
    legacy_response = client.get("/health")
    versioned_response = client.get(
        "/api/v1/system/health",
    )

    assert legacy_response.status_code == 200
    assert versioned_response.status_code == 200
    assert versioned_response.json() == legacy_response.json()


def test_liveness_matches_compatibility_health_contract() -> None:
    response = client.get("/api/v1/system/liveness")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_readiness_reports_core_local_artifacts_without_paths() -> None:
    response = client.get("/api/v1/system/readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert {check["name"] for check in payload["checks"]} == {
        "forecast_evidence",
        "portfolio_evidence",
        "rag_corpus",
        "agent_evidence",
    }
    assert all(check["status"] == "ready" for check in payload["checks"])
    assert "/Users/" not in response.text
    assert "/app/" not in response.text


def test_readiness_returns_service_unavailable_when_required_artifact_is_missing(
    tmp_path: Path,
) -> None:
    app.dependency_overrides[get_readiness_service] = lambda: ReadinessService(
        artifacts=(("forecast_evidence", tmp_path / "missing.json"),)
    )
    try:
        response = client.get("/api/v1/system/readiness")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": [
            {
                "name": "forecast_evidence",
                "status": "missing",
                "required": True,
            }
        ],
    }


def test_request_id_is_returned_and_untrusted_value_is_rejected() -> None:
    accepted = client.get(
        "/api/v1/system/health",
        headers={"X-Request-ID": "portfolio-check-123"},
    )
    rejected = client.get(
        "/api/v1/system/health",
        headers={"X-Request-ID": "unsafe request id"},
    )

    assert accepted.headers["X-Request-ID"] == "portfolio-check-123"
    generated = rejected.headers["X-Request-ID"]
    assert generated != "unsafe request id"
    assert len(generated) == 32


class AvailableOllamaStatusService:
    def evaluate(self) -> OllamaRuntimeStatus:
        return OllamaRuntimeStatus(
            status="available",
            required=False,
            fallback_available=True,
            model_name="local-test-model",
        )


def test_runtime_status_reports_safe_optional_services_and_metrics() -> None:
    app.dependency_overrides[get_ollama_status_service] = AvailableOllamaStatusService
    try:
        response = client.get("/api/v1/system/runtime-status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifacts"]["status"] == "ready"
    assert payload["ollama"] == {
        "status": "available",
        "required": False,
        "fallback_available": True,
        "model_name": "local-test-model",
    }
    assert payload["metrics"]["requests_total"] >= 1
    assert payload["metrics"]["errors_total"] >= 0
    assert payload["metrics"]["average_latency_ms"] >= 0.0
    assert "OLLAMA_HOST" not in response.text
    assert "/Users/" not in response.text


def test_runtime_status_counts_http_errors() -> None:
    application = create_app()
    application.dependency_overrides[get_ollama_status_service] = (
        AvailableOllamaStatusService
    )
    isolated_client = TestClient(application)

    assert isolated_client.get("/api/v1/unknown").status_code == 404
    response = isolated_client.get("/api/v1/system/runtime-status")

    assert response.status_code == 200
    metrics = response.json()["metrics"]
    assert metrics["requests_total"] == 1
    assert metrics["errors_total"] == 1
    assert metrics["client_errors_total"] == 1
    assert metrics["server_errors_total"] == 0


def test_unknown_api_route_returns_not_found() -> None:
    response = client.get(
        "/api/v1/unknown",
    )

    assert response.status_code == 404

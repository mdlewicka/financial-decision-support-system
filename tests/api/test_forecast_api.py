import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.src.api.forecast_repository import (
    FORECAST_MODEL_DEFINITIONS,
    ForecastArtifactRepository,
)
from backend.src.api.routes.forecast import (
    get_forecast_repository,
)
from backend.src.config import SUPPORTED_TICKERS
from backend.src.main import create_app


def _metrics(
    *,
    observations: int,
    mae: float,
    directional_accuracy: float,
) -> dict[str, int | float]:
    return {
        "observations": observations,
        "mae": mae,
        "mse": mae**2,
        "rmse": mae,
        "r2": 0.1,
        "directional_accuracy": directional_accuracy,
    }


def _write_forecast_artifact(
    benchmark_directory: Path,
    *,
    directory_name: str,
    model_name: str,
    overall_mae: float,
) -> None:
    output_directory = benchmark_directory / directory_name
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "model_name": model_name,
        "evaluation_split": "validation",
        "target_definition": "next_session_simple_return",
        "metrics": {
            "overall": _metrics(
                observations=80,
                mae=overall_mae,
                directional_accuracy=0.55,
            ),
            "by_ticker": {
                ticker: _metrics(
                    observations=10,
                    mae=overall_mae + index * 0.001,
                    directional_accuracy=0.5,
                )
                for index, ticker in enumerate(SUPPORTED_TICKERS)
            },
        },
        "observations": [],
    }

    (output_directory / "forecast_result.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_all_forecast_artifacts(
    benchmark_directory: Path,
) -> None:
    for index, definition in enumerate(FORECAST_MODEL_DEFINITIONS):
        _write_forecast_artifact(
            benchmark_directory,
            directory_name=definition.directory_name,
            model_name=definition.model_name,
            overall_mae=0.01 + index * 0.001,
        )


@pytest.fixture
def forecast_client(
    tmp_path: Path,
) -> Iterator[TestClient]:
    _write_all_forecast_artifacts(
        tmp_path,
    )

    application = create_app()
    repository = ForecastArtifactRepository(
        benchmark_directory=tmp_path,
    )

    application.dependency_overrides[get_forecast_repository] = lambda: repository

    with TestClient(application) as client:
        yield client


def test_models_endpoint_lists_configured_models(
    forecast_client: TestClient,
) -> None:
    response = forecast_client.get(
        "/api/v1/forecast/models",
    )

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {
                "model_name": "naive_zero_return",
                "display_name": "Naive zero-return",
                "artifact_available": True,
            },
            {
                "model_name": "arima_walk_forward",
                "display_name": "ARIMA walk-forward",
                "artifact_available": True,
            },
            {
                "model_name": "random_forest",
                "display_name": "Random Forest",
                "artifact_available": True,
            },
        ]
    }


def test_summary_endpoint_returns_validation_comparison(
    forecast_client: TestClient,
) -> None:
    response = forecast_client.get(
        "/api/v1/forecast/summary",
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["evaluation_split"] == "validation"
    assert payload["target_definition"] == "next_session_simple_return"
    assert payload["tickers"] == list(SUPPORTED_TICKERS)
    assert [model["model_name"] for model in payload["models"]] == [
        "naive_zero_return",
        "arima_walk_forward",
        "random_forest",
    ]
    assert payload["models"][0]["metrics"]["mae"] == 0.01


def test_ticker_endpoint_normalizes_ticker_and_returns_all_models(
    forecast_client: TestClient,
) -> None:
    response = forecast_client.get(
        "/api/v1/forecast/tickers/spy",
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["ticker"] == "SPY"
    assert payload["evaluation_split"] == "validation"
    assert payload["target_definition"] == "next_session_simple_return"
    assert len(payload["models"]) == 3
    assert {model["model_name"] for model in payload["models"]} == {
        "naive_zero_return",
        "arima_walk_forward",
        "random_forest",
    }


def test_ticker_endpoint_rejects_unsupported_ticker(
    forecast_client: TestClient,
) -> None:
    response = forecast_client.get(
        "/api/v1/forecast/tickers/UNKNOWN",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Unsupported ticker: 'UNKNOWN'.",
    }


def test_summary_returns_service_unavailable_when_artifact_is_missing(
    tmp_path: Path,
) -> None:
    first_definition = FORECAST_MODEL_DEFINITIONS[0]

    _write_forecast_artifact(
        tmp_path,
        directory_name=first_definition.directory_name,
        model_name=first_definition.model_name,
        overall_mae=0.01,
    )

    application = create_app()
    repository = ForecastArtifactRepository(
        benchmark_directory=tmp_path,
    )

    application.dependency_overrides[get_forecast_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/forecast/summary",
        )

    assert response.status_code == 503
    assert "Forecast benchmark artifact is missing" in response.json()["detail"]


def test_models_endpoint_reports_missing_artifacts_without_failing(
    tmp_path: Path,
) -> None:
    application = create_app()
    repository = ForecastArtifactRepository(
        benchmark_directory=tmp_path,
    )

    application.dependency_overrides[get_forecast_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/forecast/models",
        )

    assert response.status_code == 200
    assert all(
        model["artifact_available"] is False for model in response.json()["models"]
    )


def test_summary_rejects_artifact_with_wrong_model_name(
    tmp_path: Path,
) -> None:
    for definition in FORECAST_MODEL_DEFINITIONS:
        _write_forecast_artifact(
            tmp_path,
            directory_name=definition.directory_name,
            model_name=definition.model_name,
            overall_mae=0.01,
        )

    mismatched_definition = FORECAST_MODEL_DEFINITIONS[0]

    _write_forecast_artifact(
        tmp_path,
        directory_name=(mismatched_definition.directory_name),
        model_name="wrong_model",
        overall_mae=0.01,
    )

    application = create_app()
    repository = ForecastArtifactRepository(
        benchmark_directory=tmp_path,
    )

    application.dependency_overrides[get_forecast_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/forecast/summary",
        )

    assert response.status_code == 503
    assert "model name mismatch" in response.json()["detail"]

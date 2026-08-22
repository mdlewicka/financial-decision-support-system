import csv
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.src.api.risk_repository import (
    RiskArtifactRepository,
)
from backend.src.api.routes.risk import (
    get_risk_repository,
)
from backend.src.main import create_app


def _risk_metrics(
    *,
    observations: int,
    annualized_return: float,
    annualized_volatility: float,
) -> dict[
    str,
    int | float | None,
]:
    return {
        "observations": observations,
        "periods_per_year": 252,
        "confidence_level": 0.95,
        "annual_risk_free_rate": 0.02,
        "mean_return": (annualized_return / 252.0),
        "annualized_mean_return": (annualized_return),
        "volatility": (annualized_volatility / (252.0**0.5)),
        "annualized_volatility": (annualized_volatility),
        "sharpe_ratio": 0.4,
        "sortino_ratio": 0.6,
        "value_at_risk": 0.015,
        "conditional_value_at_risk": 0.02,
        "maximum_drawdown": 0.18,
    }


def _analysis(
    *,
    observations: int,
    annualized_return: float,
    annualized_volatility: float,
    concentration_index: float,
) -> dict[
    str,
    object,
]:
    return {
        "metrics": _risk_metrics(
            observations=observations,
            annualized_return=(annualized_return),
            annualized_volatility=(annualized_volatility),
        ),
        "concentration_index": (concentration_index),
        "effective_number_of_assets": (1.0 / concentration_index),
        "diversification_ratio": 1.2,
    }


def _scenario(
    *,
    name: str,
    construction: str,
    weights: dict[
        str,
        float,
    ],
    return_shift: float,
) -> dict[
    str,
    object,
]:
    return {
        "name": name,
        "construction": construction,
        "weights": weights,
        "training": _analysis(
            observations=120,
            annualized_return=(0.06 + return_shift),
            annualized_volatility=0.11,
            concentration_index=0.36,
        ),
        "validation": _analysis(
            observations=60,
            annualized_return=(0.05 + return_shift),
            annualized_volatility=0.12,
            concentration_index=0.36,
        ),
    }


def _write_risk_artifact(
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "return_definition": ("adjusted_close_simple_return_on_shared_market_dates"),
        "market_splits": {
            "training": {
                "observations": 120,
                "periods_per_year": 252,
                "date_min": "2023-01-03",
                "date_max": "2023-06-30",
                "tickers": [
                    "AGG",
                    "GLD",
                    "SPY",
                ],
            },
            "validation": {
                "observations": 60,
                "periods_per_year": 252,
                "date_min": "2023-07-03",
                "date_max": "2023-09-29",
                "tickers": [
                    "AGG",
                    "GLD",
                    "SPY",
                ],
            },
        },
        "scenarios": [
            _scenario(
                name="defensive",
                construction=("manual_defensive"),
                weights={
                    "AGG": 0.50,
                    "GLD": 0.30,
                    "SPY": 0.20,
                },
                return_shift=0.0,
            ),
            _scenario(
                name="equal_weight",
                construction=("manual_equal_weight"),
                weights={
                    "AGG": 1.0 / 3.0,
                    "GLD": 1.0 / 3.0,
                    "SPY": 1.0 / 3.0,
                },
                return_shift=0.01,
            ),
        ],
    }

    (output_directory / "portfolio_result.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_validation_correlation(
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_directory / "validation_correlation_matrix.csv"

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "ticker",
                "AGG",
                "GLD",
                "SPY",
            ]
        )

        writer.writerow(
            [
                "AGG",
                1.0,
                0.15,
                -0.10,
            ]
        )

        writer.writerow(
            [
                "GLD",
                0.15,
                1.0,
                0.20,
            ]
        )

        writer.writerow(
            [
                "SPY",
                -0.10,
                0.20,
                1.0,
            ]
        )


def _write_all_risk_artifacts(
    output_directory: Path,
) -> None:
    _write_risk_artifact(
        output_directory,
    )

    _write_validation_correlation(
        output_directory,
    )


@pytest.fixture
def risk_client(
    tmp_path: Path,
) -> Iterator[TestClient]:
    _write_all_risk_artifacts(
        tmp_path,
    )

    application = create_app()

    repository = RiskArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_risk_repository] = lambda: repository

    with TestClient(application) as client:
        yield client


def test_status_reports_available_risk_artifacts(
    risk_client: TestClient,
) -> None:
    response = risk_client.get(
        "/api/v1/risk/status",
    )

    assert response.status_code == 200

    assert response.json() == {
        "artifact_available": True,
        "validation_correlation_available": True,
    }


def test_market_endpoint_returns_validation_metadata_and_correlations(
    risk_client: TestClient,
) -> None:
    response = risk_client.get(
        "/api/v1/risk/market",
    )

    payload = response.json()

    assert response.status_code == 200

    assert payload["evaluation_split"] == "validation"

    assert payload["return_definition"] == (
        "adjusted_close_simple_return_on_shared_market_dates"
    )

    assert payload["observations"] == 60

    assert payload["periods_per_year"] == 252

    assert payload["date_min"] == "2023-07-03"

    assert payload["date_max"] == "2023-09-29"

    assert payload["tickers"] == [
        "AGG",
        "GLD",
        "SPY",
    ]

    assert payload["correlation_matrix"][0] == {
        "ticker": "AGG",
        "correlations": {
            "AGG": 1.0,
            "GLD": 0.15,
            "SPY": -0.1,
        },
    }


def test_scenarios_endpoint_returns_validation_risk_only(
    risk_client: TestClient,
) -> None:
    response = risk_client.get(
        "/api/v1/risk/scenarios",
    )

    payload = response.json()

    assert response.status_code == 200

    assert payload["evaluation_split"] == "validation"

    assert [scenario["name"] for scenario in (payload["scenarios"])] == [
        "defensive",
        "equal_weight",
    ]

    defensive = payload["scenarios"][0]

    assert defensive["construction"] == "manual_defensive"

    assert defensive["risk"]["metrics"]["observations"] == 60

    assert defensive["risk"]["metrics"]["annualized_mean_return"] == 0.05

    assert "training" not in defensive

    assert "weights" not in defensive


def test_scenario_detail_returns_training_and_validation_risk(
    risk_client: TestClient,
) -> None:
    response = risk_client.get(
        "/api/v1/risk/scenarios/DEFENSIVE",
    )

    payload = response.json()

    assert response.status_code == 200

    assert payload["name"] == "defensive"

    assert payload["construction"] == "manual_defensive"

    assert payload["training"]["metrics"]["observations"] == 120

    assert payload["validation"]["metrics"]["observations"] == 60

    assert payload["validation"]["concentration_index"] == 0.36


def test_scenario_detail_rejects_unknown_scenario(
    risk_client: TestClient,
) -> None:
    response = risk_client.get(
        "/api/v1/risk/scenarios/not_available",
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": ("Risk scenario is not available: 'not_available'."),
    }


def test_scenarios_return_service_unavailable_when_artifact_is_missing(
    tmp_path: Path,
) -> None:
    application = create_app()

    repository = RiskArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_risk_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/risk/scenarios",
        )

    assert response.status_code == 503

    assert "Risk benchmark artifact is missing" in response.json()["detail"]


def test_market_returns_service_unavailable_when_correlation_is_missing(
    tmp_path: Path,
) -> None:
    _write_risk_artifact(
        tmp_path,
    )

    application = create_app()

    repository = RiskArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_risk_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/risk/market",
        )

    assert response.status_code == 503

    assert "Validation correlation matrix is missing" in response.json()["detail"]


def test_scenarios_reject_duplicate_scenario_names(
    tmp_path: Path,
) -> None:
    _write_risk_artifact(
        tmp_path,
    )

    artifact_path = tmp_path / "portfolio_result.json"

    payload = json.loads(
        artifact_path.read_text(
            encoding="utf-8",
        )
    )

    payload["scenarios"].append(payload["scenarios"][0])

    artifact_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    application = create_app()

    repository = RiskArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_risk_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/risk/scenarios",
        )

    assert response.status_code == 503

    assert "invalid schema" in response.json()["detail"]


def test_market_rejects_asymmetric_correlation_matrix(
    tmp_path: Path,
) -> None:
    _write_all_risk_artifacts(
        tmp_path,
    )

    matrix_path = tmp_path / "validation_correlation_matrix.csv"

    matrix_path.write_text(
        "\n".join(
            [
                "ticker,AGG,GLD,SPY",
                "AGG,1.0,0.15,-0.10",
                "GLD,0.10,1.0,0.20",
                "SPY,-0.10,0.20,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    application = create_app()

    repository = RiskArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_risk_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/risk/market",
        )

    assert response.status_code == 503

    assert "must be symmetric" in response.json()["detail"]

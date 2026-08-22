import csv
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.src.api.portfolio_repository import (
    PortfolioArtifactRepository,
)
from backend.src.api.routes.portfolio import (
    get_portfolio_repository,
)
from backend.src.main import create_app


def _risk_metrics(
    *,
    observations: int,
    annualized_return: float,
    annualized_volatility: float,
) -> dict[str, int | float | None]:
    return {
        "observations": observations,
        "periods_per_year": 252,
        "confidence_level": 0.95,
        "annual_risk_free_rate": 0.02,
        "mean_return": (annualized_return / 252.0),
        "annualized_mean_return": (annualized_return),
        "volatility": (annualized_volatility / (252.0**0.5)),
        "annualized_volatility": (annualized_volatility),
        "sharpe_ratio": 0.5,
        "sortino_ratio": 0.7,
        "value_at_risk": 0.015,
        "conditional_value_at_risk": 0.02,
        "maximum_drawdown": 0.16,
    }


def _analysis(
    *,
    observations: int,
    annualized_return: float,
    annualized_volatility: float,
    concentration_index: float,
) -> dict[str, object]:
    return {
        "metrics": _risk_metrics(
            observations=observations,
            annualized_return=annualized_return,
            annualized_volatility=(annualized_volatility),
        ),
        "concentration_index": (concentration_index),
        "effective_number_of_assets": (1.0 / concentration_index),
        "diversification_ratio": 1.25,
    }


def _scenario(
    *,
    name: str,
    construction: str,
    weights: dict[str, float],
    return_shift: float,
) -> dict[str, object]:
    concentration = sum(weight**2 for weight in weights.values())

    return {
        "name": name,
        "construction": construction,
        "weights": weights,
        "training": _analysis(
            observations=5,
            annualized_return=(0.06 + return_shift),
            annualized_volatility=0.11,
            concentration_index=concentration,
        ),
        "validation": _analysis(
            observations=4,
            annualized_return=(0.05 + return_shift),
            annualized_volatility=0.12,
            concentration_index=concentration,
        ),
    }


SCENARIOS = (
    {
        "name": "defensive",
        "construction": "manual_defensive",
        "weights": {
            "AGG": 0.50,
            "GLD": 0.30,
            "SPY": 0.20,
        },
        "return_shift": 0.0,
    },
    {
        "name": "equal_weight",
        "construction": "manual_equal_weight",
        "weights": {
            "AGG": 1.0 / 3.0,
            "GLD": 1.0 / 3.0,
            "SPY": 1.0 / 3.0,
        },
        "return_shift": 0.01,
    },
)


def _write_portfolio_artifact(
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
                "observations": 5,
                "periods_per_year": 252,
                "date_min": "2024-01-01",
                "date_max": "2024-01-05",
                "tickers": [
                    "AGG",
                    "GLD",
                    "SPY",
                ],
            },
            "validation": {
                "observations": 4,
                "periods_per_year": 252,
                "date_min": "2024-02-01",
                "date_max": "2024-02-04",
                "tickers": [
                    "AGG",
                    "GLD",
                    "SPY",
                ],
            },
        },
        "scenarios": [_scenario(**scenario) for scenario in SCENARIOS],
    }

    (output_directory / "portfolio_result.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_contributions(
    output_directory: Path,
) -> None:
    path = output_directory / "portfolio_contributions.csv"

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "scenario",
                "construction",
                "split",
                "ticker",
                "weight",
                "annualized_return_contribution",
                "annualized_volatility_contribution",
            ]
        )

        for scenario in SCENARIOS:
            for split in (
                "training",
                "validation",
            ):
                for index, (
                    ticker,
                    weight,
                ) in enumerate(scenario["weights"].items()):
                    writer.writerow(
                        [
                            scenario["name"],
                            scenario["construction"],
                            split,
                            ticker,
                            weight,
                            0.01 * (index + 1),
                            0.02 * (index + 1),
                        ]
                    )


def _write_returns(
    output_directory: Path,
) -> None:
    path = output_directory / "portfolio_returns.csv"

    validation_returns = (
        0.01,
        -0.005,
        0.02,
        0.0,
    )

    training_returns = (
        0.01,
        0.0,
        -0.005,
        0.01,
        0.005,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "scenario",
                "construction",
                "split",
                "date",
                "portfolio_return",
            ]
        )

        for scenario in SCENARIOS:
            for index, value in enumerate(
                training_returns,
                start=1,
            ):
                writer.writerow(
                    [
                        scenario["name"],
                        scenario["construction"],
                        "training",
                        f"2024-01-{index:02d}",
                        value,
                    ]
                )

            for index, value in enumerate(
                validation_returns,
                start=1,
            ):
                writer.writerow(
                    [
                        scenario["name"],
                        scenario["construction"],
                        "validation",
                        f"2024-02-{index:02d}",
                        value,
                    ]
                )


def _write_all_portfolio_artifacts(
    output_directory: Path,
) -> None:
    _write_portfolio_artifact(output_directory)
    _write_contributions(output_directory)
    _write_returns(output_directory)


@pytest.fixture
def portfolio_client(
    tmp_path: Path,
) -> Iterator[TestClient]:
    _write_all_portfolio_artifacts(tmp_path)

    application = create_app()

    repository = PortfolioArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_portfolio_repository] = lambda: repository

    with TestClient(application) as client:
        yield client


def test_status_reports_available_portfolio_artifacts(
    portfolio_client: TestClient,
) -> None:
    response = portfolio_client.get("/api/v1/portfolio/status")

    assert response.status_code == 200
    assert response.json() == {
        "artifact_available": True,
        "contributions_available": True,
        "returns_available": True,
    }


def test_scenarios_endpoint_returns_allocations_and_validation_performance(
    portfolio_client: TestClient,
) -> None:
    response = portfolio_client.get("/api/v1/portfolio/scenarios")

    payload = response.json()

    assert response.status_code == 200
    assert payload["evaluation_split"] == "validation"
    assert payload["tickers"] == [
        "AGG",
        "GLD",
        "SPY",
    ]

    assert [scenario["name"] for scenario in payload["scenarios"]] == [
        "defensive",
        "equal_weight",
    ]

    defensive = payload["scenarios"][0]

    assert defensive["weights"] == [
        {
            "ticker": "AGG",
            "weight": 0.5,
        },
        {
            "ticker": "GLD",
            "weight": 0.3,
        },
        {
            "ticker": "SPY",
            "weight": 0.2,
        },
    ]

    assert defensive["validation"]["annualized_mean_return"] == 0.05


def test_scenario_detail_returns_validation_contributions(
    portfolio_client: TestClient,
) -> None:
    response = portfolio_client.get("/api/v1/portfolio/scenarios/defensive")

    payload = response.json()

    assert response.status_code == 200
    assert payload["name"] == "defensive"

    assert len(payload["validation_contributions"]) == 3

    first_contribution = payload["validation_contributions"][0]

    assert first_contribution == {
        "ticker": "AGG",
        "weight": 0.5,
        "annualized_return_contribution": 0.01,
        "annualized_volatility_contribution": 0.02,
    }


def test_scenario_lookup_is_case_insensitive(
    portfolio_client: TestClient,
) -> None:
    response = portfolio_client.get("/api/v1/portfolio/scenarios/DEFENSIVE")

    assert response.status_code == 200
    assert response.json()["name"] == "defensive"


def test_unknown_portfolio_scenario_returns_not_found(
    portfolio_client: TestClient,
) -> None:
    response = portfolio_client.get("/api/v1/portfolio/scenarios/not_available")

    assert response.status_code == 404
    assert response.json() == {
        "detail": ("Portfolio scenario is not available: 'not_available'.")
    }


def test_returns_endpoint_builds_cumulative_return_series(
    portfolio_client: TestClient,
) -> None:
    response = portfolio_client.get("/api/v1/portfolio/scenarios/defensive/returns")

    payload = response.json()

    assert response.status_code == 200
    assert payload["observations"] == 4
    assert len(payload["returns"]) == 4

    assert payload["returns"][0] == {
        "date": "2024-02-01",
        "portfolio_return": 0.01,
        "cumulative_return": pytest.approx(0.01),
    }

    expected_final = (1.01 * 0.995 * 1.02 * 1.0) - 1.0

    assert payload["returns"][-1]["cumulative_return"] == pytest.approx(expected_final)


def test_scenarios_return_service_unavailable_when_main_artifact_is_missing(
    tmp_path: Path,
) -> None:
    application = create_app()

    repository = PortfolioArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_portfolio_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get("/api/v1/portfolio/scenarios")

    assert response.status_code == 503
    assert "Portfolio benchmark artifact is missing" in response.json()["detail"]


def test_detail_returns_service_unavailable_when_contributions_are_missing(
    tmp_path: Path,
) -> None:
    _write_portfolio_artifact(tmp_path)
    _write_returns(tmp_path)

    application = create_app()

    repository = PortfolioArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_portfolio_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get("/api/v1/portfolio/scenarios/defensive")

    assert response.status_code == 503
    assert "Portfolio contributions artifact is missing" in response.json()["detail"]


def test_detail_rejects_contribution_weight_mismatch(
    tmp_path: Path,
) -> None:
    _write_all_portfolio_artifacts(tmp_path)

    path = tmp_path / "portfolio_contributions.csv"

    content = path.read_text(encoding="utf-8")

    content = content.replace(
        "defensive,manual_defensive,validation,AGG,0.5,",
        "defensive,manual_defensive,validation,AGG,0.4,",
        1,
    )

    path.write_text(
        content,
        encoding="utf-8",
    )

    application = create_app()

    repository = PortfolioArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_portfolio_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get("/api/v1/portfolio/scenarios/defensive")

    assert response.status_code == 503
    assert "weight does not match" in response.json()["detail"]


def test_returns_reject_unsorted_validation_dates(
    tmp_path: Path,
) -> None:
    _write_all_portfolio_artifacts(tmp_path)

    path = tmp_path / "portfolio_returns.csv"

    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))

    header = rows[0]
    data = rows[1:]

    defensive_validation_indexes = [
        index
        for index, row in enumerate(data)
        if (row[0] == "defensive" and row[2] == "validation")
    ]

    first = defensive_validation_indexes[0]
    second = defensive_validation_indexes[1]

    data[first], data[second] = (
        data[second],
        data[first],
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

    application = create_app()

    repository = PortfolioArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_portfolio_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get("/api/v1/portfolio/scenarios/defensive/returns")

    assert response.status_code == 503
    assert "dates must be sorted" in response.json()["detail"]

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.src.api.routes.xai import (
    get_xai_repository,
)
from backend.src.api.xai_repository import (
    XAIArtifactRepository,
)
from backend.src.main import create_app


def _write_xai_artifact(
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "metadata": {
            "model_name": "random_forest",
            "training_split": "training",
            "explanation_split": "validation",
            "target_definition": ("next_session_simple_return"),
            "background_rows": 100,
            "global_explanation_rows": 80,
            "effect_feature_count": 2,
            "local_ticker": "SPY",
            "local_date": "2024-01-05T00:00:00",
            "local_target_date": ("2024-01-08T00:00:00"),
            "local_actual": -0.012,
            "local_prediction": -0.006,
        },
        "methods": [
            "shap",
            "partial_dependence",
            "accumulated_local_effects",
        ],
        "local_shap": {
            "base_value": 0.001,
            "prediction": -0.006,
            "reconstructed_prediction": -0.006,
            "features": [
                {
                    "feature": "momentum_20",
                    "feature_value": -0.02,
                    "shap_value": -0.005,
                },
                {
                    "feature": "volatility_20",
                    "feature_value": 0.018,
                    "shap_value": -0.002,
                },
                {
                    "feature": "ticker_SPY",
                    "feature_value": 1.0,
                    "shap_value": 0.0,
                },
            ],
        },
        "global_shap": {
            "observations": 80,
            "mean_absolute_contributions": {
                "momentum_20": 0.004,
                "volatility_20": 0.003,
                "ticker_SPY": 0.001,
            },
        },
        "partial_dependence": {
            "momentum_20": {
                "feature_values": [
                    -0.02,
                    0.0,
                    0.02,
                ],
                "average_predictions": [
                    -0.003,
                    0.0,
                    0.002,
                ],
            },
            "volatility_20": {
                "feature_values": [
                    0.01,
                    0.02,
                    0.03,
                ],
                "average_predictions": [
                    0.002,
                    0.0,
                    -0.002,
                ],
            },
        },
        "accumulated_local_effects": {
            "momentum_20": {
                "bin_edges": [
                    -0.03,
                    -0.01,
                    0.01,
                    0.03,
                ],
                "bin_centers": [
                    -0.02,
                    0.0,
                    0.02,
                ],
                "accumulated_effects": [
                    -0.002,
                    0.0,
                    0.002,
                ],
                "sample_counts": [
                    20,
                    30,
                    30,
                ],
            },
            "volatility_20": {
                "bin_edges": [
                    0.005,
                    0.015,
                    0.025,
                    0.035,
                ],
                "bin_centers": [
                    0.01,
                    0.02,
                    0.03,
                ],
                "accumulated_effects": [
                    0.002,
                    0.0,
                    -0.002,
                ],
                "sample_counts": [
                    25,
                    30,
                    25,
                ],
            },
        },
        "plots": {
            "global_shap_bar": ("plots/global_shap_bar.png"),
            "global_shap_beeswarm": ("plots/global_shap_beeswarm.png"),
            "local_shap_waterfall": ("plots/local_shap_waterfall.png"),
        },
    }

    (output_directory / "xai_result.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


@pytest.fixture
def xai_client(
    tmp_path: Path,
) -> Iterator[TestClient]:
    _write_xai_artifact(
        tmp_path,
    )

    application = create_app()
    repository = XAIArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_xai_repository] = lambda: repository

    with TestClient(application) as client:
        yield client


def test_status_reports_available_artifact(
    xai_client: TestClient,
) -> None:
    response = xai_client.get(
        "/api/v1/xai/status",
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "random_forest",
        "artifact_available": True,
    }


def test_summary_returns_global_shap_importance(
    xai_client: TestClient,
) -> None:
    response = xai_client.get(
        "/api/v1/xai/summary",
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["model_name"] == "random_forest"
    assert payload["explanation_split"] == "validation"
    assert payload["target_definition"] == "next_session_simple_return"
    assert payload["global_observations"] == 80

    assert [feature["feature"] for feature in payload["features"]] == [
        "momentum_20",
        "volatility_20",
        "ticker_SPY",
    ]

    assert payload["effect_features"] == [
        "momentum_20",
        "volatility_20",
    ]


def test_summary_marks_features_with_pdp_and_ale(
    xai_client: TestClient,
) -> None:
    response = xai_client.get(
        "/api/v1/xai/summary",
    )

    features = {feature["feature"]: feature for feature in response.json()["features"]}

    assert features["momentum_20"]["effect_available"] is True
    assert features["volatility_20"]["effect_available"] is True
    assert features["ticker_SPY"]["effect_available"] is False


def test_local_endpoint_returns_representative_shap_explanation(
    xai_client: TestClient,
) -> None:
    response = xai_client.get(
        "/api/v1/xai/local",
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["ticker"] == "SPY"
    assert payload["actual_return"] == -0.012
    assert payload["predicted_return"] == -0.006

    assert [feature["feature"] for feature in payload["features"]] == [
        "momentum_20",
        "volatility_20",
        "ticker_SPY",
    ]


def test_feature_endpoint_returns_pdp_and_ale(
    xai_client: TestClient,
) -> None:
    response = xai_client.get(
        "/api/v1/xai/features/momentum_20",
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["feature_name"] == "momentum_20"
    assert payload["mean_absolute_shap_value"] == 0.004
    assert len(payload["partial_dependence"]) == 3
    assert len(payload["accumulated_local_effects"]) == 3

    assert payload["partial_dependence"][0] == {
        "feature_value": -0.02,
        "average_prediction": -0.003,
    }

    assert payload["accumulated_local_effects"][0] == {
        "lower_edge": -0.03,
        "upper_edge": -0.01,
        "bin_center": -0.02,
        "accumulated_effect": -0.002,
        "sample_count": 20,
    }


def test_feature_endpoint_rejects_feature_without_effect_analysis(
    xai_client: TestClient,
) -> None:
    response = xai_client.get(
        "/api/v1/xai/features/ticker_SPY",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": ("Feature effect analysis is not available for 'ticker_SPY'."),
    }


def test_summary_returns_service_unavailable_when_artifact_is_missing(
    tmp_path: Path,
) -> None:
    application = create_app()
    repository = XAIArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_xai_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/xai/summary",
        )

    assert response.status_code == 503
    assert "XAI artifact is missing" in response.json()["detail"]


def test_status_reports_missing_artifact_without_failing(
    tmp_path: Path,
) -> None:
    application = create_app()
    repository = XAIArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_xai_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/xai/status",
        )

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "random_forest",
        "artifact_available": False,
    }


def test_summary_rejects_inconsistent_pdp_and_ale_features(
    tmp_path: Path,
) -> None:
    _write_xai_artifact(
        tmp_path,
    )

    artifact_path = tmp_path / "xai_result.json"

    payload = json.loads(
        artifact_path.read_text(
            encoding="utf-8",
        )
    )

    del payload["accumulated_local_effects"]["volatility_20"]

    artifact_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    application = create_app()
    repository = XAIArtifactRepository(
        artifact_directory=tmp_path,
    )

    application.dependency_overrides[get_xai_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/xai/summary",
        )

    assert response.status_code == 503
    assert "invalid schema" in response.json()["detail"]

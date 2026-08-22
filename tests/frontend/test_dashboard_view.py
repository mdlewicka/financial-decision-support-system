from typing import Any

import pandas as pd

from frontend.api_client import APIClient
from frontend.dashboard_view import (
    AGENT_STATUS_PATH,
    BALANCED_DECISION_PATH,
    FORECAST_SUMMARY_PATH,
    MAXIMUM_SHARPE_SCENARIO,
    MINIMUM_VOLATILITY_SCENARIO,
    PORTFOLIO_SCENARIOS_PATH,
    RAG_STATUS_PATH,
    RISK_SCENARIOS_PATH,
    _build_executive_snapshot,
    _build_portfolio_snapshot,
    _find_model,
    _find_scenario,
    _load_dashboard_data,
)


class RecordingAPIClient(APIClient):
    """Deterministic API client used by dashboard tests."""

    def __init__(
        self,
        responses: dict[str, dict[str, Any]],
    ) -> None:
        super().__init__("http://test-api")

        self._responses = responses
        self.requested_paths: list[str] = []

    def get(
        self,
        path: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        del timeout_seconds

        self.requested_paths.append(path)

        return self._responses[path]


def _forecast_summary() -> dict[str, Any]:
    return {
        "evaluation_split": "validation",
        "target_definition": ("next_session_simple_return"),
        "models": [
            {
                "model_name": "random_forest",
                "display_name": "Random Forest",
                "metrics": {
                    "observations": 4786,
                    "mae": 0.0110,
                    "rmse": 0.0169,
                    "r2": 0.0048,
                    "directional_accuracy": 0.5048,
                },
            },
        ],
    }


def _risk_scenarios() -> dict[str, Any]:
    return {
        "evaluation_split": "validation",
        "scenarios": [
            {
                "name": (MAXIMUM_SHARPE_SCENARIO),
                "risk": {
                    "metrics": {
                        "observations": 565,
                        "annualized_mean_return": 0.116,
                        "annualized_volatility": 0.164,
                        "sharpe_ratio": 0.585,
                        "maximum_drawdown": 0.279,
                    },
                    "concentration_index": 0.28,
                    "effective_number_of_assets": 3.5,
                    "diversification_ratio": 1.38,
                },
            },
            {
                "name": (MINIMUM_VOLATILITY_SCENARIO),
                "risk": {
                    "metrics": {
                        "observations": 565,
                        "annualized_mean_return": 0.053,
                        "annualized_volatility": 0.095,
                        "sharpe_ratio": 0.352,
                        "maximum_drawdown": 0.153,
                    },
                    "concentration_index": 0.31,
                    "effective_number_of_assets": 3.2,
                    "diversification_ratio": 1.42,
                },
            },
        ],
    }


def _portfolio_scenarios() -> dict[str, Any]:
    return {
        "evaluation_split": "validation",
        "tickers": [
            "AGG",
            "SPY",
        ],
        "scenarios": [
            {
                "name": (MAXIMUM_SHARPE_SCENARIO),
                "weights": [
                    {
                        "ticker": "AGG",
                        "weight": 0.33,
                    },
                    {
                        "ticker": "SPY",
                        "weight": 0.67,
                    },
                ],
                "validation": {
                    "observations": 565,
                    "annualized_mean_return": 0.116,
                    "annualized_volatility": 0.164,
                    "sharpe_ratio": 0.585,
                    "maximum_drawdown": 0.279,
                    "effective_number_of_assets": 3.5,
                },
            },
            {
                "name": (MINIMUM_VOLATILITY_SCENARIO),
                "weights": [
                    {
                        "ticker": "AGG",
                        "weight": 0.72,
                    },
                    {
                        "ticker": "SPY",
                        "weight": 0.28,
                    },
                ],
                "validation": {
                    "observations": 565,
                    "annualized_mean_return": 0.053,
                    "annualized_volatility": 0.095,
                    "sharpe_ratio": 0.352,
                    "maximum_drawdown": 0.153,
                    "effective_number_of_assets": 3.2,
                },
            },
        ],
    }


def _rag_status() -> dict[str, Any]:
    return {
        "corpus_available": True,
        "benchmark_available": True,
        "model_name": "gemma3:4b",
    }


def _agent_status() -> dict[str, Any]:
    return {
        "forecast_artifact_available": True,
        "portfolio_artifact_available": True,
        "reinforcement_learning_artifact_available": True,
        "runtime_inputs_available": True,
        "test_split_used": False,
    }


def _balanced_decision() -> dict[str, Any]:
    return {
        "deterministic": True,
        "llm_used_for_ranking": False,
        "portfolio_weights_modified": False,
        "recommended_scenario": MAXIMUM_SHARPE_SCENARIO,
        "scenarios": [
            {
                "name": MAXIMUM_SHARPE_SCENARIO,
                "admissible": True,
                "rank": 1,
                "total_score": 0.82,
            },
            {
                "name": MINIMUM_VOLATILITY_SCENARIO,
                "admissible": True,
                "rank": 2,
                "total_score": 0.71,
            },
        ],
    }


def _dashboard_responses() -> dict[str, dict[str, Any]]:
    return {
        FORECAST_SUMMARY_PATH: (_forecast_summary()),
        RISK_SCENARIOS_PATH: (_risk_scenarios()),
        PORTFOLIO_SCENARIOS_PATH: (_portfolio_scenarios()),
        BALANCED_DECISION_PATH: (_balanced_decision()),
        RAG_STATUS_PATH: (_rag_status()),
        AGENT_STATUS_PATH: (_agent_status()),
    }


def test_load_dashboard_data_uses_existing_api_endpoints() -> None:
    client = RecordingAPIClient(_dashboard_responses())

    (
        forecast,
        risk,
        portfolio,
        decision,
        rag_status,
        agent_status,
    ) = _load_dashboard_data(client)

    assert forecast == (_forecast_summary())

    assert risk == (_risk_scenarios())

    assert portfolio == (_portfolio_scenarios())

    assert decision == (_balanced_decision())

    assert rag_status == (_rag_status())

    assert agent_status == (_agent_status())

    assert client.requested_paths == [
        FORECAST_SUMMARY_PATH,
        RISK_SCENARIOS_PATH,
        PORTFOLIO_SCENARIOS_PATH,
        BALANCED_DECISION_PATH,
        RAG_STATUS_PATH,
        AGENT_STATUS_PATH,
    ]


def test_executive_snapshot_combines_decision_and_validation_evidence() -> None:
    snapshot = _build_executive_snapshot(
        _balanced_decision(),
        _portfolio_scenarios(),
        _forecast_summary(),
    )

    assert snapshot == {
        "recommended_scenario": MAXIMUM_SHARPE_SCENARIO,
        "admissible": True,
        "annualized_return": 0.116,
        "annualized_volatility": 0.164,
        "sharpe_ratio": 0.585,
        "maximum_drawdown": 0.279,
        "directional_accuracy": 0.5048,
        "evaluation_split": "validation",
    }


def test_executive_snapshot_does_not_select_a_scenario_itself() -> None:
    decision = _balanced_decision()
    decision["recommended_scenario"] = None

    snapshot = _build_executive_snapshot(
        decision,
        _portfolio_scenarios(),
        _forecast_summary(),
    )

    assert snapshot["recommended_scenario"] is None
    assert snapshot["admissible"] is None


def test_find_model_returns_requested_model() -> None:
    models = _forecast_summary()["models"]

    assert isinstance(
        models,
        list,
    )

    result = _find_model(
        models,
        "random_forest",
    )

    assert result is not None

    assert result["model_name"] == "random_forest"


def test_find_model_returns_none_for_missing_model() -> None:
    result = _find_model(
        [],
        "random_forest",
    )

    assert result is None


def test_find_scenario_returns_requested_scenario() -> None:
    scenarios = _portfolio_scenarios()["scenarios"]

    assert isinstance(
        scenarios,
        list,
    )

    result = _find_scenario(
        scenarios,
        MINIMUM_VOLATILITY_SCENARIO,
    )

    assert result is not None

    assert result["name"] == MINIMUM_VOLATILITY_SCENARIO


def test_find_scenario_returns_none_for_missing_scenario() -> None:
    result = _find_scenario(
        [],
        MAXIMUM_SHARPE_SCENARIO,
    )

    assert result is None


def test_portfolio_snapshot_contains_selected_strategies() -> None:
    frame = _build_portfolio_snapshot(_portfolio_scenarios())

    assert isinstance(
        frame,
        pd.DataFrame,
    )

    assert frame.shape == (
        2,
        6,
    )

    assert frame["Strategia"].tolist() == [
        "Maksymalny Sharpe",
        "Minimalna zmienność",
    ]


def test_portfolio_snapshot_formats_validation_metrics() -> None:
    frame = _build_portfolio_snapshot(_portfolio_scenarios())

    maximum_sharpe_row = frame.iloc[0]

    assert maximum_sharpe_row["Zwrot roczny"] == "11.60%"

    assert maximum_sharpe_row["Zmienność"] == "16.40%"

    assert maximum_sharpe_row["Sharpe"] == "0.585"

    assert maximum_sharpe_row["Max drawdown"] == "27.90%"

    assert maximum_sharpe_row["Efektywna liczba aktywów"] == "3.500"


def test_portfolio_snapshot_uses_english_labels_on_request() -> None:
    frame = _build_portfolio_snapshot(
        _portfolio_scenarios(),
        language="en",
    )

    assert frame.columns.tolist() == [
        "Strategy",
        "Annualized return",
        "Volatility",
        "Sharpe",
        "Maximum drawdown",
        "Effective number of assets",
    ]
    assert frame["Strategy"].tolist() == [
        "Maximum Sharpe",
        "Minimum volatility",
    ]


def test_portfolio_snapshot_is_empty_when_scenarios_are_missing() -> None:
    frame = _build_portfolio_snapshot(
        {
            "scenarios": [],
        }
    )

    assert frame.empty

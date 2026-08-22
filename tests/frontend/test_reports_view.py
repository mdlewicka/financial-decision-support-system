from datetime import datetime
from typing import Any

from frontend.api_client import APIClient
from frontend.reports_view import (
    AGENT_STATUS_PATH,
    FORECAST_SUMMARY_PATH,
    PORTFOLIO_SCENARIOS_PATH,
    RAG_STATUS_PATH,
    RISK_SCENARIOS_PATH,
    XAI_SUMMARY_PATH,
    _build_forecast_section,
    _build_markdown_report,
    _build_portfolio_section,
    _build_report_data,
    _build_risk_section,
    _build_xai_section,
    _load_report_sources,
)

SYSTEM_INFO_PATH = "/api/v1/system/info"


class RecordingAPIClient(APIClient):
    """Deterministic API client used by report tests."""

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


def _system_info() -> dict[str, Any]:
    return {
        "title": "Financial Decision Support API",
        "version": "0.1.0",
    }


def _forecast_summary() -> dict[str, Any]:
    return {
        "evaluation_split": "validation",
        "target_definition": ("next_session_simple_return"),
        "tickers": [
            "SPY",
            "QQQ",
        ],
        "models": [
            {
                "model_name": "naive_zero_return",
                "display_name": "Naive zero-return",
                "metrics": {
                    "observations": 4786,
                    "mae": 0.0111,
                    "rmse": 0.0170,
                    "r2": -0.001,
                    "directional_accuracy": 0.003,
                },
            },
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


def _xai_summary() -> dict[str, Any]:
    return {
        "model_name": "random_forest",
        "explanation_split": "validation",
        "target_definition": ("next_session_simple_return"),
        "methods": [
            "shap",
            "partial_dependence",
            "accumulated_local_effects",
        ],
        "global_observations": 400,
        "features": [
            {
                "feature": "feature_1",
                "mean_absolute_shap_value": 0.00018,
            },
            {
                "feature": "feature_2",
                "mean_absolute_shap_value": 0.00017,
            },
            {
                "feature": "feature_3",
                "mean_absolute_shap_value": 0.00016,
            },
            {
                "feature": "feature_4",
                "mean_absolute_shap_value": 0.00015,
            },
            {
                "feature": "feature_5",
                "mean_absolute_shap_value": 0.00014,
            },
            {
                "feature": "feature_6",
                "mean_absolute_shap_value": 0.00013,
            },
        ],
    }


def _risk_scenarios() -> dict[str, Any]:
    return {
        "evaluation_split": "validation",
        "scenarios": [
            {
                "name": "maximum_sharpe",
                "risk": {
                    "metrics": {
                        "annualized_mean_return": 0.116,
                        "annualized_volatility": 0.164,
                        "sharpe_ratio": 0.585,
                        "value_at_risk": 0.0149,
                        "conditional_value_at_risk": 0.0229,
                        "maximum_drawdown": 0.279,
                    },
                    "concentration_index": 0.28,
                    "effective_number_of_assets": 3.5,
                },
            },
            {
                "name": "minimum_volatility",
                "risk": {
                    "metrics": {
                        "annualized_mean_return": 0.053,
                        "annualized_volatility": 0.095,
                        "sharpe_ratio": 0.352,
                        "value_at_risk": 0.0094,
                        "conditional_value_at_risk": 0.0126,
                        "maximum_drawdown": 0.153,
                    },
                    "concentration_index": 0.31,
                    "effective_number_of_assets": 3.2,
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
                "name": "maximum_sharpe",
                "construction": ("optimized_on_training_maximum_sharpe"),
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
                    "concentration_index": 0.28,
                    "effective_number_of_assets": 3.5,
                },
            },
            {
                "name": "minimum_volatility",
                "construction": ("optimized_on_training_minimum_volatility"),
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
                    "concentration_index": 0.31,
                    "effective_number_of_assets": 3.2,
                },
            },
        ],
    }


def _rag_status() -> dict[str, Any]:
    return {
        "corpus_available": True,
        "corpus_documents": 4,
        "benchmark_available": True,
        "model_name": "gemma3:4b",
    }


def _agent_status() -> dict[str, Any]:
    return {
        "runtime_inputs_available": True,
        "model_name": "gemma3:4b",
        "retrieval_strategy": "vector",
        "evaluation_basis": ("validation_benchmark_artifacts"),
        "test_split_used": False,
    }


def _report_sources() -> dict[
    str,
    dict[str, Any],
]:
    return {
        "system": _system_info(),
        "forecast": _forecast_summary(),
        "xai": _xai_summary(),
        "risk": _risk_scenarios(),
        "portfolio": _portfolio_scenarios(),
        "rag": _rag_status(),
        "agents": _agent_status(),
    }


def _api_responses() -> dict[
    str,
    dict[str, Any],
]:
    return {
        SYSTEM_INFO_PATH: (_system_info()),
        FORECAST_SUMMARY_PATH: (_forecast_summary()),
        XAI_SUMMARY_PATH: (_xai_summary()),
        RISK_SCENARIOS_PATH: (_risk_scenarios()),
        PORTFOLIO_SCENARIOS_PATH: (_portfolio_scenarios()),
        RAG_STATUS_PATH: (_rag_status()),
        AGENT_STATUS_PATH: (_agent_status()),
    }


def test_load_report_sources_uses_existing_api_endpoints() -> None:
    client = RecordingAPIClient(_api_responses())

    sources = _load_report_sources(client)

    assert sources == (_report_sources())

    assert client.requested_paths == [
        SYSTEM_INFO_PATH,
        FORECAST_SUMMARY_PATH,
        XAI_SUMMARY_PATH,
        RISK_SCENARIOS_PATH,
        PORTFOLIO_SCENARIOS_PATH,
        RAG_STATUS_PATH,
        AGENT_STATUS_PATH,
    ]


def test_forecast_section_uses_random_forest_metrics() -> None:
    section = _build_forecast_section(_forecast_summary())

    assert section["evaluation_split"] == "validation"

    assert section["target_definition"] == "next_session_simple_return"

    random_forest = section["random_forest"]

    assert random_forest["observations"] == 4786

    assert random_forest["directional_accuracy"] == 0.5048

    assert random_forest["r2"] == 0.0048


def test_xai_section_keeps_only_top_five_features() -> None:
    section = _build_xai_section(_xai_summary())

    top_features = section["top_features"]

    assert len(top_features) == 5

    assert top_features[0]["feature"] == "feature_1"

    assert top_features[-1]["feature"] == "feature_5"


def test_risk_section_extracts_validation_metrics() -> None:
    section = _build_risk_section(_risk_scenarios())

    scenarios = section["scenarios"]

    assert len(scenarios) == 2

    maximum_sharpe = scenarios[0]

    assert maximum_sharpe["name"] == "maximum_sharpe"

    assert maximum_sharpe["annualized_return"] == 0.116

    assert maximum_sharpe["sharpe_ratio"] == 0.585

    assert maximum_sharpe["maximum_drawdown"] == 0.279


def test_portfolio_section_preserves_weights_and_validation() -> None:
    section = _build_portfolio_section(_portfolio_scenarios())

    scenarios = section["scenarios"]

    assert len(scenarios) == 2

    maximum_sharpe = scenarios[0]

    assert maximum_sharpe["weights"] == [
        {
            "ticker": "AGG",
            "weight": 0.33,
        },
        {
            "ticker": "SPY",
            "weight": 0.67,
        },
    ]

    validation = maximum_sharpe["validation"]

    assert validation["observations"] == 565

    assert validation["annualized_volatility"] == 0.164


def test_report_data_marks_validation_scope_and_closed_test_split() -> None:
    report = _build_report_data(_report_sources())

    metadata = report["metadata"]

    assert metadata["evaluation_scope"] == "validation"

    assert metadata["test_split_used"] is False

    generated_at = datetime.fromisoformat(str(metadata["generated_at_utc"]))

    assert generated_at.tzinfo is not None

    assert report["agents"]["evaluation_basis"] == "validation_benchmark_artifacts"


def test_report_data_contains_all_main_analytical_sections() -> None:
    report = _build_report_data(_report_sources())

    expected_sections = {
        "metadata",
        "system",
        "forecasting",
        "explainability",
        "risk",
        "portfolio",
        "rag",
        "agents",
    }

    assert set(report) == expected_sections


def test_markdown_report_contains_required_sections_and_limitations() -> None:
    report = _build_report_data(_report_sources())

    markdown = _build_markdown_report(report)

    required_fragments = [
        "# Financial Decision Support System",
        "## Raport analityczny",
        "## Forecasting",
        "## Explainability and Governance",
        "## Risk and Scenarios",
        "## Optimization",
        "## AI Insights",
        "## Ograniczenia",
        "Random Forest – trafność kierunku: 50.48%",
        "Końcowy zbiór testowy pozostaje zamknięty",
        "nie stanowi indywidualnej rekomendacji inwestycyjnej",
    ]

    for fragment in required_fragments:
        assert fragment in markdown

    assert "sentinel" not in markdown.lower()


def test_markdown_report_can_be_generated_in_english() -> None:
    report = _build_report_data(_report_sources())

    markdown = _build_markdown_report(report, "en")

    required_fragments = [
        "## Analytical report",
        "Evaluation split: validation",
        "Random Forest directional accuracy: 50.48%",
        "### Maximum Sharpe",
        "#### Weights",
        "## Limitations",
        "The final test split remains closed",
        "does not constitute individualized investment advice",
    ]

    for fragment in required_fragments:
        assert fragment in markdown

    assert "Zbiór ewaluacyjny" not in markdown

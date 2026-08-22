from pathlib import Path
from typing import Any

import pytest
import requests
from streamlit.testing.v1 import AppTest

from frontend import app
from frontend.api_client import (
    APIClient,
    APIClientError,
)

APP_PATH = Path(__file__).resolve().parents[2] / "frontend" / "app.py"


class StubAPIClient(APIClient):
    """Deterministic API client used by unit tests."""

    def __init__(
        self,
        *,
        health: dict[str, Any] | None = None,
        information: dict[str, Any] | None = None,
        error: APIClientError | None = None,
    ) -> None:
        super().__init__("http://test-api")

        self._health = health or {
            "status": "healthy",
        }

        self._information = information or {
            "title": "Financial Decision Support API",
            "version": "0.1.0",
        }

        self._error = error

    def get_system_health(
        self,
    ) -> dict[str, Any]:
        if self._error is not None:
            raise self._error

        return self._health

    def get_system_info(
        self,
    ) -> dict[str, Any]:
        if self._error is not None:
            raise self._error

        return self._information


class FakeResponse:
    """Minimal requests.Response-compatible test double."""

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
    ) -> None:
        self.status_code = status_code
        self.text = ""
        self._payload = payload

    def json(
        self,
    ) -> dict[str, Any]:
        return self._payload


def _forecast_models() -> dict[str, Any]:
    return {
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
        ],
    }


def _forecast_model_results() -> list[dict[str, Any]]:
    return [
        {
            "model_name": "naive_zero_return",
            "display_name": "Naive zero-return",
            "metrics": {
                "observations": 100,
                "mae": 0.0100,
                "mse": 0.0002,
                "rmse": 0.0141,
                "r2": -0.001,
                "directional_accuracy": 0.003,
            },
        },
        {
            "model_name": "arima_walk_forward",
            "display_name": "ARIMA walk-forward",
            "metrics": {
                "observations": 100,
                "mae": 0.0101,
                "mse": 0.0002,
                "rmse": 0.0142,
                "r2": -0.010,
                "directional_accuracy": 0.501,
            },
        },
        {
            "model_name": "random_forest",
            "display_name": "Random Forest",
            "metrics": {
                "observations": 100,
                "mae": 0.0099,
                "mse": 0.0002,
                "rmse": 0.0140,
                "r2": 0.004,
                "directional_accuracy": 0.505,
            },
        },
    ]


def _forecast_summary() -> dict[str, Any]:
    return {
        "evaluation_split": "validation",
        "target_definition": "next_session_simple_return",
        "tickers": [
            "SPY",
            "QQQ",
        ],
        "models": _forecast_model_results(),
    }


def _ticker_forecast(
    ticker: str,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "evaluation_split": "validation",
        "target_definition": "next_session_simple_return",
        "models": _forecast_model_results(),
    }


def _xai_status() -> dict[str, Any]:
    return {
        "model_name": "random_forest",
        "artifact_available": True,
    }


def _xai_summary() -> dict[str, Any]:
    return {
        "model_name": "random_forest",
        "explanation_split": "validation",
        "target_definition": "next_session_simple_return",
        "methods": [
            "shap",
            "partial_dependence",
            "accumulated_local_effects",
        ],
        "global_observations": 400,
        "features": [
            {
                "feature": "adjusted_close_to_ma_20",
                "mean_absolute_shap_value": 0.00018,
                "effect_available": True,
            },
            {
                "feature": "momentum_20",
                "mean_absolute_shap_value": 0.00017,
                "effect_available": True,
            },
        ],
        "effect_features": [
            "adjusted_close_to_ma_20",
            "momentum_20",
        ],
    }


def _xai_feature() -> dict[str, Any]:
    return {
        "feature_name": "adjusted_close_to_ma_20",
        "mean_absolute_shap_value": 0.00018,
        "partial_dependence": [
            {
                "feature_value": -0.05,
                "average_prediction": 0.0004,
            },
            {
                "feature_value": 0.05,
                "average_prediction": 0.0006,
            },
        ],
        "accumulated_local_effects": [
            {
                "lower_edge": -0.10,
                "upper_edge": 0.00,
                "bin_center": -0.05,
                "accumulated_effect": -0.0002,
                "sample_count": 50,
            },
            {
                "lower_edge": 0.00,
                "upper_edge": 0.10,
                "bin_center": 0.05,
                "accumulated_effect": 0.0002,
                "sample_count": 50,
            },
        ],
    }


def _xai_local() -> dict[str, Any]:
    return {
        "model_name": "random_forest",
        "ticker": "BTC-USD",
        "feature_date": "2022-06-12T00:00:00",
        "target_date": "2022-06-13T00:00:00",
        "actual_return": -0.1597,
        "predicted_return": 0.00175,
        "base_value": 0.00065,
        "reconstructed_prediction": 0.00175,
        "features": [
            {
                "feature": "adjusted_close_to_ma_5",
                "feature_value": -0.07,
                "shap_value": 0.0028,
            },
            {
                "feature": "intraday_range",
                "feature_value": 0.06,
                "shap_value": -0.0006,
            },
        ],
    }


def _risk_metrics(
    *,
    annualized_return: float,
    annualized_volatility: float,
    sharpe_ratio: float,
    maximum_drawdown: float,
) -> dict[str, Any]:
    return {
        "observations": 565,
        "periods_per_year": 252,
        "confidence_level": 0.95,
        "annual_risk_free_rate": 0.02,
        "mean_return": annualized_return / 252.0,
        "annualized_mean_return": annualized_return,
        "volatility": annualized_volatility / (252.0**0.5),
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sharpe_ratio + 0.2,
        "value_at_risk": 0.01,
        "conditional_value_at_risk": 0.015,
        "maximum_drawdown": maximum_drawdown,
    }


def _risk_block(
    *,
    annualized_return: float,
    annualized_volatility: float,
    sharpe_ratio: float,
    maximum_drawdown: float,
) -> dict[str, Any]:
    return {
        "metrics": _risk_metrics(
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            maximum_drawdown=maximum_drawdown,
        ),
        "concentration_index": 0.28,
        "effective_number_of_assets": 3.5,
        "diversification_ratio": 1.38,
    }


def _risk_market() -> dict[str, Any]:
    return {
        "evaluation_split": "validation",
        "return_definition": ("adjusted_close_simple_return_on_shared_market_dates"),
        "observations": 565,
        "periods_per_year": 252,
        "date_min": "2022-01-26",
        "date_max": "2024-04-25",
        "tickers": [
            "AGG",
            "SPY",
        ],
        "correlation_matrix": [
            {
                "ticker": "AGG",
                "correlations": {
                    "AGG": 1.0,
                    "SPY": 0.26,
                },
            },
            {
                "ticker": "SPY",
                "correlations": {
                    "AGG": 0.26,
                    "SPY": 1.0,
                },
            },
        ],
    }


def _risk_scenario_summary(
    scenario_name: str,
) -> dict[str, Any]:
    if scenario_name == "minimum_volatility":
        annualized_return = 0.053
        annualized_volatility = 0.095
        sharpe_ratio = 0.352
        maximum_drawdown = 0.153
        construction = "optimized_on_training_minimum_volatility"
    else:
        annualized_return = 0.116
        annualized_volatility = 0.164
        sharpe_ratio = 0.585
        maximum_drawdown = 0.279
        construction = "optimized_on_training_maximum_sharpe"

    return {
        "name": scenario_name,
        "construction": construction,
        "risk": _risk_block(
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            maximum_drawdown=maximum_drawdown,
        ),
    }


def _risk_scenarios() -> dict[str, Any]:
    return {
        "evaluation_split": "validation",
        "return_definition": ("adjusted_close_simple_return_on_shared_market_dates"),
        "scenarios": [
            _risk_scenario_summary("maximum_sharpe"),
            _risk_scenario_summary("minimum_volatility"),
        ],
    }


def _risk_detail(
    scenario_name: str,
) -> dict[str, Any]:
    if scenario_name == "minimum_volatility":
        construction = "optimized_on_training_minimum_volatility"
        validation_return = 0.053
        validation_volatility = 0.095
        validation_sharpe = 0.352
        validation_drawdown = 0.153
    else:
        construction = "optimized_on_training_maximum_sharpe"
        validation_return = 0.116
        validation_volatility = 0.164
        validation_sharpe = 0.585
        validation_drawdown = 0.279

    return {
        "name": scenario_name,
        "construction": construction,
        "return_definition": ("adjusted_close_simple_return_on_shared_market_dates"),
        "training": _risk_block(
            annualized_return=0.20,
            annualized_volatility=0.15,
            sharpe_ratio=1.2,
            maximum_drawdown=0.20,
        ),
        "validation": _risk_block(
            annualized_return=validation_return,
            annualized_volatility=validation_volatility,
            sharpe_ratio=validation_sharpe,
            maximum_drawdown=validation_drawdown,
        ),
    }


def _portfolio_validation(
    *,
    annualized_return: float,
    annualized_volatility: float,
    sharpe_ratio: float,
    maximum_drawdown: float,
) -> dict[str, Any]:
    return {
        "observations": 565,
        "annualized_mean_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sharpe_ratio + 0.2,
        "maximum_drawdown": maximum_drawdown,
        "concentration_index": 0.28,
        "effective_number_of_assets": 3.5,
        "diversification_ratio": 1.38,
    }


def _portfolio_scenario_summary(
    scenario_name: str,
) -> dict[str, Any]:
    if scenario_name == "minimum_volatility":
        construction = "optimized_on_training_minimum_volatility"
        annualized_return = 0.053
        annualized_volatility = 0.095
        sharpe_ratio = 0.352
        maximum_drawdown = 0.153
    else:
        construction = "optimized_on_training_maximum_sharpe"
        annualized_return = 0.116
        annualized_volatility = 0.164
        sharpe_ratio = 0.585
        maximum_drawdown = 0.279

    return {
        "name": scenario_name,
        "construction": construction,
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
        "validation": _portfolio_validation(
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            maximum_drawdown=maximum_drawdown,
        ),
    }


def _portfolio_scenarios() -> dict[str, Any]:
    return {
        "evaluation_split": "validation",
        "return_definition": ("adjusted_close_simple_return_on_shared_market_dates"),
        "tickers": [
            "AGG",
            "SPY",
        ],
        "scenarios": [
            _portfolio_scenario_summary("maximum_sharpe"),
            _portfolio_scenario_summary("minimum_volatility"),
        ],
    }


def _portfolio_detail(
    scenario_name: str,
) -> dict[str, Any]:
    summary = _portfolio_scenario_summary(scenario_name)

    return {
        "name": summary["name"],
        "construction": summary["construction"],
        "return_definition": ("adjusted_close_simple_return_on_shared_market_dates"),
        "weights": summary["weights"],
        "training": _portfolio_validation(
            annualized_return=0.20,
            annualized_volatility=0.15,
            sharpe_ratio=1.2,
            maximum_drawdown=0.20,
        ),
        "validation": summary["validation"],
        "validation_contributions": [
            {
                "ticker": "AGG",
                "weight": 0.33,
                "annualized_return_contribution": -0.01,
                "annualized_volatility_contribution": 0.02,
            },
            {
                "ticker": "SPY",
                "weight": 0.67,
                "annualized_return_contribution": 0.12,
                "annualized_volatility_contribution": 0.14,
            },
        ],
    }


def _portfolio_returns(
    scenario_name: str,
) -> dict[str, Any]:
    construction = (
        "optimized_on_training_minimum_volatility"
        if scenario_name == "minimum_volatility"
        else "optimized_on_training_maximum_sharpe"
    )

    return {
        "name": scenario_name,
        "construction": construction,
        "evaluation_split": "validation",
        "return_definition": ("adjusted_close_simple_return_on_shared_market_dates"),
        "observations": 2,
        "returns": [
            {
                "date": "2024-04-24",
                "portfolio_return": 0.01,
                "cumulative_return": 0.20,
            },
            {
                "date": "2024-04-25",
                "portfolio_return": 0.005,
                "cumulative_return": 0.206,
            },
        ],
    }


def _rag_status() -> dict[str, Any]:
    return {
        "corpus_available": True,
        "corpus_documents": 4,
        "golden_dataset_available": True,
        "golden_dataset_items": 13,
        "benchmark_available": True,
        "default_strategy": "vector",
        "strategies": [
            "vector",
            "hybrid",
            "reranked",
        ],
        "model_name": "gemma3:4b",
    }


def _rag_benchmark() -> dict[str, Any]:
    return {
        "best_configuration": "vector",
        "configurations": [
            {
                "strategy": "vector",
                "quality_score": 0.8173,
                "recall_at_k": 1.0,
                "precision_at_k": 0.2692,
                "mean_reciprocal_rank": 1.0,
                "duration_seconds": 25.51,
            },
            {
                "strategy": "hybrid",
                "quality_score": 0.8173,
                "recall_at_k": 1.0,
                "precision_at_k": 0.2692,
                "mean_reciprocal_rank": 1.0,
                "duration_seconds": 20.63,
            },
        ],
    }


def _rag_query_response(
    question: str,
    strategy: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": ("Ryzyko finansowe to możliwość poniesienia straty finansowej."),
        "retrieval_strategy": strategy,
        "citations": [
            {
                "document_id": "financial_risk",
                "source": "reference_corpus/financial_risk",
                "title": "Financial Risk",
                "page": None,
            },
        ],
        "retrieved_chunk_count": 4,
    }


def _agent_status() -> dict[str, Any]:
    return {
        "result_artifact_available": True,
        "forecast_artifact_available": True,
        "xai_artifact_available": True,
        "portfolio_artifact_available": True,
        "reinforcement_learning_artifact_available": True,
        "corpus_available": True,
        "runtime_inputs_available": True,
        "model_name": "gemma3:4b",
        "retrieval_strategy": "vector",
        "evaluation_basis": "validation_benchmark_artifacts",
        "test_split_used": False,
    }


def _agent_response(
    *,
    agent_name: str,
    question: str,
) -> dict[str, Any]:
    return {
        "agent_name": agent_name,
        "question": question,
        "summary": "Podsumowanie wyników walidacyjnych.",
        "findings": [
            "Najważniejsze ustalenie.",
        ],
        "risks": [
            "Wynik zależy od celu analizy.",
        ],
        "recommendations": [
            "Porównaj dostępne warianty.",
        ],
        "limitations": [
            "Analiza wykorzystuje dane walidacyjne.",
        ],
        "evidence": [
            {
                "source_type": "benchmark_artifact",
                "identifier": "portfolio_result.json",
                "description": "Portfolio validation results",
            },
        ],
    }


def _agent_results() -> dict[str, Any]:
    return {
        "model_name": "gemma3:4b",
        "retrieval_strategy": "vector",
        "evaluation_basis": "validation_benchmark_artifacts",
        "test_split_used": False,
        "responses": {
            "forecast": _agent_response(
                agent_name="forecast_analysis_agent",
                question="Forecast question",
            ),
            "risk": _agent_response(
                agent_name="risk_analysis_agent",
                question="Risk question",
            ),
            "allocation": _agent_response(
                agent_name="allocation_recommendation_agent",
                question="Allocation question",
            ),
        },
    }


def _decision_profiles() -> dict[str, Any]:
    return {
        "profiles": [
            {
                "profile_type": "conservative",
                "priorities": {
                    "return": 0.15,
                    "volatility": 0.30,
                    "drawdown": 0.35,
                    "diversification": 0.20,
                },
                "constraints": {
                    "maximum_annualized_volatility": 0.14,
                    "maximum_drawdown": 0.22,
                    "maximum_concentration_index": 0.35,
                    "minimum_effective_assets": 3.0,
                    "minimum_diversification_ratio": 1.0,
                    "minimum_cash_reserve": 0.10,
                },
                "liquidity_need": "high",
                "investment_horizon_years": 3,
            },
            {
                "profile_type": "balanced",
                "priorities": {
                    "return": 0.30,
                    "volatility": 0.25,
                    "drawdown": 0.25,
                    "diversification": 0.20,
                },
                "constraints": {
                    "maximum_annualized_volatility": 0.20,
                    "maximum_drawdown": 0.32,
                    "maximum_concentration_index": 0.40,
                    "minimum_effective_assets": 2.5,
                    "minimum_diversification_ratio": 1.0,
                    "minimum_cash_reserve": 0.05,
                },
                "liquidity_need": "medium",
                "investment_horizon_years": 5,
            },
            {
                "profile_type": "growth",
                "priorities": {
                    "return": 0.50,
                    "volatility": 0.15,
                    "drawdown": 0.15,
                    "diversification": 0.20,
                },
                "constraints": {
                    "maximum_annualized_volatility": 0.30,
                    "maximum_drawdown": 0.45,
                    "maximum_concentration_index": 0.50,
                    "minimum_effective_assets": 2.0,
                    "minimum_diversification_ratio": 1.0,
                    "minimum_cash_reserve": 0.0,
                },
                "liquidity_need": "low",
                "investment_horizon_years": 10,
            },
        ]
    }


def _decision_result(profile_type: str) -> dict[str, Any]:
    recommended = (
        "minimum_volatility" if profile_type == "conservative" else "equal_weight"
    )
    return {
        "deterministic": True,
        "llm_used_for_ranking": False,
        "portfolio_weights_modified": False,
        "provenance": {
            "evaluation_split": "validation",
            "observations": 565,
            "date_min": "2022-01-26",
            "date_max": "2024-04-25",
        },
        "profile": {"profile_type": profile_type},
        "available_liquidity_fraction": 0.10,
        "normalization": [],
        "recommended_scenario": recommended,
        "scenarios": [
            {
                "name": recommended,
                "construction": "frozen_validation_scenario",
                "weights": [
                    {"ticker": "AGG", "weight": 0.5},
                    {"ticker": "SPY", "weight": 0.5},
                ],
                "admissible": True,
                "rank": 1,
                "total_score": 0.7,
                "normalized_components": {
                    "return": 0.5,
                    "volatility": 0.8,
                    "drawdown": 0.8,
                    "diversification": 0.5,
                },
                "weighted_contributions": {
                    "return": 0.15,
                    "volatility": 0.2,
                    "drawdown": 0.2,
                    "diversification": 0.1,
                },
                "constraint_violations": [],
                "tradeoffs": [],
            }
        ],
    }


def _market_intelligence() -> dict[str, Any]:
    return {
        "generated_at_utc": "2026-08-14T12:30:00Z",
        "cache_ttl_seconds": 900,
        "refresh_requested": False,
        "available_count": 2,
        "stale_count": 0,
        "real_time": False,
        "disclaimer": "Dane informacyjne, nie są notowaniami czasu rzeczywistego.",
        "quotes": [
            {
                "ticker": ticker,
                "price": price,
                "previous_close": previous_close,
                "daily_change": price - previous_close,
                "daily_change_percent": (price / previous_close) - 1.0,
                "market_date": "2026-08-13",
                "retrieved_at_utc": "2026-08-14T12:30:00Z",
                "market_timezone": (
                    "UTC" if ticker == "BTC-USD" else "America/New_York"
                ),
                "currency": "USD",
                "source": "Yahoo Finance",
                "freshness": "fresh",
                "delivery": "live",
                "age_seconds": 0,
                "error": None,
            }
            for ticker, price, previous_close in (
                ("SPY", 645.0, 640.0),
                ("BTC-USD", 118000.0, 116000.0),
            )
        ],
    }


def install_healthy_api(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Install deterministic HTTP responses."""

    requested_calls: list[dict[str, Any]] = []

    def fake_request(
        **kwargs: Any,
    ) -> FakeResponse:
        requested_calls.append(dict(kwargs))

        url = str(kwargs["url"])

        payload = kwargs.get("json")

        if url.endswith("/api/v1/system/health"):
            return FakeResponse(
                {
                    "status": "healthy",
                }
            )

        if url.endswith("/api/v1/system/info"):
            return FakeResponse(
                {
                    "title": ("Financial Decision Support API"),
                    "version": "0.1.0",
                }
            )

        if url.endswith("/api/v1/forecast/models"):
            return FakeResponse(_forecast_models())

        if url.endswith("/api/v1/forecast/summary"):
            return FakeResponse(_forecast_summary())

        if "/api/v1/forecast/tickers/" in url:
            ticker = url.rsplit(
                "/",
                maxsplit=1,
            )[-1]

            return FakeResponse(_ticker_forecast(ticker))

        if url.endswith("/api/v1/xai/status"):
            return FakeResponse(_xai_status())

        if url.endswith("/api/v1/xai/summary"):
            return FakeResponse(_xai_summary())

        if url.endswith("/api/v1/xai/local"):
            return FakeResponse(_xai_local())

        if "/api/v1/xai/features/" in url:
            return FakeResponse(_xai_feature())

        if url.endswith("/api/v1/risk/market"):
            return FakeResponse(_risk_market())

        if url.endswith("/api/v1/risk/scenarios"):
            return FakeResponse(_risk_scenarios())

        if "/api/v1/risk/scenarios/" in url:
            scenario_name = url.rsplit(
                "/",
                maxsplit=1,
            )[-1]

            return FakeResponse(_risk_detail(scenario_name))

        if url.endswith("/api/v1/portfolio/scenarios"):
            return FakeResponse(_portfolio_scenarios())

        if "/api/v1/portfolio/scenarios/" in url and url.endswith("/returns"):
            scenario_name = url.rstrip("/").split("/")[-2]

            return FakeResponse(_portfolio_returns(scenario_name))

        if "/api/v1/portfolio/scenarios/" in url:
            scenario_name = url.rsplit(
                "/",
                maxsplit=1,
            )[-1]

            return FakeResponse(_portfolio_detail(scenario_name))

        if url.endswith("/api/v1/rag/status"):
            return FakeResponse(_rag_status())

        if url.endswith("/api/v1/rag/benchmark"):
            return FakeResponse(_rag_benchmark())

        if url.endswith("/api/v1/rag/query"):
            assert isinstance(
                payload,
                dict,
            )

            question = str(payload["question"])

            strategy = str(payload["strategy"])

            return FakeResponse(
                _rag_query_response(
                    question,
                    strategy,
                )
            )

        if url.endswith("/api/v1/agents/status"):
            return FakeResponse(_agent_status())

        if url.endswith("/api/v1/agents/results"):
            return FakeResponse(_agent_results())

        if url.endswith("/api/v1/decision/profiles"):
            return FakeResponse(_decision_profiles())

        if "/api/v1/market-intelligence/quotes" in url:
            response = _market_intelligence()
            response["refresh_requested"] = "refresh=true" in url
            return FakeResponse(response)

        if "/api/v1/decision/recommendations/" in url:
            profile_type = url.split("/recommendations/", maxsplit=1)[1].split(
                "?", maxsplit=1
            )[0]
            return FakeResponse(_decision_result(profile_type))

        if url.endswith("/api/v1/decision/evaluate"):
            assert isinstance(payload, dict)
            result = _decision_result("custom")
            custom_profile = payload["profile"]
            custom_priorities = custom_profile["priorities"]
            result["profile"] = {
                "profile_type": "custom",
                "priorities": {
                    "return": custom_priorities["return_priority"],
                    "volatility": custom_priorities["volatility_priority"],
                    "drawdown": custom_priorities["drawdown_priority"],
                    "diversification": custom_priorities["diversification_priority"],
                },
                "constraints": custom_profile["constraints"],
                "liquidity_need": custom_profile["liquidity_need"],
                "investment_horizon_years": custom_profile["investment_horizon_years"],
            }
            result["available_liquidity_fraction"] = payload[
                "available_liquidity_fraction"
            ]
            return FakeResponse(result)

        if url.endswith("/api/v1/agents/forecast"):
            assert isinstance(
                payload,
                dict,
            )

            return FakeResponse(
                _agent_response(
                    agent_name=("forecast_analysis_agent"),
                    question=str(payload["question"]),
                )
            )

        if url.endswith("/api/v1/agents/risk"):
            assert isinstance(
                payload,
                dict,
            )

            return FakeResponse(
                _agent_response(
                    agent_name=("risk_analysis_agent"),
                    question=str(payload["question"]),
                )
            )

        if url.endswith("/api/v1/agents/allocation"):
            assert isinstance(
                payload,
                dict,
            )

            return FakeResponse(
                _agent_response(
                    agent_name=("allocation_recommendation_agent"),
                    question=str(payload["question"]),
                )
            )

        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(
        requests,
        "request",
        fake_request,
    )

    return requested_calls


def _navigation_key(
    page_name: str,
) -> str:
    normalized = (
        page_name.lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "&",
            "and",
        )
    )

    return "navigation_" + normalized


def _find_by_key(
    elements: Any,
    key: str,
) -> Any:
    for element in elements:
        if (
            getattr(
                element,
                "key",
                None,
            )
            == key
        ):
            return element

    raise AssertionError(f"Widget with key {key!r} not found.")


def _navigate(
    application: AppTest,
    page_name: str,
) -> None:
    button = _find_by_key(
        application.sidebar.button,
        _navigation_key(page_name),
    )

    button.click().run()


def _requested_url(
    call: dict[str, Any],
) -> str:
    return str(
        call.get(
            "url",
            "",
        )
    )


def _was_requested(
    calls: list[dict[str, Any]],
    path: str,
) -> bool:
    return any(_requested_url(call).endswith(path) for call in calls)


def _calls_for_path(
    calls: list[dict[str, Any]],
    path: str,
) -> list[dict[str, Any]]:
    return [call for call in calls if _requested_url(call).endswith(path)]


def _calls_since(
    calls: list[dict[str, Any]],
    start_index: int,
) -> list[dict[str, Any]]:
    """Return HTTP calls recorded after a selected point."""
    return calls[start_index:]


def test_navigation_contract_is_explicit() -> None:
    assert app.NAVIGATION_OPTIONS == (
        app.DASHBOARD_PAGE,
        app.FORECASTING_PAGE,
        app.RISK_SCENARIOS_PAGE,
        app.OPTIMIZATION_PAGE,
        app.DECISION_WORKSPACE_PAGE,
        app.MARKET_INTELLIGENCE_PAGE,
        app.AI_INSIGHTS_PAGE,
        app.GOVERNANCE_PAGE,
        app.REPORTS_PAGE,
    )


def test_create_api_client_uses_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        app.API_URL_ENVIRONMENT_VARIABLE,
        "http://localhost:9999/",
    )

    client = app.create_api_client()

    assert client.base_url == "http://localhost:9999"


def test_create_api_client_uses_default_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        app.API_URL_ENVIRONMENT_VARIABLE,
        raising=False,
    )

    client = app.create_api_client()

    assert client.base_url == app.DEFAULT_API_URL


def test_load_api_status_returns_backend_data() -> None:
    client = StubAPIClient()

    (
        available,
        health,
        information,
        error_message,
    ) = app.load_api_status(client)

    assert available is True

    assert health == {
        "status": "healthy",
    }

    assert information == {
        "title": ("Financial Decision Support API"),
        "version": "0.1.0",
    }

    assert error_message is None


def test_load_api_status_handles_api_failure() -> None:
    client = StubAPIClient(error=APIClientError("Backend unavailable"))

    (
        available,
        health,
        information,
        error_message,
    ) = app.load_api_status(client)

    assert available is False
    assert health is None
    assert information is None

    assert error_message == "Backend unavailable"


def test_dashboard_renders_with_healthy_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    assert not application.exception

    assert application.main.title[0].value == "Dashboard"

    assert any(item.value == "API dostępne" for item in (application.sidebar.success))


def test_dashboard_action_opens_decision_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_healthy_api(monkeypatch)
    application = AppTest.from_file(str(APP_PATH)).run()

    _find_by_key(
        application.main.button,
        "dashboard_action_decision_workspace",
    ).click()
    application.run()

    assert not application.exception
    assert application.main.title[0].value == "Przestrzeń decyzyjna"


def test_sidebar_uses_module_buttons_instead_of_radio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    assert not application.exception

    assert len(application.sidebar.radio) == 0

    for page_name in app.NAVIGATION_OPTIONS:
        _find_by_key(
            application.sidebar.button,
            _navigation_key(page_name),
        )


def test_global_language_and_timezone_preferences_are_persistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_healthy_api(monkeypatch)
    application = AppTest.from_file(str(APP_PATH)).run()

    language = _find_by_key(
        application.sidebar.selectbox,
        app.LANGUAGE_STATE_KEY,
    )
    presentation_timezone = _find_by_key(
        application.sidebar.selectbox,
        app.TIMEZONE_STATE_KEY,
    )

    assert language.value == "pl"
    assert presentation_timezone.value == "Europe/Warsaw"

    language.set_value("en").run()
    presentation_timezone = _find_by_key(
        application.sidebar.selectbox,
        app.TIMEZONE_STATE_KEY,
    )
    presentation_timezone.set_value("America/New_York").run()
    _navigate(application, app.MARKET_INTELLIGENCE_PAGE)

    assert not application.exception
    assert application.main.title[0].value == "Market Intelligence"
    assert (
        _find_by_key(
            application.sidebar.selectbox,
            app.LANGUAGE_STATE_KEY,
        ).value
        == "en"
    )
    assert (
        _find_by_key(
            application.sidebar.selectbox,
            app.TIMEZONE_STATE_KEY,
        ).value
        == "America/New_York"
    )
    assert any(item.value == "API available" for item in application.sidebar.success)

    _navigate(application, app.DECISION_WORKSPACE_PAGE)

    assert not application.exception
    assert application.main.title[0].value == "Decision Workspace"


@pytest.mark.parametrize(
    (
        "page_name",
        "expected_title",
    ),
    [
        (
            app.FORECASTING_PAGE,
            "Prognozowanie",
        ),
        (
            app.RISK_SCENARIOS_PAGE,
            "Ryzyko i scenariusze",
        ),
        (
            app.OPTIMIZATION_PAGE,
            "Optymalizacja",
        ),
        (
            app.DECISION_WORKSPACE_PAGE,
            "Przestrzeń decyzyjna",
        ),
        (
            app.MARKET_INTELLIGENCE_PAGE,
            "Informacje rynkowe",
        ),
        (
            app.AI_INSIGHTS_PAGE,
            "Wnioski AI",
        ),
        (
            app.GOVERNANCE_PAGE,
            "Wyjaśnialność i nadzór nad modelem",
        ),
        (
            app.REPORTS_PAGE,
            "Raporty",
        ),
    ],
)
def test_button_navigation_renders_selected_module(
    monkeypatch: pytest.MonkeyPatch,
    page_name: str,
    expected_title: str,
) -> None:
    install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(
        application,
        page_name,
    )

    assert not application.exception
    assert application.main.title[0].value == expected_title


def test_language_change_localizes_core_analytical_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_healthy_api(monkeypatch)
    application = AppTest.from_file(str(APP_PATH)).run()

    language = _find_by_key(
        application.sidebar.selectbox,
        app.LANGUAGE_STATE_KEY,
    )
    language.set_value("en").run()

    assert application.main.title[0].value == "Dashboard"
    assert any(item.value == "Decision summary" for item in application.main.subheader)

    _navigate(application, app.FORECASTING_PAGE)

    assert not application.exception
    assert any(
        item.value == "Instrument analysis" for item in application.main.subheader
    )

    _navigate(application, app.RISK_SCENARIOS_PAGE)

    assert not application.exception
    assert application.main.title[0].value == "Risk and scenarios"
    assert any(item.value == "Risk profile" for item in application.main.subheader)

    _navigate(application, app.OPTIMIZATION_PAGE)

    assert not application.exception
    assert application.main.title[0].value == "Optimization"
    assert any(
        item.value == "Portfolio structure" for item in application.main.subheader
    )

    _navigate(application, app.GOVERNANCE_PAGE)

    assert not application.exception
    assert application.main.title[0].value == "Explainability and governance"
    assert any(
        item.value == "Global explainability" for item in application.main.subheader
    )

    _navigate(application, app.AI_INSIGHTS_PAGE)

    assert not application.exception
    assert application.main.title[0].value == "AI Insights"
    assert any(item.value == "Ask the sources" for item in application.main.subheader)

    _navigate(application, app.REPORTS_PAGE)

    assert not application.exception
    assert application.main.title[0].value == "Reports"
    assert any(item.label == "Generate report" for item in application.main.button)


def test_decision_workspace_submits_custom_profile_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_healthy_api(monkeypatch)
    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(application, app.DECISION_WORKSPACE_PAGE)

    submit = _find_by_key(
        application.main.button,
        "decision_custom_submit_balanced",
    )
    submit.click().run()

    assert not application.exception
    evaluation_calls = _calls_for_path(calls, "/api/v1/decision/evaluate")
    assert len(evaluation_calls) == 1
    payload = evaluation_calls[0]["json"]
    assert payload["profile"]["profile_type"] == "custom"
    assert sum(payload["profile"]["priorities"].values()) == pytest.approx(1.0)
    assert payload["available_liquidity_fraction"] == pytest.approx(0.10)
    assert any(
        item.value == "Wpływ zmiany założeń" for item in application.main.subheader
    )


def test_market_intelligence_manual_refresh_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_healthy_api(monkeypatch)
    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(application, app.MARKET_INTELLIGENCE_PAGE)

    assert not application.exception
    assert _was_requested(calls, "/api/v1/market-intelligence/quotes")

    refresh = _find_by_key(
        application.main.button,
        "market_intelligence_refresh",
    )
    refresh.click().run()

    assert not application.exception
    assert any(
        _requested_url(call).endswith("/api/v1/market-intelligence/quotes?refresh=true")
        for call in calls
    )


def test_forecasting_uses_only_forecast_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(
        application,
        app.FORECASTING_PAGE,
    )

    assert not application.exception

    assert _was_requested(
        calls,
        "/api/v1/forecast/models",
    )

    assert _was_requested(
        calls,
        "/api/v1/forecast/summary",
    )

    assert _was_requested(
        calls,
        "/api/v1/forecast/tickers/SPY",
    )

    assert not any("/api/v1/xai/" in _requested_url(call) for call in calls)


def test_forecasting_defaults_to_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(
        application,
        app.FORECASTING_PAGE,
    )

    ticker_select = _find_by_key(
        application.main.selectbox,
        "forecast_ticker",
    )

    assert ticker_select.value == "SPY"


def test_forecasting_can_change_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(
        application,
        app.FORECASTING_PAGE,
    )

    ticker_select = _find_by_key(
        application.main.selectbox,
        "forecast_ticker",
    )

    ticker_select.set_value("QQQ").run()

    assert not application.exception

    assert _was_requested(
        calls,
        "/api/v1/forecast/tickers/QQQ",
    )


def test_governance_uses_xai_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    navigation_start = len(calls)

    _navigate(
        application,
        app.GOVERNANCE_PAGE,
    )

    navigation_calls = _calls_since(
        calls,
        navigation_start,
    )

    assert not application.exception

    assert _was_requested(
        navigation_calls,
        "/api/v1/xai/status",
    )

    assert _was_requested(
        navigation_calls,
        "/api/v1/xai/summary",
    )

    assert _was_requested(
        navigation_calls,
        ("/api/v1/xai/features/adjusted_close_to_ma_20"),
    )

    assert _was_requested(
        navigation_calls,
        "/api/v1/xai/local",
    )

    assert not any(
        "/api/v1/forecast/" in _requested_url(call) for call in navigation_calls
    )


def test_risk_page_uses_only_risk_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    navigation_start = len(calls)

    _navigate(
        application,
        app.RISK_SCENARIOS_PAGE,
    )

    navigation_calls = _calls_since(
        calls,
        navigation_start,
    )

    assert not application.exception

    assert _was_requested(
        navigation_calls,
        "/api/v1/risk/market",
    )

    assert _was_requested(
        navigation_calls,
        "/api/v1/risk/scenarios",
    )

    assert _was_requested(
        navigation_calls,
        ("/api/v1/risk/scenarios/maximum_sharpe"),
    )

    assert not any(
        "/api/v1/portfolio/" in _requested_url(call) for call in navigation_calls
    )


def test_risk_page_defaults_to_maximum_sharpe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(
        application,
        app.RISK_SCENARIOS_PAGE,
    )

    scenario_select = _find_by_key(
        application.main.selectbox,
        "risk_scenario",
    )

    assert scenario_select.value == "maximum_sharpe"


def test_risk_page_can_change_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(
        application,
        app.RISK_SCENARIOS_PAGE,
    )

    scenario_select = _find_by_key(
        application.main.selectbox,
        "risk_scenario",
    )

    scenario_select.set_value("minimum_volatility").run()

    assert not application.exception

    assert _was_requested(
        calls,
        ("/api/v1/risk/scenarios/minimum_volatility"),
    )


def test_optimization_uses_only_portfolio_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    navigation_start = len(calls)

    _navigate(
        application,
        app.OPTIMIZATION_PAGE,
    )

    navigation_calls = _calls_since(
        calls,
        navigation_start,
    )

    assert not application.exception

    assert _was_requested(
        navigation_calls,
        "/api/v1/portfolio/scenarios",
    )

    assert _was_requested(
        navigation_calls,
        ("/api/v1/portfolio/scenarios/maximum_sharpe"),
    )

    assert _was_requested(
        navigation_calls,
        ("/api/v1/portfolio/scenarios/maximum_sharpe/returns"),
    )

    assert not any("/api/v1/risk/" in _requested_url(call) for call in navigation_calls)


def test_optimization_defaults_to_maximum_sharpe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(
        application,
        app.OPTIMIZATION_PAGE,
    )

    scenario_select = _find_by_key(
        application.main.selectbox,
        "optimization_scenario",
    )

    assert scenario_select.value == "maximum_sharpe"


def test_optimization_can_change_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(
        application,
        app.OPTIMIZATION_PAGE,
    )

    scenario_select = _find_by_key(
        application.main.selectbox,
        "optimization_scenario",
    )

    scenario_select.set_value("minimum_volatility").run()

    assert not application.exception

    assert _was_requested(
        calls,
        ("/api/v1/portfolio/scenarios/minimum_volatility"),
    )

    assert _was_requested(
        calls,
        ("/api/v1/portfolio/scenarios/minimum_volatility/returns"),
    )


def test_ai_insights_loads_rag_and_agent_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(
        application,
        app.AI_INSIGHTS_PAGE,
    )

    assert not application.exception

    expected_paths = [
        "/api/v1/rag/status",
        "/api/v1/rag/benchmark",
        "/api/v1/agents/status",
        "/api/v1/agents/results",
    ]

    for path in expected_paths:
        assert _was_requested(
            calls,
            path,
        )


def test_ai_insights_questions_are_empty_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(
        application,
        app.AI_INSIGHTS_PAGE,
    )

    rag_question = _find_by_key(
        application.main.text_area,
        "ai_insights_rag_question_v2",
    )

    agent_question = _find_by_key(
        application.main.text_area,
        "ai_insights_agent_question_v2",
    )

    assert rag_question.value == ""

    assert agent_question.value == ""


def test_ai_insights_rag_query_uses_long_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(
        application,
        app.AI_INSIGHTS_PAGE,
    )

    rag_question = _find_by_key(
        application.main.text_area,
        "ai_insights_rag_question_v2",
    )

    rag_question.set_value("Co to jest ryzyko finansowe?").run()

    submit_button = _find_by_key(
        application.main.button,
        "ai_insights_rag_submit_v2",
    )

    submit_button.click().run()

    assert not application.exception

    calls_for_query = _calls_for_path(
        calls,
        "/api/v1/rag/query",
    )

    assert calls_for_query

    last_call = calls_for_query[-1]

    assert last_call["timeout"] == 60.0

    assert last_call["json"] == {
        "question": ("Co to jest ryzyko finansowe?"),
        "strategy": "vector",
    }

    assert any(item.value == "Odpowiedź" for item in (application.main.subheader))


def test_ai_insights_allocation_agent_uses_long_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_healthy_api(monkeypatch)

    application = AppTest.from_file(str(APP_PATH)).run()

    _navigate(
        application,
        app.AI_INSIGHTS_PAGE,
    )

    agent_type = _find_by_key(
        application.main.selectbox,
        "ai_insights_agent_type_v2",
    )

    agent_type.set_value("allocation").run()

    agent_question = _find_by_key(
        application.main.text_area,
        "ai_insights_agent_question_v2",
    )

    agent_question.set_value(
        "Która strategia najlepiej odpowiada zrównoważonej preferencji ryzyka?"
    ).run()

    submit_button = _find_by_key(
        application.main.button,
        "ai_insights_agent_submit_v2",
    )

    submit_button.click().run()

    assert not application.exception

    allocation_calls = _calls_for_path(
        calls,
        "/api/v1/agents/allocation",
    )

    assert allocation_calls

    last_call = allocation_calls[-1]

    assert last_call["timeout"] == 60.0

    assert last_call["json"]["risk_preference"] == "balanced"

    assert last_call["json"]["response_language"] == "pl"

    assert last_call["json"]["question"] == (
        "Która strategia najlepiej odpowiada zrównoważonej preferencji ryzyka?"
    )

    assert any(
        item.value == "Agent analizy alokacji" for item in (application.main.subheader)
    )


def test_application_handles_unavailable_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_request(
        **kwargs: Any,
    ) -> FakeResponse:
        del kwargs

        raise (requests.exceptions.ConnectionError("Connection refused"))

    monkeypatch.setattr(
        requests,
        "request",
        failing_request,
    )

    application = AppTest.from_file(str(APP_PATH)).run()

    assert not application.exception

    assert any(item.value == "API niedostępne" for item in (application.sidebar.error))

    assert any(
        ("Nie udało się pobrać danych wymaganych przez Dashboard.") in item.value
        for item in (application.main.error)
    )


def test_application_title_contains_no_legacy_brand() -> None:
    assert "sentinel" not in (app.APPLICATION_TITLE.lower())

    assert app.APPLICATION_TITLE == ("System wspomagania decyzji finansowych")

    assert app.APPLICATION_SUBTITLE == ("Financial Decision Support System")

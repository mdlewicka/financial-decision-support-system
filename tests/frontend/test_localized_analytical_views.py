from frontend.forecast_view import (
    _build_metrics_frame,
    _format_metrics_frame,
)
from frontend.risk_view import (
    _build_scenario_comparison,
    _build_split_comparison,
    _format_scenario_comparison,
)


def _forecast_models() -> list[dict[str, object]]:
    return [
        {
            "model_name": "random_forest",
            "display_name": "Random Forest",
            "metrics": {
                "observations": 100,
                "mae": 0.01,
                "rmse": 0.02,
                "r2": 0.03,
                "directional_accuracy": 0.54,
            },
        }
    ]


def _risk_block() -> dict[str, object]:
    return {
        "metrics": {
            "observations": 100,
            "annualized_mean_return": 0.08,
            "annualized_volatility": 0.12,
            "sharpe_ratio": 0.5,
            "sortino_ratio": 0.7,
            "value_at_risk": 0.01,
            "conditional_value_at_risk": 0.02,
            "maximum_drawdown": 0.15,
        },
        "concentration_index": 0.3,
        "effective_number_of_assets": 3.2,
        "diversification_ratio": 1.4,
    }


def test_forecast_metrics_use_requested_language() -> None:
    polish = _format_metrics_frame(_build_metrics_frame(_forecast_models(), "pl"), "pl")
    english = _format_metrics_frame(
        _build_metrics_frame(_forecast_models(), "en"), "en"
    )

    assert "Obserwacje" in polish.columns
    assert "Trafność kierunku" in polish.columns
    assert "Observations" in english.columns
    assert "Directional accuracy" in english.columns
    assert english.loc[0, "Directional accuracy"] == "54.00%"


def test_risk_tables_use_requested_language_without_changing_values() -> None:
    scenarios = [
        {
            "name": "minimum_volatility",
            "risk": _risk_block(),
        }
    ]
    detail = {
        "training": _risk_block(),
        "validation": _risk_block(),
    }

    comparison = _format_scenario_comparison(
        _build_scenario_comparison(scenarios, "en"),
        "en",
    )
    splits = _build_split_comparison(detail, "en")

    assert comparison.loc[0, "Scenario"] == "Minimum volatility"
    assert comparison.loc[0, "Annualized return"] == "8.00%"
    assert splits["Split"].tolist() == ["training", "validation"]
    assert splits["Maximum drawdown"].tolist() == ["15.00%", "15.00%"]

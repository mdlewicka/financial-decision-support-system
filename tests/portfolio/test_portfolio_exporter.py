import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.src.portfolio.portfolio_analyzer import (
    PortfolioAnalysisConfig,
    PortfolioAnalyzer,
)
from backend.src.portfolio.portfolio_exporter import export_portfolio_benchmark
from backend.src.portfolio.portfolio_metrics import PortfolioScenarioEvaluation
from backend.src.risk.risk_analyzer import (
    MultiAssetRiskAnalysisConfig,
    MultiAssetRiskAnalyzer,
    RiskAnalysisConfig,
)
from backend.src.risk.risk_metrics import MultiAssetRiskResult


def create_market_data(multiplier: float = 1.0) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=7, freq="D")
    prices = {
        "SPY": [100.0, 101.0, 100.0, 103.0, 102.0, 104.0, 105.0],
        "AGG": [100.0, 100.2, 100.1, 100.4, 100.5, 100.6, 100.8],
        "GLD": [100.0, 99.0, 101.0, 100.5, 102.0, 101.5, 103.0],
    }
    rows = []
    for ticker, values in prices.items():
        for date_value, price in zip(dates, values, strict=True):
            rows.append(
                {
                    "date": date_value,
                    "ticker": ticker,
                    "adjusted_close": price * multiplier,
                }
            )
    return pd.DataFrame(rows)


def create_market_risk(multiplier: float = 1.0) -> MultiAssetRiskResult:
    return MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(
            periods_per_year=252,
            minimum_common_observations=6,
        )
    ).analyze(create_market_data(multiplier))


def create_scenarios(
    training_market_risk: MultiAssetRiskResult,
    validation_market_risk: MultiAssetRiskResult,
) -> tuple[PortfolioScenarioEvaluation, ...]:
    analyzer = PortfolioAnalyzer(
        config=PortfolioAnalysisConfig(
            risk_config=RiskAnalysisConfig(
                confidence_level=0.95,
                periods_per_year=252,
                minimum_observations=6,
            )
        )
    )
    definitions = {
        "Equal weight": pd.Series(
            {"AGG": 1 / 3, "GLD": 1 / 3, "SPY": 1 / 3},
            dtype=float,
        ),
        "Defensive": pd.Series(
            {"AGG": 0.60, "GLD": 0.30, "SPY": 0.10},
            dtype=float,
        ),
    }
    return tuple(
        PortfolioScenarioEvaluation(
            name=name,
            construction="manual",
            training=analyzer.analyze(
                training_market_risk.aligned_returns,
                weights,
            ),
            validation=analyzer.analyze(
                validation_market_risk.aligned_returns,
                weights,
            ),
        )
        for name, weights in definitions.items()
    )


def test_export_writes_complete_deterministic_artifacts(tmp_path: Path) -> None:
    training_market_risk = create_market_risk()
    validation_market_risk = create_market_risk(multiplier=1.05)
    scenarios = create_scenarios(training_market_risk, validation_market_risk)

    paths = export_portfolio_benchmark(
        scenarios=scenarios,
        training_market_risk=training_market_risk,
        validation_market_risk=validation_market_risk,
        output_directory=tmp_path,
    )

    for path in (
        paths.summary_path,
        paths.weights_path,
        paths.contributions_path,
        paths.returns_path,
        paths.training_correlation_path,
        paths.validation_correlation_path,
        paths.training_covariance_path,
        paths.validation_covariance_path,
        paths.json_path,
    ):
        assert path.exists()

    summary = pd.read_csv(paths.summary_path)
    weights = pd.read_csv(paths.weights_path)
    contributions = pd.read_csv(paths.contributions_path)
    returns = pd.read_csv(paths.returns_path)

    assert len(summary) == 4
    assert list(summary[["scenario", "split"]].itertuples(index=False, name=None)) == [
        ("Defensive", "training"),
        ("Defensive", "validation"),
        ("Equal weight", "training"),
        ("Equal weight", "validation"),
    ]
    assert len(weights) == 6
    assert len(contributions) == 12
    assert len(returns) == 24
    optional_columns = [
        "sharpe_ratio",
        "sortino_ratio",
        "diversification_ratio",
    ]
    required_numeric = summary.select_dtypes(include="number").drop(
        columns=optional_columns
    )
    assert np.isfinite(required_numeric.to_numpy(dtype=float)).all()
    optional_values = summary[optional_columns].to_numpy(dtype=float)
    assert (np.isfinite(optional_values) | np.isnan(optional_values)).all()


def test_export_json_records_schema_splits_and_scenarios(tmp_path: Path) -> None:
    training_market_risk = create_market_risk()
    validation_market_risk = create_market_risk(multiplier=1.05)

    paths = export_portfolio_benchmark(
        scenarios=create_scenarios(training_market_risk, validation_market_risk),
        training_market_risk=training_market_risk,
        validation_market_risk=validation_market_risk,
        output_directory=tmp_path,
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["return_definition"] == (
        "adjusted_close_simple_return_on_shared_market_dates"
    )
    assert payload["market_splits"]["training"]["observations"] == 6
    assert payload["market_splits"]["validation"]["tickers"] == [
        "AGG",
        "GLD",
        "SPY",
    ]
    assert [scenario["name"] for scenario in payload["scenarios"]] == [
        "Defensive",
        "Equal weight",
    ]


def test_export_rejects_empty_or_duplicate_scenarios(tmp_path: Path) -> None:
    training_market_risk = create_market_risk()
    validation_market_risk = create_market_risk(multiplier=1.05)
    scenarios = create_scenarios(training_market_risk, validation_market_risk)

    with pytest.raises(ValueError, match="scenarios must not be empty"):
        export_portfolio_benchmark(
            scenarios=(),
            training_market_risk=training_market_risk,
            validation_market_risk=validation_market_risk,
            output_directory=tmp_path,
        )

    duplicate = PortfolioScenarioEvaluation(
        name=scenarios[0].name,
        construction="duplicate",
        training=scenarios[1].training,
        validation=scenarios[1].validation,
    )
    with pytest.raises(ValueError, match="scenario names must be unique"):
        export_portfolio_benchmark(
            scenarios=(scenarios[0], duplicate),
            training_market_risk=training_market_risk,
            validation_market_risk=validation_market_risk,
            output_directory=tmp_path,
        )


def test_export_rejects_weights_changed_between_splits(tmp_path: Path) -> None:
    training_market_risk = create_market_risk()
    validation_market_risk = create_market_risk(multiplier=1.05)
    scenarios = create_scenarios(training_market_risk, validation_market_risk)
    changed = PortfolioScenarioEvaluation(
        name="Changed",
        construction="invalid",
        training=scenarios[0].training,
        validation=scenarios[1].validation,
    )

    with pytest.raises(ValueError, match="weights must remain fixed"):
        export_portfolio_benchmark(
            scenarios=(changed,),
            training_market_risk=training_market_risk,
            validation_market_risk=validation_market_risk,
            output_directory=tmp_path,
        )

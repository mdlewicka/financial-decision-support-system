from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.src.portfolio.portfolio_metrics import (
    PortfolioAnalysisResult,
    PortfolioScenarioEvaluation,
)
from backend.src.risk.risk_metrics import MultiAssetRiskResult


@dataclass(frozen=True, slots=True)
class PortfolioExportPaths:
    """Paths created by one portfolio benchmark export."""

    summary_path: Path
    weights_path: Path
    contributions_path: Path
    returns_path: Path
    training_correlation_path: Path
    validation_correlation_path: Path
    training_covariance_path: Path
    validation_covariance_path: Path
    json_path: Path


def export_portfolio_benchmark(
    scenarios: tuple[PortfolioScenarioEvaluation, ...],
    training_market_risk: MultiAssetRiskResult,
    validation_market_risk: MultiAssetRiskResult,
    output_directory: Path,
) -> PortfolioExportPaths:
    """Export deterministic portfolio comparisons for notebooks and UI."""

    validated_scenarios = _validate_export_inputs(
        scenarios=scenarios,
        training_market_risk=training_market_risk,
        validation_market_risk=validation_market_risk,
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = PortfolioExportPaths(
        summary_path=output_directory / "portfolio_summary.csv",
        weights_path=output_directory / "portfolio_weights.csv",
        contributions_path=output_directory / "portfolio_contributions.csv",
        returns_path=output_directory / "portfolio_returns.csv",
        training_correlation_path=output_directory / "training_correlation_matrix.csv",
        validation_correlation_path=output_directory
        / "validation_correlation_matrix.csv",
        training_covariance_path=output_directory
        / "training_annualized_covariance_matrix.csv",
        validation_covariance_path=output_directory
        / "validation_annualized_covariance_matrix.csv",
        json_path=output_directory / "portfolio_result.json",
    )

    _build_summary(validated_scenarios).to_csv(paths.summary_path, index=False)
    _build_weights(validated_scenarios).to_csv(paths.weights_path, index=False)
    _build_contributions(validated_scenarios).to_csv(
        paths.contributions_path,
        index=False,
    )
    _build_returns(validated_scenarios).to_csv(paths.returns_path, index=False)

    training_market_risk.correlation_matrix.to_csv(
        paths.training_correlation_path,
        index=True,
        index_label="ticker",
    )
    validation_market_risk.correlation_matrix.to_csv(
        paths.validation_correlation_path,
        index=True,
        index_label="ticker",
    )
    training_market_risk.annualized_covariance_matrix.to_csv(
        paths.training_covariance_path,
        index=True,
        index_label="ticker",
    )
    validation_market_risk.annualized_covariance_matrix.to_csv(
        paths.validation_covariance_path,
        index=True,
        index_label="ticker",
    )

    payload = _build_json_payload(
        scenarios=validated_scenarios,
        training_market_risk=training_market_risk,
        validation_market_risk=validation_market_risk,
    )
    paths.json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return paths


def _validate_export_inputs(
    scenarios: tuple[PortfolioScenarioEvaluation, ...],
    training_market_risk: MultiAssetRiskResult,
    validation_market_risk: MultiAssetRiskResult,
) -> tuple[PortfolioScenarioEvaluation, ...]:
    if not scenarios:
        raise ValueError("scenarios must not be empty.")

    if not isinstance(training_market_risk, MultiAssetRiskResult) or not isinstance(
        validation_market_risk,
        MultiAssetRiskResult,
    ):
        raise TypeError("market risk inputs must be MultiAssetRiskResult instances.")

    if training_market_risk.tickers != validation_market_risk.tickers:
        raise ValueError("training and validation market tickers must match.")

    names: set[str] = set()

    for scenario in scenarios:
        if not isinstance(scenario, PortfolioScenarioEvaluation):
            raise TypeError(
                "scenarios must contain PortfolioScenarioEvaluation instances."
            )

        if not isinstance(scenario.name, str) or not scenario.name.strip():
            raise ValueError("scenario names must be non-empty.")

        if scenario.name in names:
            raise ValueError("scenario names must be unique.")

        names.add(scenario.name)

        if (
            not isinstance(scenario.construction, str)
            or not scenario.construction.strip()
        ):
            raise ValueError("scenario construction labels must be non-empty.")

        expected_tickers = training_market_risk.tickers
        training_tickers = tuple(scenario.training.weights.index)
        validation_tickers = tuple(scenario.validation.weights.index)

        if (
            training_tickers != expected_tickers
            or validation_tickers != expected_tickers
        ):
            raise ValueError(
                "scenario weights must match the exported market ticker order."
            )

        if not np.allclose(
            scenario.training.weights.to_numpy(dtype=float),
            scenario.validation.weights.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "scenario weights must remain fixed between training and validation."
            )

    return tuple(sorted(scenarios, key=lambda scenario: scenario.name))


def _build_summary(
    scenarios: tuple[PortfolioScenarioEvaluation, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for scenario in scenarios:
        for split, analysis in _iter_splits(scenario):
            metrics = analysis.risk_metrics

            rows.append(
                {
                    "scenario": scenario.name,
                    "construction": scenario.construction,
                    "split": split,
                    "observations": metrics.observations,
                    "periods_per_year": metrics.periods_per_year,
                    "confidence_level": metrics.confidence_level,
                    "annual_risk_free_rate": metrics.annual_risk_free_rate,
                    "mean_return": metrics.mean_return,
                    "annualized_mean_return": metrics.annualized_mean_return,
                    "volatility": metrics.volatility,
                    "annualized_volatility": metrics.annualized_volatility,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "sortino_ratio": metrics.sortino_ratio,
                    "value_at_risk": metrics.value_at_risk,
                    "conditional_value_at_risk": (metrics.conditional_value_at_risk),
                    "maximum_drawdown": metrics.maximum_drawdown,
                    "concentration_index": analysis.concentration_index,
                    "effective_number_of_assets": (analysis.effective_number_of_assets),
                    "diversification_ratio": analysis.diversification_ratio,
                }
            )

    return pd.DataFrame(rows)


def _build_weights(
    scenarios: tuple[PortfolioScenarioEvaluation, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for scenario in scenarios:
        for ticker, weight in scenario.training.weights.items():
            rows.append(
                {
                    "scenario": scenario.name,
                    "construction": scenario.construction,
                    "ticker": str(ticker),
                    "weight": _finite_float(weight),
                }
            )

    return pd.DataFrame(rows)


def _build_contributions(
    scenarios: tuple[PortfolioScenarioEvaluation, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for scenario in scenarios:
        for split, analysis in _iter_splits(scenario):
            for ticker in analysis.weights.index:
                rows.append(
                    {
                        "scenario": scenario.name,
                        "construction": scenario.construction,
                        "split": split,
                        "ticker": str(ticker),
                        "weight": _finite_float(analysis.weights.loc[ticker]),
                        "annualized_return_contribution": _finite_float(
                            analysis.annualized_return_contributions.loc[ticker]
                        ),
                        "annualized_volatility_contribution": _finite_float(
                            analysis.annualized_volatility_contributions.loc[ticker]
                        ),
                    }
                )

    return pd.DataFrame(rows)


def _build_returns(
    scenarios: tuple[PortfolioScenarioEvaluation, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for scenario in scenarios:
        for split, analysis in _iter_splits(scenario):
            for date_value, return_value in analysis.portfolio_returns.items():
                date_text = str(date_value)

                rows.append(
                    {
                        "scenario": scenario.name,
                        "construction": scenario.construction,
                        "split": split,
                        "date": pd.Timestamp(date_text).date().isoformat(),
                        "portfolio_return": _finite_float(return_value),
                    }
                )

    return pd.DataFrame(rows)


def _build_json_payload(
    scenarios: tuple[PortfolioScenarioEvaluation, ...],
    training_market_risk: MultiAssetRiskResult,
    validation_market_risk: MultiAssetRiskResult,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "return_definition": ("adjusted_close_simple_return_on_shared_market_dates"),
        "market_splits": {
            "training": _market_split_payload(training_market_risk),
            "validation": _market_split_payload(validation_market_risk),
        },
        "scenarios": [
            {
                "name": scenario.name,
                "construction": scenario.construction,
                "weights": {
                    str(ticker): _finite_float(weight)
                    for ticker, weight in scenario.training.weights.items()
                },
                "training": _analysis_payload(scenario.training),
                "validation": _analysis_payload(scenario.validation),
            }
            for scenario in scenarios
        ],
    }


def _market_split_payload(
    result: MultiAssetRiskResult,
) -> dict[str, object]:
    return {
        "observations": result.observations,
        "periods_per_year": result.periods_per_year,
        "date_min": result.date_min.date().isoformat(),
        "date_max": result.date_max.date().isoformat(),
        "tickers": list(result.tickers),
    }


def _analysis_payload(
    analysis: PortfolioAnalysisResult,
) -> dict[str, object]:
    metrics = {
        key: (
            _optional_finite_float(value)
            if isinstance(value, float) or value is None
            else value
        )
        for key, value in asdict(analysis.risk_metrics).items()
    }

    return {
        "metrics": metrics,
        "concentration_index": _finite_float(analysis.concentration_index),
        "effective_number_of_assets": _finite_float(
            analysis.effective_number_of_assets
        ),
        "diversification_ratio": _optional_finite_float(analysis.diversification_ratio),
    }


def _iter_splits(
    scenario: PortfolioScenarioEvaluation,
) -> tuple[
    tuple[
        str,
        PortfolioAnalysisResult,
    ],
    ...,
]:
    return (
        (
            "training",
            scenario.training,
        ),
        (
            "validation",
            scenario.validation,
        ),
    )


def _finite_float(
    value: float,
) -> float:
    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError("portfolio exports must contain only finite values.")

    return numeric_value


def _optional_finite_float(
    value: float | None,
) -> float | None:
    return None if value is None else _finite_float(value)

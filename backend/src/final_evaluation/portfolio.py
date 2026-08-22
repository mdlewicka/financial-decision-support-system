from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.src.portfolio.portfolio_analyzer import (
    PortfolioAnalysisConfig,
    PortfolioAnalyzer,
)
from backend.src.risk.risk_analyzer import (
    MultiAssetRiskAnalysisConfig,
    MultiAssetRiskAnalyzer,
    RiskAnalysisConfig,
)


@dataclass(frozen=True)
class FrozenPortfolio:
    name: str
    construction: str
    weights: pd.Series


def load_frozen_portfolios(
    path: Path,
) -> tuple[FrozenPortfolio, ...]:
    frame = pd.read_csv(Path(path))

    required_columns = {
        "scenario",
        "construction",
        "ticker",
        "weight",
    }

    missing_columns = sorted(required_columns.difference(frame.columns))

    if missing_columns:
        raise ValueError(
            f"Portfolio weights artifact is missing required columns: {missing_columns}"
        )

    if frame.empty:
        raise ValueError("Portfolio weights artifact must not be empty.")

    if (
        frame[
            [
                "scenario",
                "ticker",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise ValueError("Portfolio weights contain duplicate scenario/ticker rows.")

    portfolios: list[FrozenPortfolio] = []

    for scenario_name, scenario_frame in frame.groupby(
        "scenario",
        sort=True,
    ):
        constructions = scenario_frame["construction"].astype(str).unique().tolist()

        if len(constructions) != 1:
            raise ValueError(
                "Every portfolio scenario must have exactly one construction value."
            )

        weights = pd.Series(
            data=pd.to_numeric(
                scenario_frame["weight"],
                errors="raise",
            ).to_numpy(
                dtype=float,
            ),
            index=(scenario_frame["ticker"].astype(str).tolist()),
            dtype=float,
            name="weight",
        )

        if abs(float(weights.sum()) - 1.0) > 1e-8:
            raise ValueError(f"Frozen weights for {scenario_name} must sum to 1.")

        portfolios.append(
            FrozenPortfolio(
                name=str(scenario_name),
                construction=(constructions[0]),
                weights=weights,
            )
        )

    return tuple(portfolios)


def evaluate_frozen_portfolios(
    evaluation_dataset: pd.DataFrame,
    weights_path: Path,
    confidence_level: float = 0.95,
    annual_risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    minimum_common_observations: int = 100,
) -> pd.DataFrame:
    frozen_portfolios = load_frozen_portfolios(
        path=weights_path,
    )

    market_risk = MultiAssetRiskAnalyzer(
        config=MultiAssetRiskAnalysisConfig(
            periods_per_year=periods_per_year,
            minimum_common_observations=(minimum_common_observations),
        )
    ).analyze(evaluation_dataset)

    analyzer = PortfolioAnalyzer(
        config=PortfolioAnalysisConfig(
            risk_config=RiskAnalysisConfig(
                confidence_level=(confidence_level),
                annual_risk_free_rate=(annual_risk_free_rate),
                periods_per_year=(periods_per_year),
                minimum_observations=(minimum_common_observations),
            )
        )
    )

    expected_tickers = set(market_risk.tickers)

    rows: list[dict[str, object]] = []

    for portfolio in frozen_portfolios:
        portfolio_tickers = set(portfolio.weights.index)

        if portfolio_tickers != expected_tickers:
            missing = sorted(expected_tickers.difference(portfolio_tickers))
            unexpected = sorted(portfolio_tickers.difference(expected_tickers))

            raise ValueError(
                "Frozen portfolio tickers do not match "
                "the evaluation dataset "
                f"(missing={missing}; "
                f"unexpected={unexpected})."
            )

        analysis = analyzer.analyze(
            aligned_returns=(market_risk.aligned_returns),
            weights=portfolio.weights,
        )

        metrics = analysis.risk_metrics

        rows.append(
            {
                "scenario": portfolio.name,
                "construction": (portfolio.construction),
                "observations": (metrics.observations),
                "periods_per_year": (metrics.periods_per_year),
                "confidence_level": (metrics.confidence_level),
                "annual_risk_free_rate": (metrics.annual_risk_free_rate),
                "mean_return": (metrics.mean_return),
                "annualized_mean_return": (metrics.annualized_mean_return),
                "volatility": (metrics.volatility),
                "annualized_volatility": (metrics.annualized_volatility),
                "sharpe_ratio": (metrics.sharpe_ratio),
                "sortino_ratio": (metrics.sortino_ratio),
                "value_at_risk": (metrics.value_at_risk),
                "conditional_value_at_risk": (metrics.conditional_value_at_risk),
                "maximum_drawdown": (metrics.maximum_drawdown),
                "concentration_index": (analysis.concentration_index),
                "effective_number_of_assets": (analysis.effective_number_of_assets),
                "diversification_ratio": (analysis.diversification_ratio),
            }
        )

    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)

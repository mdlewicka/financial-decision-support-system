from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.src.config import MARKET_DATASETS_DIR, PROJECT_ROOT, SUPPORTED_TICKERS
from backend.src.portfolio.markowitz_optimizer import (
    MarkowitzOptimizationConfig,
    MarkowitzOptimizer,
)
from backend.src.portfolio.portfolio_analyzer import (
    PortfolioAnalysisConfig,
    PortfolioAnalyzer,
)
from backend.src.portfolio.portfolio_exporter import (
    PortfolioExportPaths,
    export_portfolio_benchmark,
)
from backend.src.portfolio.portfolio_metrics import (
    PortfolioObjective,
    PortfolioScenarioEvaluation,
)
from backend.src.risk.risk_analyzer import (
    MultiAssetRiskAnalysisConfig,
    MultiAssetRiskAnalyzer,
    RiskAnalysisConfig,
)
from backend.src.risk.risk_metrics import MultiAssetRiskResult
from backend.src.storage.market_dataset_reader import MarketDatasetReader

DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "benchmark_results" / "risk" / "portfolio"


@dataclass(frozen=True, slots=True)
class PortfolioBenchmarkResult:
    """In-memory result and exported paths from one benchmark run."""

    scenarios: tuple[PortfolioScenarioEvaluation, ...]
    training_market_risk: MultiAssetRiskResult
    validation_market_risk: MultiAssetRiskResult
    export_paths: PortfolioExportPaths


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare manual and training-optimized portfolios on validation data."
        )
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=MARKET_DATASETS_DIR,
        help="Directory containing train.csv, validation.csv and test.csv.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for reproducible CSV and JSON artifacts.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--annual-risk-free-rate",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
    )
    parser.add_argument(
        "--minimum-common-observations",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--maximum-weight",
        type=float,
        default=0.35,
    )

    return parser.parse_args(arguments)


def run_portfolio_benchmark(
    input_directory: Path = MARKET_DATASETS_DIR,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    confidence_level: float = 0.95,
    annual_risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    minimum_common_observations: int = 100,
    maximum_weight: float = 0.35,
) -> PortfolioBenchmarkResult:
    """Select weights on training and evaluate fixed allocations on validation."""

    training_dataset, validation_dataset, _ = MarketDatasetReader(
        input_directory=input_directory
    ).read()

    multi_asset_config = MultiAssetRiskAnalysisConfig(
        periods_per_year=periods_per_year,
        minimum_common_observations=minimum_common_observations,
    )
    multi_asset_analyzer = MultiAssetRiskAnalyzer(config=multi_asset_config)
    training_market_risk = multi_asset_analyzer.analyze(training_dataset)
    validation_market_risk = multi_asset_analyzer.analyze(validation_dataset)

    if training_market_risk.tickers != validation_market_risk.tickers:
        raise ValueError("Training and validation tickers must match.")

    risk_config = RiskAnalysisConfig(
        confidence_level=confidence_level,
        annual_risk_free_rate=annual_risk_free_rate,
        periods_per_year=periods_per_year,
        minimum_observations=minimum_common_observations,
    )
    analysis_config = PortfolioAnalysisConfig(risk_config=risk_config)
    portfolio_analyzer = PortfolioAnalyzer(config=analysis_config)
    optimizer = MarkowitzOptimizer(
        config=MarkowitzOptimizationConfig(
            analysis_config=analysis_config,
            maximum_weight=maximum_weight,
        )
    )

    scenarios = list(
        _build_manual_scenarios(
            tickers=training_market_risk.tickers,
            training_returns=training_market_risk.aligned_returns,
            validation_returns=validation_market_risk.aligned_returns,
            analyzer=portfolio_analyzer,
        )
    )
    scenarios.extend(
        _build_optimized_scenarios(
            training_returns=training_market_risk.aligned_returns,
            validation_returns=validation_market_risk.aligned_returns,
            analyzer=portfolio_analyzer,
            optimizer=optimizer,
        )
    )
    ordered_scenarios = tuple(sorted(scenarios, key=lambda scenario: scenario.name))

    export_paths = export_portfolio_benchmark(
        scenarios=ordered_scenarios,
        training_market_risk=training_market_risk,
        validation_market_risk=validation_market_risk,
        output_directory=output_directory,
    )

    return PortfolioBenchmarkResult(
        scenarios=ordered_scenarios,
        training_market_risk=training_market_risk,
        validation_market_risk=validation_market_risk,
        export_paths=export_paths,
    )


def _build_manual_scenarios(
    tickers: tuple[str, ...],
    training_returns: pd.DataFrame,
    validation_returns: pd.DataFrame,
    analyzer: PortfolioAnalyzer,
) -> tuple[PortfolioScenarioEvaluation, ...]:
    expected_tickers = set(SUPPORTED_TICKERS)
    if set(tickers) != expected_tickers:
        missing = sorted(expected_tickers.difference(tickers))
        unexpected = sorted(set(tickers).difference(expected_tickers))
        raise ValueError(
            "Portfolio benchmark requires the configured supported tickers "
            f"(missing: {missing}; unexpected: {unexpected})."
        )

    definitions = {
        "equal_weight": (
            "manual_equal_weight",
            pd.Series(1.0 / len(tickers), index=tickers, dtype=float),
        ),
        "growth": (
            "manual_growth",
            pd.Series(
                {
                    "AGG": 0.10,
                    "BTC-USD": 0.10,
                    "EPOL": 0.10,
                    "GLD": 0.05,
                    "QQQ": 0.20,
                    "SPY": 0.20,
                    "VGK": 0.10,
                    "VT": 0.15,
                },
                dtype=float,
            ),
        ),
        "defensive": (
            "manual_defensive",
            pd.Series(
                {
                    "AGG": 0.35,
                    "BTC-USD": 0.03,
                    "EPOL": 0.05,
                    "GLD": 0.20,
                    "QQQ": 0.07,
                    "SPY": 0.10,
                    "VGK": 0.08,
                    "VT": 0.12,
                },
                dtype=float,
            ),
        ),
    }

    return tuple(
        _evaluate_fixed_weights(
            name=name,
            construction=construction,
            weights=weights,
            training_returns=training_returns,
            validation_returns=validation_returns,
            analyzer=analyzer,
        )
        for name, (construction, weights) in definitions.items()
    )


def _build_optimized_scenarios(
    training_returns: pd.DataFrame,
    validation_returns: pd.DataFrame,
    analyzer: PortfolioAnalyzer,
    optimizer: MarkowitzOptimizer,
) -> tuple[PortfolioScenarioEvaluation, ...]:
    definitions = (
        (
            "minimum_volatility",
            "optimized_on_training_minimum_volatility",
            PortfolioObjective.MINIMUM_VOLATILITY,
        ),
        (
            "maximum_sharpe",
            "optimized_on_training_maximum_sharpe",
            PortfolioObjective.MAXIMUM_SHARPE,
        ),
    )
    scenarios = []
    for name, construction, objective in definitions:
        optimization = optimizer.optimize(
            aligned_returns=training_returns,
            objective=objective,
        )
        training_analysis = optimization.analysis
        validation_analysis = analyzer.analyze(
            aligned_returns=validation_returns,
            weights=training_analysis.weights,
        )
        scenarios.append(
            PortfolioScenarioEvaluation(
                name=name,
                construction=construction,
                training=training_analysis,
                validation=validation_analysis,
            )
        )

    return tuple(scenarios)


def _evaluate_fixed_weights(
    name: str,
    construction: str,
    weights: pd.Series,
    training_returns: pd.DataFrame,
    validation_returns: pd.DataFrame,
    analyzer: PortfolioAnalyzer,
) -> PortfolioScenarioEvaluation:
    training_analysis = analyzer.analyze(
        aligned_returns=training_returns,
        weights=weights,
    )
    validation_analysis = analyzer.analyze(
        aligned_returns=validation_returns,
        weights=training_analysis.weights,
    )
    return PortfolioScenarioEvaluation(
        name=name,
        construction=construction,
        training=training_analysis,
        validation=validation_analysis,
    )


def main(
    arguments: Sequence[str] | None = None,
) -> None:
    parsed = parse_arguments(arguments)
    result = run_portfolio_benchmark(
        input_directory=parsed.input_directory,
        output_directory=parsed.output_directory,
        confidence_level=parsed.confidence_level,
        annual_risk_free_rate=parsed.annual_risk_free_rate,
        periods_per_year=parsed.periods_per_year,
        minimum_common_observations=parsed.minimum_common_observations,
        maximum_weight=parsed.maximum_weight,
    )

    print("Weight selection split: training")
    print("Evaluation split: validation")
    print("Scenarios:")
    for scenario in result.scenarios:
        metrics = scenario.validation.risk_metrics
        sharpe = (
            "undefined"
            if metrics.sharpe_ratio is None
            else f"{metrics.sharpe_ratio:.6f}"
        )
        print(
            f"  {scenario.name}: "
            f"return={metrics.annualized_mean_return:.6f}, "
            f"volatility={metrics.annualized_volatility:.6f}, "
            f"Sharpe={sharpe}, "
            f"max_drawdown={metrics.maximum_drawdown:.6f}"
        )

    paths = result.export_paths
    print(f"Summary: {paths.summary_path}")
    print(f"Weights: {paths.weights_path}")
    print(f"Contributions: {paths.contributions_path}")
    print(f"Portfolio returns: {paths.returns_path}")
    print(f"JSON result: {paths.json_path}")


if __name__ == "__main__":
    main()

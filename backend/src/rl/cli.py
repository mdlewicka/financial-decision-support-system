from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.src.config import MARKET_DATASETS_DIR, PROJECT_ROOT
from backend.src.portfolio.markowitz_optimizer import (
    MarkowitzOptimizationConfig,
    MarkowitzOptimizer,
)
from backend.src.portfolio.portfolio_analyzer import (
    PortfolioAnalysisConfig,
    PortfolioAnalyzer,
)
from backend.src.portfolio.portfolio_metrics import (
    PortfolioObjective,
    PortfolioScenarioEvaluation,
)
from backend.src.risk.risk_analyzer import RiskAnalysisConfig
from backend.src.rl.environment_factory import (
    PortfolioAllocationEnvironmentBundle,
    PortfolioAllocationEnvironmentFactory,
)
from backend.src.rl.policy_evaluator import (
    RLPolicyEvaluationResult,
    RLPolicyEvaluator,
)
from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnvConfig,
)
from backend.src.rl.ppo_allocation_agent import (
    PPOAllocationAgent,
    PPOAllocationAgentConfig,
)
from backend.src.rl.rl_exporter import (
    RLBenchmarkExportPaths,
    export_rl_benchmark,
)
from backend.src.storage.market_dataset_reader import MarketDatasetReader

DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "benchmark_results" / "rl" / "ppo"


@dataclass(frozen=True, slots=True)
class RLBenchmarkResult:
    evaluation: RLPolicyEvaluationResult
    baselines: tuple[PortfolioScenarioEvaluation, ...]
    export_paths: RLBenchmarkExportPaths


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PPO allocation policy on training data and compare it "
            "with fixed classical portfolios on validation data."
        )
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=MARKET_DATASETS_DIR,
        help="Directory containing train.csv and validation.csv.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for the PPO model and reproducible benchmark artifacts.",
    )
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lookback-window", type=int, default=20)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.25)
    parser.add_argument("--maximum-weight", type=float, default=0.35)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--annual-risk-free-rate", type=float, default=0.02)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument(
        "--minimum-common-observations",
        type=int,
        default=100,
    )

    return parser.parse_args(arguments)


def run_rl_benchmark(
    input_directory: Path = MARKET_DATASETS_DIR,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    total_timesteps: int = 10_000,
    learning_rate: float = 0.0003,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 10,
    seed: int = 42,
    device: str = "cpu",
    lookback_window: int = 20,
    transaction_cost_bps: float = 10.0,
    drawdown_penalty: float = 0.25,
    maximum_weight: float = 0.35,
    confidence_level: float = 0.95,
    annual_risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    minimum_common_observations: int = 100,
) -> RLBenchmarkResult:
    training_dataset, validation_dataset = MarketDatasetReader(
        input_directory=input_directory
    ).read_training_and_validation()

    environment_config = PortfolioAllocationEnvConfig(
        lookback_window=lookback_window,
        transaction_cost_bps=transaction_cost_bps,
        drawdown_penalty=drawdown_penalty,
        max_weight=maximum_weight,
    )
    bundle = PortfolioAllocationEnvironmentFactory(
        environment_config=environment_config,
        minimum_common_observations=minimum_common_observations,
        periods_per_year=periods_per_year,
    ).create(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
    )

    agent_config = PPOAllocationAgentConfig(
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        seed=seed,
        device=device,
    )
    agent = PPOAllocationAgent(config=agent_config).fit(bundle.training_environment)
    model_path = agent.save(Path(output_directory) / "ppo_allocation_agent.zip")

    risk_config = RiskAnalysisConfig(
        confidence_level=confidence_level,
        annual_risk_free_rate=annual_risk_free_rate,
        periods_per_year=periods_per_year,
        minimum_observations=minimum_common_observations,
    )
    evaluation = RLPolicyEvaluator(risk_config=risk_config).evaluate(
        policy=agent,
        environment=bundle.validation_environment,
        seed=seed,
    )
    baselines = _build_baselines(
        bundle=bundle,
        risk_config=risk_config,
        maximum_weight=maximum_weight,
    )
    export_paths = export_rl_benchmark(
        evaluation=evaluation,
        baselines=baselines,
        agent_config=agent_config,
        environment_config=environment_config,
        training_observations=len(bundle.training_returns),
        model_path=model_path,
        output_directory=output_directory,
    )

    return RLBenchmarkResult(
        evaluation=evaluation,
        baselines=baselines,
        export_paths=export_paths,
    )


def _build_baselines(
    bundle: PortfolioAllocationEnvironmentBundle,
    risk_config: RiskAnalysisConfig,
    maximum_weight: float,
) -> tuple[PortfolioScenarioEvaluation, ...]:
    analysis_config = PortfolioAnalysisConfig(risk_config=risk_config)
    analyzer = PortfolioAnalyzer(config=analysis_config)
    optimizer = MarkowitzOptimizer(
        config=MarkowitzOptimizationConfig(
            analysis_config=analysis_config,
            maximum_weight=maximum_weight,
        )
    )

    equal_weights = pd.Series(
        1.0 / len(bundle.tickers),
        index=bundle.tickers,
        dtype=float,
    )
    scenarios = [
        _evaluate_fixed_weights(
            name="equal_weight",
            construction="manual_equal_weight",
            weights=equal_weights,
            bundle=bundle,
            analyzer=analyzer,
        )
    ]
    definitions = (
        (
            "maximum_sharpe",
            "optimized_on_training_maximum_sharpe",
            PortfolioObjective.MAXIMUM_SHARPE,
        ),
        (
            "minimum_volatility",
            "optimized_on_training_minimum_volatility",
            PortfolioObjective.MINIMUM_VOLATILITY,
        ),
    )
    for name, construction, objective in definitions:
        optimization = optimizer.optimize(
            aligned_returns=bundle.training_returns,
            objective=objective,
        )
        training_analysis = optimization.analysis
        validation_analysis = analyzer.analyze(
            aligned_returns=bundle.validation_returns,
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
    bundle: PortfolioAllocationEnvironmentBundle,
    analyzer: PortfolioAnalyzer,
) -> PortfolioScenarioEvaluation:
    training_analysis = analyzer.analyze(
        aligned_returns=bundle.training_returns,
        weights=weights,
    )
    validation_analysis = analyzer.analyze(
        aligned_returns=bundle.validation_returns,
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
    result = run_rl_benchmark(
        input_directory=parsed.input_directory,
        output_directory=parsed.output_directory,
        total_timesteps=parsed.total_timesteps,
        learning_rate=parsed.learning_rate,
        n_steps=parsed.n_steps,
        batch_size=parsed.batch_size,
        n_epochs=parsed.n_epochs,
        seed=parsed.seed,
        device=parsed.device,
        lookback_window=parsed.lookback_window,
        transaction_cost_bps=parsed.transaction_cost_bps,
        drawdown_penalty=parsed.drawdown_penalty,
        maximum_weight=parsed.maximum_weight,
        confidence_level=parsed.confidence_level,
        annual_risk_free_rate=parsed.annual_risk_free_rate,
        periods_per_year=parsed.periods_per_year,
        minimum_common_observations=(parsed.minimum_common_observations),
    )

    metrics = result.evaluation.risk_metrics
    sharpe = (
        "undefined" if metrics.sharpe_ratio is None else f"{metrics.sharpe_ratio:.6f}"
    )
    print("Model: ppo")
    print("Training split: training")
    print("Evaluation split: validation")
    print("Test split used: no")
    print(f"Observations: {result.evaluation.observations}")
    print(f"Annualized return: {metrics.annualized_mean_return:.6f}")
    print(f"Annualized volatility: {metrics.annualized_volatility:.6f}")
    print(f"Sharpe ratio: {sharpe}")
    print(f"Maximum drawdown: {metrics.maximum_drawdown:.6f}")
    print(f"Final portfolio value: {result.evaluation.final_portfolio_value:.6f}")
    print("Baselines: " + ", ".join(baseline.name for baseline in result.baselines))

    paths = result.export_paths
    print(f"Summary: {paths.summary_path}")
    print(f"Trajectory: {paths.trajectory_path}")
    print(f"Weights: {paths.weights_path}")
    print(f"JSON result: {paths.json_path}")
    print(f"Model archive: {paths.model_path}")


if __name__ == "__main__":
    main()

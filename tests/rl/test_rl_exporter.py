import json
from pathlib import Path

import pandas as pd
import pytest

from backend.src.portfolio.portfolio_analyzer import (
    PortfolioAnalysisConfig,
    PortfolioAnalyzer,
)
from backend.src.portfolio.portfolio_metrics import (
    PortfolioScenarioEvaluation,
)
from backend.src.risk.risk_analyzer import (
    RiskAnalysisConfig,
    RiskAnalyzer,
)
from backend.src.rl.policy_evaluator import (
    RLPolicyEvaluationResult,
)
from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnvConfig,
)
from backend.src.rl.ppo_allocation_agent import (
    PPOAllocationAgentConfig,
)
from backend.src.rl.rl_exporter import (
    RLBenchmarkExportPaths,
    export_rl_benchmark,
)


def create_risk_config() -> RiskAnalysisConfig:
    return RiskAnalysisConfig(
        confidence_level=0.95,
        annual_risk_free_rate=0.02,
        periods_per_year=252,
        minimum_observations=2,
    )


def create_rl_result() -> RLPolicyEvaluationResult:
    index = pd.DatetimeIndex(
        [
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",
        ],
        name="date",
    )
    net_returns = pd.Series(
        [
            0.010,
            -0.005,
            0.002,
        ],
        index=index,
        name="net_return",
        dtype=float,
    )
    portfolio_values = (1.0 + net_returns).cumprod()
    running_peak = portfolio_values.cummax()
    drawdowns = 1.0 - portfolio_values / running_peak
    trajectory = pd.DataFrame(
        {
            "gross_return": [
                0.0102,
                -0.0048,
                0.0021,
            ],
            "transaction_cost": [
                0.0002,
                0.0002,
                0.0001,
            ],
            "net_return": net_returns,
            "turnover": [
                0.20,
                0.20,
                0.10,
            ],
            "portfolio_value": portfolio_values,
            "drawdown": drawdowns,
            "reward": [
                0.0099,
                -0.0051,
                0.0020,
            ],
        },
        index=index,
    )
    weights = pd.DataFrame(
        {
            "AGG": [
                0.40,
                0.35,
                0.30,
            ],
            "GLD": [
                0.30,
                0.35,
                0.35,
            ],
            "SPY": [
                0.30,
                0.30,
                0.35,
            ],
        },
        index=index,
    )
    risk_metrics = RiskAnalyzer(config=create_risk_config()).analyze(net_returns)

    return RLPolicyEvaluationResult(
        tickers=(
            "AGG",
            "GLD",
            "SPY",
        ),
        observations=3,
        date_min=index[0],
        date_max=index[-1],
        trajectory=trajectory,
        weights=weights,
        net_returns=net_returns,
        risk_metrics=risk_metrics,
        total_reward=float(trajectory["reward"].sum()),
        cumulative_return=float(portfolio_values.iloc[-1] - 1.0),
        final_portfolio_value=float(portfolio_values.iloc[-1]),
        total_transaction_cost=float(trajectory["transaction_cost"].sum()),
        mean_turnover=float(trajectory["turnover"].mean()),
    )


def create_baseline(
    name: str = "equal_weight",
) -> PortfolioScenarioEvaluation:
    index = pd.date_range(
        "2024-01-01",
        periods=5,
        freq="D",
    )
    training_returns = pd.DataFrame(
        {
            "AGG": [
                0.001,
                0.002,
                -0.001,
                0.003,
                0.001,
            ],
            "GLD": [
                0.002,
                -0.001,
                0.003,
                0.001,
                0.002,
            ],
            "SPY": [
                0.008,
                -0.004,
                0.006,
                0.003,
                -0.002,
            ],
        },
        index=index,
    )
    validation_returns = training_returns * 0.80
    weights = pd.Series(
        {
            "AGG": 1.0 / 3.0,
            "GLD": 1.0 / 3.0,
            "SPY": 1.0 / 3.0,
        },
        dtype=float,
    )
    analyzer = PortfolioAnalyzer(
        config=PortfolioAnalysisConfig(risk_config=create_risk_config())
    )

    return PortfolioScenarioEvaluation(
        name=name,
        construction="fixed_training_weights",
        training=analyzer.analyze(
            aligned_returns=training_returns,
            weights=weights,
        ),
        validation=analyzer.analyze(
            aligned_returns=validation_returns,
            weights=weights,
        ),
    )


def create_agent_config() -> PPOAllocationAgentConfig:
    return PPOAllocationAgentConfig(
        total_timesteps=32,
        n_steps=16,
        batch_size=8,
        n_epochs=2,
        hidden_layer_sizes=(
            16,
            16,
        ),
        seed=42,
        device="cpu",
    )


def create_environment_config() -> PortfolioAllocationEnvConfig:
    return PortfolioAllocationEnvConfig(
        lookback_window=3,
        transaction_cost_bps=10.0,
        drawdown_penalty=0.25,
        max_weight=0.60,
    )


def create_model_archive(
    directory: Path,
) -> Path:
    model_path = directory / "ppo_allocation_agent.zip"
    model_path.write_bytes(b"test-model-archive")

    return model_path


def test_export_creates_expected_artifacts(
    tmp_path: Path,
) -> None:
    model_path = create_model_archive(tmp_path)

    paths = export_rl_benchmark(
        evaluation=create_rl_result(),
        baselines=(create_baseline(),),
        agent_config=create_agent_config(),
        environment_config=(create_environment_config()),
        training_observations=20,
        model_path=model_path,
        output_directory=tmp_path,
    )

    assert isinstance(
        paths,
        RLBenchmarkExportPaths,
    )
    assert paths.summary_path.exists()
    assert paths.trajectory_path.exists()
    assert paths.weights_path.exists()
    assert paths.json_path.exists()
    assert paths.model_path == model_path


def test_exported_csv_files_have_stable_contract(
    tmp_path: Path,
) -> None:
    paths = export_rl_benchmark(
        evaluation=create_rl_result(),
        baselines=(create_baseline(),),
        agent_config=create_agent_config(),
        environment_config=(create_environment_config()),
        training_observations=20,
        model_path=create_model_archive(tmp_path),
        output_directory=tmp_path,
    )

    summary = pd.read_csv(paths.summary_path)
    trajectory = pd.read_csv(paths.trajectory_path)
    weights = pd.read_csv(paths.weights_path)

    assert summary["strategy"].tolist() == [
        "ppo",
        "equal_weight",
    ]
    assert set(
        [
            "annualized_mean_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "value_at_risk",
            "conditional_value_at_risk",
            "maximum_drawdown",
            "cumulative_return",
        ]
    ).issubset(summary.columns)
    assert trajectory["date"].tolist() == [
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
    ]
    assert weights.columns.tolist() == [
        "date",
        "AGG",
        "GLD",
        "SPY",
    ]
    assert len(trajectory) == 3
    assert len(weights) == 3


def test_exported_json_contains_reproducibility_metadata(
    tmp_path: Path,
) -> None:
    paths = export_rl_benchmark(
        evaluation=create_rl_result(),
        baselines=(create_baseline(),),
        agent_config=create_agent_config(),
        environment_config=(create_environment_config()),
        training_observations=20,
        model_path=create_model_archive(tmp_path),
        output_directory=tmp_path,
    )

    text = paths.json_path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert payload["model_name"] == "ppo"
    assert payload["training_split"] == "training"
    assert payload["evaluation_split"] == "validation"
    assert payload["test_split_used"] is False
    assert payload["training_observations"] == 20
    assert payload["evaluation"]["observations"] == 3
    assert payload["agent_config"]["seed"] == 42
    assert payload["agent_config"]["device"] == "cpu"
    assert payload["environment_config"]["transaction_cost_bps"] == pytest.approx(10.0)
    assert payload["model_file"] == ("ppo_allocation_agent.zip")
    assert [row["strategy"] for row in payload["comparison"]] == [
        "ppo",
        "equal_weight",
    ]
    assert "NaN" not in text
    assert "Infinity" not in text


def test_export_rejects_missing_model_archive(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="model",
    ):
        export_rl_benchmark(
            evaluation=create_rl_result(),
            baselines=(create_baseline(),),
            agent_config=create_agent_config(),
            environment_config=(create_environment_config()),
            training_observations=20,
            model_path=(tmp_path / "missing-model.zip"),
            output_directory=tmp_path,
        )


def test_export_rejects_duplicate_baseline_names(
    tmp_path: Path,
) -> None:
    duplicate_baseline = create_baseline()

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        export_rl_benchmark(
            evaluation=create_rl_result(),
            baselines=(
                duplicate_baseline,
                duplicate_baseline,
            ),
            agent_config=create_agent_config(),
            environment_config=(create_environment_config()),
            training_observations=20,
            model_path=create_model_archive(tmp_path),
            output_directory=tmp_path,
        )

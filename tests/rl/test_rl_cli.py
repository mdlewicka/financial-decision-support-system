import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.src.rl.cli import (
    main,
    parse_arguments,
    run_rl_benchmark,
)

TICKERS = (
    "AGG",
    "GLD",
    "SPY",
)


def create_dataset(
    start_date: str,
    periods: int,
    position_offset: int,
) -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=periods, freq="D")
    rows = []
    for ticker_index, ticker in enumerate(TICKERS):
        base_price = 100.0 + 40.0 * ticker_index
        for local_position, date_value in enumerate(dates):
            position = position_offset + local_position
            log_price = (
                np.log(base_price)
                + (0.0004 + ticker_index * 0.0001) * position
                + 0.012
                * np.sin(
                    0.17 * position + ticker_index * 0.8,
                )
            )
            rows.append(
                {
                    "date": date_value,
                    "target_date": date_value + pd.Timedelta(days=1),
                    "ticker": ticker,
                    "adjusted_close": float(np.exp(log_price)),
                    "target": 0.0,
                }
            )

    return pd.DataFrame(rows)


def write_training_and_validation(directory: Path) -> None:
    create_dataset(
        start_date="2024-01-01",
        periods=80,
        position_offset=0,
    ).to_csv(
        directory / "train.csv",
        index=False,
    )
    create_dataset(
        start_date="2024-03-21",
        periods=40,
        position_offset=80,
    ).to_csv(
        directory / "validation.csv",
        index=False,
    )


def benchmark_arguments(
    input_directory: Path,
    output_directory: Path,
) -> list[str]:
    return [
        "--input-directory",
        str(input_directory),
        "--output-directory",
        str(output_directory),
        "--total-timesteps",
        "16",
        "--n-steps",
        "16",
        "--batch-size",
        "8",
        "--n-epochs",
        "1",
        "--lookback-window",
        "5",
        "--transaction-cost-bps",
        "5",
        "--drawdown-penalty",
        "0.10",
        "--maximum-weight",
        "0.60",
        "--minimum-common-observations",
        "10",
    ]


def test_parse_arguments_reads_rl_options(tmp_path: Path) -> None:
    output_directory = tmp_path / "results"

    arguments = parse_arguments(
        benchmark_arguments(
            input_directory=tmp_path,
            output_directory=output_directory,
        )
    )

    assert arguments.input_directory == tmp_path
    assert arguments.output_directory == output_directory
    assert arguments.total_timesteps == 16
    assert arguments.n_steps == 16
    assert arguments.batch_size == 8
    assert arguments.n_epochs == 1
    assert arguments.lookback_window == 5
    assert arguments.transaction_cost_bps == pytest.approx(5.0)
    assert arguments.drawdown_penalty == pytest.approx(0.10)
    assert arguments.maximum_weight == pytest.approx(0.60)
    assert arguments.minimum_common_observations == 10
    assert arguments.seed == 42
    assert arguments.device == "cpu"


def test_run_exports_rl_benchmark_without_test_dataset(
    tmp_path: Path,
) -> None:
    write_training_and_validation(tmp_path)
    output_directory = tmp_path / "results"

    assert not (tmp_path / "test.csv").exists()

    result = run_rl_benchmark(
        input_directory=tmp_path,
        output_directory=output_directory,
        total_timesteps=16,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        lookback_window=5,
        transaction_cost_bps=5.0,
        drawdown_penalty=0.10,
        maximum_weight=0.60,
        minimum_common_observations=10,
    )

    assert result.evaluation.observations == 40
    assert result.evaluation.tickers == TICKERS
    assert [baseline.name for baseline in result.baselines] == [
        "equal_weight",
        "maximum_sharpe",
        "minimum_volatility",
    ]
    assert result.export_paths.model_path.exists()
    assert result.export_paths.summary_path.exists()
    assert result.export_paths.trajectory_path.exists()
    assert result.export_paths.weights_path.exists()
    assert result.export_paths.json_path.exists()

    summary = pd.read_csv(result.export_paths.summary_path)
    payload = json.loads(result.export_paths.json_path.read_text(encoding="utf-8"))

    assert summary["strategy"].tolist() == [
        "ppo",
        "equal_weight",
        "maximum_sharpe",
        "minimum_volatility",
    ]
    assert payload["training_split"] == "training"
    assert payload["evaluation_split"] == "validation"
    assert payload["test_split_used"] is False
    assert payload["agent_config"]["device"] == "cpu"


def test_main_prints_validation_summary_and_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_training_and_validation(tmp_path)
    output_directory = tmp_path / "results"

    main(
        benchmark_arguments(
            input_directory=tmp_path,
            output_directory=output_directory,
        )
    )

    captured = capsys.readouterr()
    assert "Model: ppo" in captured.out
    assert "Training split: training" in captured.out
    assert "Evaluation split: validation" in captured.out
    assert "Test split used: no" in captured.out
    assert "equal_weight" in captured.out
    assert "maximum_sharpe" in captured.out
    assert "minimum_volatility" in captured.out
    assert str(output_directory / "rl_result.json") in captured.out
    assert str(output_directory / "ppo_allocation_agent.zip") in captured.out

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.src.portfolio.cli import (
    main,
    parse_arguments,
    run_portfolio_benchmark,
)

TICKERS = (
    "EPOL",
    "VGK",
    "SPY",
    "VT",
    "QQQ",
    "GLD",
    "BTC-USD",
    "AGG",
)


def create_dataset(
    start_date: str,
    price_shift: float,
) -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=80, freq="D")
    rows = []
    for ticker_index, ticker in enumerate(TICKERS):
        base_price = 100.0 + ticker_index * 20.0 + price_shift
        adjusted_close = base_price
        for date_index, date_value in enumerate(dates):
            if date_index > 0:
                daily_return = (
                    0.0002
                    + ticker_index * 0.00003
                    + 0.006 * np.sin((date_index + ticker_index) * 0.7)
                    + 0.003 * np.cos(date_index * (ticker_index + 1) * 0.13)
                )
                adjusted_close *= 1.0 + daily_return
            rows.append(
                {
                    "date": date_value,
                    "target_date": date_value + pd.Timedelta(days=1),
                    "ticker": ticker,
                    "adjusted_close": adjusted_close,
                    "target": 0.0,
                }
            )
    return pd.DataFrame(rows)


def write_datasets(directory: Path) -> None:
    create_dataset("2024-01-01", 0.0).to_csv(
        directory / "train.csv",
        index=False,
    )
    create_dataset("2024-02-01", 5.0).to_csv(
        directory / "validation.csv",
        index=False,
    )
    test_dataset = create_dataset("2024-03-01", 10.0)
    test_dataset["adjusted_close"] = np.inf
    test_dataset.to_csv(directory / "test.csv", index=False)


def test_parse_arguments_reads_portfolio_options(tmp_path: Path) -> None:
    output_directory = tmp_path / "results"

    arguments = parse_arguments(
        [
            "--input-directory",
            str(tmp_path),
            "--output-directory",
            str(output_directory),
            "--confidence-level",
            "0.90",
            "--annual-risk-free-rate",
            "0.01",
            "--periods-per-year",
            "365",
            "--minimum-common-observations",
            "10",
            "--maximum-weight",
            "0.40",
        ]
    )

    assert arguments.input_directory == tmp_path
    assert arguments.output_directory == output_directory
    assert arguments.confidence_level == 0.90
    assert arguments.annual_risk_free_rate == 0.01
    assert arguments.periods_per_year == 365
    assert arguments.minimum_common_observations == 10
    assert arguments.maximum_weight == 0.40


def test_run_exports_five_scenarios_without_using_test_split(
    tmp_path: Path,
) -> None:
    write_datasets(tmp_path)
    output_directory = tmp_path / "results"

    result = run_portfolio_benchmark(
        input_directory=tmp_path,
        output_directory=output_directory,
        minimum_common_observations=10,
        maximum_weight=0.35,
    )

    assert [scenario.name for scenario in result.scenarios] == [
        "defensive",
        "equal_weight",
        "growth",
        "maximum_sharpe",
        "minimum_volatility",
    ]
    assert result.training_market_risk.observations == 79
    assert result.validation_market_risk.observations == 79
    assert len(pd.read_csv(result.export_paths.summary_path)) == 10
    assert len(pd.read_csv(result.export_paths.weights_path)) == 40
    assert result.export_paths.json_path.exists()

    for scenario in result.scenarios:
        assert scenario.training.weights.equals(scenario.validation.weights)
        assert scenario.validation.risk_metrics.observations == 79


def test_main_prints_validation_summary_and_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_datasets(tmp_path)
    output_directory = tmp_path / "results"

    main(
        [
            "--input-directory",
            str(tmp_path),
            "--output-directory",
            str(output_directory),
            "--minimum-common-observations",
            "10",
            "--maximum-weight",
            "0.35",
        ]
    )

    captured = capsys.readouterr()
    assert "Weight selection split: training" in captured.out
    assert "Evaluation split: validation" in captured.out
    assert "maximum_sharpe" in captured.out
    assert str(output_directory / "portfolio_result.json") in captured.out

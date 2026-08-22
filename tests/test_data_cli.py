import json
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from backend.src.config import (
    HISTORY_START_DATE,
    MINIMUM_MODEL_OBSERVATIONS_PER_TICKER,
    SUPPORTED_TICKERS,
)
from backend.src.data_cli import (
    DataWorkflowResult,
    parse_arguments,
    run_data_workflow,
    write_data_manifest,
)


def write_dataset(
    path: Path,
    tickers: list[str],
    dates: list[str],
    target_dates: list[str] | None = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = pd.DataFrame(
        {
            "ticker": tickers,
            "date": pd.to_datetime(dates),
        }
    )

    if target_dates is not None:
        dataset["target_date"] = pd.to_datetime(target_dates)

    dataset.to_csv(
        path,
        index=False,
    )


def test_supported_universe_contains_eight_assets() -> None:
    assert SUPPORTED_TICKERS == (
        "EPOL",
        "VGK",
        "SPY",
        "VT",
        "QQQ",
        "GLD",
        "BTC-USD",
        "AGG",
    )


def test_parse_arguments_uses_reproducible_defaults() -> None:
    arguments = parse_arguments(
        [
            "--end-date",
            "2026-08-01",
        ]
    )

    assert arguments.start_date == HISTORY_START_DATE
    assert arguments.end_date == date(2026, 8, 1)
    assert arguments.minimum_observations == MINIMUM_MODEL_OBSERVATIONS_PER_TICKER


def test_parse_arguments_rejects_invalid_date_range() -> None:
    with pytest.raises(SystemExit):
        parse_arguments(
            [
                "--start-date",
                "2026-08-01",
                "--end-date",
                "2026-08-01",
            ]
        )


def test_run_data_workflow_connects_existing_pipelines() -> None:
    market_data_pipeline = Mock()
    dataset_pipeline = Mock()

    processed_data_path = Path("data/processed/market/market_features.csv")
    dataset_paths = (
        Path("data/datasets/market/train.csv"),
        Path("data/datasets/market/validation.csv"),
        Path("data/datasets/market/test.csv"),
    )

    market_data_pipeline.run.return_value = processed_data_path
    dataset_pipeline.run.return_value = dataset_paths

    result = run_data_workflow(
        start_date=date(2010, 1, 1),
        end_date=date(2026, 8, 1),
        minimum_observations=1000,
        market_data_pipeline=market_data_pipeline,
        dataset_pipeline=dataset_pipeline,
    )

    market_data_pipeline.run.assert_called_once_with(
        tickers=SUPPORTED_TICKERS,
        start_date=date(2010, 1, 1),
        end_date=date(2026, 8, 1),
    )
    dataset_pipeline.run.assert_called_once_with(
        processed_data_path=processed_data_path,
    )

    assert result == DataWorkflowResult(
        processed_data_path=processed_data_path,
        train_path=dataset_paths[0],
        validation_path=dataset_paths[1],
        test_path=dataset_paths[2],
    )


def test_write_data_manifest_records_parameters_and_counts(
    tmp_path: Path,
) -> None:
    processed_data_path = tmp_path / "processed.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    output_path = tmp_path / "manifest.json"

    write_dataset(
        processed_data_path,
        tickers=[
            "SPY",
            "SPY",
            "QQQ",
            "QQQ",
        ],
        dates=[
            "2024-01-01",
            "2024-01-02",
            "2024-01-01",
            "2024-01-02",
        ],
    )
    write_dataset(
        train_path,
        tickers=["SPY", "QQQ"],
        dates=["2024-01-01", "2024-01-01"],
        target_dates=["2024-01-02", "2024-01-02"],
    )
    write_dataset(
        validation_path,
        tickers=["SPY"],
        dates=["2024-01-02"],
        target_dates=["2024-01-03"],
    )
    write_dataset(
        test_path,
        tickers=["QQQ"],
        dates=["2024-01-03"],
        target_dates=["2024-01-04"],
    )

    result = DataWorkflowResult(
        processed_data_path=processed_data_path,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
    )

    manifest_path = write_data_manifest(
        result=result,
        start_date=date(2010, 1, 1),
        end_date=date(2026, 8, 1),
        minimum_observations=1000,
        output_path=output_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path == output_path
    assert manifest["schema_version"] == 1
    assert manifest["generated_at_utc"].endswith("+00:00")

    assert manifest["request"] == {
        "start_date": "2010-01-01",
        "end_date_exclusive": "2026-08-01",
        "interval": "1d",
        "minimum_observations_per_ticker": 1000,
        "tickers": list(SUPPORTED_TICKERS),
    }
    assert manifest["target"] == {
        "column": "target",
        "definition": "next_session_simple_return",
        "target_date_column": "target_date",
    }

    processed_summary = manifest["outputs"]["processed"]

    assert processed_summary["rows"] == 4
    assert processed_summary["rows_per_ticker"] == {
        "QQQ": 2,
        "SPY": 2,
    }
    assert processed_summary["date_min"] == "2024-01-01"
    assert processed_summary["date_max"] == "2024-01-02"
    assert "target_date_min" not in processed_summary
    assert "target_date_max" not in processed_summary

    train_summary = manifest["outputs"]["train"]
    validation_summary = manifest["outputs"]["validation"]
    test_summary = manifest["outputs"]["test"]

    assert train_summary["rows"] == 2
    assert train_summary["target_date_min"] == "2024-01-02"
    assert train_summary["target_date_max"] == "2024-01-02"

    assert validation_summary["rows"] == 1
    assert validation_summary["target_date_min"] == "2024-01-03"
    assert validation_summary["target_date_max"] == "2024-01-03"

    assert test_summary["rows"] == 1
    assert test_summary["target_date_min"] == "2024-01-04"
    assert test_summary["target_date_max"] == "2024-01-04"

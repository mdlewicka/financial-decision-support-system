import json
from pathlib import Path

import pandas as pd

from backend.src.forecast.arima_cli import (
    ARIMA_MODEL_NAME,
    ArimaBenchmarkResult,
    parse_arguments,
    run_arima_benchmark,
)


def write_dataset(
    path: Path,
    start_date: str,
    targets: list[float],
) -> pd.DataFrame:
    dates = pd.date_range(
        start=start_date,
        periods=len(targets),
        freq="D",
    )
    dataset = pd.DataFrame(
        {
            "ticker": ["SPY"] * len(targets),
            "date": dates,
            "target_date": dates + pd.Timedelta(days=1),
            "target": targets,
        }
    )
    dataset.to_csv(
        path,
        index=False,
    )

    return dataset


def write_market_datasets(
    directory: Path,
) -> pd.DataFrame:
    training_targets = [
        0.010,
        -0.005,
        0.004,
        -0.002,
        0.006,
        -0.003,
        0.002,
        -0.001,
        0.005,
        -0.004,
        0.003,
        -0.002,
        0.007,
        -0.003,
        0.001,
        -0.001,
        0.004,
        -0.002,
        0.003,
        -0.001,
    ]
    validation_targets = [
        0.006,
        -0.004,
        0.002,
        -0.001,
    ]

    write_dataset(
        path=directory / "train.csv",
        start_date="2024-01-01",
        targets=training_targets,
    )
    validation_dataset = write_dataset(
        path=directory / "validation.csv",
        start_date="2024-02-01",
        targets=validation_targets,
    )
    write_dataset(
        path=directory / "test.csv",
        start_date="2024-03-01",
        targets=[
            999.0,
            999.0,
        ],
    )

    return validation_dataset


def test_parse_arguments_accepts_arima_benchmark_options(
    tmp_path: Path,
) -> None:
    dataset_directory = tmp_path / "datasets"
    output_directory = tmp_path / "benchmark"

    arguments = parse_arguments(
        [
            "--dataset-directory",
            str(dataset_directory),
            "--output-directory",
            str(output_directory),
            "--criterion",
            "bic",
            "--minimum-observations",
            "20",
        ]
    )

    assert arguments.dataset_directory == dataset_directory
    assert arguments.output_directory == output_directory
    assert arguments.criterion == "bic"
    assert arguments.minimum_observations == 20


def test_run_arima_benchmark_selects_and_evaluates_model(
    tmp_path: Path,
) -> None:
    validation_dataset = write_market_datasets(tmp_path)
    output_directory = tmp_path / "benchmark"

    benchmark = run_arima_benchmark(
        dataset_directory=tmp_path,
        output_directory=output_directory,
        candidate_orders=((0, 0, 0),),
        criterion="aic",
        minimum_observations=10,
    )

    assert isinstance(
        benchmark,
        ArimaBenchmarkResult,
    )
    assert benchmark.order_selection.selected_orders == {
        "SPY": (0, 0, 0),
    }
    assert benchmark.forecast_result.overall_metrics.observations == 4

    assert (
        benchmark.forecast_result.observations["actual"].tolist()
        == validation_dataset["target"].tolist()
    )

    assert benchmark.export_paths.observations_csv.is_file()
    assert benchmark.export_paths.metrics_csv.is_file()
    assert benchmark.export_paths.result_json.is_file()
    assert benchmark.selection_scores_csv.is_file()
    assert benchmark.selection_summary_json.is_file()


def test_run_arima_benchmark_exports_selection_metadata(
    tmp_path: Path,
) -> None:
    write_market_datasets(tmp_path)
    output_directory = tmp_path / "benchmark"

    benchmark = run_arima_benchmark(
        dataset_directory=tmp_path,
        output_directory=output_directory,
        candidate_orders=((0, 0, 0),),
        criterion="aic",
        minimum_observations=10,
    )

    selection_payload = json.loads(
        benchmark.selection_summary_json.read_text(
            encoding="utf-8",
        )
    )
    forecast_payload = json.loads(
        benchmark.export_paths.result_json.read_text(
            encoding="utf-8",
        )
    )

    assert selection_payload["schema_version"] == 1
    assert selection_payload["selection_dataset"] == "training"
    assert selection_payload["criterion"] == "aic"
    assert selection_payload["selected_orders"] == {
        "SPY": [0, 0, 0],
    }
    assert len(selection_payload["candidate_scores"]) == 1

    assert forecast_payload["model_name"] == ARIMA_MODEL_NAME
    assert forecast_payload["evaluation_split"] == "validation"

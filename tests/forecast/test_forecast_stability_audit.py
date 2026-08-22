import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.forecast.forecast_stability_audit import (
    MODEL_NAMES,
    export_forecast_stability_audit,
    main,
    run_forecast_stability_audit,
)
from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)


def create_dataset(
    start: str,
    periods: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for ticker_index, ticker in enumerate(("QQQ", "SPY")):
        dates = pd.date_range(start=start, periods=periods, freq="D")
        for index, date in enumerate(dates):
            adjusted_close = 100.0 + 50.0 * ticker_index + index
            daily_return = 0.004 * ((index % 5) - 2)
            momentum_5 = 0.006 * ((index % 7) - 3)
            momentum_20 = 0.004 * ((index % 9) - 4)
            target = 0.55 * momentum_5 - 0.25 * daily_return + 0.0005 * ticker_index
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": adjusted_close * 0.999,
                    "high": adjusted_close * 1.005,
                    "low": adjusted_close * 0.995,
                    "close": adjusted_close,
                    "adjusted_close": adjusted_close,
                    "volume": 1_000_000 + index,
                    "daily_return": daily_return,
                    "log_return": np.log1p(daily_return),
                    "price_change": adjusted_close - 100.0,
                    "volume_change": 0.02 * ((index % 4) - 1.5),
                    "moving_average_5": adjusted_close / (1.0 + momentum_5),
                    "moving_average_20": adjusted_close / (1.0 + momentum_20),
                    "volatility_5": 0.01 + 0.001 * (index % 3),
                    "volatility_20": 0.015 + 0.001 * (index % 4),
                    "momentum_5": momentum_5,
                    "momentum_20": momentum_20,
                    "target_date": date + pd.Timedelta(days=1),
                    "target": target,
                }
            )

    return pd.DataFrame(rows)


def create_small_model() -> RandomForestForecastModel:
    return RandomForestForecastModel(
        n_estimators=20,
        max_depth=4,
        min_samples_leaf=2,
        minimum_observations_per_ticker=10,
        random_state=42,
        n_jobs=1,
    )


def run_small_audit():
    return run_forecast_stability_audit(
        training_dataset=create_dataset("2023-01-01", 30),
        validation_dataset=create_dataset("2023-02-01", 12),
        folds=3,
        bootstrap_iterations=50,
        random_state=42,
        model_factory=create_small_model,
    )


def test_audit_builds_chronological_leakage_safe_folds() -> None:
    result = run_small_audit()

    assert len(result.folds) == 3
    assert all(
        fold.training_target_date_max <= fold.evaluation_date_min
        for fold in result.folds
    )
    assert [fold.training_rows for fold in result.folds] == sorted(
        fold.training_rows for fold in result.folds
    )


def test_audit_compares_all_models_per_fold_and_ticker() -> None:
    result = run_small_audit()

    assert set(result.observations["model"]) == set(MODEL_NAMES)
    assert set(result.metrics["scope"]) == {"overall", "ticker"}
    assert set(result.metrics["fold"]) == {1, 2, 3}
    assert set(result.stability_summary["model"]) == set(MODEL_NAMES)
    assert set(result.bootstrap_comparison["model"]) == set(MODEL_NAMES)
    assert np.isfinite(result.observations["prediction"]).all()


def test_audit_is_reproducible() -> None:
    first = run_small_audit()
    second = run_small_audit()

    pdt.assert_frame_equal(first.observations, second.observations)
    pdt.assert_frame_equal(first.metrics, second.metrics)
    pdt.assert_frame_equal(
        first.bootstrap_comparison,
        second.bootstrap_comparison,
    )


def test_audit_rejects_overlapping_splits() -> None:
    with pytest.raises(ValueError, match="Training dates must end"):
        run_forecast_stability_audit(
            training_dataset=create_dataset("2023-01-01", 30),
            validation_dataset=create_dataset("2023-01-20", 12),
            folds=3,
            bootstrap_iterations=10,
            model_factory=create_small_model,
        )


def test_export_records_that_test_split_was_not_read(tmp_path: Path) -> None:
    training = create_dataset("2023-01-01", 30)
    validation = create_dataset("2023-02-01", 12)
    result = run_forecast_stability_audit(
        training_dataset=training,
        validation_dataset=validation,
        folds=3,
        bootstrap_iterations=20,
        model_factory=create_small_model,
    )

    paths = export_forecast_stability_audit(
        result=result,
        output_directory=tmp_path,
        training_dataset=training,
        validation_dataset=validation,
        bootstrap_iterations=20,
        random_state=42,
    )

    assert all(
        path.is_file()
        for path in (
            paths.observations_csv,
            paths.metrics_csv,
            paths.stability_summary_csv,
            paths.bootstrap_csv,
            paths.metadata_json,
        )
    )
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["development_splits_used"] == ["training", "validation"]
    assert metadata["test_split_read"] is False
    assert metadata["fold_count"] == 3


def test_cli_runs_without_test_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_directory = tmp_path / "datasets"
    output_directory = tmp_path / "output"
    dataset_directory.mkdir()
    create_dataset("2023-01-01", 30).to_csv(
        dataset_directory / "train.csv",
        index=False,
    )
    create_dataset("2023-02-01", 12).to_csv(
        dataset_directory / "validation.csv",
        index=False,
    )

    monkeypatch.setattr(
        "backend.src.forecast.forecast_stability_audit.RandomForestForecastModel",
        create_small_model,
    )

    exit_code = main(
        [
            "--dataset-directory",
            str(dataset_directory),
            "--output-directory",
            str(output_directory),
            "--folds",
            "3",
            "--bootstrap-iterations",
            "20",
        ]
    )

    assert exit_code == 0
    assert not (dataset_directory / "test.csv").exists()
    assert (output_directory / "audit_metadata.json").is_file()

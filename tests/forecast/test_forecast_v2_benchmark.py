import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.forecast.forecast_v2_benchmark import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    ForecastV2BenchmarkConfig,
    export_forecast_v2_benchmark,
    main,
    run_forecast_v2_benchmark,
)
from backend.src.forecast.multi_horizon_target_builder import ForecastTargetConfig
from backend.src.models.random_forest_forecast_model import RandomForestForecastModel
from backend.src.models.regularized_forecast_models import (
    LogisticForecastConfig,
    RegularizedDirectionForecastModel,
    RegularizedReturnForecastModel,
    RidgeForecastConfig,
)

TICKERS = ("QQQ", "SPY")


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    raise TypeError(f"Expected a numeric value, got {type(value).__name__}.")


def _as_timestamp(value: object) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    raise TypeError(f"Expected a timestamp value, got {type(value).__name__}.")


def create_development_dataset(start: str, periods: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pattern = np.asarray([-0.012] * 6 + [0.0] * 6 + [0.012] * 6)
    for ticker_index, ticker in enumerate(TICKERS):
        price = 100.0 + 25.0 * ticker_index
        dates = pd.date_range(start, periods=periods, freq="D")
        ticker_rows: list[dict[str, object]] = []
        for index, date in enumerate(dates):
            daily_return = float(pattern[(index + 3 * ticker_index) % len(pattern)])
            price *= 1.0 + daily_return
            momentum_5 = 0.008 * ((index % 7) - 3)
            momentum_20 = 0.004 * ((index % 9) - 4)
            ticker_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price * 0.999,
                    "high": price * 1.005,
                    "low": price * 0.995,
                    "close": price,
                    "adjusted_close": price,
                    "volume": 1_000_000.0 + 10_000 * ticker_index + index,
                    "daily_return": daily_return,
                    "log_return": np.log1p(daily_return),
                    "price_change": price - 100.0,
                    "volume_change": 0.02 * ((index % 4) - 1.5),
                    "moving_average_5": price / (1.0 + momentum_5),
                    "moving_average_20": price / (1.0 + momentum_20),
                    "volatility_5": 0.01 + 0.001 * (index % 3),
                    "volatility_20": 0.015 + 0.001 * (index % 4),
                    "momentum_5": momentum_5,
                    "momentum_20": momentum_20,
                }
            )
        for index, row in enumerate(ticker_rows):
            next_index = min(index + 1, len(ticker_rows) - 1)
            next_row = ticker_rows[next_index]
            row["target_date"] = _as_timestamp(next_row["date"])
            row["target"] = (
                _as_float(next_row["adjusted_close"]) / _as_float(row["adjusted_close"])
                - 1.0
            )
            rows.append(row)
    return pd.DataFrame(rows)


def development_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = create_development_dataset("2023-01-01", 125)
    cutoff = pd.Timestamp("2023-04-11")
    training = combined.loc[combined["date"] < cutoff].copy()
    validation = combined.loc[combined["date"] >= cutoff].copy()
    return training, validation


def target_config() -> ForecastTargetConfig:
    return ForecastTargetConfig(
        one_session_threshold=0.0025,
        five_session_threshold=0.005,
    )


def ridge_factory(horizon: int) -> RegularizedReturnForecastModel:
    return RegularizedReturnForecastModel(
        config=RidgeForecastConfig(horizon=horizon, alpha=2.0),  # type: ignore[arg-type]
        target_config=target_config(),
    )


def logistic_factory() -> RegularizedDirectionForecastModel:
    return RegularizedDirectionForecastModel(
        config=LogisticForecastConfig(
            horizon=5,
            inverse_regularization_strength=0.1,
            minimum_confidence=0.45,
            max_iterations=500,
            random_state=42,
        ),
        target_config=target_config(),
    )


def random_forest_factory() -> RandomForestForecastModel:
    return RandomForestForecastModel(
        n_estimators=10,
        max_depth=3,
        min_samples_leaf=2,
        minimum_observations_per_ticker=10,
        random_state=42,
        n_jobs=1,
    )


def small_config() -> ForecastV2BenchmarkConfig:
    return ForecastV2BenchmarkConfig(
        folds=3,
        bootstrap_iterations=30,
        random_state=42,
        transaction_cost=0.001,
    )


def run_small_benchmark():
    training, validation = development_splits()
    return run_forecast_v2_benchmark(
        training_dataset=training,
        validation_dataset=validation,
        config=small_config(),
        ridge_factory=ridge_factory,
        logistic_factory=logistic_factory,
        random_forest_factory=random_forest_factory,
    )


def test_benchmark_uses_chronological_expanding_folds() -> None:
    result = run_small_benchmark()

    assert len(result.folds) == 3
    assert [fold.training_rows for fold in result.folds] == sorted(
        fold.training_rows for fold in result.folds
    )
    assert all(
        fold.training_date_max < fold.evaluation_date_min for fold in result.folds
    )
    assert all(
        fold.training_target_date_max_horizon_5 <= fold.evaluation_date_min
        for fold in result.folds
    )


def test_benchmark_compares_expected_models_and_tasks() -> None:
    result = run_small_benchmark()
    regression = result.observations.loc[result.observations["task"] == "regression"]
    classification = result.observations.loc[
        result.observations["task"] == "classification"
    ]

    for horizon, model_names in REGRESSION_MODELS.items():
        assert set(regression.loc[regression["horizon"] == horizon, "model"]) == set(
            model_names
        )
    assert set(classification["model"]) == set(CLASSIFICATION_MODELS)
    assert set(classification["actual_class"]) == {"down", "abstain", "up"}
    assert set(result.metrics["scope"]) == {"overall", "ticker"}
    assert set(result.bootstrap_comparison["task"]) == {
        "regression",
        "classification",
    }
    assert set(result.candidate_decisions["model"]) == {
        "ridge_v2",
        "logistic_v2",
    }
    assert (
        result.candidate_decisions["bootstrap_probability_better"]
        .between(0.0, 1.0)
        .all()
    )


def test_probabilities_and_economic_screening_values_are_valid() -> None:
    result = run_small_benchmark()
    classification = result.observations.loc[
        result.observations["task"] == "classification"
    ]
    probabilities = classification[
        [f"probability_{name}" for name in ("down", "abstain", "up")]
    ]

    assert np.isfinite(probabilities.to_numpy(dtype=float)).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(result.observations["net_utility"]).all()
    assert result.metrics["coverage"].between(0.0, 1.0).all()
    logistic = classification.loc[classification["model"] == "logistic_v2"]
    low_confidence = (
        logistic[[f"probability_{name}" for name in ("down", "abstain", "up")]].max(
            axis=1
        )
        < small_config().minimum_classification_confidence
    )
    assert logistic.loc[low_confidence, "predicted_class"].eq("abstain").all()


def test_benchmark_is_reproducible() -> None:
    first = run_small_benchmark()
    second = run_small_benchmark()

    pdt.assert_frame_equal(first.observations, second.observations)
    pdt.assert_frame_equal(first.metrics, second.metrics)
    pdt.assert_frame_equal(first.bootstrap_comparison, second.bootstrap_comparison)
    pdt.assert_frame_equal(first.candidate_decisions, second.candidate_decisions)


def test_changed_existing_target_is_rejected() -> None:
    training, validation = development_splits()
    row_index = validation.index[3]
    validation.at[row_index, "target"] = (
        _as_float(validation.at[row_index, "target"]) + 0.1
    )

    with pytest.raises(ValueError, match="do not match"):
        run_forecast_v2_benchmark(
            training,
            validation,
            config=small_config(),
            ridge_factory=ridge_factory,
            logistic_factory=logistic_factory,
            random_forest_factory=random_forest_factory,
        )


def test_export_records_provenance_and_no_test_access(tmp_path: Path) -> None:
    training, validation = development_splits()
    result = run_small_benchmark()
    paths = export_forecast_v2_benchmark(
        result=result,
        output_directory=tmp_path,
        training_dataset=training,
        validation_dataset=validation,
        config=small_config(),
    )

    assert all(
        path.is_file()
        for path in (
            paths.observations_csv,
            paths.metrics_csv,
            paths.stability_summary_csv,
            paths.bootstrap_csv,
            paths.candidate_decisions_csv,
            paths.metadata_json,
        )
    )
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["development_splits_used"] == ["training", "validation"]
    assert metadata["test_split_read"] is False
    assert metadata["known_v1_test_used_for_selection"] is False
    assert metadata["features_fitted_per_fold_on_training_only"] is True


def test_cli_does_not_require_test_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, validation = development_splits()
    dataset_directory = tmp_path / "datasets"
    output_directory = tmp_path / "results"
    dataset_directory.mkdir()
    training.to_csv(dataset_directory / "train.csv", index=False)
    validation.to_csv(dataset_directory / "validation.csv", index=False)

    monkeypatch.setattr(
        "backend.src.forecast.forecast_v2_benchmark.RandomForestForecastModel",
        random_forest_factory,
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
    assert (output_directory / "benchmark_metadata.json").is_file()


def test_invalid_config_and_overlapping_splits_are_rejected() -> None:
    with pytest.raises(ValueError, match="folds"):
        ForecastV2BenchmarkConfig(folds=1)
    with pytest.raises(ValueError, match="minimum_classification_confidence"):
        ForecastV2BenchmarkConfig(minimum_classification_confidence=1.1)

    training, validation = development_splits()
    validation["date"] = validation["date"] - pd.Timedelta(days=30)
    with pytest.raises(ValueError, match="Training dates must end"):
        run_forecast_v2_benchmark(training, validation, config=small_config())

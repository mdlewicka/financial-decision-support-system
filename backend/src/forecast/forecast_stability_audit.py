import argparse
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from backend.src.config import MARKET_DATASETS_DIR, PROJECT_ROOT
from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)
from backend.src.storage.market_dataset_reader import MarketDatasetReader

DEFAULT_FORECAST_STABILITY_AUDIT_DIR = (
    PROJECT_ROOT / "benchmark_results" / "forecast_v2" / "stability_audit"
)

MODEL_NAMES: tuple[str, ...] = (
    "zero_return",
    "historical_mean",
    "last_observed_return",
    "random_forest_v1",
)

REQUIRED_COLUMNS = frozenset(
    {
        "date",
        "ticker",
        "target_date",
        "target",
    }
)


@dataclass(frozen=True, slots=True)
class AuditFold:
    fold: int
    training_rows: int
    evaluation_rows: int
    training_date_min: str
    training_date_max: str
    training_target_date_max: str
    evaluation_date_min: str
    evaluation_date_max: str
    evaluation_target_date_max: str


@dataclass(frozen=True, slots=True)
class ForecastStabilityAuditPaths:
    observations_csv: Path
    metrics_csv: Path
    stability_summary_csv: Path
    bootstrap_csv: Path
    metadata_json: Path


@dataclass(frozen=True, slots=True)
class ForecastStabilityAuditResult:
    observations: pd.DataFrame
    metrics: pd.DataFrame
    stability_summary: pd.DataFrame
    bootstrap_comparison: pd.DataFrame
    folds: tuple[AuditFold, ...]


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the stability of V1 forecasting performance using only "
            "training and validation data. The V1 test split is never read."
        )
    )
    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=MARKET_DATASETS_DIR,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_FORECAST_STABILITY_AUDIT_DIR,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    return parser


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    return create_argument_parser().parse_args(arguments)


def run_forecast_stability_audit(
    training_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    folds: int = 5,
    bootstrap_iterations: int = 1000,
    random_state: int = 42,
    model_factory: Callable[[], RandomForestForecastModel] | None = None,
) -> ForecastStabilityAuditResult:
    if folds < 2:
        raise ValueError("folds must be at least 2.")

    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be greater than zero.")

    training = _prepare_dataset(training_dataset, "training")
    validation = _prepare_dataset(validation_dataset, "validation")
    _validate_split_order(training, validation)

    selected_model_factory = model_factory or RandomForestForecastModel
    fold_datasets = _build_fold_datasets(
        training=training,
        validation=validation,
        folds=folds,
    )

    observation_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    fold_definitions: list[AuditFold] = []

    for fold_number, fold_training, fold_evaluation in fold_datasets:
        predictions = _generate_fold_predictions(
            training=fold_training,
            evaluation=fold_evaluation,
            model_factory=selected_model_factory,
        )

        fold_definitions.append(
            _build_fold_definition(
                fold=fold_number,
                training=fold_training,
                evaluation=fold_evaluation,
            )
        )

        for model_name in MODEL_NAMES:
            model_observations = _build_observations(
                fold=fold_number,
                model_name=model_name,
                evaluation=fold_evaluation,
                predictions=predictions[model_name],
            )
            observation_frames.append(model_observations)
            metric_rows.extend(_build_metric_rows(model_observations))

    observations = pd.concat(
        observation_frames,
        ignore_index=True,
    ).sort_values(
        ["fold", "model", "date", "ticker"],
        kind="stable",
        ignore_index=True,
    )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["fold", "scope", "ticker", "model"],
        kind="stable",
        ignore_index=True,
    )
    stability_summary = _build_stability_summary(metrics)
    bootstrap_comparison = _build_bootstrap_comparison(
        observations=observations,
        iterations=bootstrap_iterations,
        random_state=random_state,
    )

    return ForecastStabilityAuditResult(
        observations=observations,
        metrics=metrics,
        stability_summary=stability_summary,
        bootstrap_comparison=bootstrap_comparison,
        folds=tuple(fold_definitions),
    )


def export_forecast_stability_audit(
    result: ForecastStabilityAuditResult,
    output_directory: Path,
    training_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    bootstrap_iterations: int,
    random_state: int,
) -> ForecastStabilityAuditPaths:
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = ForecastStabilityAuditPaths(
        observations_csv=output_directory / "audit_observations.csv",
        metrics_csv=output_directory / "audit_metrics.csv",
        stability_summary_csv=output_directory / "stability_summary.csv",
        bootstrap_csv=output_directory / "bootstrap_comparison.csv",
        metadata_json=output_directory / "audit_metadata.json",
    )

    _prepare_csv_frame(result.observations).to_csv(
        paths.observations_csv,
        index=False,
    )
    _prepare_csv_frame(result.metrics).to_csv(
        paths.metrics_csv,
        index=False,
    )
    _prepare_csv_frame(result.stability_summary).to_csv(
        paths.stability_summary_csv,
        index=False,
    )
    _prepare_csv_frame(result.bootstrap_comparison).to_csv(
        paths.bootstrap_csv,
        index=False,
    )

    metadata = {
        "schema_version": 1,
        "audit_name": "forecast_v1_stability_audit",
        "development_splits_used": ["training", "validation"],
        "test_split_read": False,
        "target_definition": "next_session_simple_return",
        "models": list(MODEL_NAMES),
        "fold_count": len(result.folds),
        "bootstrap_iterations": bootstrap_iterations,
        "random_state": random_state,
        "training_fingerprint": _dataset_fingerprint(training_dataset),
        "validation_fingerprint": _dataset_fingerprint(validation_dataset),
        "folds": [asdict(fold) for fold in result.folds],
    }
    paths.metadata_json.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return paths


def run_audit_from_directory(
    dataset_directory: Path = MARKET_DATASETS_DIR,
    output_directory: Path = DEFAULT_FORECAST_STABILITY_AUDIT_DIR,
    folds: int = 5,
    bootstrap_iterations: int = 1000,
    random_state: int = 42,
) -> tuple[ForecastStabilityAuditResult, ForecastStabilityAuditPaths]:
    reader = MarketDatasetReader(input_directory=dataset_directory)
    training, validation = reader.read_training_and_validation()

    result = run_forecast_stability_audit(
        training_dataset=training,
        validation_dataset=validation,
        folds=folds,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )
    paths = export_forecast_stability_audit(
        result=result,
        output_directory=output_directory,
        training_dataset=training,
        validation_dataset=validation,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )
    return result, paths


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    result, paths = run_audit_from_directory(
        dataset_directory=parsed.dataset_directory,
        output_directory=parsed.output_directory,
        folds=parsed.folds,
        bootstrap_iterations=parsed.bootstrap_iterations,
        random_state=parsed.random_state,
    )

    print("Forecasting V2 stability audit")
    print("Development splits: training, validation")
    print("V1 test split read: no")
    print(f"Chronological folds: {len(result.folds)}")
    print("\nStability summary:")
    print(result.stability_summary.to_string(index=False))
    print(f"\nObservations: {paths.observations_csv}")
    print(f"Metrics: {paths.metrics_csv}")
    print(f"Summary: {paths.stability_summary_csv}")
    print(f"Bootstrap: {paths.bootstrap_csv}")
    print(f"Metadata: {paths.metadata_json}")
    return 0


def _prepare_dataset(dataset: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if dataset.empty:
        raise ValueError(f"{dataset_name} dataset must not be empty.")

    missing_columns = sorted(REQUIRED_COLUMNS.difference(dataset.columns))
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"{dataset_name} dataset is missing columns: {joined}.")

    prepared = dataset.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    prepared["target_date"] = pd.to_datetime(
        prepared["target_date"],
        errors="raise",
    )
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["target"] = pd.to_numeric(prepared["target"], errors="raise").astype(float)

    if prepared[["date", "target_date", "ticker", "target"]].isna().any().any():
        raise ValueError(f"{dataset_name} dataset contains missing key values.")

    if not np.isfinite(prepared["target"].to_numpy(dtype=float)).all():
        raise ValueError(f"{dataset_name} targets must be finite.")

    if (prepared["target_date"] <= prepared["date"]).any():
        raise ValueError(
            f"{dataset_name} target_date must be later than the feature date."
        )

    return prepared.sort_values(
        ["date", "ticker"],
        kind="stable",
        ignore_index=True,
    )


def _validate_split_order(training: pd.DataFrame, validation: pd.DataFrame) -> None:
    if training["date"].max() >= validation["date"].min():
        raise ValueError("Training dates must end before validation dates begin.")

    training_tickers = set(training["ticker"].astype(str))
    validation_tickers = set(validation["ticker"].astype(str))
    missing_tickers = sorted(validation_tickers.difference(training_tickers))
    if missing_tickers:
        raise ValueError(
            "Validation contains tickers absent from training: "
            + ", ".join(missing_tickers)
            + "."
        )


def _build_fold_datasets(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    folds: int,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    validation_dates = np.asarray(
        sorted(pd.to_datetime(validation["date"].unique())),
        dtype="datetime64[ns]",
    )
    if len(validation_dates) < folds:
        raise ValueError("Validation contains fewer unique dates than folds.")

    date_groups = np.array_split(validation_dates, folds)
    combined_development = pd.concat(
        [training, validation],
        ignore_index=True,
    )
    fold_datasets: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []

    for fold_number, date_group in enumerate(date_groups, start=1):
        evaluation_start = pd.Timestamp(date_group[0])
        evaluation_end = pd.Timestamp(date_group[-1])

        fold_training = combined_development.loc[
            (combined_development["date"] < evaluation_start)
            & (combined_development["target_date"] <= evaluation_start)
        ].copy()
        fold_evaluation = validation.loc[
            validation["date"].between(evaluation_start, evaluation_end)
        ].copy()

        if fold_training.empty or fold_evaluation.empty:
            raise RuntimeError(f"Fold {fold_number} is empty.")

        if fold_training["target_date"].max() > fold_evaluation["date"].min():
            raise RuntimeError(f"Fold {fold_number} leaks future targets.")

        fold_datasets.append(
            (
                fold_number,
                fold_training.reset_index(drop=True),
                fold_evaluation.reset_index(drop=True),
            )
        )

    return fold_datasets


def _generate_fold_predictions(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    model_factory: Callable[[], RandomForestForecastModel],
) -> dict[str, pd.Series]:
    historical_means = training.groupby("ticker", sort=True)["target"].mean()

    random_forest = model_factory()
    random_forest.fit(training)

    return {
        "zero_return": pd.Series(0.0, index=evaluation.index, dtype=float),
        "historical_mean": evaluation["ticker"].map(historical_means).astype(float),
        "last_observed_return": _last_observed_return_predictions(
            training,
            evaluation,
        ),
        "random_forest_v1": random_forest.predict(evaluation),
    }


def _last_observed_return_predictions(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> pd.Series:
    predictions = pd.Series(index=evaluation.index, dtype=float)

    for ticker, ticker_evaluation in evaluation.groupby("ticker", sort=True):
        ticker_training = training.loc[training["ticker"] == ticker].sort_values(
            ["target_date", "date"],
            kind="stable",
        )

        if ticker_training.empty:
            raise ValueError(f"No training history for ticker {ticker}.")

        known_targets = [
            (
                _as_timestamp(row.target_date, "target_date"),
                _as_float(row.target, "target"),
            )
            for row in ticker_training.itertuples()
        ]
        pending_targets: list[tuple[pd.Timestamp, float]] = []

        for row in ticker_evaluation.sort_values(
            "date",
            kind="stable",
        ).itertuples():
            current_date = _as_timestamp(row.date, "date")

            newly_known = [
                target for target in pending_targets if target[0] <= current_date
            ]

            if newly_known:
                known_targets.extend(newly_known)
                pending_targets = [
                    target for target in pending_targets if target[0] > current_date
                ]

            available = [
                target for target in known_targets if target[0] <= current_date
            ]

            if not available:
                raise RuntimeError(
                    f"No known target for ticker {ticker} at {current_date}."
                )

            row_index = _as_integer_index(row.Index)
            predictions.iloc[row_index] = max(
                available,
                key=lambda item: item[0],
            )[1]

            pending_targets.append(
                (
                    _as_timestamp(row.target_date, "target_date"),
                    _as_float(row.target, "target"),
                )
            )

    return predictions


def _build_observations(
    fold: int,
    model_name: str,
    evaluation: pd.DataFrame,
    predictions: pd.Series,
) -> pd.DataFrame:
    if not predictions.index.equals(evaluation.index):
        raise ValueError("Prediction index must match evaluation index.")

    observations = evaluation[["ticker", "date", "target_date"]].copy()
    observations.insert(0, "fold", fold)
    observations.insert(1, "model", model_name)
    observations["actual"] = evaluation["target"].astype(float)
    observations["prediction"] = predictions.astype(float)
    observations["absolute_error"] = (
        observations["actual"] - observations["prediction"]
    ).abs()
    observations["direction_correct"] = np.sign(observations["actual"]) == np.sign(
        observations["prediction"]
    )
    return observations


def _build_metric_rows(observations: pd.DataFrame) -> list[dict[str, Any]]:
    fold = int(observations["fold"].iloc[0])
    model_name = str(observations["model"].iloc[0])
    rows = [
        {
            "fold": fold,
            "model": model_name,
            "scope": "overall",
            "ticker": "",
            **_calculate_metrics(observations),
        }
    ]

    rows.extend(
        {
            "fold": fold,
            "model": model_name,
            "scope": "ticker",
            "ticker": str(ticker),
            **_calculate_metrics(ticker_observations),
        }
        for ticker, ticker_observations in observations.groupby("ticker", sort=True)
    )
    return rows


def _calculate_metrics(observations: pd.DataFrame) -> dict[str, int | float]:
    actual = observations["actual"].to_numpy(dtype=float)
    prediction = observations["prediction"].to_numpy(dtype=float)
    if np.unique(actual).size < 2 or np.unique(prediction).size < 2:
        spearman = math.nan
    else:
        spearman = pd.Series(actual).corr(
            pd.Series(prediction),
            method="spearman",
        )

    return {
        "observations": len(observations),
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(root_mean_squared_error(actual, prediction)),
        "r2": float(r2_score(actual, prediction)),
        "directional_accuracy": float((np.sign(actual) == np.sign(prediction)).mean()),
        "spearman": float(spearman) if pd.notna(spearman) else math.nan,
    }


def _build_stability_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    overall = metrics.loc[metrics["scope"] == "overall"].copy()
    fold_winners = (
        overall.loc[overall.groupby("fold")["mae"].transform("min") == overall["mae"]]
        .groupby("model")
        .size()
    )
    zero_mae = overall.loc[overall["model"] == "zero_return"].set_index("fold")["mae"]

    rows: list[dict[str, Any]] = []
    for model_name, model_metrics in overall.groupby("model", sort=True):
        ordered = model_metrics.sort_values("fold")
        aligned_zero = ordered["fold"].map(zero_mae)
        improvement = 100.0 * (aligned_zero - ordered["mae"]) / aligned_zero
        rows.append(
            {
                "model": model_name,
                "folds": len(ordered),
                "mean_mae": float(ordered["mae"].mean()),
                "median_mae": float(ordered["mae"].median()),
                "worst_mae": float(ordered["mae"].max()),
                "mean_rmse": float(ordered["rmse"].mean()),
                "mean_r2": float(ordered["r2"].mean()),
                "mean_directional_accuracy": float(
                    ordered["directional_accuracy"].mean()
                ),
                "mean_spearman": float(ordered["spearman"].mean()),
                "mae_fold_wins": int(fold_winners.get(model_name, 0)),
                "mean_mae_improvement_vs_zero_percent": float(improvement.mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["median_mae", "model"],
        kind="stable",
        ignore_index=True,
    )


def _build_bootstrap_comparison(
    observations: pd.DataFrame,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    zero = observations.loc[observations["model"] == "zero_return"][
        [
            "fold",
            "ticker",
            "target_date",
            "absolute_error",
        ]
    ].rename(columns={"absolute_error": "zero_absolute_error"})
    random_generator = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        model = observations.loc[observations["model"] == model_name][
            [
                "fold",
                "ticker",
                "target_date",
                "absolute_error",
            ]
        ]
        paired = model.merge(
            zero,
            on=["fold", "ticker", "target_date"],
            how="inner",
            validate="one_to_one",
        )
        paired["mae_difference"] = (
            paired["absolute_error"] - paired["zero_absolute_error"]
        )
        date_means = paired.groupby("target_date", sort=True)["mae_difference"].mean()
        values = date_means.to_numpy(dtype=float)
        samples = np.empty(iterations, dtype=float)

        for iteration in range(iterations):
            sampled_positions = random_generator.integers(
                0,
                len(values),
                size=len(values),
            )
            samples[iteration] = float(values[sampled_positions].mean())

        rows.append(
            {
                "model": model_name,
                "comparison_model": "zero_return",
                "mean_mae_difference": float(values.mean()),
                "ci_lower_95": float(np.quantile(samples, 0.025)),
                "ci_upper_95": float(np.quantile(samples, 0.975)),
                "bootstrap_probability_better_than_zero": float((samples < 0.0).mean()),
                "cluster_dates": len(values),
            }
        )

    return pd.DataFrame(rows)


def _build_fold_definition(
    fold: int,
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> AuditFold:
    return AuditFold(
        fold=fold,
        training_rows=len(training),
        evaluation_rows=len(evaluation),
        training_date_min=_date_text(training["date"].min()),
        training_date_max=_date_text(training["date"].max()),
        training_target_date_max=_date_text(training["target_date"].max()),
        evaluation_date_min=_date_text(evaluation["date"].min()),
        evaluation_date_max=_date_text(evaluation["date"].max()),
        evaluation_target_date_max=_date_text(evaluation["target_date"].max()),
    )


def _prepare_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in ("date", "target_date"):
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(prepared[column]).dt.strftime("%Y-%m-%d")
    return prepared


def _dataset_fingerprint(dataset: pd.DataFrame) -> str:
    columns = [
        column
        for column in ("date", "ticker", "target_date", "target")
        if column in dataset.columns
    ]
    hashed = pd.util.hash_pandas_object(
        dataset[columns].reset_index(drop=True),
        index=True,
    ).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _as_timestamp(
    value: object,
    name: str,
) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value

    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)

    raise TypeError(f"{name} must contain pandas or NumPy datetime values.")


def _as_float(
    value: object,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must contain numeric values.")

    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)

    raise TypeError(f"{name} must contain numeric values.")


def _as_integer_index(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("DataFrame index must contain integer values.")

    if isinstance(value, int | np.integer):
        return int(value)

    raise TypeError("DataFrame index must contain integer values.")


def _date_text(value: object) -> str:
    return _as_timestamp(value, "date").strftime("%Y-%m-%d")


if __name__ == "__main__":
    raise SystemExit(main())

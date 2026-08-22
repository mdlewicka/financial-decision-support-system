from __future__ import annotations

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
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

from backend.src.config import MARKET_DATASETS_DIR, PROJECT_ROOT
from backend.src.forecast.multi_horizon_target_builder import (
    TARGET_CLASSES,
    ForecastHorizon,
    ForecastTargetConfig,
    MultiHorizonForecastTargetBuilder,
)
from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)
from backend.src.models.regularized_forecast_models import (
    LogisticForecastConfig,
    RegularizedDirectionForecastModel,
    RegularizedReturnForecastModel,
    RidgeForecastConfig,
)
from backend.src.storage.market_dataset_reader import MarketDatasetReader

DEFAULT_FORECAST_V2_BENCHMARK_DIR = (
    PROJECT_ROOT / "benchmark_results" / "forecast_v2" / "model_benchmark"
)

REGRESSION_MODELS: dict[int, tuple[str, ...]] = {
    1: (
        "zero_return",
        "historical_mean",
        "random_forest_v1",
        "ridge_v2",
    ),
    5: (
        "zero_return",
        "historical_mean",
        "ridge_v2",
    ),
}

CLASSIFICATION_MODELS: tuple[str, ...] = (
    "always_abstain",
    "historical_prior",
    "logistic_v2",
)

REQUIRED_COLUMNS = frozenset(
    {
        "date",
        "ticker",
        "adjusted_close",
        "volume",
        "daily_return",
        "target_date",
        "target",
    }
)

RidgeFactory = Callable[[ForecastHorizon], RegularizedReturnForecastModel]
LogisticFactory = Callable[[], RegularizedDirectionForecastModel]
RandomForestFactory = Callable[[], RandomForestForecastModel]


@dataclass(frozen=True, slots=True)
class ForecastV2BenchmarkConfig:
    folds: int = 5
    bootstrap_iterations: int = 1000
    random_state: int = 42
    one_session_threshold: float = 0.0025
    five_session_threshold: float = 0.005
    minimum_classification_confidence: float = 0.45
    transaction_cost: float = 0.001

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("folds must be at least 2.")

        if self.bootstrap_iterations <= 0:
            raise ValueError("bootstrap_iterations must be greater than zero.")

        for name, value in (
            ("one_session_threshold", self.one_session_threshold),
            ("five_session_threshold", self.five_session_threshold),
            ("transaction_cost", self.transaction_cost),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

        if (
            not math.isfinite(self.minimum_classification_confidence)
            or not 0.0 <= self.minimum_classification_confidence <= 1.0
        ):
            raise ValueError("minimum_classification_confidence must be in [0, 1].")


@dataclass(frozen=True, slots=True)
class BenchmarkFold:
    fold: int
    training_rows: int
    training_date_min: str
    training_date_max: str
    evaluation_date_min: str
    evaluation_date_max: str
    evaluation_rows_horizon_1: int
    evaluation_rows_horizon_5: int
    training_target_date_max_horizon_1: str
    training_target_date_max_horizon_5: str


@dataclass(frozen=True, slots=True)
class ForecastV2BenchmarkResult:
    observations: pd.DataFrame
    metrics: pd.DataFrame
    stability_summary: pd.DataFrame
    bootstrap_comparison: pd.DataFrame
    candidate_decisions: pd.DataFrame
    folds: tuple[BenchmarkFold, ...]


@dataclass(frozen=True, slots=True)
class ForecastV2BenchmarkPaths:
    observations_csv: Path
    metrics_csv: Path
    stability_summary_csv: Path
    bootstrap_csv: Path
    candidate_decisions_csv: Path
    metadata_json: Path


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare V1 and V2 forecasting candidates on chronological "
            "development folds without reading the test split."
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
        default=DEFAULT_FORECAST_V2_BENCHMARK_DIR,
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
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )
    return parser


def run_forecast_v2_benchmark(
    training_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    config: ForecastV2BenchmarkConfig | None = None,
    ridge_factory: RidgeFactory | None = None,
    logistic_factory: LogisticFactory | None = None,
    random_forest_factory: RandomForestFactory | None = None,
) -> ForecastV2BenchmarkResult:
    selected_config = config or ForecastV2BenchmarkConfig()

    target_config = ForecastTargetConfig(
        one_session_threshold=selected_config.one_session_threshold,
        five_session_threshold=selected_config.five_session_threshold,
    )
    target_builder = MultiHorizonForecastTargetBuilder(target_config)

    training = _prepare_dataset(
        training_dataset,
        "training",
    )
    validation = _prepare_dataset(
        validation_dataset,
        "validation",
    )
    _validate_split_order(training, validation)

    training = training.assign(__split="training")
    validation = validation.assign(__split="validation")

    development = pd.concat(
        [training, validation],
        ignore_index=True,
    ).sort_values(
        ["date", "ticker"],
        kind="stable",
        ignore_index=True,
    )

    development_targets = target_builder.build(development)

    _validate_v1_target_compatibility(
        development,
        development_targets,
    )

    date_groups = _build_validation_date_groups(
        validation,
        selected_config.folds,
    )

    selected_ridge_factory = ridge_factory or _default_ridge_factory(target_config)
    selected_logistic_factory = logistic_factory or _default_logistic_factory(
        target_config,
        selected_config.minimum_classification_confidence,
    )
    selected_random_forest_factory = random_forest_factory or RandomForestForecastModel

    observation_frames: list[pd.DataFrame] = []
    fold_definitions: list[BenchmarkFold] = []

    for fold_number, date_group in enumerate(
        date_groups,
        start=1,
    ):
        evaluation_start = pd.Timestamp(date_group[0])
        evaluation_end = pd.Timestamp(date_group[-1])

        fold_training = development.loc[development["date"] < evaluation_start].drop(
            columns="__split"
        )
        fold_training = fold_training.reset_index(drop=True)

        fold_training_targets = target_builder.build(fold_training)

        fold_evaluation_mask = development["__split"].eq("validation") & development[
            "date"
        ].between(
            evaluation_start,
            evaluation_end,
        )

        fold_evaluation = development.loc[fold_evaluation_mask].drop(columns="__split")

        fold_targets = development_targets.loc[fold_evaluation.index]

        evaluation_by_horizon = {
            horizon: _evaluation_for_horizon(
                fold_evaluation,
                fold_targets,
                horizon,
            )
            for horizon in (1, 5)
        }

        fold_definitions.append(
            _build_fold_definition(
                fold=fold_number,
                training=fold_training,
                training_targets=fold_training_targets,
                evaluation_by_horizon=evaluation_by_horizon,
            )
        )

        for horizon in (1, 5):
            typed_horizon: ForecastHorizon = horizon
            evaluation = evaluation_by_horizon[horizon]

            predictions = _regression_predictions(
                training=fold_training,
                training_targets=fold_training_targets,
                evaluation=evaluation,
                horizon=typed_horizon,
                ridge_factory=selected_ridge_factory,
                random_forest_factory=(selected_random_forest_factory),
            )

            threshold = _threshold(
                selected_config,
                typed_horizon,
            )

            for model_name in REGRESSION_MODELS[horizon]:
                observation_frames.append(
                    _regression_observations(
                        fold=fold_number,
                        horizon=typed_horizon,
                        model_name=model_name,
                        evaluation=evaluation,
                        predictions=predictions[model_name],
                        threshold=threshold,
                        transaction_cost=(selected_config.transaction_cost),
                    )
                )

        classification_evaluation = evaluation_by_horizon[5]

        probability_frames = _classification_probabilities(
            training=fold_training,
            training_targets=fold_training_targets,
            evaluation=classification_evaluation,
            logistic_factory=selected_logistic_factory,
        )

        for model_name in CLASSIFICATION_MODELS:
            observation_frames.append(
                _classification_observations(
                    fold=fold_number,
                    model_name=model_name,
                    evaluation=classification_evaluation,
                    probabilities=probability_frames[model_name],
                    minimum_confidence=(
                        selected_config.minimum_classification_confidence
                        if model_name == "logistic_v2"
                        else 0.0
                    ),
                    transaction_cost=(selected_config.transaction_cost),
                )
            )

    observations = pd.concat(
        observation_frames,
        ignore_index=True,
    ).sort_values(
        [
            "task",
            "horizon",
            "fold",
            "model",
            "date",
            "ticker",
        ],
        kind="stable",
        ignore_index=True,
    )

    metrics = _build_metrics(observations)
    stability_summary = _build_stability_summary(metrics)

    bootstrap_comparison = _build_bootstrap_comparison(
        observations=observations,
        iterations=selected_config.bootstrap_iterations,
        random_state=selected_config.random_state,
    )

    candidate_decisions = _build_candidate_decisions(
        stability_summary,
        bootstrap_comparison,
    )

    return ForecastV2BenchmarkResult(
        observations=observations,
        metrics=metrics,
        stability_summary=stability_summary,
        bootstrap_comparison=bootstrap_comparison,
        candidate_decisions=candidate_decisions,
        folds=tuple(fold_definitions),
    )


def export_forecast_v2_benchmark(
    result: ForecastV2BenchmarkResult,
    output_directory: Path,
    training_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    config: ForecastV2BenchmarkConfig,
) -> ForecastV2BenchmarkPaths:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = ForecastV2BenchmarkPaths(
        observations_csv=(output_directory / "benchmark_observations.csv"),
        metrics_csv=(output_directory / "benchmark_metrics.csv"),
        stability_summary_csv=(output_directory / "stability_summary.csv"),
        bootstrap_csv=(output_directory / "bootstrap_comparison.csv"),
        candidate_decisions_csv=(output_directory / "candidate_decisions.csv"),
        metadata_json=(output_directory / "benchmark_metadata.json"),
    )

    for frame, path in (
        (result.observations, paths.observations_csv),
        (result.metrics, paths.metrics_csv),
        (result.stability_summary, paths.stability_summary_csv),
        (result.bootstrap_comparison, paths.bootstrap_csv),
        (
            result.candidate_decisions,
            paths.candidate_decisions_csv,
        ),
    ):
        _prepare_csv_frame(frame).to_csv(
            path,
            index=False,
        )

    metadata = {
        "schema_version": 1,
        "benchmark_name": "forecast_v2_model_benchmark",
        "development_splits_used": [
            "training",
            "validation",
        ],
        "test_split_read": False,
        "known_v1_test_used_for_selection": False,
        "config": asdict(config),
        "regression_models": {
            str(horizon): list(models) for horizon, models in REGRESSION_MODELS.items()
        },
        "classification_models": list(CLASSIFICATION_MODELS),
        "model_configurations": {
            "ridge_v2": {
                "alpha": 10.0,
                "horizons": [1, 5],
            },
            "logistic_v2": {
                "inverse_regularization_strength": 0.5,
                "horizon": 5,
                "minimum_confidence": (config.minimum_classification_confidence),
                "class_weight": "balanced",
                "max_iterations": 1000,
            },
        },
        "features_fitted_per_fold_on_training_only": True,
        "five_session_utility_warning": (
            "Per-decision utility uses overlapping five-session targets "
            "and is a screening metric, not a portfolio backtest."
        ),
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


def run_benchmark_from_directory(
    dataset_directory: Path = MARKET_DATASETS_DIR,
    output_directory: Path = DEFAULT_FORECAST_V2_BENCHMARK_DIR,
    config: ForecastV2BenchmarkConfig | None = None,
) -> tuple[
    ForecastV2BenchmarkResult,
    ForecastV2BenchmarkPaths,
]:
    selected_config = config or ForecastV2BenchmarkConfig()

    training, validation = MarketDatasetReader(
        input_directory=dataset_directory
    ).read_training_and_validation()

    result = run_forecast_v2_benchmark(
        training_dataset=training,
        validation_dataset=validation,
        config=selected_config,
    )

    paths = export_forecast_v2_benchmark(
        result=result,
        output_directory=output_directory,
        training_dataset=training,
        validation_dataset=validation,
        config=selected_config,
    )

    return result, paths


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = create_argument_parser().parse_args(arguments)

    config = ForecastV2BenchmarkConfig(
        folds=parsed.folds,
        bootstrap_iterations=parsed.bootstrap_iterations,
        random_state=parsed.random_state,
        transaction_cost=parsed.transaction_cost,
    )

    result, paths = run_benchmark_from_directory(
        dataset_directory=parsed.dataset_directory,
        output_directory=parsed.output_directory,
        config=config,
    )

    print("Forecasting V2 model benchmark")
    print("Development splits: training, validation")
    print("V1 test split read: no")
    print(f"Chronological folds: {len(result.folds)}")
    print("\nCandidate decisions:")
    print(result.candidate_decisions.to_string(index=False))
    print(f"\nObservations: {paths.observations_csv}")
    print(f"Metrics: {paths.metrics_csv}")
    print(f"Summary: {paths.stability_summary_csv}")
    print(f"Bootstrap: {paths.bootstrap_csv}")
    print(f"Decisions: {paths.candidate_decisions_csv}")
    print(f"Metadata: {paths.metadata_json}")

    return 0


def _default_ridge_factory(
    target_config: ForecastTargetConfig,
) -> RidgeFactory:
    def create(
        horizon: ForecastHorizon,
    ) -> RegularizedReturnForecastModel:
        return RegularizedReturnForecastModel(
            config=RidgeForecastConfig(
                horizon=horizon,
                alpha=10.0,
            ),
            target_config=target_config,
        )

    return create


def _default_logistic_factory(
    target_config: ForecastTargetConfig,
    minimum_confidence: float,
) -> LogisticFactory:
    def create() -> RegularizedDirectionForecastModel:
        return RegularizedDirectionForecastModel(
            config=LogisticForecastConfig(
                horizon=5,
                inverse_regularization_strength=0.5,
                minimum_confidence=minimum_confidence,
                max_iterations=1000,
                random_state=42,
            ),
            target_config=target_config,
        )

    return create


def _prepare_dataset(
    dataset: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    if dataset.empty:
        raise ValueError(f"{name} dataset must not be empty.")

    missing = sorted(REQUIRED_COLUMNS.difference(dataset.columns))
    if missing:
        raise ValueError(f"{name} dataset is missing columns: {', '.join(missing)}.")

    prepared = dataset.copy()

    prepared["date"] = pd.to_datetime(
        prepared["date"],
        errors="raise",
    )
    prepared["target_date"] = pd.to_datetime(
        prepared["target_date"],
        errors="raise",
    )
    prepared["ticker"] = prepared["ticker"].astype("string")

    for column in (
        "adjusted_close",
        "volume",
        "daily_return",
        "target",
    ):
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="raise",
        ).astype(float)

    if prepared[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(f"{name} dataset contains missing required values.")

    numeric = prepared[
        [
            "adjusted_close",
            "volume",
            "daily_return",
            "target",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} dataset contains non-finite values.")

    return prepared.sort_values(
        ["date", "ticker"],
        kind="stable",
        ignore_index=True,
    )


def _validate_split_order(
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    if training["date"].max() >= validation["date"].min():
        raise ValueError("Training dates must end before validation dates begin.")

    missing_tickers = sorted(set(validation["ticker"]) - set(training["ticker"]))

    if missing_tickers:
        raise ValueError(
            "Validation contains tickers absent from training: "
            + ", ".join(str(value) for value in missing_tickers)
            + "."
        )


def _validate_v1_target_compatibility(
    development: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    ready = targets["target_ready_1"].astype(bool)

    expected = targets.loc[
        ready,
        "target_return_1",
    ].to_numpy(dtype=float)

    existing = development.loc[
        ready,
        "target",
    ].to_numpy(dtype=float)

    if not np.allclose(
        expected,
        existing,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError("Rebuilt one-session targets do not match the V1 target.")


def _build_validation_date_groups(
    validation: pd.DataFrame,
    folds: int,
) -> list[np.ndarray]:
    dates = np.asarray(
        sorted(pd.to_datetime(validation["date"].unique())),
        dtype="datetime64[ns]",
    )

    if len(dates) < folds:
        raise ValueError("Validation contains fewer unique dates than folds.")

    return [
        group
        for group in np.array_split(
            dates,
            folds,
        )
        if len(group)
    ]


def _evaluation_for_horizon(
    evaluation: pd.DataFrame,
    targets: pd.DataFrame,
    horizon: ForecastHorizon,
) -> pd.DataFrame:
    ready = targets[f"target_ready_{horizon}"].astype(bool)

    selected = evaluation.loc[ready].copy().reset_index(drop=True)

    selected["actual_return"] = targets.loc[
        ready,
        f"target_return_{horizon}",
    ].to_numpy(dtype=float)

    selected["actual_class"] = (
        targets.loc[
            ready,
            f"target_class_{horizon}",
        ]
        .astype(str)
        .to_numpy()
    )

    selected["evaluation_target_date"] = pd.to_datetime(
        targets.loc[
            ready,
            f"target_date_{horizon}",
        ]
    ).to_numpy()

    if selected.empty:
        raise RuntimeError(f"No ready evaluation targets for horizon {horizon}.")

    return selected


def _regression_predictions(
    training: pd.DataFrame,
    training_targets: pd.DataFrame,
    evaluation: pd.DataFrame,
    horizon: ForecastHorizon,
    ridge_factory: RidgeFactory,
    random_forest_factory: RandomForestFactory,
) -> dict[str, pd.Series]:
    ready = training_targets[f"target_ready_{horizon}"].astype(bool)

    means = (
        training_targets.loc[ready]
        .groupby(
            "ticker",
            sort=True,
        )[f"target_return_{horizon}"]
        .mean()
    )

    ridge = ridge_factory(horizon)
    ridge.fit(training)

    predictions = {
        "zero_return": pd.Series(
            0.0,
            index=evaluation.index,
            dtype=float,
        ),
        "historical_mean": (evaluation["ticker"].map(means).astype(float)),
        "ridge_v2": ridge.predict(evaluation),
    }

    if horizon == 1:
        rf_training = training.loc[
            training["target_date"] <= evaluation["date"].min()
        ].reset_index(drop=True)

        random_forest = random_forest_factory()
        random_forest.fit(rf_training)

        predictions["random_forest_v1"] = random_forest.predict(evaluation)

    return predictions


def _classification_probabilities(
    training: pd.DataFrame,
    training_targets: pd.DataFrame,
    evaluation: pd.DataFrame,
    logistic_factory: LogisticFactory,
) -> dict[str, pd.DataFrame]:
    ready = training_targets["target_ready_5"].astype(bool)

    class_counts = training_targets.loc[
        ready,
        "target_class_5",
    ].value_counts()

    priors = np.asarray(
        [float(class_counts.get(name, 0)) for name in TARGET_CLASSES],
        dtype=float,
    )
    priors /= priors.sum()

    always_abstain = np.zeros(
        (
            len(evaluation),
            len(TARGET_CLASSES),
        ),
        dtype=float,
    )

    always_abstain[
        :,
        TARGET_CLASSES.index("abstain"),
    ] = 1.0

    historical_prior = np.tile(
        priors,
        (
            len(evaluation),
            1,
        ),
    )

    logistic = logistic_factory()
    logistic.fit(training)

    return {
        "always_abstain": pd.DataFrame(
            always_abstain,
            columns=TARGET_CLASSES,
            index=evaluation.index,
        ),
        "historical_prior": pd.DataFrame(
            historical_prior,
            columns=TARGET_CLASSES,
            index=evaluation.index,
        ),
        "logistic_v2": logistic.predict_proba(evaluation),
    }


def _regression_observations(
    fold: int,
    horizon: ForecastHorizon,
    model_name: str,
    evaluation: pd.DataFrame,
    predictions: pd.Series,
    threshold: float,
    transaction_cost: float,
) -> pd.DataFrame:
    prediction = predictions.reindex(evaluation.index).to_numpy(dtype=float)

    actual = evaluation["actual_return"].to_numpy(dtype=float)

    predicted_class = _return_classes(
        prediction,
        threshold,
    )

    active = predicted_class != "abstain"

    signal = np.where(
        predicted_class == "up",
        1.0,
        -1.0,
    )

    net_utility = np.where(
        active,
        signal * actual - transaction_cost,
        0.0,
    )

    return _observation_frame(
        fold=fold,
        horizon=horizon,
        task="regression",
        model_name=model_name,
        evaluation=evaluation,
        prediction=prediction,
        predicted_class=predicted_class,
        probabilities=None,
        net_utility=net_utility,
    )


def _classification_observations(
    fold: int,
    model_name: str,
    evaluation: pd.DataFrame,
    probabilities: pd.DataFrame,
    minimum_confidence: float,
    transaction_cost: float,
) -> pd.DataFrame:
    aligned = probabilities.reindex(evaluation.index)[list(TARGET_CLASSES)]

    predicted_class = aligned.idxmax(axis=1).to_numpy(dtype=str)

    confidence = aligned.max(axis=1).to_numpy(dtype=float)

    predicted_class[confidence < minimum_confidence] = "abstain"

    actual = evaluation["actual_return"].to_numpy(dtype=float)

    active = predicted_class != "abstain"

    signal = np.where(
        predicted_class == "up",
        1.0,
        -1.0,
    )

    net_utility = np.where(
        active,
        signal * actual - transaction_cost,
        0.0,
    )

    return _observation_frame(
        fold=fold,
        horizon=5,
        task="classification",
        model_name=model_name,
        evaluation=evaluation,
        prediction=np.full(
            len(evaluation),
            np.nan,
            dtype=float,
        ),
        predicted_class=predicted_class,
        probabilities=aligned,
        net_utility=net_utility,
    )


def _observation_frame(
    fold: int,
    horizon: int,
    task: str,
    model_name: str,
    evaluation: pd.DataFrame,
    prediction: np.ndarray,
    predicted_class: np.ndarray,
    probabilities: pd.DataFrame | None,
    net_utility: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "fold": fold,
            "horizon": horizon,
            "task": task,
            "model": model_name,
            "ticker": (evaluation["ticker"].astype(str).to_numpy()),
            "date": evaluation["date"].to_numpy(),
            "target_date": evaluation["evaluation_target_date"].to_numpy(),
            "actual_return": evaluation["actual_return"].to_numpy(dtype=float),
            "actual_class": (evaluation["actual_class"].astype(str).to_numpy()),
            "prediction": prediction,
            "predicted_class": predicted_class,
            "active": (predicted_class != "abstain"),
            "net_utility": net_utility,
        }
    )

    for class_name in TARGET_CLASSES:
        frame[f"probability_{class_name}"] = (
            probabilities[class_name].to_numpy(dtype=float)
            if probabilities is not None
            else np.nan
        )

    return frame


def _build_metrics(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    groups = observations.groupby(
        [
            "task",
            "horizon",
            "fold",
            "model",
        ],
        sort=True,
    )

    for (
        task,
        horizon,
        fold,
        model,
    ), group in groups:
        rows.append(
            {
                "task": task,
                "horizon": horizon,
                "fold": fold,
                "model": model,
                "scope": "overall",
                "ticker": "",
                **_calculate_metrics(group),
            }
        )

        for ticker, ticker_group in group.groupby(
            "ticker",
            sort=True,
        ):
            rows.append(
                {
                    "task": task,
                    "horizon": horizon,
                    "fold": fold,
                    "model": model,
                    "scope": "ticker",
                    "ticker": ticker,
                    **_calculate_metrics(ticker_group),
                }
            )

    return pd.DataFrame(rows).sort_values(
        [
            "task",
            "horizon",
            "fold",
            "scope",
            "ticker",
            "model",
        ],
        kind="stable",
        ignore_index=True,
    )


def _calculate_metrics(
    observations: pd.DataFrame,
) -> dict[str, int | float]:
    actual = observations["actual_return"].to_numpy(dtype=float)

    active = observations["active"].to_numpy(dtype=bool)

    predicted_class = observations["predicted_class"].to_numpy(dtype=str)

    actual_class = observations["actual_class"].to_numpy(dtype=str)

    base: dict[str, int | float] = {
        "observations": len(observations),
        "coverage": float(active.mean()),
        "active_hit_rate": (
            float((predicted_class[active] == actual_class[active]).mean())
            if active.any()
            else math.nan
        ),
        "mean_net_utility": float(observations["net_utility"].mean()),
        "total_net_utility": float(observations["net_utility"].sum()),
        "mae": math.nan,
        "rmse": math.nan,
        "r2": math.nan,
        "directional_accuracy": math.nan,
        "spearman": math.nan,
        "balanced_accuracy": math.nan,
        "useful_precision": math.nan,
        "useful_recall": math.nan,
        "multiclass_brier": math.nan,
        "roc_auc_ovr": math.nan,
    }

    if observations["task"].iloc[0] == "regression":
        prediction = observations["prediction"].to_numpy(dtype=float)

        base.update(
            {
                "mae": float(
                    mean_absolute_error(
                        actual,
                        prediction,
                    )
                ),
                "rmse": float(
                    root_mean_squared_error(
                        actual,
                        prediction,
                    )
                ),
                "r2": float(
                    r2_score(
                        actual,
                        prediction,
                    )
                ),
                "directional_accuracy": float(
                    (np.sign(actual) == np.sign(prediction)).mean()
                ),
                "spearman": _spearman(
                    actual,
                    prediction,
                ),
            }
        )

        return base

    useful_actual = actual_class != "abstain"
    useful_predicted = predicted_class != "abstain"

    correct_useful = useful_actual & (predicted_class == actual_class)

    probability_columns = [f"probability_{name}" for name in TARGET_CLASSES]

    probabilities = observations[probability_columns].to_numpy(dtype=float)

    one_hot = np.column_stack(
        [(actual_class == name).astype(float) for name in TARGET_CLASSES]
    )

    base.update(
        {
            "balanced_accuracy": _balanced_accuracy(
                actual_class,
                predicted_class,
            ),
            "useful_precision": (
                float(correct_useful.sum() / useful_predicted.sum())
                if useful_predicted.any()
                else math.nan
            ),
            "useful_recall": (
                float(correct_useful.sum() / useful_actual.sum())
                if useful_actual.any()
                else math.nan
            ),
            "multiclass_brier": float(
                np.mean(
                    np.sum(
                        (probabilities - one_hot) ** 2,
                        axis=1,
                    )
                )
            ),
            "roc_auc_ovr": _multiclass_auc(
                actual_class,
                probabilities,
            ),
        }
    )

    return base


def _build_stability_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    overall = metrics.loc[metrics["scope"] == "overall"].copy()

    rows: list[dict[str, Any]] = []

    for (
        task,
        horizon,
        model,
    ), group in overall.groupby(
        [
            "task",
            "horizon",
            "model",
        ],
        sort=True,
    ):
        ordered = group.sort_values("fold")

        if task == "regression":
            baseline = overall.loc[
                (overall["task"] == task)
                & (overall["horizon"] == horizon)
                & (overall["model"] == "zero_return")
            ].set_index("fold")

            aligned = ordered["fold"].map(baseline["mae"])

            improvement = 100.0 * (aligned - ordered["mae"]) / aligned

            fold_wins = int((ordered["mae"].to_numpy() < aligned.to_numpy()).sum())

            primary_mean = float(ordered["mae"].mean())
            primary_median = float(ordered["mae"].median())
            primary_worst = float(ordered["mae"].max())

            improvement_mean = float(improvement.mean())
            improvement_median = float(improvement.median())

        else:
            baseline = overall.loc[
                (overall["task"] == task)
                & (overall["horizon"] == horizon)
                & (overall["model"] == "historical_prior")
            ].set_index("fold")

            aligned = ordered["fold"].map(baseline["balanced_accuracy"])

            improvement = ordered["balanced_accuracy"] - aligned

            fold_wins = int(
                (ordered["balanced_accuracy"].to_numpy() > aligned.to_numpy()).sum()
            )

            primary_mean = float(ordered["balanced_accuracy"].mean())
            primary_median = float(ordered["balanced_accuracy"].median())
            primary_worst = float(ordered["balanced_accuracy"].min())

            improvement_mean = float(improvement.mean())
            improvement_median = float(improvement.median())

        rows.append(
            {
                "task": task,
                "horizon": horizon,
                "model": model,
                "folds": len(ordered),
                "primary_metric": (
                    "mae" if task == "regression" else "balanced_accuracy"
                ),
                "primary_mean": primary_mean,
                "primary_median": primary_median,
                "primary_worst": primary_worst,
                "mean_improvement_vs_baseline": (improvement_mean),
                "median_improvement_vs_baseline": (improvement_median),
                "fold_wins_vs_baseline": fold_wins,
                "mean_rmse": float(ordered["rmse"].mean()),
                "mean_r2": float(ordered["r2"].mean()),
                "mean_spearman": float(ordered["spearman"].mean()),
                "mean_balanced_accuracy": float(ordered["balanced_accuracy"].mean()),
                "mean_coverage": float(ordered["coverage"].mean()),
                "mean_active_hit_rate": float(ordered["active_hit_rate"].mean()),
                "mean_useful_precision": float(ordered["useful_precision"].mean()),
                "mean_useful_recall": float(ordered["useful_recall"].mean()),
                "mean_multiclass_brier": float(ordered["multiclass_brier"].mean()),
                "mean_net_utility": float(ordered["mean_net_utility"].mean()),
                "positive_utility_folds": int(
                    (ordered["mean_net_utility"] > 0.0).sum()
                ),
            }
        )

    result = pd.DataFrame(rows)

    result["__primary_sort"] = np.where(
        result["task"].eq("classification"),
        -result["primary_median"],
        result["primary_median"],
    )

    return result.sort_values(
        [
            "task",
            "horizon",
            "__primary_sort",
            "model",
        ],
        kind="stable",
        ignore_index=True,
    ).drop(columns="__primary_sort")


def _build_bootstrap_comparison(
    observations: pd.DataFrame,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    random_generator = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []

    regression = observations.loc[observations["task"] == "regression"]

    for horizon in (1, 5):
        horizon_data = regression.loc[regression["horizon"] == horizon].copy()

        zero = horizon_data.loc[horizon_data["model"] == "zero_return"].copy()

        zero["absolute_error_zero"] = zero["actual_return"].abs()

        for model_name in REGRESSION_MODELS[horizon]:
            model = horizon_data.loc[horizon_data["model"] == model_name].copy()

            model["absolute_error_model"] = (
                model["actual_return"] - model["prediction"]
            ).abs()

            paired = model.merge(
                zero[
                    [
                        "fold",
                        "ticker",
                        "target_date",
                        "absolute_error_zero",
                    ]
                ],
                on=[
                    "fold",
                    "ticker",
                    "target_date",
                ],
                validate="one_to_one",
            )

            paired["difference"] = (
                paired["absolute_error_model"] - paired["absolute_error_zero"]
            )

            date_means = paired.groupby(
                "target_date",
                sort=True,
            )["difference"].mean()

            values = date_means.to_numpy(dtype=float)

            samples = np.asarray(
                [
                    values[
                        random_generator.integers(
                            0,
                            len(values),
                            size=len(values),
                        )
                    ].mean()
                    for _ in range(iterations)
                ],
                dtype=float,
            )

            rows.append(
                {
                    "task": "regression",
                    "horizon": horizon,
                    "model": model_name,
                    "comparison_model": ("zero_return"),
                    "mean_primary_difference": float(values.mean()),
                    "ci_lower_95": float(
                        np.quantile(
                            samples,
                            0.025,
                        )
                    ),
                    "ci_upper_95": float(
                        np.quantile(
                            samples,
                            0.975,
                        )
                    ),
                    "probability_better_than_baseline": float((samples < 0.0).mean()),
                    "cluster_dates": len(values),
                }
            )

    classification = observations.loc[observations["task"] == "classification"]

    historical_prior = classification.loc[classification["model"] == "historical_prior"]

    baseline_counts = _classification_counts_by_date(historical_prior)

    for model_name in CLASSIFICATION_MODELS:
        model = classification.loc[classification["model"] == model_name]

        model_counts = _classification_counts_by_date(model)

        if not model_counts.index.equals(baseline_counts.index):
            raise RuntimeError("Classification bootstrap dates are not aligned.")

        model_values = model_counts.to_numpy(dtype=float)
        baseline_values = baseline_counts.to_numpy(dtype=float)

        point_difference = _balanced_accuracy_from_counts(
            model_values.sum(axis=0)
        ) - _balanced_accuracy_from_counts(baseline_values.sum(axis=0))

        samples = np.empty(
            iterations,
            dtype=float,
        )

        for iteration in range(iterations):
            positions = random_generator.integers(
                0,
                len(model_values),
                size=len(model_values),
            )

            samples[iteration] = _balanced_accuracy_from_counts(
                model_values[positions].sum(axis=0)
            ) - _balanced_accuracy_from_counts(baseline_values[positions].sum(axis=0))

        rows.append(
            {
                "task": "classification",
                "horizon": 5,
                "model": model_name,
                "comparison_model": ("historical_prior"),
                "mean_primary_difference": float(point_difference),
                "ci_lower_95": float(
                    np.quantile(
                        samples,
                        0.025,
                    )
                ),
                "ci_upper_95": float(
                    np.quantile(
                        samples,
                        0.975,
                    )
                ),
                "probability_better_than_baseline": float((samples > 0.0).mean()),
                "cluster_dates": len(model_values),
            }
        )

    return pd.DataFrame(rows)


def _build_candidate_decisions(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    candidates = summary.loc[
        summary["model"].isin(
            [
                "ridge_v2",
                "logistic_v2",
            ]
        )
    ].copy()

    rows: list[dict[str, Any]] = []

    for row in candidates.itertuples():
        task = _as_text(
            row.task,
            "task",
        )
        horizon = _as_integer(
            row.horizon,
            "horizon",
        )
        model = _as_text(
            row.model,
            "model",
        )
        median_improvement = _as_float(
            row.median_improvement_vs_baseline,
            "median_improvement_vs_baseline",
        )
        fold_wins = _as_integer(
            row.fold_wins_vs_baseline,
            "fold_wins_vs_baseline",
        )
        folds = _as_integer(
            row.folds,
            "folds",
        )
        mean_net_utility = _as_float(
            row.mean_net_utility,
            "mean_net_utility",
        )

        bootstrap_row = bootstrap.loc[
            (bootstrap["task"] == task)
            & (bootstrap["horizon"] == horizon)
            & (bootstrap["model"] == model)
        ]

        if len(bootstrap_row) != 1:
            raise RuntimeError("Candidate bootstrap result is missing or duplicated.")

        bootstrap_probability = _as_float(
            bootstrap_row["probability_better_than_baseline"].iloc[0],
            "probability_better_than_baseline",
        )

        primary_improves = median_improvement > 0.0

        majority_wins = fold_wins >= folds // 2 + 1

        bootstrap_support = bootstrap_probability >= 0.8

        positive_utility = mean_net_utility > 0.0

        promote = (
            primary_improves
            and majority_wins
            and bootstrap_support
            and positive_utility
        )

        rows.append(
            {
                "task": task,
                "horizon": horizon,
                "model": model,
                "primary_metric_improves": (primary_improves),
                "majority_fold_wins": (majority_wins),
                "bootstrap_probability_better": (bootstrap_probability),
                "bootstrap_support_at_80_percent": (bootstrap_support),
                "positive_screening_utility": (positive_utility),
                "decision": (
                    "promote_to_economic_validation"
                    if promote
                    else "do_not_promote_in_current_form"
                ),
                "decision_is_final_test_claim": False,
            }
        )

    return pd.DataFrame(rows)


def _classification_counts_by_date(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for target_date, group in observations.groupby(
        "target_date",
        sort=True,
    ):
        actual = group["actual_class"].to_numpy(dtype=str)

        predicted = group["predicted_class"].to_numpy(dtype=str)

        row: dict[str, Any] = {"target_date": target_date}

        for class_name in TARGET_CLASSES:
            class_mask = actual == class_name

            row[f"actual_{class_name}"] = int(class_mask.sum())

            row[f"correct_{class_name}"] = int(
                (class_mask & (predicted == class_name)).sum()
            )

        rows.append(row)

    return pd.DataFrame(rows).set_index("target_date")


def _balanced_accuracy_from_counts(
    values: np.ndarray,
) -> float:
    recalls: list[float] = []

    for class_index in range(len(TARGET_CLASSES)):
        actual_count = float(values[2 * class_index])

        correct_count = float(values[2 * class_index + 1])

        if actual_count > 0.0:
            recalls.append(correct_count / actual_count)

    return float(np.mean(recalls)) if recalls else math.nan


def _build_fold_definition(
    fold: int,
    training: pd.DataFrame,
    training_targets: pd.DataFrame,
    evaluation_by_horizon: dict[int, pd.DataFrame],
) -> BenchmarkFold:
    return BenchmarkFold(
        fold=fold,
        training_rows=len(training),
        training_date_min=_date_text(training["date"].min()),
        training_date_max=_date_text(training["date"].max()),
        evaluation_date_min=_date_text(evaluation_by_horizon[1]["date"].min()),
        evaluation_date_max=_date_text(evaluation_by_horizon[1]["date"].max()),
        evaluation_rows_horizon_1=len(evaluation_by_horizon[1]),
        evaluation_rows_horizon_5=len(evaluation_by_horizon[5]),
        training_target_date_max_horizon_1=(
            _ready_target_max(
                training_targets,
                1,
            )
        ),
        training_target_date_max_horizon_5=(
            _ready_target_max(
                training_targets,
                5,
            )
        ),
    )


def _ready_target_max(
    targets: pd.DataFrame,
    horizon: ForecastHorizon,
) -> str:
    ready = targets[f"target_ready_{horizon}"].astype(bool)

    return _date_text(
        targets.loc[
            ready,
            f"target_date_{horizon}",
        ].max()
    )


def _return_classes(
    values: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return np.where(
        values > threshold,
        "up",
        np.where(
            values < -threshold,
            "down",
            "abstain",
        ),
    ).astype(str)


def _threshold(
    config: ForecastV2BenchmarkConfig,
    horizon: ForecastHorizon,
) -> float:
    return (
        config.one_session_threshold if horizon == 1 else config.five_session_threshold
    )


def _spearman(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    if np.unique(actual).size < 2 or np.unique(prediction).size < 2:
        return math.nan

    result = pd.Series(actual).corr(
        pd.Series(prediction),
        method="spearman",
    )

    return float(result) if pd.notna(result) else math.nan


def _multiclass_auc(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if set(actual) != set(TARGET_CLASSES):
        return math.nan

    try:
        ordered_classes = tuple(sorted(TARGET_CLASSES))

        ordered_probabilities = np.column_stack(
            [
                probabilities[
                    :,
                    TARGET_CLASSES.index(class_name),
                ]
                for class_name in ordered_classes
            ]
        )

        return float(
            roc_auc_score(
                actual,
                ordered_probabilities,
                labels=list(ordered_classes),
                multi_class="ovr",
                average="macro",
            )
        )

    except ValueError:
        return math.nan


def _balanced_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    recalls = [
        float((predicted[actual == class_name] == class_name).mean())
        for class_name in TARGET_CLASSES
        if (actual == class_name).any()
    ]

    return float(np.mean(recalls)) if recalls else math.nan


def _prepare_csv_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    prepared = frame.copy()

    for column in (
        "date",
        "target_date",
    ):
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(prepared[column]).dt.strftime("%Y-%m-%d")

    return prepared


def _dataset_fingerprint(
    dataset: pd.DataFrame,
) -> str:
    columns = [
        column
        for column in (
            "date",
            "ticker",
            "target_date",
            "target",
        )
        if column in dataset.columns
    ]

    hashed = pd.util.hash_pandas_object(
        dataset[columns].reset_index(drop=True),
        index=True,
    ).to_numpy(dtype="uint64")

    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _as_float(
    value: object,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must contain numeric values.")

    if isinstance(
        value,
        int | float | np.integer | np.floating,
    ):
        return float(value)

    raise TypeError(f"{name} must contain numeric values.")


def _as_integer(
    value: object,
    name: str,
) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must contain integer values.")

    if isinstance(
        value,
        int | np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        float | np.floating,
    ):
        numeric = float(value)

        if numeric.is_integer():
            return int(numeric)

    raise TypeError(f"{name} must contain integer values.")


def _as_text(
    value: object,
    name: str,
) -> str:
    if isinstance(value, str):
        return value

    raise TypeError(f"{name} must contain text values.")


def _date_text(
    value: object,
) -> str:
    if isinstance(
        value,
        pd.Timestamp,
    ):
        timestamp = value

    elif isinstance(
        value,
        np.datetime64,
    ):
        timestamp = pd.Timestamp(value)

    else:
        raise TypeError("Date value must be a pandas Timestamp or NumPy datetime64.")

    return timestamp.strftime("%Y-%m-%d")


if __name__ == "__main__":
    raise SystemExit(main())

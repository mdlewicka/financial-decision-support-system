"""Nested chronological evaluation of the Forecasting V2 boosting challenger."""

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

from backend.src.config import MARKET_DATASETS_DIR, PROJECT_ROOT
from backend.src.forecast.multi_horizon_target_builder import (
    TARGET_CLASSES,
    ForecastTargetConfig,
    MultiHorizonForecastTargetBuilder,
)
from backend.src.models.gradient_boosted_forecast_model import (
    GradientBoostedDirectionConfig,
    GradientBoostedDirectionForecastModel,
)
from backend.src.storage.market_dataset_reader import MarketDatasetReader

DEFAULT_MODEL_BENCHMARK_DIRECTORY = (
    PROJECT_ROOT / "benchmark_results" / "forecast_v2" / "model_benchmark"
)
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "benchmark_results" / "forecast_v2" / "boosting_challenger"
)


@dataclass(frozen=True, slots=True)
class BoostingGridCandidate:
    learning_rate: float
    max_leaf_nodes: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and greater than zero.")
        if self.max_leaf_nodes < 2:
            raise ValueError("max_leaf_nodes must be at least 2.")

    @property
    def candidate_id(self) -> str:
        rate = str(self.learning_rate).replace(".", "p")
        return f"lr_{rate}_leaves_{self.max_leaf_nodes}"


DEFAULT_CANDIDATE_GRID: tuple[BoostingGridCandidate, ...] = (
    BoostingGridCandidate(learning_rate=0.03, max_leaf_nodes=7),
    BoostingGridCandidate(learning_rate=0.03, max_leaf_nodes=15),
    BoostingGridCandidate(learning_rate=0.06, max_leaf_nodes=7),
    BoostingGridCandidate(learning_rate=0.06, max_leaf_nodes=15),
)


@dataclass(frozen=True, slots=True)
class GradientBoostingChallengerConfig:
    inner_validation_fraction: float = 0.2
    max_iter: int = 150
    min_samples_leaf: int = 40
    l2_regularization: float = 2.0
    minimum_confidence: float = 0.45
    transaction_cost: float = 0.001
    bootstrap_iterations: int = 2000
    random_state: int = 42
    candidate_grid: tuple[BoostingGridCandidate, ...] = DEFAULT_CANDIDATE_GRID

    def __post_init__(self) -> None:
        if not 0.1 <= self.inner_validation_fraction <= 0.4:
            raise ValueError("inner_validation_fraction must be in [0.1, 0.4].")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than zero.")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than zero.")
        if not math.isfinite(self.l2_regularization) or self.l2_regularization < 0:
            raise ValueError("l2_regularization must be finite and non-negative.")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1].")
        if not math.isfinite(self.transaction_cost) or self.transaction_cost < 0:
            raise ValueError("transaction_cost must be finite and non-negative.")
        if self.bootstrap_iterations <= 0:
            raise ValueError("bootstrap_iterations must be greater than zero.")
        if not 1 <= len(self.candidate_grid) <= 8:
            raise ValueError("candidate_grid must contain between 1 and 8 candidates.")
        identifiers = [candidate.candidate_id for candidate in self.candidate_grid]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("candidate_grid identifiers must be unique.")


@dataclass(frozen=True, slots=True)
class GradientBoostingChallengerResult:
    observations: pd.DataFrame
    selection_scores: pd.DataFrame
    fold_metrics: pd.DataFrame
    stability_summary: pd.DataFrame
    bootstrap_comparison: pd.DataFrame
    ticker_comparison: pd.DataFrame
    decision: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GradientBoostingChallengerPaths:
    observations_csv: Path
    selection_scores_csv: Path
    fold_metrics_csv: Path
    stability_summary_csv: Path
    bootstrap_csv: Path
    ticker_comparison_csv: Path
    decision_json: Path
    metadata_json: Path


ModelFactory = Callable[
    [GradientBoostedDirectionConfig, ForecastTargetConfig],
    GradientBoostedDirectionForecastModel,
]


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune a small gradient boosting grid inside each training fold and "
            "compare the challenger with the frozen Logistic V2 observations."
        )
    )
    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=MARKET_DATASETS_DIR,
    )
    parser.add_argument(
        "--model-benchmark-directory",
        type=Path,
        default=DEFAULT_MODEL_BENCHMARK_DIRECTORY,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def run_gradient_boosting_challenger(
    training_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    logistic_observations: pd.DataFrame,
    benchmark_metadata: dict[str, Any],
    config: GradientBoostingChallengerConfig | None = None,
    model_factory: ModelFactory | None = None,
) -> GradientBoostingChallengerResult:
    selected_config = config or GradientBoostingChallengerConfig()
    _validate_benchmark_metadata(
        benchmark_metadata,
        training_dataset,
        validation_dataset,
    )
    target_config = ForecastTargetConfig(
        one_session_threshold=float(
            benchmark_metadata["config"]["one_session_threshold"]
        ),
        five_session_threshold=float(
            benchmark_metadata["config"]["five_session_threshold"]
        ),
    )
    target_builder = MultiHorizonForecastTargetBuilder(target_config)
    factory = model_factory or _default_model_factory

    training = _prepare_dataset(training_dataset, "training").assign(__split="training")
    validation = _prepare_dataset(validation_dataset, "validation").assign(
        __split="validation"
    )
    if training["date"].max() >= validation["date"].min():
        raise ValueError("Training dates must end before validation dates begin.")

    development = pd.concat([training, validation], ignore_index=True).sort_values(
        ["date", "ticker"], kind="stable", ignore_index=True
    )
    development_targets = target_builder.build(development)
    fold_definitions = benchmark_metadata.get("folds")
    if not isinstance(fold_definitions, list) or not fold_definitions:
        raise ValueError("Benchmark metadata must contain chronological folds.")

    selection_rows: list[dict[str, Any]] = []
    observation_frames: list[pd.DataFrame] = []
    for fold_definition in fold_definitions:
        fold = int(fold_definition["fold"])
        evaluation_start = pd.Timestamp(fold_definition["evaluation_date_min"])
        evaluation_end = pd.Timestamp(fold_definition["evaluation_date_max"])
        outer_training = development.loc[development["date"] < evaluation_start].drop(
            columns="__split"
        )
        outer_training = outer_training.reset_index(drop=True)
        evaluation_mask = development["__split"].eq("validation") & development[
            "date"
        ].between(evaluation_start, evaluation_end)
        evaluation = _evaluation_rows(
            development.loc[evaluation_mask].drop(columns="__split"),
            development_targets.loc[evaluation_mask],
        )

        selected_candidate, scores = _select_candidate(
            outer_training,
            target_builder,
            target_config,
            selected_config,
            factory,
        )
        for score in scores:
            selection_rows.append({"fold": fold, **score})

        final_model_config = _model_config(
            selected_candidate,
            selected_config,
        )
        final_model = factory(final_model_config, target_config)
        final_model.fit(outer_training)
        probabilities = final_model.predict_proba(evaluation)
        observation_frames.append(
            _build_observations(
                fold=fold,
                evaluation=evaluation,
                probabilities=probabilities,
                confidence=selected_config.minimum_confidence,
                transaction_cost=selected_config.transaction_cost,
                candidate_id=selected_candidate.candidate_id,
            )
        )

    observations = pd.concat(observation_frames, ignore_index=True).sort_values(
        ["fold", "target_date", "ticker"], kind="stable", ignore_index=True
    )
    logistic = _prepare_logistic_observations(logistic_observations, observations)
    combined = pd.concat([logistic, observations], ignore_index=True)
    fold_metrics = _build_group_metrics(combined, ["fold", "model"])
    stability_summary = _build_stability_summary(fold_metrics)
    bootstrap = _paired_bootstrap(
        logistic,
        observations,
        iterations=selected_config.bootstrap_iterations,
        random_state=selected_config.random_state,
    )
    ticker_metrics = _build_group_metrics(combined, ["ticker", "model"])
    ticker_comparison = _build_ticker_comparison(ticker_metrics)
    decision = _build_decision(
        fold_metrics,
        stability_summary,
        bootstrap,
        ticker_comparison,
    )
    return GradientBoostingChallengerResult(
        observations=observations,
        selection_scores=pd.DataFrame(selection_rows).sort_values(
            ["fold", "selection_rank", "candidate_id"],
            kind="stable",
            ignore_index=True,
        ),
        fold_metrics=fold_metrics,
        stability_summary=stability_summary,
        bootstrap_comparison=bootstrap,
        ticker_comparison=ticker_comparison,
        decision=decision,
    )


def export_gradient_boosting_challenger(
    result: GradientBoostingChallengerResult,
    output_directory: Path,
    config: GradientBoostingChallengerConfig,
    benchmark_metadata: dict[str, Any],
) -> GradientBoostingChallengerPaths:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = GradientBoostingChallengerPaths(
        observations_csv=output_directory / "challenger_observations.csv",
        selection_scores_csv=output_directory / "inner_selection_scores.csv",
        fold_metrics_csv=output_directory / "fold_metrics.csv",
        stability_summary_csv=output_directory / "stability_summary.csv",
        bootstrap_csv=output_directory / "bootstrap_comparison.csv",
        ticker_comparison_csv=output_directory / "ticker_comparison.csv",
        decision_json=output_directory / "challenger_decision.json",
        metadata_json=output_directory / "challenger_metadata.json",
    )
    _prepare_csv(result.observations).to_csv(paths.observations_csv, index=False)
    result.selection_scores.to_csv(paths.selection_scores_csv, index=False)
    result.fold_metrics.to_csv(paths.fold_metrics_csv, index=False)
    result.stability_summary.to_csv(paths.stability_summary_csv, index=False)
    result.bootstrap_comparison.to_csv(paths.bootstrap_csv, index=False)
    result.ticker_comparison.to_csv(paths.ticker_comparison_csv, index=False)
    paths.decision_json.write_text(
        json.dumps(result.decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "experiment": "forecast_v2_gradient_boosting_challenger",
        "development_splits_used": ["training", "validation"],
        "test_split_read": False,
        "known_v1_test_used_for_selection": False,
        "selection_protocol": (
            f"{len(config.candidate_grid)} frozen candidates are ranked on the "
            f"latest chronological {config.inner_validation_fraction:.0%} of "
            "each outer training fold. The outer fold is not used for "
            "hyperparameter selection."
        ),
        "config": {
            **asdict(config),
            "candidate_grid": [
                asdict(candidate) for candidate in config.candidate_grid
            ],
        },
        "source_benchmark": benchmark_metadata.get("benchmark_name"),
        "source_training_fingerprint": benchmark_metadata.get("training_fingerprint"),
        "source_validation_fingerprint": benchmark_metadata.get(
            "validation_fingerprint"
        ),
        "final_test_claim": False,
    }
    paths.metadata_json.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def run_from_directories(
    dataset_directory: Path,
    model_benchmark_directory: Path,
    output_directory: Path,
    config: GradientBoostingChallengerConfig,
) -> tuple[GradientBoostingChallengerResult, GradientBoostingChallengerPaths]:
    reader = MarketDatasetReader(input_directory=dataset_directory)
    training, validation = reader.read_training_and_validation()
    metadata = json.loads(
        (model_benchmark_directory / "benchmark_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    source_observations = pd.read_csv(
        model_benchmark_directory / "benchmark_observations.csv",
        parse_dates=["date", "target_date"],
    )
    logistic = source_observations.loc[
        (source_observations["task"] == "classification")
        & (source_observations["horizon"] == 5)
        & (source_observations["model"] == "logistic_v2")
    ].copy()
    result = run_gradient_boosting_challenger(
        training,
        validation,
        logistic,
        metadata,
        config,
    )
    paths = export_gradient_boosting_challenger(
        result,
        output_directory,
        config,
        metadata,
    )
    return result, paths


def main(arguments: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(arguments)
    config = GradientBoostingChallengerConfig(
        bootstrap_iterations=args.bootstrap_iterations,
        random_state=args.random_state,
    )
    result, paths = run_from_directories(
        args.dataset_directory,
        args.model_benchmark_directory,
        args.output_directory,
        config,
    )
    print("Forecasting V2 gradient boosting challenger")
    print("Grid candidates:", len(config.candidate_grid))
    print("Outer folds:", result.fold_metrics["fold"].nunique())
    print("Test split read: no")
    print("\nInner selection:")
    selected = result.selection_scores.loc[result.selection_scores["selected"]]
    print(selected.to_string(index=False))
    print("\nStability comparison:")
    print(result.stability_summary.to_string(index=False))
    print("\nDecision:")
    print(json.dumps(result.decision, indent=2, ensure_ascii=False))
    print("\nArtifacts:")
    for path in asdict(paths).values():
        print(path)
    return 0


def _select_candidate(
    outer_training: pd.DataFrame,
    target_builder: MultiHorizonForecastTargetBuilder,
    target_config: ForecastTargetConfig,
    config: GradientBoostingChallengerConfig,
    factory: ModelFactory,
) -> tuple[BoostingGridCandidate, list[dict[str, Any]]]:
    inner_training, inner_evaluation = _chronological_inner_split(
        outer_training,
        config.inner_validation_fraction,
    )
    outer_targets = target_builder.build(outer_training)
    evaluation_index = outer_training.index[
        outer_training["date"].isin(inner_evaluation["date"].unique())
    ]
    evaluation = _evaluation_rows(
        outer_training.loc[evaluation_index],
        outer_targets.loc[evaluation_index],
    )
    rows: list[dict[str, Any]] = []
    for candidate in config.candidate_grid:
        model = factory(_model_config(candidate, config), target_config)
        model.fit(inner_training)
        probabilities = model.predict_proba(evaluation)
        predicted = _predicted_classes(probabilities, config.minimum_confidence)
        actual = evaluation["actual_class"].to_numpy(dtype=str)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "learning_rate": candidate.learning_rate,
                "max_leaf_nodes": candidate.max_leaf_nodes,
                "inner_training_date_max": _date_text(inner_training["date"].max()),
                "inner_evaluation_date_min": _date_text(evaluation["date"].min()),
                "inner_evaluation_date_max": _date_text(evaluation["date"].max()),
                "inner_observations": len(evaluation),
                "balanced_accuracy": _balanced_accuracy(actual, predicted),
                "multiclass_brier": _multiclass_brier(actual, probabilities),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["balanced_accuracy"]),
            float(row["multiclass_brier"]),
            str(row["candidate_id"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank
        row["selected"] = rank == 1
    selected_id = str(ranked[0]["candidate_id"])
    selected = next(
        candidate
        for candidate in config.candidate_grid
        if candidate.candidate_id == selected_id
    )
    return selected, ranked


def _chronological_inner_split(
    outer_training: pd.DataFrame,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.asarray(sorted(pd.to_datetime(outer_training["date"]).unique()))
    if len(dates) < 100:
        raise ValueError("At least 100 dates are required for nested validation.")
    split_position = int(math.floor(len(dates) * (1.0 - validation_fraction)))
    split_position = min(max(split_position, 60), len(dates) - 20)
    validation_start = pd.Timestamp(dates[split_position])
    inner_training = outer_training.loc[
        outer_training["date"] < validation_start
    ].reset_index(drop=True)
    inner_evaluation = outer_training.loc[
        outer_training["date"] >= validation_start
    ].reset_index(drop=True)
    if inner_training["date"].max() >= inner_evaluation["date"].min():
        raise RuntimeError("Inner training and evaluation dates overlap.")
    return inner_training, inner_evaluation


def _model_config(
    candidate: BoostingGridCandidate,
    config: GradientBoostingChallengerConfig,
) -> GradientBoostedDirectionConfig:
    return GradientBoostedDirectionConfig(
        learning_rate=candidate.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=candidate.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        minimum_confidence=config.minimum_confidence,
        random_state=config.random_state,
    )


def _default_model_factory(
    model_config: GradientBoostedDirectionConfig,
    target_config: ForecastTargetConfig,
) -> GradientBoostedDirectionForecastModel:
    return GradientBoostedDirectionForecastModel(model_config, target_config)


def _evaluation_rows(
    dataset: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    ready = targets["target_ready_5"].astype(bool)
    evaluation = dataset.loc[ready].copy()
    evaluation["evaluation_target_date"] = targets.loc[ready, "target_date_5"]
    evaluation["actual_return"] = targets.loc[ready, "target_return_5"].astype(float)
    evaluation["actual_class"] = targets.loc[ready, "target_class_5"].astype(str)
    return evaluation


def _build_observations(
    fold: int,
    evaluation: pd.DataFrame,
    probabilities: pd.DataFrame,
    confidence: float,
    transaction_cost: float,
    candidate_id: str,
) -> pd.DataFrame:
    aligned = probabilities.reindex(evaluation.index)[list(TARGET_CLASSES)]
    predicted = _predicted_classes(aligned, confidence)
    actual_return = evaluation["actual_return"].to_numpy(dtype=float)
    active = predicted != "abstain"
    signal = np.where(predicted == "up", 1.0, -1.0)
    net_utility = np.where(active, signal * actual_return - transaction_cost, 0.0)
    result = pd.DataFrame(
        {
            "fold": fold,
            "horizon": 5,
            "task": "classification",
            "model": "hist_gradient_boosting_v2",
            "selected_candidate": candidate_id,
            "ticker": evaluation["ticker"].astype(str).to_numpy(),
            "date": evaluation["date"].to_numpy(),
            "target_date": evaluation["evaluation_target_date"].to_numpy(),
            "actual_return": actual_return,
            "actual_class": evaluation["actual_class"].astype(str).to_numpy(),
            "predicted_class": predicted,
            "active": active,
            "net_utility": net_utility,
        }
    )
    for name in TARGET_CLASSES:
        result[f"probability_{name}"] = aligned[name].to_numpy(dtype=float)
    return result


def _prepare_logistic_observations(
    logistic: pd.DataFrame,
    challenger: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "fold",
        "ticker",
        "date",
        "target_date",
        "actual_return",
        "actual_class",
        "predicted_class",
        "active",
        "net_utility",
        *(f"probability_{name}" for name in TARGET_CLASSES),
    }
    missing = sorted(required.difference(logistic.columns))
    if missing:
        raise ValueError(
            "Logistic observations are missing columns: " + ", ".join(missing)
        )
    prepared = logistic.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared["target_date"] = pd.to_datetime(prepared["target_date"])
    keys = ["fold", "ticker", "date", "target_date"]
    alignment = prepared[keys + ["actual_return", "actual_class"]].merge(
        challenger[keys + ["actual_return", "actual_class"]],
        on=keys,
        suffixes=("_logistic", "_challenger"),
        validate="one_to_one",
    )
    if len(alignment) != len(challenger) or len(prepared) != len(challenger):
        raise ValueError("Logistic and challenger observations are not aligned.")
    if not np.allclose(
        alignment["actual_return_logistic"], alignment["actual_return_challenger"]
    ):
        raise ValueError("Logistic and challenger actual returns differ.")
    if not (
        alignment["actual_class_logistic"] == alignment["actual_class_challenger"]
    ).all():
        raise ValueError("Logistic and challenger actual classes differ.")
    prepared["model"] = "logistic_v2"
    prepared["selected_candidate"] = "frozen_v2"
    return prepared[list(challenger.columns)]


def _build_group_metrics(
    observations: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in observations.groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        actual = group["actual_class"].to_numpy(dtype=str)
        predicted = group["predicted_class"].to_numpy(dtype=str)
        probabilities = group[
            [f"probability_{name}" for name in TARGET_CLASSES]
        ].rename(columns={f"probability_{name}": name for name in TARGET_CLASSES})
        row = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "observations": len(group),
                "balanced_accuracy": _balanced_accuracy(actual, predicted),
                "multiclass_brier": _multiclass_brier(actual, probabilities),
                "coverage": float(group["active"].astype(bool).mean()),
                "mean_net_utility": float(group["net_utility"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [*group_columns], kind="stable", ignore_index=True
    )


def _build_stability_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in fold_metrics.groupby("model", sort=True):
        rows.append(
            {
                "model": model,
                "folds": len(group),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "median_balanced_accuracy": float(group["balanced_accuracy"].median()),
                "worst_balanced_accuracy": float(group["balanced_accuracy"].min()),
                "mean_multiclass_brier": float(group["multiclass_brier"].mean()),
                "mean_coverage": float(group["coverage"].mean()),
                "mean_net_utility": float(group["mean_net_utility"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("model", ignore_index=True)


def _paired_bootstrap(
    logistic: pd.DataFrame,
    challenger: pd.DataFrame,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    logistic_dates = _date_metric_components(logistic)
    challenger_dates = _date_metric_components(challenger)
    if not logistic_dates.index.equals(challenger_dates.index):
        raise ValueError("Paired bootstrap dates are not aligned.")
    generator = np.random.default_rng(random_state)
    logistic_values = logistic_dates.to_numpy(dtype=float)
    challenger_values = challenger_dates.to_numpy(dtype=float)
    ba_samples = np.empty(iterations, dtype=float)
    brier_samples = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        positions = generator.integers(0, len(logistic_values), len(logistic_values))
        logistic_sample = logistic_values[positions].sum(axis=0)
        challenger_sample = challenger_values[positions].sum(axis=0)
        ba_samples[iteration] = _balanced_accuracy_from_components(
            challenger_sample
        ) - _balanced_accuracy_from_components(logistic_sample)
        brier_samples[iteration] = (
            challenger_sample[-2] / challenger_sample[-1]
            - logistic_sample[-2] / logistic_sample[-1]
        )
    logistic_total = logistic_values.sum(axis=0)
    challenger_total = challenger_values.sum(axis=0)
    return pd.DataFrame(
        [
            {
                "metric": "balanced_accuracy",
                "challenger_minus_logistic": _balanced_accuracy_from_components(
                    challenger_total
                )
                - _balanced_accuracy_from_components(logistic_total),
                "ci_lower_95": float(np.quantile(ba_samples, 0.025)),
                "ci_upper_95": float(np.quantile(ba_samples, 0.975)),
                "probability_challenger_better": float((ba_samples > 0).mean()),
                "cluster_dates": len(logistic_values),
            },
            {
                "metric": "multiclass_brier",
                "challenger_minus_logistic": (
                    challenger_total[-2] / challenger_total[-1]
                    - logistic_total[-2] / logistic_total[-1]
                ),
                "ci_lower_95": float(np.quantile(brier_samples, 0.025)),
                "ci_upper_95": float(np.quantile(brier_samples, 0.975)),
                "probability_challenger_better": float((brier_samples < 0).mean()),
                "cluster_dates": len(logistic_values),
            },
        ]
    )


def _date_metric_components(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_date, group in observations.groupby("target_date", sort=True):
        actual = group["actual_class"].to_numpy(dtype=str)
        predicted = group["predicted_class"].to_numpy(dtype=str)
        probabilities = group[
            [f"probability_{name}" for name in TARGET_CLASSES]
        ].to_numpy(dtype=float)
        one_hot = np.column_stack(
            [(actual == name).astype(float) for name in TARGET_CLASSES]
        )
        row: dict[str, Any] = {"target_date": target_date}
        for name in TARGET_CLASSES:
            mask = actual == name
            row[f"actual_{name}"] = int(mask.sum())
            row[f"correct_{name}"] = int((mask & (predicted == name)).sum())
        row["brier_sum"] = float(np.sum((probabilities - one_hot) ** 2))
        row["observations"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows).set_index("target_date")


def _balanced_accuracy_from_components(values: np.ndarray) -> float:
    recalls = []
    for position in range(len(TARGET_CLASSES)):
        actual = float(values[2 * position])
        correct = float(values[2 * position + 1])
        if actual > 0:
            recalls.append(correct / actual)
    return float(np.mean(recalls))


def _build_ticker_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(index="ticker", columns="model", values="balanced_accuracy")
    result = pivot.reset_index()
    result["challenger_minus_logistic"] = (
        result["hist_gradient_boosting_v2"] - result["logistic_v2"]
    )
    result["challenger_wins"] = result["challenger_minus_logistic"] > 0
    return result.sort_values("ticker", ignore_index=True)


def _build_decision(
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    ticker_comparison: pd.DataFrame,
) -> dict[str, Any]:
    logistic = summary.loc[summary["model"] == "logistic_v2"].iloc[0]
    challenger = summary.loc[summary["model"] == "hist_gradient_boosting_v2"].iloc[0]
    fold_pivot = fold_metrics.pivot(
        index="fold", columns="model", values="balanced_accuracy"
    )
    fold_wins = int(
        (fold_pivot["hist_gradient_boosting_v2"] > fold_pivot["logistic_v2"]).sum()
    )
    ba_bootstrap = bootstrap.loc[bootstrap["metric"] == "balanced_accuracy"].iloc[0]
    ticker_wins = int(ticker_comparison["challenger_wins"].sum())
    gates = {
        "median_balanced_accuracy_improves": bool(
            challenger["median_balanced_accuracy"]
            > logistic["median_balanced_accuracy"]
        ),
        "wins_at_least_three_of_five_folds": fold_wins >= 3,
        "bootstrap_support_at_80_percent": bool(
            ba_bootstrap["probability_challenger_better"] >= 0.8
        ),
        "mean_brier_not_worse": bool(
            challenger["mean_multiclass_brier"] <= logistic["mean_multiclass_brier"]
        ),
        "wins_at_least_half_of_tickers": ticker_wins
        >= math.ceil(len(ticker_comparison) / 2),
    }
    predictive_gate = all(gates.values())
    return {
        "challenger": "hist_gradient_boosting_v2",
        "comparison_model": "logistic_v2",
        "fold_wins": fold_wins,
        "folds": int(fold_pivot.shape[0]),
        "ticker_wins": ticker_wins,
        "tickers": len(ticker_comparison),
        "balanced_accuracy_bootstrap_probability_better": float(
            ba_bootstrap["probability_challenger_better"]
        ),
        "gates": gates,
        "predictive_gate_passed": predictive_gate,
        "decision": (
            "promote_to_economic_validation"
            if predictive_gate
            else "do_not_promote_in_current_form"
        ),
        "test_split_read": False,
        "final_test_claim": False,
    }


def _predicted_classes(probabilities: pd.DataFrame, confidence: float) -> np.ndarray:
    predicted = probabilities[list(TARGET_CLASSES)].idxmax(axis=1).to_numpy(dtype=str)
    maximum = probabilities[list(TARGET_CLASSES)].max(axis=1).to_numpy(dtype=float)
    predicted[maximum < confidence] = "abstain"
    return predicted


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    recalls = [
        float((predicted[actual == name] == name).mean())
        for name in TARGET_CLASSES
        if (actual == name).any()
    ]
    return float(np.mean(recalls))


def _multiclass_brier(actual: np.ndarray, probabilities: pd.DataFrame) -> float:
    values = probabilities[list(TARGET_CLASSES)].to_numpy(dtype=float)
    one_hot = np.column_stack(
        [(actual == name).astype(float) for name in TARGET_CLASSES]
    )
    return float(np.mean(np.sum((values - one_hot) ** 2, axis=1)))


def _validate_benchmark_metadata(
    metadata: dict[str, Any],
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    if metadata.get("test_split_read") is not False:
        raise ValueError("Source benchmark must confirm test_split_read=false.")
    if metadata.get("known_v1_test_used_for_selection") is not False:
        raise ValueError(
            "Source benchmark must not use the known V1 test for selection."
        )
    if metadata.get("training_fingerprint") != _dataset_fingerprint(training):
        raise ValueError("Training dataset fingerprint differs from the benchmark.")
    if metadata.get("validation_fingerprint") != _dataset_fingerprint(validation):
        raise ValueError("Validation dataset fingerprint differs from the benchmark.")


def _prepare_dataset(dataset: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {
        "date",
        "ticker",
        "adjusted_close",
        "volume",
        "daily_return",
        "target_date",
        "target",
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"{name} dataset is missing columns: {', '.join(missing)}.")
    result = dataset.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["target_date"] = pd.to_datetime(result["target_date"], errors="raise")
    return result.sort_values(["date", "ticker"], kind="stable", ignore_index=True)


def _dataset_fingerprint(dataset: pd.DataFrame) -> str:
    columns = [
        column
        for column in ("date", "ticker", "target_date", "target")
        if column in dataset.columns
    ]
    hashed = pd.util.hash_pandas_object(
        dataset[columns].reset_index(drop=True), index=True
    ).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _prepare_csv(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in ("date", "target_date"):
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(prepared[column]).dt.strftime("%Y-%m-%d")
    return prepared


def _date_text(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        timestamp = value
    elif isinstance(value, np.datetime64):
        timestamp = pd.Timestamp(value)
    else:
        raise TypeError("Date value must be a pandas Timestamp or NumPy datetime64.")

    return timestamp.strftime("%Y-%m-%d")


if __name__ == "__main__":
    raise SystemExit(main())

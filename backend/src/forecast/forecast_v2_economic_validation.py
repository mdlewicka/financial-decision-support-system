from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.src.config import PROJECT_ROOT

DEFAULT_MODEL_BENCHMARK_DIR = (
    PROJECT_ROOT / "benchmark_results" / "forecast_v2" / "model_benchmark"
)
DEFAULT_ECONOMIC_VALIDATION_DIR = (
    PROJECT_ROOT / "benchmark_results" / "forecast_v2" / "economic_validation"
)
POLICY_NAME = "logistic_5d_long_cash"
BENCHMARK_NAME = "buy_and_hold_5d"


@dataclass(frozen=True, slots=True)
class EconomicValidationConfig:
    confidence_thresholds: tuple[float, ...] = (
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    )
    default_threshold: float = 0.45
    transaction_cost: float = 0.001
    periods_per_year: float = 252.0 / 5.0
    bootstrap_iterations: int = 2000
    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.confidence_thresholds:
            raise ValueError("confidence_thresholds must not be empty.")

        if tuple(sorted(set(self.confidence_thresholds))) != self.confidence_thresholds:
            raise ValueError("confidence_thresholds must be unique and sorted.")

        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.confidence_thresholds
        ):
            raise ValueError("confidence thresholds must be in [0, 1].")

        if self.default_threshold not in self.confidence_thresholds:
            raise ValueError("default_threshold must be in confidence_thresholds.")

        if (
            not math.isfinite(self.transaction_cost)
            or not 0.0 <= self.transaction_cost < 1.0
        ):
            raise ValueError("transaction_cost must be in [0, 1).")

        if not math.isfinite(self.periods_per_year) or self.periods_per_year <= 0.0:
            raise ValueError("periods_per_year must be finite and positive.")

        if self.bootstrap_iterations <= 0:
            raise ValueError("bootstrap_iterations must be greater than zero.")


@dataclass(frozen=True, slots=True)
class EconomicValidationResult:
    period_observations: pd.DataFrame
    path_metrics: pd.DataFrame
    threshold_summary: pd.DataFrame
    bootstrap_comparison: pd.DataFrame
    decision: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EconomicValidationPaths:
    period_observations_csv: Path
    path_metrics_csv: Path
    threshold_summary_csv: Path
    bootstrap_csv: Path
    decision_json: Path
    metadata_json: Path


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the five-session logistic signal as a long/cash policy "
            "using non-overlapping out-of-fold observations."
        )
    )
    parser.add_argument(
        "--benchmark-directory",
        type=Path,
        default=DEFAULT_MODEL_BENCHMARK_DIR,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_ECONOMIC_VALIDATION_DIR,
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    return parser


def run_economic_validation(
    benchmark_observations: pd.DataFrame,
    config: EconomicValidationConfig | None = None,
) -> EconomicValidationResult:
    selected_config = config or EconomicValidationConfig()
    observations = _prepare_observations(benchmark_observations)
    non_overlapping = _select_non_overlapping_periods(observations)

    period_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for threshold in selected_config.confidence_thresholds:
        for (
            fold,
            ticker,
        ), group in non_overlapping.groupby(
            ["fold", "ticker"],
            sort=True,
        ):
            fold_value = _as_integer(
                fold,
                "fold",
            )
            ticker_value = str(ticker)

            policy_actions = _policy_actions(
                group,
                threshold,
            )
            buy_hold_actions = np.ones(
                len(group),
                dtype=float,
            )

            for strategy, actions in (
                (
                    POLICY_NAME,
                    policy_actions,
                ),
                (
                    BENCHMARK_NAME,
                    buy_hold_actions,
                ),
            ):
                periods, metrics = _evaluate_path(
                    group=group,
                    actions=actions,
                    transaction_cost=(selected_config.transaction_cost),
                    periods_per_year=(selected_config.periods_per_year),
                )

                periods.insert(
                    0,
                    "threshold",
                    threshold,
                )
                periods.insert(
                    1,
                    "strategy",
                    strategy,
                )
                periods.insert(
                    2,
                    "fold",
                    fold_value,
                )
                periods.insert(
                    3,
                    "ticker",
                    ticker_value,
                )

                period_frames.append(periods)

                metric_rows.append(
                    {
                        "threshold": threshold,
                        "strategy": strategy,
                        "fold": fold_value,
                        "ticker": ticker_value,
                        **metrics,
                    }
                )

    period_observations = pd.concat(
        period_frames,
        ignore_index=True,
    ).sort_values(
        [
            "threshold",
            "strategy",
            "fold",
            "ticker",
            "date",
        ],
        kind="stable",
        ignore_index=True,
    )

    path_metrics = pd.DataFrame(metric_rows).sort_values(
        [
            "threshold",
            "strategy",
            "fold",
            "ticker",
        ],
        kind="stable",
        ignore_index=True,
    )

    threshold_summary = _build_threshold_summary(path_metrics)

    bootstrap_comparison = _bootstrap_comparison(
        path_metrics=path_metrics,
        iterations=selected_config.bootstrap_iterations,
        random_state=selected_config.random_state,
    )

    decision = _build_decision(
        threshold_summary=threshold_summary,
        bootstrap_comparison=bootstrap_comparison,
        config=selected_config,
    )

    return EconomicValidationResult(
        period_observations=period_observations,
        path_metrics=path_metrics,
        threshold_summary=threshold_summary,
        bootstrap_comparison=bootstrap_comparison,
        decision=decision,
    )


def export_economic_validation(
    result: EconomicValidationResult,
    output_directory: Path,
    benchmark_observations: pd.DataFrame,
    benchmark_metadata: dict[str, Any],
    config: EconomicValidationConfig,
) -> EconomicValidationPaths:
    _validate_benchmark_metadata(benchmark_metadata)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = EconomicValidationPaths(
        period_observations_csv=(output_directory / "policy_period_observations.csv"),
        path_metrics_csv=(output_directory / "policy_path_metrics.csv"),
        threshold_summary_csv=(output_directory / "threshold_summary.csv"),
        bootstrap_csv=(output_directory / "bootstrap_comparison.csv"),
        decision_json=(output_directory / "economic_decision.json"),
        metadata_json=(output_directory / "economic_validation_metadata.json"),
    )

    for frame, path in (
        (
            result.period_observations,
            paths.period_observations_csv,
        ),
        (
            result.path_metrics,
            paths.path_metrics_csv,
        ),
        (
            result.threshold_summary,
            paths.threshold_summary_csv,
        ),
        (
            result.bootstrap_comparison,
            paths.bootstrap_csv,
        ),
    ):
        _prepare_csv_frame(frame).to_csv(
            path,
            index=False,
        )

    paths.decision_json.write_text(
        json.dumps(
            result.decision,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "schema_version": 1,
        "validation_name": "forecast_v2_economic_validation",
        "development_splits_used": [
            "training",
            "validation",
        ],
        "test_split_read": False,
        "known_v1_test_used_for_selection": False,
        "source_benchmark": benchmark_metadata.get("benchmark_name"),
        "source_observations_fingerprint": (_frame_fingerprint(benchmark_observations)),
        "config": asdict(config),
        "policy": {
            "name": POLICY_NAME,
            "positions": [
                "cash",
                "long",
            ],
            "short_selling": False,
            "decision_rule": (
                "Long only when P(up) is the largest class probability "
                "and is at least the fixed confidence threshold; "
                "otherwise cash."
            ),
            "periods_overlap": False,
            "entry_exit_turnover_costed": True,
        },
        "limitation": (
            "This is an out-of-fold development screening analysis, "
            "not the one-time V2 final test and not a claim of "
            "live profitability."
        ),
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


def run_from_directory(
    benchmark_directory: Path = DEFAULT_MODEL_BENCHMARK_DIR,
    output_directory: Path = DEFAULT_ECONOMIC_VALIDATION_DIR,
    config: EconomicValidationConfig | None = None,
) -> tuple[
    EconomicValidationResult,
    EconomicValidationPaths,
]:
    selected_config = config or EconomicValidationConfig()

    observations_path = benchmark_directory / "benchmark_observations.csv"
    metadata_path = benchmark_directory / "benchmark_metadata.json"

    observations = pd.read_csv(
        observations_path,
        parse_dates=[
            "date",
            "target_date",
        ],
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    _validate_benchmark_metadata(metadata)

    result = run_economic_validation(
        observations,
        selected_config,
    )

    paths = export_economic_validation(
        result=result,
        output_directory=output_directory,
        benchmark_observations=observations,
        benchmark_metadata=metadata,
        config=selected_config,
    )

    return result, paths


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    parsed = create_argument_parser().parse_args(arguments)

    config = EconomicValidationConfig(
        transaction_cost=parsed.transaction_cost,
        bootstrap_iterations=parsed.bootstrap_iterations,
        random_state=parsed.random_state,
    )

    result, paths = run_from_directory(
        benchmark_directory=parsed.benchmark_directory,
        output_directory=parsed.output_directory,
        config=config,
    )

    print("Forecasting V2 economic validation")
    print("Policy: five-session logistic long/cash")
    print("Periods overlap: no")
    print("Short selling: no")
    print("V1 test split read: no")
    print("\nThreshold sensitivity:")
    print(result.threshold_summary.to_string(index=False))
    print("\nDecision:")
    print(
        json.dumps(
            result.decision,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nPeriod observations: {paths.period_observations_csv}")
    print(f"Path metrics: {paths.path_metrics_csv}")
    print(f"Threshold summary: {paths.threshold_summary_csv}")
    print(f"Bootstrap: {paths.bootstrap_csv}")
    print(f"Decision: {paths.decision_json}")
    print(f"Metadata: {paths.metadata_json}")

    return 0


def _prepare_observations(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "fold",
        "horizon",
        "task",
        "model",
        "ticker",
        "date",
        "target_date",
        "actual_return",
        "probability_down",
        "probability_abstain",
        "probability_up",
    }

    missing = sorted(required.difference(observations.columns))

    if missing:
        raise ValueError(
            "Benchmark observations are missing columns: " + ", ".join(missing) + "."
        )

    selected = observations.loc[
        observations["task"].eq("classification")
        & observations["model"].eq("logistic_v2")
        & observations["horizon"].eq(5)
    ].copy()

    if selected.empty:
        raise ValueError("No logistic_v2 five-session observations were found.")

    selected["date"] = pd.to_datetime(
        selected["date"],
        errors="raise",
    )
    selected["target_date"] = pd.to_datetime(
        selected["target_date"],
        errors="raise",
    )
    selected["ticker"] = selected["ticker"].astype("string")

    numeric_columns = [
        "actual_return",
        "probability_down",
        "probability_abstain",
        "probability_up",
    ]

    for column in numeric_columns:
        selected[column] = pd.to_numeric(
            selected[column],
            errors="raise",
        ).astype(float)

    if not np.isfinite(selected[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Economic validation inputs must be finite.")

    probabilities = selected[
        [
            "probability_down",
            "probability_abstain",
            "probability_up",
        ]
    ]

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-9,
    ):
        raise ValueError("Class probabilities must sum to one.")

    if (selected["target_date"] <= selected["date"]).any():
        raise ValueError("Target dates must be later than decision dates.")

    return selected.sort_values(
        [
            "fold",
            "ticker",
            "date",
        ],
        kind="stable",
        ignore_index=True,
    )


def _select_non_overlapping_periods(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []

    for _, group in observations.groupby(
        [
            "fold",
            "ticker",
        ],
        sort=True,
    ):
        next_available: pd.Timestamp | None = None

        for _, row in group.sort_values(
            "date",
            kind="stable",
        ).iterrows():
            decision_date = _as_timestamp(
                row["date"],
                "date",
            )

            if next_available is not None and decision_date < next_available:
                continue

            selected_rows.append(row)

            next_available = _as_timestamp(
                row["target_date"],
                "target_date",
            )

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)

    if selected.empty:
        raise RuntimeError("No non-overlapping economic periods were selected.")

    for _, group in selected.groupby(
        [
            "fold",
            "ticker",
        ],
        sort=True,
    ):
        ordered = group.sort_values("date")
        previous_end = ordered["target_date"].shift(1)

        if (
            ordered["date"].iloc[1:].to_numpy() < previous_end.iloc[1:].to_numpy()
        ).any():
            raise RuntimeError("Economic validation periods overlap.")

    return selected


def _policy_actions(
    group: pd.DataFrame,
    threshold: float,
) -> np.ndarray:
    up = group["probability_up"].to_numpy(dtype=float)

    down = group["probability_down"].to_numpy(dtype=float)

    abstain = group["probability_abstain"].to_numpy(dtype=float)

    return ((up >= threshold) & (up >= down) & (up >= abstain)).astype(float)


def _evaluate_path(
    group: pd.DataFrame,
    actions: np.ndarray,
    transaction_cost: float,
    periods_per_year: float,
) -> tuple[
    pd.DataFrame,
    dict[str, int | float],
]:
    ordered = group.sort_values(
        "date",
        kind="stable",
    ).reset_index(drop=True)

    actual = ordered["actual_return"].to_numpy(dtype=float)

    previous = np.concatenate(
        (
            [0.0],
            actions[:-1],
        )
    )

    turnover = np.abs(actions - previous)

    factors = (1.0 + actions * actual) * (1.0 - transaction_cost * turnover)

    if actions[-1] > 0.0:
        factors[-1] *= 1.0 - transaction_cost * actions[-1]
        turnover[-1] += actions[-1]

    if (factors <= 0.0).any() or not np.isfinite(factors).all():
        raise ValueError("Policy produced invalid wealth factors.")

    wealth = np.cumprod(factors)
    net_returns = factors - 1.0

    peaks = np.maximum.accumulate(
        np.concatenate(
            (
                [1.0],
                wealth,
            )
        )
    )

    drawdowns = (
        1.0
        - np.concatenate(
            (
                [1.0],
                wealth,
            )
        )
        / peaks
    )

    volatility = (
        float(
            np.std(
                net_returns,
                ddof=1,
            )
        )
        if len(net_returns) > 1
        else 0.0
    )

    mean_return = float(np.mean(net_returns))

    sharpe = (
        math.sqrt(periods_per_year) * mean_return / volatility
        if volatility > 0.0
        else math.nan
    )

    periods = ordered[
        [
            "date",
            "target_date",
            "actual_return",
        ]
    ].copy()

    periods["action"] = actions
    periods["turnover"] = turnover
    periods["net_period_return"] = net_returns
    periods["wealth"] = wealth

    return periods, {
        "periods": len(ordered),
        "active_periods": int(actions.sum()),
        "coverage": float(actions.mean()),
        "turnover": float(turnover.sum()),
        "cumulative_return": float(wealth[-1] - 1.0),
        "mean_period_return": mean_return,
        "annualized_volatility": (volatility * math.sqrt(periods_per_year)),
        "sharpe": float(sharpe),
        "maximum_drawdown": float(np.max(drawdowns)),
        "active_hit_rate": (
            float((actual[actions > 0.0] > 0.0).mean())
            if (actions > 0.0).any()
            else math.nan
        ),
    }


def _build_threshold_summary(
    path_metrics: pd.DataFrame,
) -> pd.DataFrame:
    policy = path_metrics.loc[path_metrics["strategy"] == POLICY_NAME]

    benchmark = path_metrics.loc[path_metrics["strategy"] == BENCHMARK_NAME]

    paired = policy.merge(
        benchmark[
            [
                "threshold",
                "fold",
                "ticker",
                "cumulative_return",
                "maximum_drawdown",
            ]
        ].rename(
            columns={
                "cumulative_return": ("benchmark_cumulative_return"),
                "maximum_drawdown": ("benchmark_maximum_drawdown"),
            }
        ),
        on=[
            "threshold",
            "fold",
            "ticker",
        ],
        validate="one_to_one",
    )

    paired["excess_return"] = (
        paired["cumulative_return"] - paired["benchmark_cumulative_return"]
    )

    paired["drawdown_reduction"] = (
        paired["benchmark_maximum_drawdown"] - paired["maximum_drawdown"]
    )

    rows: list[dict[str, Any]] = []

    for threshold, group in paired.groupby(
        "threshold",
        sort=True,
    ):
        threshold_value = _as_float(
            threshold,
            "threshold",
        )

        rows.append(
            {
                "threshold": threshold_value,
                "groups": len(group),
                "mean_coverage": float(group["coverage"].mean()),
                "mean_turnover": float(group["turnover"].mean()),
                "mean_policy_return": float(group["cumulative_return"].mean()),
                "median_policy_return": float(group["cumulative_return"].median()),
                "mean_benchmark_return": float(
                    group["benchmark_cumulative_return"].mean()
                ),
                "mean_excess_return": float(group["excess_return"].mean()),
                "median_excess_return": float(group["excess_return"].median()),
                "group_wins_vs_benchmark": int((group["excess_return"] > 0.0).sum()),
                "mean_sharpe": float(group["sharpe"].mean()),
                "mean_drawdown_reduction": float(group["drawdown_reduction"].mean()),
            }
        )

    return pd.DataFrame(rows)


def _bootstrap_comparison(
    path_metrics: pd.DataFrame,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    policy = path_metrics.loc[path_metrics["strategy"] == POLICY_NAME]

    benchmark = path_metrics.loc[path_metrics["strategy"] == BENCHMARK_NAME]

    paired = policy.merge(
        benchmark[
            [
                "threshold",
                "fold",
                "ticker",
                "cumulative_return",
            ]
        ].rename(columns={"cumulative_return": ("benchmark_return")}),
        on=[
            "threshold",
            "fold",
            "ticker",
        ],
        validate="one_to_one",
    )

    paired["excess_return"] = paired["cumulative_return"] - paired["benchmark_return"]

    generator = np.random.default_rng(random_state)

    rows: list[dict[str, Any]] = []

    for threshold, group in paired.groupby(
        "threshold",
        sort=True,
    ):
        threshold_value = _as_float(
            threshold,
            "threshold",
        )

        values = group["excess_return"].to_numpy(dtype=float)

        samples = np.asarray(
            [
                values[
                    generator.integers(
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
                "threshold": threshold_value,
                "mean_excess_return": float(values.mean()),
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
                "probability_better_than_buy_hold": float((samples > 0.0).mean()),
                "groups": len(values),
            }
        )

    return pd.DataFrame(rows)


def _build_decision(
    threshold_summary: pd.DataFrame,
    bootstrap_comparison: pd.DataFrame,
    config: EconomicValidationConfig,
) -> dict[str, Any]:
    default = threshold_summary.loc[
        threshold_summary["threshold"] == config.default_threshold
    ].iloc[0]

    default_bootstrap = bootstrap_comparison.loc[
        bootstrap_comparison["threshold"] == config.default_threshold
    ].iloc[0]

    robust_thresholds = int((threshold_summary["mean_excess_return"] > 0.0).sum())

    group_wins = _as_integer(
        default["group_wins_vs_benchmark"],
        "group_wins_vs_benchmark",
    )

    groups = _as_integer(
        default["groups"],
        "groups",
    )

    bootstrap_probability = _as_float(
        default_bootstrap["probability_better_than_buy_hold"],
        "probability_better_than_buy_hold",
    )

    mean_excess_return = _as_float(
        default["mean_excess_return"],
        "mean_excess_return",
    )

    majority_group_wins = group_wins > groups / 2

    bootstrap_support = bootstrap_probability >= 0.8

    positive_default_excess = mean_excess_return > 0.0

    robust_sensitivity = robust_thresholds >= math.ceil(
        len(config.confidence_thresholds) / 2
    )

    promote = (
        positive_default_excess
        and majority_group_wins
        and bootstrap_support
        and robust_sensitivity
    )

    return {
        "candidate": POLICY_NAME,
        "default_threshold": (config.default_threshold),
        "positive_default_excess": (positive_default_excess),
        "majority_group_wins": (majority_group_wins),
        "bootstrap_probability_better": (bootstrap_probability),
        "bootstrap_support_at_80_percent": (bootstrap_support),
        "thresholds_with_positive_mean_excess": (robust_thresholds),
        "thresholds_evaluated": len(config.confidence_thresholds),
        "robust_across_thresholds": (robust_sensitivity),
        "decision": (
            "promote_as_allocation_signal"
            if promote
            else "retain_as_predictive_decision_support_only"
        ),
        "final_test_claim": False,
        "reason": (
            "The five-session classifier has predictive class signal, "
            "but its fixed long/cash policy must beat buy-and-hold "
            "robustly after costs before it can be treated as an "
            "allocation signal."
        ),
    }


def _validate_benchmark_metadata(
    metadata: dict[str, Any],
) -> None:
    if metadata.get("test_split_read") is not False:
        raise ValueError("Source benchmark must not read the test split.")

    if metadata.get("known_v1_test_used_for_selection") is not False:
        raise ValueError("Known V1 test must not be used for selection.")


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


def _frame_fingerprint(
    frame: pd.DataFrame,
) -> str:
    hashed = pd.util.hash_pandas_object(
        frame.reset_index(drop=True),
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


def _as_timestamp(
    value: object,
    name: str,
) -> pd.Timestamp:
    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value

    if isinstance(
        value,
        np.datetime64,
    ):
        return pd.Timestamp(value)

    raise TypeError(f"{name} must contain pandas or NumPy datetime values.")


if __name__ == "__main__":
    raise SystemExit(main())

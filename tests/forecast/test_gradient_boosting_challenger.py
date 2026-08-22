import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.forecast.gradient_boosting_challenger import (
    DEFAULT_CANDIDATE_GRID,
    BoostingGridCandidate,
    GradientBoostingChallengerConfig,
    GradientBoostingChallengerResult,
    _chronological_inner_split,
    _paired_bootstrap,
    _prepare_logistic_observations,
    _select_candidate,
    export_gradient_boosting_challenger,
)
from backend.src.forecast.multi_horizon_target_builder import ForecastTargetConfig


class FakeTargetBuilder:
    def build(self, dataset: pd.DataFrame) -> pd.DataFrame:
        class_to_return = {"down": -0.02, "abstain": 0.0, "up": 0.02}
        return pd.DataFrame(
            {
                "target_ready_5": True,
                "target_date_5": dataset["date"] + pd.Timedelta(days=5),
                "target_return_5": dataset["actual_class"].map(class_to_return),
                "target_class_5": dataset["actual_class"],
            },
            index=dataset.index,
        )


class FakeModel:
    def __init__(
        self,
        model_config,
        calls: list[tuple[pd.Timestamp, pd.Timestamp]],
    ) -> None:
        self.config = model_config
        self.calls = calls
        self.training_date_max: pd.Timestamp | None = None

    def fit(self, dataset: pd.DataFrame) -> None:
        self.training_date_max = pd.Timestamp(dataset["date"].max())

    def predict_proba(self, dataset: pd.DataFrame) -> pd.DataFrame:
        assert self.training_date_max is not None
        self.calls.append((self.training_date_max, pd.Timestamp(dataset["date"].min())))
        strong = self.config.learning_rate == 0.06 and self.config.max_leaf_nodes == 15
        rows: list[dict[str, float]] = []
        for actual in dataset["actual_class"].astype(str):
            if strong:
                probabilities = {"down": 0.1, "abstain": 0.1, "up": 0.1}
                probabilities[actual] = 0.8
            else:
                probabilities = {"down": 0.34, "abstain": 0.33, "up": 0.33}
            rows.append(probabilities)
        return pd.DataFrame(rows, index=dataset.index)


def create_outer_training(periods: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=periods, freq="D")
    classes = np.resize(np.asarray(["down", "abstain", "up"]), periods)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "SPY",
            "adjusted_close": 100.0,
            "volume": 1_000_000.0,
            "daily_return": 0.0,
            "actual_class": classes,
        }
    )


def create_observations(model: str, advantage: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(1, 6):
        for index, actual in enumerate(("down", "abstain", "up") * 4):
            predicted = actual if advantage or index % 2 == 0 else "abstain"
            probabilities = {"down": 0.1, "abstain": 0.1, "up": 0.1}
            probabilities[predicted] = 0.8
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(
                days=(fold - 1) * 20 + index
            )
            rows.append(
                {
                    "fold": fold,
                    "horizon": 5,
                    "task": "classification",
                    "model": model,
                    "selected_candidate": "frozen_v2",
                    "ticker": "SPY",
                    "date": date,
                    "target_date": date + pd.Timedelta(days=5),
                    "actual_return": 0.01 if actual == "up" else -0.01,
                    "actual_class": actual,
                    "predicted_class": predicted,
                    "active": predicted != "abstain",
                    "net_utility": 0.0,
                    **{
                        f"probability_{name}": value
                        for name, value in probabilities.items()
                    },
                }
            )
    return pd.DataFrame(rows)


def test_small_grid_is_frozen_and_unique() -> None:
    assert len(DEFAULT_CANDIDATE_GRID) == 4
    assert len({candidate.candidate_id for candidate in DEFAULT_CANDIDATE_GRID}) == 4
    assert {candidate.learning_rate for candidate in DEFAULT_CANDIDATE_GRID} == {
        0.03,
        0.06,
    }
    assert {candidate.max_leaf_nodes for candidate in DEFAULT_CANDIDATE_GRID} == {
        7,
        15,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"learning_rate": 0.0, "max_leaf_nodes": 7}, "learning_rate"),
        ({"learning_rate": 0.03, "max_leaf_nodes": 1}, "max_leaf_nodes"),
    ],
)
def test_invalid_grid_candidate_is_rejected(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BoostingGridCandidate(**kwargs)  # type: ignore[arg-type]


def test_inner_split_is_strictly_chronological() -> None:
    training, evaluation = _chronological_inner_split(
        create_outer_training(),
        validation_fraction=0.2,
    )

    assert len(training["date"].unique()) == 96
    assert len(evaluation["date"].unique()) == 24
    assert training["date"].max() < evaluation["date"].min()


def test_grid_selection_uses_only_past_data_and_picks_best_candidate() -> None:
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def factory(model_config, _target_config):
        return FakeModel(model_config, calls)

    selected, rows = _select_candidate(
        outer_training=create_outer_training(),
        target_builder=FakeTargetBuilder(),  # type: ignore[arg-type]
        target_config=ForecastTargetConfig(),
        config=GradientBoostingChallengerConfig(bootstrap_iterations=20),
        factory=factory,  # type: ignore[arg-type]
    )

    assert selected == BoostingGridCandidate(0.06, 15)
    assert len(rows) == 4
    assert rows[0]["selected"] is True
    assert rows[0]["selection_rank"] == 1
    assert all(training_max < evaluation_min for training_max, evaluation_min in calls)


def test_logistic_alignment_adds_frozen_candidate_without_changing_rows() -> None:
    challenger = create_observations("hist_gradient_boosting_v2", advantage=True)
    logistic = create_observations("logistic_v2", advantage=False).drop(
        columns="selected_candidate"
    )

    prepared = _prepare_logistic_observations(logistic, challenger)

    assert list(prepared.columns) == list(challenger.columns)
    assert prepared["selected_candidate"].eq("frozen_v2").all()
    assert prepared["model"].eq("logistic_v2").all()


def test_paired_bootstrap_is_reproducible_and_supports_better_challenger() -> None:
    logistic = create_observations("logistic_v2", advantage=False)
    challenger = create_observations("hist_gradient_boosting_v2", advantage=True)

    first = _paired_bootstrap(logistic, challenger, 100, 42)
    second = _paired_bootstrap(logistic, challenger, 100, 42)

    pdt.assert_frame_equal(first, second)
    balanced = first.loc[first["metric"] == "balanced_accuracy"].iloc[0]
    assert balanced["challenger_minus_logistic"] > 0.0
    assert balanced["probability_challenger_better"] == 1.0


def test_export_records_nested_selection_and_closed_test_split(
    tmp_path: Path,
) -> None:
    observations = create_observations("hist_gradient_boosting_v2", advantage=True)
    result = GradientBoostingChallengerResult(
        observations=observations,
        selection_scores=pd.DataFrame(
            [{"fold": 1, "candidate_id": "lr_0p06_leaves_15"}]
        ),
        fold_metrics=pd.DataFrame([{"fold": 1, "model": "logistic_v2"}]),
        stability_summary=pd.DataFrame([{"model": "logistic_v2"}]),
        bootstrap_comparison=pd.DataFrame([{"metric": "balanced_accuracy"}]),
        ticker_comparison=pd.DataFrame([{"ticker": "SPY"}]),
        decision={"predictive_gate_passed": True},
    )
    config = GradientBoostingChallengerConfig(bootstrap_iterations=20)
    paths = export_gradient_boosting_challenger(
        result,
        tmp_path,
        config,
        {
            "benchmark_name": "forecast_v2_model_benchmark",
            "training_fingerprint": "training",
            "validation_fingerprint": "validation",
        },
    )

    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert all(Path(path).is_file() for path in asdict(paths).values())
    assert metadata["test_split_read"] is False
    assert metadata["known_v1_test_used_for_selection"] is False
    assert "outer fold is not used" in metadata["selection_protocol"]
    assert len(metadata["config"]["candidate_grid"]) == 4


def test_invalid_challenger_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="inner_validation_fraction"):
        GradientBoostingChallengerConfig(inner_validation_fraction=0.5)
    with pytest.raises(ValueError, match="candidate_grid"):
        GradientBoostingChallengerConfig(candidate_grid=())

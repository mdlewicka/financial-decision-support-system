import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.src.config import (
    MARKET_DATASETS_DIR,
    PROJECT_ROOT,
)
from backend.src.forecast.ml_feature_transformer import (
    NUMERIC_FEATURE_COLUMNS,
)
from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)
from backend.src.storage.market_dataset_reader import (
    MarketDatasetReader,
)
from backend.src.xai.random_forest_feature_effects import (
    RandomForestFeatureEffectAnalyzer,
)
from backend.src.xai.random_forest_shap_explainer import (
    RandomForestShapExplainer,
)
from backend.src.xai.xai_exporter import (
    XAIExportPaths,
    export_xai_results,
)

DEFAULT_XAI_OUTPUT_DIR = PROJECT_ROOT / "benchmark_results" / "xai" / "random_forest"

XAI_MODEL_NAME = "random_forest"
GLOBAL_EXPLANATION_ROWS = 400
BACKGROUND_ROWS = 100
EFFECT_FEATURE_COUNT = 4
PDP_GRID_POINTS = 20
ALE_BINS = 10
RANDOM_STATE = 42


@dataclass(frozen=True, slots=True)
class RandomForestXAIWorkflowResult:
    """Store results of the reproducible XAI workflow."""

    export_paths: XAIExportPaths
    analyzed_features: tuple[str, ...]
    global_observations: int
    local_ticker: str
    local_date: pd.Timestamp
    local_target_date: pd.Timestamp
    local_actual: float
    local_prediction: float


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the compact Random Forest model on the "
            "training split and generate SHAP, PDP and ALE "
            "explanations for the validation split."
        )
    )
    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=MARKET_DATASETS_DIR,
        help=(
            "Directory containing train.csv, validation.csv "
            f"and test.csv. Default: {MARKET_DATASETS_DIR}."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_XAI_OUTPUT_DIR,
        help=(
            f"Directory for XAI tables and plots. Default: {DEFAULT_XAI_OUTPUT_DIR}."
        ),
    )

    return parser


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    return create_argument_parser().parse_args(
        arguments,
    )


def run_random_forest_xai(
    dataset_directory: Path = MARKET_DATASETS_DIR,
    output_directory: Path = DEFAULT_XAI_OUTPUT_DIR,
    model: RandomForestForecastModel | None = None,
    dataset_reader: MarketDatasetReader | None = None,
) -> RandomForestXAIWorkflowResult:
    """Run the complete Random Forest XAI workflow."""
    selected_reader = (
        dataset_reader
        if dataset_reader is not None
        else MarketDatasetReader(
            input_directory=dataset_directory,
        )
    )

    (
        training_dataset,
        validation_dataset,
        _,
    ) = selected_reader.read()

    training_dataset = training_dataset.reset_index(
        drop=True,
    )
    validation_dataset = validation_dataset.reset_index(
        drop=True,
    )

    selected_model = model if model is not None else RandomForestForecastModel()
    selected_model.fit(
        training_dataset,
    )

    validation_predictions = selected_model.predict(
        validation_dataset,
    )

    local_observation, local_position = _select_local_observation(
        validation_dataset=(validation_dataset),
        predictions=validation_predictions,
    )
    global_observations = _select_global_observations(
        validation_dataset=(validation_dataset),
        maximum_rows=(GLOBAL_EXPLANATION_ROWS),
        random_state=RANDOM_STATE,
    )

    shap_explainer = RandomForestShapExplainer(
        model=selected_model,
        background_dataset=(training_dataset),
        max_background_rows=(BACKGROUND_ROWS),
        random_state=RANDOM_STATE,
    )

    local_explanation = shap_explainer.explain_local(
        local_observation,
    )
    global_explanation = shap_explainer.explain_global(
        global_observations,
    )

    analyzed_features = _select_effect_features(
        model=selected_model,
        dataset=global_observations,
        mean_absolute_contributions=(global_explanation.mean_absolute_contributions),
        feature_count=(EFFECT_FEATURE_COUNT),
    )

    effect_analyzer = RandomForestFeatureEffectAnalyzer(
        selected_model,
    )

    partial_dependence_results = {
        feature_name: (
            effect_analyzer.partial_dependence(
                dataset=validation_dataset,
                feature_name=feature_name,
                grid_points=PDP_GRID_POINTS,
            )
        )
        for feature_name in analyzed_features
    }
    accumulated_local_effects_results = {
        feature_name: (
            effect_analyzer.accumulated_local_effects(
                dataset=validation_dataset,
                feature_name=feature_name,
                bins=ALE_BINS,
            )
        )
        for feature_name in analyzed_features
    }

    local_row = validation_dataset.iloc[local_position]
    local_actual = float(
        pd.to_numeric(
            local_row["target"],
            errors="raise",
        )
    )
    local_prediction = float(validation_predictions.iloc[local_position])
    local_date = pd.Timestamp(
        local_row["date"],
    )
    local_target_date = pd.Timestamp(
        local_row["target_date"],
    )
    local_ticker = str(
        local_row["ticker"],
    )

    export_paths = export_xai_results(
        local_explanation=(local_explanation),
        global_explanation=(global_explanation),
        partial_dependence_results=(partial_dependence_results),
        accumulated_local_effects_results=(accumulated_local_effects_results),
        output_directory=output_directory,
        metadata={
            "model_name": XAI_MODEL_NAME,
            "training_split": "training",
            "explanation_split": "validation",
            "target_definition": ("next_session_simple_return"),
            "background_rows": len(shap_explainer.background_feature_values),
            "global_explanation_rows": len(
                global_observations,
            ),
            "effect_feature_count": len(
                analyzed_features,
            ),
            "local_ticker": local_ticker,
            "local_date": local_date,
            "local_target_date": (local_target_date),
            "local_actual": local_actual,
            "local_prediction": (local_prediction),
        },
    )

    return RandomForestXAIWorkflowResult(
        export_paths=export_paths,
        analyzed_features=(analyzed_features),
        global_observations=len(
            global_observations,
        ),
        local_ticker=local_ticker,
        local_date=local_date,
        local_target_date=local_target_date,
        local_actual=local_actual,
        local_prediction=local_prediction,
    )


def _select_local_observation(
    validation_dataset: pd.DataFrame,
    predictions: pd.Series,
) -> tuple[pd.DataFrame, int]:
    if validation_dataset.empty:
        raise ValueError(
            "validation_dataset must not be empty.",
        )

    if len(predictions) != len(validation_dataset):
        raise ValueError(
            "predictions must match the validation dataset length.",
        )

    actual_values = np.asarray(
        pd.to_numeric(
            validation_dataset["target"],
            errors="raise",
        ),
        dtype=float,
    )
    prediction_values = np.asarray(
        predictions,
        dtype=float,
    )

    if not np.isfinite(actual_values).all():
        raise ValueError(
            "Validation targets must be finite.",
        )

    if not np.isfinite(prediction_values).all():
        raise ValueError(
            "Validation predictions must be finite.",
        )

    ranking = pd.DataFrame(
        {
            "row_position": np.arange(
                len(validation_dataset),
                dtype=int,
            ),
            "ticker": (validation_dataset["ticker"].astype(str).to_numpy()),
            "date": pd.to_datetime(
                validation_dataset["date"],
                errors="raise",
            ).to_numpy(),
            "absolute_error": np.abs(actual_values - prediction_values),
        }
    )

    ranking = ranking.sort_values(
        by=[
            "absolute_error",
            "date",
            "ticker",
            "row_position",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
        kind="stable",
    )

    selected_position = int(ranking.iloc[0]["row_position"])

    return (
        validation_dataset.iloc[[selected_position]].copy(
            deep=True,
        ),
        selected_position,
    )


def _select_global_observations(
    validation_dataset: pd.DataFrame,
    maximum_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if validation_dataset.empty:
        raise ValueError(
            "validation_dataset must not be empty.",
        )

    if maximum_rows <= 0:
        raise ValueError(
            "maximum_rows must be greater than zero.",
        )

    ticker_names = sorted(validation_dataset["ticker"].astype(str).unique().tolist())

    if maximum_rows < len(ticker_names):
        raise ValueError(
            "maximum_rows must allow at least one observation per ticker.",
        )

    if len(validation_dataset) <= maximum_rows:
        return validation_dataset.sort_values(
            by=[
                "date",
                "ticker",
            ],
            kind="stable",
        ).copy(
            deep=True,
        )

    base_rows = maximum_rows // len(ticker_names)
    remaining_rows = maximum_rows % len(ticker_names)

    sampled_groups: list[pd.DataFrame] = []

    ticker_values = validation_dataset["ticker"].astype(str)

    for ticker_index, ticker in enumerate(ticker_names):
        ticker_data = validation_dataset.loc[ticker_values == ticker]

        requested_rows = base_rows + int(ticker_index < remaining_rows)
        selected_rows = min(
            len(ticker_data),
            requested_rows,
        )

        sampled_groups.append(
            ticker_data.sample(
                n=selected_rows,
                random_state=(random_state + ticker_index),
                replace=False,
            )
        )

    return (
        pd.concat(
            sampled_groups,
            axis=0,
        )
        .sort_values(
            by=[
                "date",
                "ticker",
            ],
            kind="stable",
        )
        .copy(
            deep=True,
        )
    )


def _select_effect_features(
    model: RandomForestForecastModel,
    dataset: pd.DataFrame,
    mean_absolute_contributions: pd.Series,
    feature_count: int,
) -> tuple[str, ...]:
    if feature_count <= 0:
        raise ValueError(
            "feature_count must be greater than zero.",
        )

    transformed = model.transform_features(
        dataset,
    )

    numeric_feature_names = set(
        NUMERIC_FEATURE_COLUMNS,
    )

    eligible_features = [
        str(feature_name)
        for feature_name in (mean_absolute_contributions.index)
        if (
            str(feature_name) in numeric_feature_names
            and transformed[str(feature_name)].nunique(
                dropna=False,
            )
            >= 3
        )
    ]

    if len(eligible_features) < feature_count:
        raise RuntimeError(
            "Not enough continuous model features are available for PDP and ALE.",
        )

    return tuple(eligible_features[:feature_count])


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    parsed_arguments = parse_arguments(
        arguments,
    )

    result = run_random_forest_xai(
        dataset_directory=(parsed_arguments.dataset_directory),
        output_directory=(parsed_arguments.output_directory),
    )

    print(f"Model: {XAI_MODEL_NAME}")
    print("Training split: training")
    print("Explanation split: validation")
    print(f"Global SHAP observations: {result.global_observations}")
    print(
        "Local explanation: "
        f"{result.local_ticker}, "
        f"{result.local_date.date()} -> "
        f"{result.local_target_date.date()}"
    )
    print(f"Local actual return: {result.local_actual:.8f}")
    print(f"Local predicted return: {result.local_prediction:.8f}")
    print("PDP/ALE features: " + ", ".join(result.analyzed_features))
    print(f"XAI JSON result: {result.export_paths.result_json}")
    print(f"Global SHAP importance: {result.export_paths.global_shap_importance_csv}")
    print(f"Partial dependence: {result.export_paths.partial_dependence_csv}")
    print(
        "Accumulated local effects: "
        f"{result.export_paths.accumulated_local_effects_csv}"
    )
    print(f"Generated plots: {len(result.export_paths.plot_files)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
from pathlib import Path

import pandas as pd
import pytest

from backend.src.xai.xai_exporter import (
    XAIExportPaths,
    export_xai_results,
)
from backend.src.xai.xai_result import (
    AccumulatedLocalEffectsResult,
    GlobalShapExplanation,
    LocalShapExplanation,
    PartialDependenceResult,
)


def create_local_explanation() -> LocalShapExplanation:
    return LocalShapExplanation(
        feature_values=pd.Series(
            {
                "momentum_5": 0.02,
                "volatility_20": 0.015,
            },
            dtype=float,
        ),
        feature_contributions=pd.Series(
            {
                "momentum_5": 0.004,
                "volatility_20": -0.002,
            },
            dtype=float,
        ),
        base_value=0.001,
        prediction=0.003,
    )


def create_global_explanation() -> GlobalShapExplanation:
    feature_values = pd.DataFrame(
        {
            "momentum_5": [
                -0.01,
                0.00,
                0.02,
            ],
            "volatility_20": [
                0.010,
                0.015,
                0.020,
            ],
        },
        index=[
            10,
            20,
            30,
        ],
    )
    feature_contributions = pd.DataFrame(
        {
            "momentum_5": [
                -0.003,
                0.000,
                0.004,
            ],
            "volatility_20": [
                0.001,
                -0.001,
                -0.002,
            ],
        },
        index=feature_values.index,
    )
    base_values = pd.Series(
        [
            0.001,
            0.001,
            0.001,
        ],
        index=feature_values.index,
        dtype=float,
        name="base_value",
    )
    predictions = (
        base_values
        + feature_contributions.sum(
            axis=1,
        )
    ).rename(
        "prediction",
    )
    mean_absolute_contributions = (
        feature_contributions.abs()
        .mean(axis=0)
        .sort_values(
            ascending=False,
        )
        .rename(
            "mean_absolute_shap_value",
        )
    )

    return GlobalShapExplanation(
        feature_values=feature_values,
        feature_contributions=(feature_contributions),
        base_values=base_values,
        predictions=predictions,
        mean_absolute_contributions=(mean_absolute_contributions),
    )


def create_partial_dependence_result() -> PartialDependenceResult:
    return PartialDependenceResult(
        feature_name="momentum_5",
        feature_values=pd.Series(
            [
                -0.02,
                0.00,
                0.02,
            ],
            dtype=float,
            name="feature_value",
        ),
        average_predictions=pd.Series(
            [
                -0.001,
                0.001,
                0.003,
            ],
            dtype=float,
            name="average_prediction",
        ),
    )


def create_ale_result() -> AccumulatedLocalEffectsResult:
    return AccumulatedLocalEffectsResult(
        feature_name="momentum_5",
        bin_edges=pd.Series(
            [
                -0.02,
                0.00,
                0.02,
                0.04,
            ],
            dtype=float,
            name="bin_edge",
        ),
        bin_centers=pd.Series(
            [
                -0.01,
                0.01,
                0.03,
            ],
            dtype=float,
            name="bin_center",
        ),
        local_effects=pd.Series(
            [
                -0.002,
                0.001,
                0.003,
            ],
            dtype=float,
            name="local_effect",
        ),
        accumulated_effects=pd.Series(
            [
                -0.001,
                0.000,
                0.001,
            ],
            dtype=float,
            name="accumulated_effect",
        ),
        sample_counts=pd.Series(
            [
                10,
                10,
                10,
            ],
            dtype=int,
            name="sample_count",
        ),
    )


def test_export_xai_results_writes_tables_plots_and_json(
    tmp_path: Path,
) -> None:
    paths = export_xai_results(
        local_explanation=(create_local_explanation()),
        global_explanation=(create_global_explanation()),
        partial_dependence_results={"momentum_5": (create_partial_dependence_result())},
        accumulated_local_effects_results={"momentum_5": create_ale_result()},
        output_directory=tmp_path,
        metadata={
            "model_name": "random_forest",
            "explanation_split": "validation",
            "ticker": "SPY",
            "date": pd.Timestamp(
                "2024-01-05",
            ),
        },
    )

    assert isinstance(
        paths,
        XAIExportPaths,
    )

    table_paths = [
        paths.local_shap_csv,
        paths.global_shap_importance_csv,
        paths.global_shap_feature_values_csv,
        paths.global_shap_contributions_csv,
        paths.partial_dependence_csv,
        paths.accumulated_local_effects_csv,
        paths.result_json,
    ]

    assert all(path.is_file() and path.stat().st_size > 0 for path in table_paths)
    assert len(paths.plot_files) == 5
    assert all(
        path.is_file() and path.stat().st_size > 0 for path in paths.plot_files.values()
    )

    local_shap = pd.read_csv(
        paths.local_shap_csv,
    )
    assert local_shap.columns.tolist() == [
        "feature",
        "feature_value",
        "shap_value",
    ]
    assert local_shap["feature"].tolist() == [
        "momentum_5",
        "volatility_20",
    ]

    global_importance = pd.read_csv(
        paths.global_shap_importance_csv,
    )
    assert global_importance.columns.tolist() == [
        "feature",
        "mean_absolute_shap_value",
    ]

    partial_dependence = pd.read_csv(
        paths.partial_dependence_csv,
    )
    assert partial_dependence["feature"].tolist() == [
        "momentum_5",
        "momentum_5",
        "momentum_5",
    ]

    accumulated_local_effects = pd.read_csv(
        paths.accumulated_local_effects_csv,
    )
    assert accumulated_local_effects["sample_count"].sum() == 30

    payload = json.loads(
        paths.result_json.read_text(
            encoding="utf-8",
        )
    )

    assert payload["schema_version"] == 1
    assert payload["methods"] == [
        "shap",
        "partial_dependence",
        "accumulated_local_effects",
    ]
    assert payload["metadata"] == {
        "date": "2024-01-05T00:00:00",
        "explanation_split": "validation",
        "model_name": "random_forest",
        "ticker": "SPY",
    }
    assert payload["global_shap"]["observations"] == 3
    assert set(payload["plots"]) == set(
        paths.plot_files,
    )


def test_export_rejects_mismatched_feature_effects(
    tmp_path: Path,
) -> None:
    ale_result = create_ale_result()

    mismatched_ale_result = AccumulatedLocalEffectsResult(
        feature_name="volatility_20",
        bin_edges=ale_result.bin_edges,
        bin_centers=ale_result.bin_centers,
        local_effects=ale_result.local_effects,
        accumulated_effects=(ale_result.accumulated_effects),
        sample_counts=ale_result.sample_counts,
    )

    with pytest.raises(
        ValueError,
        match=("PDP and ALE results must contain the same feature names"),
    ):
        export_xai_results(
            local_explanation=(create_local_explanation()),
            global_explanation=(create_global_explanation()),
            partial_dependence_results={
                "momentum_5": (create_partial_dependence_result())
            },
            accumulated_local_effects_results={
                "volatility_20": (mismatched_ale_result)
            },
            output_directory=tmp_path,
        )


def test_export_rejects_non_positive_display_limit(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=("max_display_features must be greater than zero"),
    ):
        export_xai_results(
            local_explanation=(create_local_explanation()),
            global_explanation=(create_global_explanation()),
            partial_dependence_results={
                "momentum_5": (create_partial_dependence_result())
            },
            accumulated_local_effects_results={"momentum_5": create_ale_result()},
            output_directory=tmp_path,
            max_display_features=0,
        )

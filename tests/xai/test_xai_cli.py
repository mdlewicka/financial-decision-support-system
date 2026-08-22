import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from backend.src.models.random_forest_forecast_model import (
    RandomForestForecastModel,
)
from backend.src.xai.cli import (
    EFFECT_FEATURE_COUNT,
    RandomForestXAIWorkflowResult,
    _select_global_observations,
    parse_arguments,
    run_random_forest_xai,
)


def create_market_dataset(
    start_date: str,
    rows_per_ticker: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for ticker_index, ticker in enumerate(
        (
            "QQQ",
            "SPY",
        )
    ):
        dates = pd.date_range(
            start=start_date,
            periods=rows_per_ticker,
            freq="B",
        )
        base_price = 100.0 + 50.0 * ticker_index

        for position, feature_date in enumerate(dates):
            signal = np.sin(
                position / 3.0 + ticker_index,
            )
            adjusted_close = base_price + position * 0.12 + 1.25 * signal
            open_price = adjusted_close * (1.0 - 0.002 * np.cos(position / 4.0))
            close_price = adjusted_close * (1.0 + 0.001 * np.sin(position / 5.0))
            high_price = (
                max(
                    open_price,
                    close_price,
                )
                * 1.005
            )
            low_price = (
                min(
                    open_price,
                    close_price,
                )
                * 0.995
            )

            daily_return = 0.0025 * signal + 0.0003 * ticker_index
            volatility_5 = 0.010 + 0.001 * abs(signal)
            volatility_20 = 0.015 + 0.0015 * abs(signal)
            momentum_5 = 0.012 * np.sin(position / 6.0)
            momentum_20 = 0.020 * np.sin(position / 9.0)
            volume_change = 0.05 * np.cos(position / 5.0)
            target = (
                0.50 * daily_return
                + 0.10 * momentum_5
                - 0.06 * volatility_20
                + 0.0002 * ticker_index
            )

            rows.append(
                {
                    "date": feature_date,
                    "ticker": ticker,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "adjusted_close": (adjusted_close),
                    "volume": (1_000_000 + position * 10_000 + ticker_index * 100_000),
                    "daily_return": daily_return,
                    "log_return": float(
                        np.log1p(daily_return),
                    ),
                    "price_change": (close_price - open_price),
                    "volume_change": (volume_change),
                    "moving_average_5": (adjusted_close / (1.0 + 0.003 * signal)),
                    "moving_average_20": (adjusted_close / (1.0 + 0.006 * signal)),
                    "volatility_5": (volatility_5),
                    "volatility_20": (volatility_20),
                    "momentum_5": momentum_5,
                    "momentum_20": (momentum_20),
                    "target_date": (feature_date + pd.offsets.BDay(1)),
                    "target": target,
                }
            )

    return pd.DataFrame(rows)


def write_market_datasets(
    directory: Path,
) -> None:
    training_dataset = create_market_dataset(
        start_date="2023-01-02",
        rows_per_ticker=40,
    )
    validation_dataset = create_market_dataset(
        start_date="2024-01-02",
        rows_per_ticker=12,
    )
    test_dataset = create_market_dataset(
        start_date="2025-01-02",
        rows_per_ticker=12,
    )

    training_dataset.to_csv(
        directory / "train.csv",
        index=False,
    )
    validation_dataset.to_csv(
        directory / "validation.csv",
        index=False,
    )
    test_dataset.to_csv(
        directory / "test.csv",
        index=False,
    )


def test_parse_arguments_accepts_custom_directories(
    tmp_path: Path,
) -> None:
    dataset_directory = tmp_path / "datasets"
    output_directory = tmp_path / "xai"

    arguments = parse_arguments(
        [
            "--dataset-directory",
            str(dataset_directory),
            "--output-directory",
            str(output_directory),
        ]
    )

    assert arguments.dataset_directory == dataset_directory
    assert arguments.output_directory == output_directory


def test_global_sample_is_stratified_and_deterministic() -> None:
    dataset = create_market_dataset(
        start_date="2024-01-02",
        rows_per_ticker=20,
    )

    first_sample = _select_global_observations(
        validation_dataset=dataset,
        maximum_rows=10,
        random_state=42,
    )
    second_sample = _select_global_observations(
        validation_dataset=dataset,
        maximum_rows=10,
        random_state=42,
    )

    assert len(first_sample) == 10
    assert first_sample.groupby("ticker").size().to_dict() == {
        "QQQ": 5,
        "SPY": 5,
    }

    pdt.assert_frame_equal(
        first_sample,
        second_sample,
    )


def test_run_xai_workflow_exports_real_explanations(
    tmp_path: Path,
) -> None:
    dataset_directory = tmp_path / "datasets"
    output_directory = tmp_path / "xai"
    dataset_directory.mkdir()

    write_market_datasets(
        dataset_directory,
    )

    model = RandomForestForecastModel(
        n_estimators=30,
        max_depth=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=1,
    )

    result = run_random_forest_xai(
        dataset_directory=dataset_directory,
        output_directory=output_directory,
        model=model,
    )

    assert isinstance(
        result,
        RandomForestXAIWorkflowResult,
    )
    assert model.is_fitted
    assert len(result.analyzed_features) == EFFECT_FEATURE_COUNT
    assert result.global_observations == 24
    assert result.local_ticker in {
        "QQQ",
        "SPY",
    }
    assert np.isfinite(
        result.local_actual,
    )
    assert np.isfinite(
        result.local_prediction,
    )

    export_paths = result.export_paths

    assert export_paths.result_json.is_file()
    assert export_paths.local_shap_csv.is_file()
    assert export_paths.global_shap_importance_csv.is_file()
    assert export_paths.partial_dependence_csv.is_file()
    assert export_paths.accumulated_local_effects_csv.is_file()

    expected_plot_count = 3 + 2 * EFFECT_FEATURE_COUNT
    assert len(export_paths.plot_files) == expected_plot_count
    assert all(
        path.is_file() and path.stat().st_size > 0
        for path in (export_paths.plot_files.values())
    )

    payload = json.loads(
        export_paths.result_json.read_text(
            encoding="utf-8",
        )
    )

    assert payload["metadata"]["model_name"] == "random_forest"
    assert payload["metadata"]["training_split"] == "training"
    assert payload["metadata"]["explanation_split"] == "validation"
    assert payload["metadata"]["global_explanation_rows"] == 24
    assert set(payload["partial_dependence"]) == set(result.analyzed_features)
    assert set(payload["accumulated_local_effects"]) == set(result.analyzed_features)

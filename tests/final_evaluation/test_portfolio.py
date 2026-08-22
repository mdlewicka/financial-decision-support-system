from pathlib import Path

import pandas as pd
import pytest

from backend.src.final_evaluation.portfolio import (
    load_frozen_portfolios,
)


def write_weights(
    path: Path,
) -> None:
    pd.DataFrame(
        [
            {
                "scenario": "equal_weight",
                "construction": "manual_equal_weight",
                "ticker": "AAA",
                "weight": 0.5,
            },
            {
                "scenario": "equal_weight",
                "construction": "manual_equal_weight",
                "ticker": "BBB",
                "weight": 0.5,
            },
            {
                "scenario": "defensive",
                "construction": "manual_defensive",
                "ticker": "AAA",
                "weight": 0.7,
            },
            {
                "scenario": "defensive",
                "construction": "manual_defensive",
                "ticker": "BBB",
                "weight": 0.3,
            },
        ]
    ).to_csv(
        path,
        index=False,
    )


def test_load_frozen_portfolios(
    tmp_path: Path,
) -> None:
    path = tmp_path / "portfolio_weights.csv"

    write_weights(path)

    portfolios = load_frozen_portfolios(path)

    assert [portfolio.name for portfolio in portfolios] == [
        "defensive",
        "equal_weight",
    ]

    assert portfolios[0].construction == "manual_defensive"

    assert portfolios[0].weights.to_dict() == {
        "AAA": 0.7,
        "BBB": 0.3,
    }


def test_load_frozen_portfolios_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "portfolio_weights.csv"

    pd.DataFrame({"scenario": ["equal_weight"]}).to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        load_frozen_portfolios(path)


def test_load_frozen_portfolios_rejects_duplicate_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "portfolio_weights.csv"

    pd.DataFrame(
        [
            {
                "scenario": "example",
                "construction": "manual",
                "ticker": "AAA",
                "weight": 0.5,
            },
            {
                "scenario": "example",
                "construction": "manual",
                "ticker": "AAA",
                "weight": 0.5,
            },
        ]
    ).to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        load_frozen_portfolios(path)


def test_load_frozen_portfolios_rejects_invalid_weight_sum(
    tmp_path: Path,
) -> None:
    path = tmp_path / "portfolio_weights.csv"

    pd.DataFrame(
        [
            {
                "scenario": "example",
                "construction": "manual",
                "ticker": "AAA",
                "weight": 0.4,
            },
            {
                "scenario": "example",
                "construction": "manual",
                "ticker": "BBB",
                "weight": 0.4,
            },
        ]
    ).to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="must sum to 1",
    ):
        load_frozen_portfolios(path)

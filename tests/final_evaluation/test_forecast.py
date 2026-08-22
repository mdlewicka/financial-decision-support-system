import json
from pathlib import Path

import pytest

from backend.src.final_evaluation.forecast import (
    load_frozen_arima_orders,
)


def test_load_frozen_arima_orders(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "arima_order_selection.json"

    artifact_path.write_text(
        json.dumps(
            {
                "selected_orders": {
                    "SPY": [1, 0, 1],
                    "BTC-USD": [2, 0, 0],
                }
            }
        ),
        encoding="utf-8",
    )

    result = load_frozen_arima_orders(artifact_path)

    assert result == {
        "BTC-USD": (2, 0, 0),
        "SPY": (1, 0, 1),
    }


def test_load_frozen_arima_orders_rejects_missing_mapping(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "arima_order_selection.json"

    artifact_path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="selected_orders",
    ):
        load_frozen_arima_orders(artifact_path)


def test_load_frozen_arima_orders_rejects_invalid_order(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "arima_order_selection.json"

    artifact_path.write_text(
        json.dumps(
            {
                "selected_orders": {
                    "SPY": [1, -1, 1],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="three non-negative integers",
    ):
        load_frozen_arima_orders(artifact_path)

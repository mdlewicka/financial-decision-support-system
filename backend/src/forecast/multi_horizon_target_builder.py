from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

ForecastHorizon = Literal[1, 5]

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "adjusted_close",
)

TARGET_COLUMNS: tuple[str, ...] = (
    "target_date_1",
    "target_return_1",
    "target_direction_1",
    "target_class_1",
    "target_ready_1",
    "target_date_5",
    "target_return_5",
    "target_direction_5",
    "target_class_5",
    "target_ready_5",
)

TARGET_CLASSES: tuple[str, ...] = (
    "down",
    "abstain",
    "up",
)


@dataclass(frozen=True, slots=True)
class ForecastTargetConfig:
    """Economic thresholds for the V2 forecasting targets.

    A return with an absolute value below or equal to its threshold receives
    the class ``abstain``. The label means that the realised move was too small
    to justify an active directional decision under the configured assumption.
    It is distinct from an unavailable target at the end of a time series.
    """

    one_session_threshold: float = 0.0025
    five_session_threshold: float = 0.0050

    def __post_init__(self) -> None:
        for name, value in (
            ("one_session_threshold", self.one_session_threshold),
            ("five_session_threshold", self.five_session_threshold),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value.")


@dataclass(frozen=True, slots=True)
class ForecastTargetContract:
    horizons: tuple[int, ...] = (1, 5)
    target_classes: tuple[str, ...] = TARGET_CLASSES
    uses_adjusted_close: bool = True
    uses_existing_target_column: bool = False
    requires_target_date_cutoff_for_folds: bool = True


class MultiHorizonForecastTargetBuilder:
    """Build one- and five-session forecasting targets per ticker."""

    contract = ForecastTargetContract()

    def __init__(
        self,
        config: ForecastTargetConfig | None = None,
    ) -> None:
        self.config = config or ForecastTargetConfig()

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepare_input(data)
        prepared["__original_order"] = np.arange(
            len(prepared),
            dtype=int,
        )

        ordered = prepared.sort_values(
            ["ticker", "date"],
            kind="stable",
            ignore_index=True,
        )

        grouped = ordered.groupby(
            "ticker",
            sort=False,
            group_keys=False,
        )

        self._add_horizon_targets(
            data=ordered,
            grouped=grouped,
            horizon=1,
            threshold=self.config.one_session_threshold,
        )

        self._add_horizon_targets(
            data=ordered,
            grouped=grouped,
            horizon=5,
            threshold=self.config.five_session_threshold,
        )

        result = ordered.sort_values(
            "__original_order",
            kind="stable",
        ).drop(columns="__original_order")

        result.index = data.index

        return result[
            [
                *REQUIRED_INPUT_COLUMNS,
                *TARGET_COLUMNS,
            ]
        ].copy()

    @staticmethod
    def ready_rows(
        targets: pd.DataFrame,
        horizon: ForecastHorizon,
    ) -> pd.DataFrame:
        ready_column = MultiHorizonForecastTargetBuilder._column(
            "target_ready",
            horizon,
        )

        MultiHorizonForecastTargetBuilder._require_columns(
            targets,
            [ready_column],
        )

        return targets.loc[targets[ready_column].astype(bool)].copy()

    @staticmethod
    def targets_available_by(
        targets: pd.DataFrame,
        horizon: ForecastHorizon,
        cutoff: object,
    ) -> pd.DataFrame:
        """Return labels whose complete horizon is known by ``cutoff``.

        This method must be used before fitting a chronological fold. It keeps
        targets ending on the cutoff date because the project predicts the
        following session after observing the current session close.
        """

        ready_column = MultiHorizonForecastTargetBuilder._column(
            "target_ready",
            horizon,
        )

        date_column = MultiHorizonForecastTargetBuilder._column(
            "target_date",
            horizon,
        )

        MultiHorizonForecastTargetBuilder._require_columns(
            targets,
            [
                ready_column,
                date_column,
            ],
        )

        cutoff_date = _as_timestamp(
            cutoff,
            "cutoff",
        )

        target_dates = pd.to_datetime(
            targets[date_column],
            errors="raise",
        )

        mask = targets[ready_column].astype(bool) & target_dates.le(cutoff_date)

        return targets.loc[mask].copy()

    @staticmethod
    def _add_horizon_targets(
        data: pd.DataFrame,
        grouped: Any,
        horizon: ForecastHorizon,
        threshold: float,
    ) -> None:
        future_price = grouped["adjusted_close"].shift(-horizon)

        future_date = grouped["date"].shift(-horizon)

        target_return = future_price / data["adjusted_close"] - 1.0

        target_ready = future_price.notna() & future_date.notna()

        date_column = MultiHorizonForecastTargetBuilder._column(
            "target_date",
            horizon,
        )

        return_column = MultiHorizonForecastTargetBuilder._column(
            "target_return",
            horizon,
        )

        direction_column = MultiHorizonForecastTargetBuilder._column(
            "target_direction",
            horizon,
        )

        class_column = MultiHorizonForecastTargetBuilder._column(
            "target_class",
            horizon,
        )

        ready_column = MultiHorizonForecastTargetBuilder._column(
            "target_ready",
            horizon,
        )

        data[date_column] = future_date

        data[return_column] = target_return.astype(float)

        data[direction_column] = MultiHorizonForecastTargetBuilder._direction_labels(
            target_return=target_return,
            target_ready=target_ready,
        )

        data[class_column] = MultiHorizonForecastTargetBuilder._economic_classes(
            target_return=target_return,
            target_ready=target_ready,
            threshold=threshold,
        )

        data[ready_column] = target_ready.astype(bool)

    @staticmethod
    def _direction_labels(
        target_return: pd.Series,
        target_ready: pd.Series,
    ) -> pd.Series:
        labels = pd.Series(
            pd.NA,
            index=target_return.index,
            dtype="Int8",
        )

        labels.loc[target_ready & target_return.gt(0.0)] = 1

        labels.loc[target_ready & target_return.eq(0.0)] = 0

        labels.loc[target_ready & target_return.lt(0.0)] = -1

        return labels

    @staticmethod
    def _economic_classes(
        target_return: pd.Series,
        target_ready: pd.Series,
        threshold: float,
    ) -> pd.Series:
        labels = pd.Series(
            pd.NA,
            index=target_return.index,
            dtype="string",
        )

        labels.loc[target_ready & target_return.gt(threshold)] = "up"

        labels.loc[target_ready & target_return.lt(-threshold)] = "down"

        labels.loc[
            target_ready
            & target_return.between(
                -threshold,
                threshold,
            )
        ] = "abstain"

        return labels

    @staticmethod
    def _prepare_input(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        if data.empty:
            raise ValueError("Forecast target input data must not be empty.")

        missing_columns = sorted(set(REQUIRED_INPUT_COLUMNS).difference(data.columns))

        if missing_columns:
            joined = ", ".join(missing_columns)

            raise ValueError(f"Forecast target input is missing columns: {joined}.")

        prepared = data[list(REQUIRED_INPUT_COLUMNS)].copy()

        prepared["date"] = pd.to_datetime(
            prepared["date"],
            errors="raise",
        )

        prepared["ticker"] = prepared["ticker"].astype("string")

        prepared["adjusted_close"] = pd.to_numeric(
            prepared["adjusted_close"],
            errors="raise",
        ).astype(float)

        if prepared[list(REQUIRED_INPUT_COLUMNS)].isna().any().any():
            raise ValueError("Forecast target input contains missing values.")

        if not np.isfinite(prepared["adjusted_close"].to_numpy(dtype=float)).all():
            raise ValueError("Adjusted close values must be finite.")

        if (prepared["adjusted_close"] <= 0.0).any():
            raise ValueError("Adjusted close values must be greater than zero.")

        if prepared["ticker"].str.strip().eq("").any():
            raise ValueError("Ticker values must be non-empty.")

        if prepared.duplicated(
            [
                "date",
                "ticker",
            ]
        ).any():
            raise ValueError(
                "Forecast target input contains duplicate date/ticker rows."
            )

        return prepared

    @staticmethod
    def _column(
        prefix: str,
        horizon: ForecastHorizon,
    ) -> str:
        if horizon not in (1, 5):
            raise ValueError("Supported target horizons are 1 and 5 sessions.")

        return f"{prefix}_{horizon}"

    @staticmethod
    def _require_columns(
        data: pd.DataFrame,
        required_columns: list[str],
    ) -> None:
        missing_columns = sorted(set(required_columns).difference(data.columns))

        if missing_columns:
            raise ValueError(
                "Target data is missing columns: " + ", ".join(missing_columns) + "."
            )


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

    if isinstance(
        value,
        datetime,
    ):
        return pd.Timestamp(value)

    if isinstance(
        value,
        date,
    ):
        return pd.Timestamp(value)

    if isinstance(
        value,
        str,
    ):
        return pd.Timestamp(value)

    raise TypeError(
        f"{name} must be a date, datetime, string, "
        "pandas Timestamp or NumPy datetime64."
    )

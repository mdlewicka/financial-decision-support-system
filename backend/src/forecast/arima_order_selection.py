import warnings
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
)
from statsmodels.tsa.arima.model import ARIMA

ArimaOrder = tuple[int, int, int]

DEFAULT_ARIMA_CANDIDATE_ORDERS: tuple[
    ArimaOrder,
    ...,
] = (
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
    (1, 0, 1),
)

SCORE_COLUMNS = [
    "ticker",
    "p",
    "d",
    "q",
    "aic",
    "bic",
    "converged",
    "selected",
    "error",
]


class CandidateScore(TypedDict):
    ticker: str
    p: int
    d: int
    q: int
    aic: float
    bic: float
    converged: bool
    selected: bool
    error: str


@dataclass(frozen=True, slots=True)
class ArimaOrderSelectionResult:
    selected_orders: dict[str, ArimaOrder]
    scores: pd.DataFrame


def select_arima_orders(
    training_dataset: pd.DataFrame,
    candidate_orders: tuple[
        ArimaOrder,
        ...,
    ] = DEFAULT_ARIMA_CANDIDATE_ORDERS,
    criterion: str = "aic",
) -> ArimaOrderSelectionResult:
    _validate_training_dataset(training_dataset)
    _validate_candidate_orders(candidate_orders)

    if criterion not in {
        "aic",
        "bic",
    }:
        raise ValueError("criterion must be either 'aic' or 'bic'.")

    rows: list[CandidateScore] = []
    selected_orders: dict[str, ArimaOrder] = {}

    for ticker, ticker_data in training_dataset.groupby(
        "ticker",
        sort=True,
    ):
        ticker_name = str(ticker)
        ordered_ticker_data = ticker_data.sort_values(
            by="date",
            kind="stable",
        )
        targets = pd.to_numeric(
            ordered_ticker_data["target"],
            errors="raise",
        ).to_numpy(
            dtype=float,
        )

        if not np.isfinite(targets).all():
            raise ValueError(f"Ticker {ticker_name} contains non-finite target values.")

        ticker_rows = [
            _evaluate_candidate(
                ticker=ticker_name,
                targets=targets,
                order=order,
            )
            for order in candidate_orders
        ]
        rows.extend(ticker_rows)

        eligible_rows = [
            row
            for row in ticker_rows
            if row["converged"]
            and np.isfinite(
                _criterion_value(
                    row=row,
                    criterion=criterion,
                )
            )
        ]

        if not eligible_rows:
            raise RuntimeError(
                f"No converged ARIMA candidate for ticker: {ticker_name}."
            )

        selected_row = min(
            eligible_rows,
            key=lambda row: _criterion_value(
                row=row,
                criterion=criterion,
            ),
        )
        selected_row["selected"] = True

        selected_orders[ticker_name] = (
            selected_row["p"],
            selected_row["d"],
            selected_row["q"],
        )

    scores = pd.DataFrame(
        rows,
        columns=SCORE_COLUMNS,
    )

    return ArimaOrderSelectionResult(
        selected_orders=selected_orders,
        scores=scores,
    )


def _criterion_value(
    row: CandidateScore,
    criterion: str,
) -> float:
    if criterion == "aic":
        return row["aic"]

    return row["bic"]


def _evaluate_candidate(
    ticker: str,
    targets: np.ndarray,
    order: ArimaOrder,
) -> CandidateScore:
    row = CandidateScore(
        ticker=ticker,
        p=order[0],
        d=order[1],
        q=order[2],
        aic=float("nan"),
        bic=float("nan"),
        converged=False,
        selected=False,
        error="",
    )

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )

            fitted_model = ARIMA(
                endog=targets,
                order=order,
            ).fit()

        convergence_warning = any(
            issubclass(
                warning.category,
                ConvergenceWarning,
            )
            for warning in caught_warnings
        )
        fit_details = getattr(
            fitted_model,
            "mle_retvals",
            {},
        )
        optimizer_converged = bool(
            fit_details.get(
                "converged",
                True,
            )
        )

        row["aic"] = float(fitted_model.aic)
        row["bic"] = float(fitted_model.bic)
        row["converged"] = optimizer_converged and not convergence_warning

        if not row["converged"]:
            row["error"] = "Model did not converge."

    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"

    return row


def _validate_training_dataset(
    training_dataset: pd.DataFrame,
) -> None:
    if training_dataset.empty:
        raise ValueError("Training dataset must not be empty.")

    required_columns = {
        "ticker",
        "date",
        "target",
    }
    missing_columns = sorted(required_columns.difference(training_dataset.columns))

    if missing_columns:
        joined_columns = ", ".join(missing_columns)
        raise ValueError(
            f"Training dataset is missing required columns: {joined_columns}."
        )


def _validate_candidate_orders(
    candidate_orders: tuple[
        ArimaOrder,
        ...,
    ],
) -> None:
    if not candidate_orders:
        raise ValueError("At least one candidate ARIMA order is required.")

    for order in candidate_orders:
        if len(order) != 3 or any(
            not isinstance(value, int) or value < 0 for value in order
        ):
            raise ValueError(
                "Every ARIMA order must contain three non-negative integers."
            )

    if len(set(candidate_orders)) != len(candidate_orders):
        raise ValueError("Candidate ARIMA orders must be unique.")

"""Read frozen portfolio risk benchmark artifacts for the HTTP API."""

import csv
import json
from json import JSONDecodeError
from math import isclose, isfinite
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from backend.src.api.risk_schemas import (
    RiskArtifactPayload,
    RiskCorrelationRow,
)
from backend.src.config import PROJECT_ROOT

DEFAULT_RISK_ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "benchmark_results" / "risk" / "portfolio"
)

RISK_ARTIFACT_FILENAME = "portfolio_result.json"
VALIDATION_CORRELATION_FILENAME = "validation_correlation_matrix.csv"


class RiskArtifactError(RuntimeError):
    """Base error raised while reading frozen risk benchmark artifacts."""


class RiskArtifactNotFoundError(RiskArtifactError):
    """Raised when an expected risk artifact does not exist."""


class InvalidRiskArtifactError(RiskArtifactError):
    """Raised when a risk artifact violates its expected contract."""


class RiskArtifactRepository:
    """Load previously exported portfolio risk benchmark results."""

    def __init__(
        self,
        artifact_directory: Path = (DEFAULT_RISK_ARTIFACT_DIRECTORY),
    ) -> None:
        self.artifact_directory = artifact_directory

    @property
    def artifact_path(
        self,
    ) -> Path:
        """Return the main portfolio risk benchmark artifact path."""
        return self.artifact_directory / RISK_ARTIFACT_FILENAME

    @property
    def validation_correlation_path(
        self,
    ) -> Path:
        """Return the validation correlation matrix path."""
        return self.artifact_directory / VALIDATION_CORRELATION_FILENAME

    def artifact_exists(
        self,
    ) -> bool:
        """Return whether the main risk benchmark artifact exists."""
        return self.artifact_path.is_file()

    def validation_correlation_exists(
        self,
    ) -> bool:
        """Return whether the validation correlation matrix exists."""
        return self.validation_correlation_path.is_file()

    def load(
        self,
    ) -> RiskArtifactPayload:
        """Load and validate the main portfolio risk benchmark artifact."""
        artifact_path = self.artifact_path

        if not artifact_path.is_file():
            raise RiskArtifactNotFoundError(
                f"Risk benchmark artifact is missing: {artifact_path}"
            )

        try:
            raw_payload = json.loads(
                artifact_path.read_text(
                    encoding="utf-8",
                )
            )
        except (
            OSError,
            JSONDecodeError,
        ) as error:
            raise InvalidRiskArtifactError(
                f"Risk benchmark artifact could not be read: {artifact_path}"
            ) from error

        try:
            return RiskArtifactPayload.model_validate(raw_payload)
        except ValidationError as error:
            raise InvalidRiskArtifactError(
                f"Risk benchmark artifact has an invalid schema: {artifact_path}"
            ) from error

    def load_validation_correlation(
        self,
        tickers: tuple[str, ...],
    ) -> tuple[
        RiskCorrelationRow,
        ...,
    ]:
        """Load and validate the validation correlation matrix."""
        matrix_path = self.validation_correlation_path

        if not matrix_path.is_file():
            raise RiskArtifactNotFoundError(
                f"Validation correlation matrix is missing: {matrix_path}"
            )

        try:
            with matrix_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                reader = csv.DictReader(file)

                fieldnames = reader.fieldnames

                rows = list(reader)
        except (
            OSError,
            csv.Error,
        ) as error:
            raise InvalidRiskArtifactError(
                f"Validation correlation matrix could not be read: {matrix_path}"
            ) from error

        expected_fieldnames = (
            "ticker",
            *tickers,
        )

        if tuple(fieldnames or ()) != expected_fieldnames:
            raise InvalidRiskArtifactError(
                "Validation correlation matrix columns do not match "
                "risk market tickers."
            )

        if len(rows) != len(tickers):
            raise InvalidRiskArtifactError(
                "Validation correlation matrix row count does not match "
                "risk market tickers."
            )

        parsed_rows: list[RiskCorrelationRow] = []

        for (
            expected_ticker,
            row,
        ) in zip(
            tickers,
            rows,
            strict=True,
        ):
            row_ticker = cast(
                str | None,
                row.get("ticker"),
            )

            if row_ticker != expected_ticker:
                raise InvalidRiskArtifactError(
                    "Validation correlation matrix row order does not match "
                    "risk market tickers."
                )

            correlations: dict[
                str,
                float,
            ] = {}

            for column_ticker in tickers:
                raw_value = cast(
                    str | None,
                    row.get(column_ticker),
                )

                if raw_value is None:
                    raise InvalidRiskArtifactError(
                        "Validation correlation matrix contains a missing value."
                    )

                try:
                    value = float(raw_value)
                except ValueError as error:
                    raise InvalidRiskArtifactError(
                        "Validation correlation matrix contains a non-numeric value."
                    ) from error

                if not isfinite(value):
                    raise InvalidRiskArtifactError(
                        "Validation correlation matrix contains a non-finite value."
                    )

                if not (-1.000000001 <= value <= 1.000000001):
                    raise InvalidRiskArtifactError(
                        "Validation correlation matrix contains "
                        "a value outside [-1, 1]."
                    )

                correlations[column_ticker] = value

            parsed_rows.append(
                RiskCorrelationRow(
                    ticker=expected_ticker,
                    correlations=correlations,
                )
            )

        self._validate_correlation_matrix(
            rows=tuple(parsed_rows),
            tickers=tickers,
        )

        return tuple(parsed_rows)

    @staticmethod
    def _validate_correlation_matrix(
        rows: tuple[
            RiskCorrelationRow,
            ...,
        ],
        tickers: tuple[
            str,
            ...,
        ],
    ) -> None:
        """Validate correlation-matrix diagonal and symmetry."""
        matrix = {row.ticker: (row.correlations) for row in rows}

        for ticker in tickers:
            if not isclose(
                matrix[ticker][ticker],
                1.0,
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise InvalidRiskArtifactError(
                    "Validation correlation matrix diagonal must equal one."
                )

        for row_ticker in tickers:
            for column_ticker in tickers:
                if not isclose(
                    matrix[row_ticker][column_ticker],
                    matrix[column_ticker][row_ticker],
                    rel_tol=0.0,
                    abs_tol=1e-8,
                ):
                    raise InvalidRiskArtifactError(
                        "Validation correlation matrix must be symmetric."
                    )

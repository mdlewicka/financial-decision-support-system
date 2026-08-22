"""Read frozen portfolio artifacts for the HTTP API."""

import csv
import json
from datetime import date
from json import JSONDecodeError
from math import isclose, isfinite
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from backend.src.api.portfolio_schemas import (
    PortfolioContribution,
    PortfolioReturnPoint,
)
from backend.src.api.risk_schemas import RiskArtifactPayload
from backend.src.config import PROJECT_ROOT

DEFAULT_PORTFOLIO_ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "benchmark_results" / "risk" / "portfolio"
)

PORTFOLIO_ARTIFACT_FILENAME = "portfolio_result.json"
PORTFOLIO_CONTRIBUTIONS_FILENAME = "portfolio_contributions.csv"
PORTFOLIO_RETURNS_FILENAME = "portfolio_returns.csv"


class PortfolioArtifactError(RuntimeError):
    """Base error raised while reading portfolio artifacts."""


class PortfolioArtifactNotFoundError(PortfolioArtifactError):
    """Raised when an expected portfolio artifact does not exist."""


class InvalidPortfolioArtifactError(PortfolioArtifactError):
    """Raised when a portfolio artifact violates its contract."""


class PortfolioArtifactRepository:
    """Load previously exported portfolio benchmark results."""

    def __init__(
        self,
        artifact_directory: Path = DEFAULT_PORTFOLIO_ARTIFACT_DIRECTORY,
    ) -> None:
        self.artifact_directory = artifact_directory

    @property
    def artifact_path(self) -> Path:
        """Return the main portfolio JSON artifact path."""
        return self.artifact_directory / PORTFOLIO_ARTIFACT_FILENAME

    @property
    def contributions_path(self) -> Path:
        """Return the portfolio contributions CSV path."""
        return self.artifact_directory / PORTFOLIO_CONTRIBUTIONS_FILENAME

    @property
    def returns_path(self) -> Path:
        """Return the portfolio returns CSV path."""
        return self.artifact_directory / PORTFOLIO_RETURNS_FILENAME

    def artifact_exists(self) -> bool:
        """Return whether the main portfolio artifact exists."""
        return self.artifact_path.is_file()

    def contributions_exist(self) -> bool:
        """Return whether portfolio contribution data exists."""
        return self.contributions_path.is_file()

    def returns_exist(self) -> bool:
        """Return whether portfolio return data exists."""
        return self.returns_path.is_file()

    def load(self) -> RiskArtifactPayload:
        """Load and validate the main portfolio benchmark artifact."""
        artifact_path = self.artifact_path

        if not artifact_path.is_file():
            raise PortfolioArtifactNotFoundError(
                f"Portfolio benchmark artifact is missing: {artifact_path}"
            )

        try:
            raw_payload = json.loads(
                artifact_path.read_text(
                    encoding="utf-8",
                )
            )
        except (OSError, JSONDecodeError) as error:
            raise InvalidPortfolioArtifactError(
                f"Portfolio benchmark artifact could not be read: {artifact_path}"
            ) from error

        try:
            return RiskArtifactPayload.model_validate(raw_payload)
        except ValidationError as error:
            raise InvalidPortfolioArtifactError(
                f"Portfolio benchmark artifact has an invalid schema: {artifact_path}"
            ) from error

    def load_validation_contributions(
        self,
        *,
        scenario_name: str,
        construction: str,
        expected_weights: dict[str, float],
    ) -> tuple[PortfolioContribution, ...]:
        """Load validation asset contributions for one portfolio scenario."""
        path = self.contributions_path

        if not path.is_file():
            raise PortfolioArtifactNotFoundError(
                f"Portfolio contributions artifact is missing: {path}"
            )

        expected_columns = (
            "scenario",
            "construction",
            "split",
            "ticker",
            "weight",
            "annualized_return_contribution",
            "annualized_volatility_contribution",
        )

        try:
            with path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                reader = csv.DictReader(file)

                if tuple(reader.fieldnames or ()) != expected_columns:
                    raise InvalidPortfolioArtifactError(
                        "Portfolio contributions columns are invalid."
                    )

                rows = [
                    row
                    for row in reader
                    if (
                        row.get("scenario") == scenario_name
                        and row.get("split") == "validation"
                    )
                ]
        except (OSError, csv.Error) as error:
            raise InvalidPortfolioArtifactError(
                f"Portfolio contributions artifact could not be read: {path}"
            ) from error

        if len(rows) != len(expected_weights):
            raise InvalidPortfolioArtifactError(
                "Portfolio validation contribution count does not "
                "match portfolio weights."
            )

        contributions_by_ticker: dict[str, PortfolioContribution] = {}

        for row in rows:
            if row.get("construction") != construction:
                raise InvalidPortfolioArtifactError(
                    "Portfolio contribution construction does not "
                    "match the scenario definition."
                )

            ticker = cast(
                str | None,
                row.get("ticker"),
            )

            if ticker is None or not ticker.strip():
                raise InvalidPortfolioArtifactError(
                    "Portfolio contribution ticker must be present."
                )

            ticker = ticker.strip()

            if ticker not in expected_weights:
                raise InvalidPortfolioArtifactError(
                    "Portfolio contribution contains an unexpected ticker."
                )

            if ticker in contributions_by_ticker:
                raise InvalidPortfolioArtifactError(
                    "Portfolio contribution tickers must be unique."
                )

            weight = self._parse_finite_float(
                cast(
                    str | None,
                    row.get("weight"),
                ),
                field_name="weight",
            )

            if not isclose(
                weight,
                expected_weights[ticker],
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise InvalidPortfolioArtifactError(
                    "Portfolio contribution weight does not match "
                    "the scenario allocation."
                )

            return_contribution = self._parse_finite_float(
                cast(
                    str | None,
                    row.get("annualized_return_contribution"),
                ),
                field_name=("annualized_return_contribution"),
            )

            volatility_contribution = self._parse_finite_float(
                cast(
                    str | None,
                    row.get("annualized_volatility_contribution"),
                ),
                field_name=("annualized_volatility_contribution"),
            )

            contributions_by_ticker[ticker] = PortfolioContribution(
                ticker=ticker,
                weight=weight,
                annualized_return_contribution=(return_contribution),
                annualized_volatility_contribution=(volatility_contribution),
            )

        if set(contributions_by_ticker) != set(expected_weights):
            raise InvalidPortfolioArtifactError(
                "Portfolio contribution tickers do not match the scenario allocation."
            )

        return tuple(contributions_by_ticker[ticker] for ticker in expected_weights)

    def load_validation_returns(
        self,
        *,
        scenario_name: str,
        construction: str,
        expected_observations: int,
        expected_date_min: date,
        expected_date_max: date,
    ) -> tuple[PortfolioReturnPoint, ...]:
        """Load validation returns and calculate cumulative return."""
        path = self.returns_path

        if not path.is_file():
            raise PortfolioArtifactNotFoundError(
                f"Portfolio returns artifact is missing: {path}"
            )

        expected_columns = (
            "scenario",
            "construction",
            "split",
            "date",
            "portfolio_return",
        )

        try:
            with path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                reader = csv.DictReader(file)

                if tuple(reader.fieldnames or ()) != expected_columns:
                    raise InvalidPortfolioArtifactError(
                        "Portfolio returns columns are invalid."
                    )

                rows = [
                    row
                    for row in reader
                    if (
                        row.get("scenario") == scenario_name
                        and row.get("split") == "validation"
                    )
                ]
        except (OSError, csv.Error) as error:
            raise InvalidPortfolioArtifactError(
                f"Portfolio returns artifact could not be read: {path}"
            ) from error

        if len(rows) != expected_observations:
            raise InvalidPortfolioArtifactError(
                "Portfolio validation return count does not match "
                "validation market observations."
            )

        parsed: list[
            tuple[
                date,
                float,
            ]
        ] = []

        for row in rows:
            if row.get("construction") != construction:
                raise InvalidPortfolioArtifactError(
                    "Portfolio return construction does not "
                    "match the scenario definition."
                )

            raw_date = cast(
                str | None,
                row.get("date"),
            )

            if raw_date is None:
                raise InvalidPortfolioArtifactError(
                    "Portfolio returns contain a missing date."
                )

            try:
                observation_date = date.fromisoformat(raw_date)
            except ValueError as error:
                raise InvalidPortfolioArtifactError(
                    "Portfolio returns contain an invalid date."
                ) from error

            return_value = self._parse_finite_float(
                cast(
                    str | None,
                    row.get("portfolio_return"),
                ),
                field_name="portfolio_return",
            )

            if return_value < -1.0:
                raise InvalidPortfolioArtifactError(
                    "Portfolio return cannot be lower than -1."
                )

            parsed.append(
                (
                    observation_date,
                    return_value,
                )
            )

        if not parsed:
            raise InvalidPortfolioArtifactError(
                "Portfolio validation returns must not be empty."
            )

        dates = tuple(observation_date for observation_date, _ in parsed)

        if len(dates) != len(set(dates)):
            raise InvalidPortfolioArtifactError(
                "Portfolio return dates must be unique."
            )

        if dates != tuple(sorted(dates)):
            raise InvalidPortfolioArtifactError(
                "Portfolio return dates must be sorted in ascending order."
            )

        if dates[0] != expected_date_min or dates[-1] != expected_date_max:
            raise InvalidPortfolioArtifactError(
                "Portfolio return date range does not match validation market metadata."
            )

        cumulative_growth = 1.0
        points: list[PortfolioReturnPoint] = []

        for (
            observation_date,
            return_value,
        ) in parsed:
            cumulative_growth *= 1.0 + return_value

            points.append(
                PortfolioReturnPoint(
                    date=observation_date,
                    portfolio_return=return_value,
                    cumulative_return=(cumulative_growth - 1.0),
                )
            )

        return tuple(points)

    @staticmethod
    def _parse_finite_float(
        raw_value: str | None,
        *,
        field_name: str,
    ) -> float:
        """Parse one required finite numeric CSV field."""
        if raw_value is None:
            raise InvalidPortfolioArtifactError(
                f"Portfolio {field_name} must be present."
            )

        try:
            value = float(raw_value)
        except ValueError as error:
            raise InvalidPortfolioArtifactError(
                f"Portfolio {field_name} must be numeric."
            ) from error

        if not isfinite(value):
            raise InvalidPortfolioArtifactError(
                f"Portfolio {field_name} must be finite."
            )

        return value

"""Read frozen forecast validation benchmark artifacts for the API."""

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from backend.src.api.forecast_schemas import ForecastArtifactPayload
from backend.src.config import PROJECT_ROOT

DEFAULT_FORECAST_ARTIFACT_DIRECTORY = PROJECT_ROOT / "benchmark_results" / "forecast"


@dataclass(frozen=True, slots=True)
class ForecastModelDefinition:
    """Describe one forecast benchmark exposed by the API."""

    model_name: str
    display_name: str
    directory_name: str


FORECAST_MODEL_DEFINITIONS: tuple[ForecastModelDefinition, ...] = (
    ForecastModelDefinition(
        model_name="naive_zero_return",
        display_name="Naive zero-return",
        directory_name="naive",
    ),
    ForecastModelDefinition(
        model_name="arima_walk_forward",
        display_name="ARIMA walk-forward",
        directory_name="arima",
    ),
    ForecastModelDefinition(
        model_name="random_forest",
        display_name="Random Forest",
        directory_name="random_forest",
    ),
)


class ForecastArtifactError(RuntimeError):
    """Base error raised while reading forecast benchmark artifacts."""


class ForecastArtifactNotFoundError(ForecastArtifactError):
    """Raised when an expected forecast artifact does not exist."""


class InvalidForecastArtifactError(ForecastArtifactError):
    """Raised when a forecast artifact does not satisfy its contract."""


class ForecastArtifactRepository:
    """Load previously exported validation benchmark results."""

    def __init__(
        self,
        benchmark_directory: Path = DEFAULT_FORECAST_ARTIFACT_DIRECTORY,
    ) -> None:
        self.benchmark_directory = benchmark_directory

    @property
    def model_definitions(self) -> tuple[ForecastModelDefinition, ...]:
        """Return the ordered forecast model registry."""
        return FORECAST_MODEL_DEFINITIONS

    def artifact_path(
        self,
        definition: ForecastModelDefinition,
    ) -> Path:
        """Return the JSON artifact path for one model."""
        return (
            self.benchmark_directory
            / definition.directory_name
            / "forecast_result.json"
        )

    def artifact_exists(
        self,
        definition: ForecastModelDefinition,
    ) -> bool:
        """Return whether the benchmark artifact exists."""
        return self.artifact_path(definition).is_file()

    def load(
        self,
        definition: ForecastModelDefinition,
    ) -> ForecastArtifactPayload:
        """Load and validate one forecast benchmark artifact."""
        artifact_path = self.artifact_path(definition)

        if not artifact_path.is_file():
            raise ForecastArtifactNotFoundError(
                "Forecast benchmark artifact is missing for "
                f"{definition.display_name}: {artifact_path}"
            )

        try:
            raw_payload = json.loads(
                artifact_path.read_text(
                    encoding="utf-8",
                )
            )
        except (OSError, JSONDecodeError) as error:
            raise InvalidForecastArtifactError(
                "Forecast benchmark artifact could not be read for "
                f"{definition.display_name}: {artifact_path}"
            ) from error

        try:
            payload = ForecastArtifactPayload.model_validate(
                raw_payload,
            )
        except ValidationError as error:
            raise InvalidForecastArtifactError(
                "Forecast benchmark artifact has an invalid schema for "
                f"{definition.display_name}: {artifact_path}"
            ) from error

        if payload.model_name != definition.model_name:
            raise InvalidForecastArtifactError(
                "Forecast benchmark artifact model name mismatch for "
                f"{definition.display_name}: expected "
                f"{definition.model_name!r}, got "
                f"{payload.model_name!r}."
            )

        return payload

    def load_all(
        self,
    ) -> tuple[
        tuple[
            ForecastModelDefinition,
            ForecastArtifactPayload,
        ],
        ...,
    ]:
        """Load all configured forecast benchmark artifacts."""
        return tuple(
            (
                definition,
                self.load(definition),
            )
            for definition in self.model_definitions
        )

"""Read frozen XAI validation artifacts for the HTTP API."""

import json
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from backend.src.api.xai_schemas import (
    XAIArtifactPayload,
)
from backend.src.config import PROJECT_ROOT

DEFAULT_XAI_ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "benchmark_results" / "xai" / "random_forest"
)

XAI_ARTIFACT_FILENAME = "xai_result.json"


class XAIArtifactError(RuntimeError):
    """Base error raised while reading an XAI artifact."""


class XAIArtifactNotFoundError(XAIArtifactError):
    """Raised when the expected XAI artifact does not exist."""


class InvalidXAIArtifactError(XAIArtifactError):
    """Raised when an XAI artifact violates its contract."""


class XAIArtifactRepository:
    """Load the previously exported Random Forest XAI result."""

    def __init__(
        self,
        artifact_directory: Path = DEFAULT_XAI_ARTIFACT_DIRECTORY,
    ) -> None:
        self.artifact_directory = artifact_directory

    @property
    def artifact_path(self) -> Path:
        """Return the path to the exported XAI JSON result."""
        return self.artifact_directory / XAI_ARTIFACT_FILENAME

    def artifact_exists(self) -> bool:
        """Return whether the XAI artifact exists."""
        return self.artifact_path.is_file()

    def load(self) -> XAIArtifactPayload:
        """Load and validate the XAI artifact."""
        artifact_path = self.artifact_path

        if not artifact_path.is_file():
            raise XAIArtifactNotFoundError(f"XAI artifact is missing: {artifact_path}")

        try:
            raw_payload = json.loads(
                artifact_path.read_text(
                    encoding="utf-8",
                )
            )
        except (OSError, JSONDecodeError) as error:
            raise InvalidXAIArtifactError(
                f"XAI artifact could not be read: {artifact_path}"
            ) from error

        try:
            return XAIArtifactPayload.model_validate(
                raw_payload,
            )
        except ValidationError as error:
            raise InvalidXAIArtifactError(
                f"XAI artifact has an invalid schema: {artifact_path}"
            ) from error

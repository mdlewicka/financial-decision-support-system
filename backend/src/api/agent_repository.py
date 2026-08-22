"""Read frozen decision-support agent artifacts for the HTTP API."""

import json
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from backend.src.agents.artifact_repository import (
    AgentArtifactPaths,
)
from backend.src.api.agent_schemas import (
    AgentResultsArtifact,
)
from backend.src.config import PROJECT_ROOT


class AgentArtifactError(RuntimeError):
    """Base error raised while reading decision-support agent artifacts."""


class AgentArtifactNotFoundError(AgentArtifactError):
    """Raised when an expected agent artifact does not exist."""


class InvalidAgentArtifactError(AgentArtifactError):
    """Raised when an agent artifact violates its expected contract."""


class AgentResultRepository:
    """Load frozen agent results and report runtime input availability."""

    def __init__(
        self,
        *,
        project_root: Path = PROJECT_ROOT,
        results_path: Path | None = None,
        corpus_path: Path | None = None,
    ) -> None:
        self.project_root = project_root

        self.results_path = (
            results_path
            if results_path is not None
            else (project_root / "benchmark_results" / "agents" / "agent_results.json")
        )

        self.corpus_path = (
            corpus_path
            if corpus_path is not None
            else (
                project_root / "data" / "rag" / "evaluation" / "reference_corpus.json"
            )
        )

        self.agent_artifact_paths = AgentArtifactPaths.from_project_root(project_root)

    def result_exists(self) -> bool:
        """Return whether the frozen three-agent result exists."""
        return self.results_path.is_file()

    def forecast_artifact_exists(self) -> bool:
        """Return whether the Random Forest benchmark artifact exists."""
        return self.agent_artifact_paths.forecast.is_file()

    def xai_artifact_exists(self) -> bool:
        """Return whether the Random Forest XAI artifact exists."""
        return self.agent_artifact_paths.xai.is_file()

    def portfolio_artifact_exists(self) -> bool:
        """Return whether the portfolio benchmark artifact exists."""
        return self.agent_artifact_paths.portfolio.is_file()

    def reinforcement_learning_artifact_exists(
        self,
    ) -> bool:
        """Return whether the reinforcement-learning artifact exists."""
        return self.agent_artifact_paths.reinforcement_learning.is_file()

    def corpus_exists(self) -> bool:
        """Return whether the reference corpus used by agents exists."""
        return self.corpus_path.is_file()

    def runtime_inputs_available(self) -> bool:
        """Return whether all files needed by the live agents are present."""
        return all(
            (
                self.result_exists(),
                self.forecast_artifact_exists(),
                self.xai_artifact_exists(),
                self.portfolio_artifact_exists(),
                self.reinforcement_learning_artifact_exists(),
                self.corpus_exists(),
            )
        )

    def load_results(self) -> AgentResultsArtifact:
        """Load and validate the frozen three-agent workflow result."""
        path = self.results_path

        if not path.is_file():
            raise AgentArtifactNotFoundError(
                f"Agent result artifact is missing: {path}"
            )

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8",
                )
            )
        except (
            OSError,
            JSONDecodeError,
        ) as error:
            raise InvalidAgentArtifactError(
                f"Agent result artifact could not be read: {path}"
            ) from error

        try:
            return AgentResultsArtifact.model_validate(
                payload,
            )
        except ValidationError as error:
            raise InvalidAgentArtifactError(
                f"Agent result artifact has an invalid schema: {path}"
            ) from error

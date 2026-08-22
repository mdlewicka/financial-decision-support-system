"""Read frozen RAG data and benchmark artifacts for the HTTP API."""

import json
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from backend.src.api.rag_schemas import (
    RAGComparisonArtifact,
)
from backend.src.config import PROJECT_ROOT
from backend.src.rag.loaders import (
    load_reference_corpus,
)
from backend.src.rag.models import Document

DEFAULT_RAG_DATA_DIRECTORY = PROJECT_ROOT / "data" / "rag" / "evaluation"

DEFAULT_RAG_BENCHMARK_DIRECTORY = PROJECT_ROOT / "benchmark_results" / "rag"

DEFAULT_RAG_CORPUS_PATH = DEFAULT_RAG_DATA_DIRECTORY / "reference_corpus.json"

DEFAULT_RAG_GOLDEN_DATASET_PATH = (
    DEFAULT_RAG_DATA_DIRECTORY / "golden_dataset_expanded.json"
)

DEFAULT_RAG_COMPARISON_PATH = DEFAULT_RAG_BENCHMARK_DIRECTORY / "comparison.json"


class RAGArtifactError(RuntimeError):
    """Base error raised while reading RAG artifacts."""


class RAGArtifactNotFoundError(RAGArtifactError):
    """Raised when an expected RAG artifact does not exist."""


class InvalidRAGArtifactError(RAGArtifactError):
    """Raised when an RAG artifact violates its expected contract."""


class RAGArtifactRepository:
    """Load RAG corpus, Golden Dataset metadata and benchmark results."""

    def __init__(
        self,
        *,
        corpus_path: Path = DEFAULT_RAG_CORPUS_PATH,
        golden_dataset_path: Path = (DEFAULT_RAG_GOLDEN_DATASET_PATH),
        comparison_path: Path = (DEFAULT_RAG_COMPARISON_PATH),
    ) -> None:
        self.corpus_path = corpus_path
        self.golden_dataset_path = golden_dataset_path
        self.comparison_path = comparison_path

    def corpus_exists(self) -> bool:
        """Return whether the reference corpus exists."""
        return self.corpus_path.is_file()

    def golden_dataset_exists(self) -> bool:
        """Return whether the Golden Dataset exists."""
        return self.golden_dataset_path.is_file()

    def comparison_exists(self) -> bool:
        """Return whether the benchmark comparison exists."""
        return self.comparison_path.is_file()

    def count_corpus_documents(self) -> int:
        """Return the number of documents in the reference corpus."""
        payload = self._load_json_object(
            self.corpus_path,
            artifact_name="RAG reference corpus",
        )

        documents = payload.get("documents")

        if not isinstance(
            documents,
            list,
        ):
            raise InvalidRAGArtifactError(
                "RAG reference corpus must contain a documents list."
            )

        return len(documents)

    def count_golden_dataset_items(self) -> int:
        """Return the number of Golden Dataset benchmark items."""
        payload = self._load_json_object(
            self.golden_dataset_path,
            artifact_name="RAG Golden Dataset",
        )

        items = payload.get("items")

        if not isinstance(
            items,
            list,
        ):
            raise InvalidRAGArtifactError(
                "RAG Golden Dataset must contain an items list."
            )

        return len(items)

    def load_documents(
        self,
    ) -> list[Document]:
        """Load and validate reference corpus documents."""
        if not self.corpus_path.is_file():
            raise RAGArtifactNotFoundError(
                f"RAG reference corpus is missing: {self.corpus_path}"
            )

        try:
            documents = load_reference_corpus(self.corpus_path)
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            JSONDecodeError,
        ) as error:
            raise InvalidRAGArtifactError(
                f"RAG reference corpus has an invalid schema: {self.corpus_path}"
            ) from error

        if not documents:
            raise InvalidRAGArtifactError("RAG reference corpus must not be empty.")

        return documents

    def load_comparison(
        self,
    ) -> RAGComparisonArtifact:
        """Load and validate the RAG benchmark comparison."""
        payload = self._load_json_object(
            self.comparison_path,
            artifact_name=("RAG benchmark comparison"),
        )

        try:
            return RAGComparisonArtifact.model_validate(payload)
        except ValidationError as error:
            raise InvalidRAGArtifactError(
                "RAG benchmark comparison has an invalid schema: "
                f"{self.comparison_path}"
            ) from error

    def _load_json_object(
        self,
        path: Path,
        *,
        artifact_name: str,
    ) -> dict[str, object]:
        if not path.is_file():
            raise RAGArtifactNotFoundError(f"{artifact_name} is missing: {path}")

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
            raise InvalidRAGArtifactError(
                f"{artifact_name} could not be read: {path}"
            ) from error

        if not isinstance(
            payload,
            dict,
        ):
            raise InvalidRAGArtifactError(
                f"{artifact_name} must contain a JSON object."
            )

        return payload

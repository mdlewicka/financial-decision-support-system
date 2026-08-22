"""Bounded availability probe for the optional local Ollama runtime."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Final, Literal, Protocol
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from backend.src.rag.config import RAGConfig

OLLAMA_HOST_ENVIRONMENT_VARIABLE: Final = "OLLAMA_HOST"
DEFAULT_OLLAMA_HOST: Final = "http://127.0.0.1:11434"


class URLResponse(Protocol):
    """Minimum response interface used by the local status probe."""

    status: int

    def read(self, amount: int = -1) -> bytes: ...

    def __enter__(self) -> URLResponse: ...

    def __exit__(self, *args: object) -> None: ...


@dataclass(frozen=True)
class OllamaRuntimeStatus:
    """Public status that intentionally omits host and local paths."""

    status: Literal["available", "unavailable"]
    required: bool
    fallback_available: bool
    model_name: str


class OllamaStatusService:
    """Check Ollama quickly without making it an application dependency."""

    def __init__(
        self,
        *,
        host: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float = 0.35,
    ) -> None:
        selected_host = host or os.getenv(
            OLLAMA_HOST_ENVIRONMENT_VARIABLE,
            DEFAULT_OLLAMA_HOST,
        )
        parsed_host = urlsplit(selected_host)
        if parsed_host.scheme not in {"http", "https"} or not parsed_host.hostname:
            raise ValueError("Ollama host must be an HTTP or HTTPS URL.")
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be greater than zero.")

        self._status_url = f"{selected_host.rstrip('/')}/api/version"
        self._model_name = model_name or RAGConfig().language_model
        self._timeout_seconds = timeout_seconds

    def evaluate(self) -> OllamaRuntimeStatus:
        """Return optional-service status; connectivity errors are expected."""
        available = False
        request = Request(
            self._status_url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read(16_384).decode("utf-8"))
                available = response.status == 200 and bool(payload.get("version"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            available = False

        return OllamaRuntimeStatus(
            status="available" if available else "unavailable",
            required=False,
            fallback_available=True,
            model_name=self._model_name,
        )

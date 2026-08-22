"""HTTP client used by the Streamlit frontend."""

from typing import Any, cast

import requests

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 10.0


class APIClientError(RuntimeError):
    """Base exception raised by the frontend API client."""


class APIConnectionError(APIClientError):
    """Raised when the FastAPI backend cannot be reached."""


class APITimeoutError(APIClientError):
    """Raised when an API request exceeds its configured timeout."""


class APIResponseError(APIClientError):
    """Raised when the backend returns an invalid response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIClient:
    """Small HTTP client for the Financial Decision Support API."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")

        if not normalized_base_url:
            raise ValueError("base_url must be a non-empty string.")

        self._validate_timeout(timeout_seconds)

        self._base_url = normalized_base_url
        self._timeout_seconds = timeout_seconds

    @property
    def base_url(self) -> str:
        """Return the normalized backend base URL."""
        return self._base_url

    @property
    def timeout_seconds(self) -> float:
        """Return the configured default request timeout."""
        return self._timeout_seconds

    def get(
        self,
        path: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Perform a GET request and return a JSON object."""
        return self._request(
            method="GET",
            path=path,
            timeout_seconds=timeout_seconds,
        )

    def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Perform a POST request and return a JSON object."""
        return self._request(
            method="POST",
            path=path,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )

    def get_system_info(
        self,
    ) -> dict[str, Any]:
        """Return versioned API information."""
        return self.get("/api/v1/system/info")

    def get_system_health(
        self,
    ) -> dict[str, Any]:
        """Return versioned API health information."""
        return self.get("/api/v1/system/health")

    def _request(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        url = self._build_url(path)

        effective_timeout = self._resolve_timeout(timeout_seconds)

        request_arguments: dict[str, Any] = {
            "method": method,
            "url": url,
            "timeout": effective_timeout,
        }

        if payload is not None:
            request_arguments["json"] = payload

        try:
            response = requests.request(**request_arguments)
        except requests.exceptions.Timeout as exc:
            raise APITimeoutError(
                "Przekroczono limit czasu oczekiwania "
                "na odpowiedź API "
                f"({effective_timeout:g} s)."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise APIConnectionError(
                f"Nie udało się połączyć z API pod adresem {self._base_url}."
            ) from exc

        if not (200 <= response.status_code < 300):
            detail = self._extract_error_detail(response)

            raise APIResponseError(
                (f"API zwróciło HTTP {response.status_code}: {detail}"),
                status_code=(response.status_code),
            )

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise APIResponseError(
                "API zwróciło odpowiedź, która nie jest poprawnym JSON-em.",
                status_code=(response.status_code),
            ) from exc

        if not isinstance(
            response_payload,
            dict,
        ):
            raise APIResponseError(
                "API zwróciło JSON o nieoczekiwanej strukturze.",
                status_code=(response.status_code),
            )

        return cast(
            dict[str, Any],
            response_payload,
        )

    def _resolve_timeout(
        self,
        timeout_seconds: float | None,
    ) -> float:
        """Return request timeout after validation."""
        if timeout_seconds is None:
            return self._timeout_seconds

        self._validate_timeout(timeout_seconds)

        return timeout_seconds

    def _build_url(
        self,
        path: str,
    ) -> str:
        normalized_path = path.strip()

        if not normalized_path:
            raise ValueError("path must be a non-empty string.")

        if not normalized_path.startswith("/"):
            normalized_path = f"/{normalized_path}"

        return f"{self._base_url}{normalized_path}"

    @staticmethod
    def _validate_timeout(
        timeout_seconds: float,
    ) -> None:
        """Validate a request timeout."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

    @staticmethod
    def _extract_error_detail(
        response: requests.Response,
    ) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(
            payload,
            dict,
        ):
            detail = payload.get("detail")

            if (
                isinstance(
                    detail,
                    str,
                )
                and detail.strip()
            ):
                return detail.strip()

        response_text = response.text.strip()

        if response_text:
            return response_text[:300]

        return "Brak szczegółów błędu."

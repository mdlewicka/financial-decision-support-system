"""Privacy-conscious structured HTTP request logging."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from time import perf_counter
from typing import Final
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.src.observability.runtime_metrics import RuntimeMetrics

REQUEST_ID_HEADER: Final = "X-Request-ID"
REQUEST_ID_HEADER_BYTES: Final = b"x-request-id"
MAXIMUM_REQUEST_ID_LENGTH: Final = 128
VALID_REQUEST_ID: Final = re.compile(r"^[A-Za-z0-9._-]+$")
LOG_LEVEL_ENVIRONMENT_VARIABLE: Final = "FDSS_LOG_LEVEL"
LOGGER_NAME: Final = "fdss"


class JSONLogFormatter(logging.Formatter):
    """Render stable machine-readable logs using the standard library."""

    EXTRA_FIELDS: Final = (
        "request_id",
        "http_method",
        "http_path",
        "http_status",
        "duration_ms",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for field_name in self.EXTRA_FIELDS:
            value = getattr(record, field_name, None)
            if value is not None:
                payload[field_name] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_structured_logging() -> logging.Logger:
    """Configure one idempotent local JSON log handler."""
    logger = logging.getLogger(LOGGER_NAME)
    level_name = os.getenv(LOG_LEVEL_ENVIRONMENT_VARIABLE, "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    if not any(handler.get_name() == "fdss-json" for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.set_name("fdss-json")
        handler.setFormatter(JSONLogFormatter())
        logger.addHandler(handler)

    return logger


def _request_id_from_scope(scope: Scope) -> str:
    for name, value in scope.get("headers", []):
        if name.lower() != REQUEST_ID_HEADER_BYTES:
            continue
        candidate = value.decode("ascii", errors="ignore")
        if (
            candidate
            and len(candidate) <= MAXIMUM_REQUEST_ID_LENGTH
            and VALID_REQUEST_ID.fullmatch(candidate)
        ):
            return candidate
    return uuid4().hex


class RequestLoggingMiddleware:
    """Log HTTP request metadata and return a correlation identifier."""

    def __init__(
        self,
        app: ASGIApp,
        logger: logging.Logger | None = None,
        metrics: RuntimeMetrics | None = None,
    ) -> None:
        self.app = app
        self.logger = logger or logging.getLogger(f"{LOGGER_NAME}.http")
        self.metrics = metrics or RuntimeMetrics()

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_scope(scope)
        started_at = perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = list(message.get("headers", []))
                headers.append(
                    (
                        REQUEST_ID_HEADER_BYTES,
                        request_id.encode("ascii"),
                    )
                )
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration_ms = round((perf_counter() - started_at) * 1000, 3)
            self.metrics.record(
                status_code=status_code,
                duration_ms=duration_ms,
            )
            self.logger.info(
                "http_request_completed",
                extra={
                    "request_id": request_id,
                    "http_method": scope.get("method", ""),
                    "http_path": scope.get("path", ""),
                    "http_status": status_code,
                    "duration_ms": duration_ms,
                },
            )

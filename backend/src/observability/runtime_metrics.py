"""Small process-local HTTP metrics without an external monitoring service."""

from __future__ import annotations

import math
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RuntimeMetricsSnapshot:
    """Immutable aggregate runtime metrics safe for API presentation."""

    requests_total: int
    errors_total: int
    client_errors_total: int
    server_errors_total: int
    average_latency_ms: float
    maximum_latency_ms: float


class RuntimeMetrics:
    """Collect bounded aggregate request metrics for one API process."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests_total = 0
        self._client_errors_total = 0
        self._server_errors_total = 0
        self._total_latency_ms = 0.0
        self._maximum_latency_ms = 0.0

    def record(
        self,
        *,
        status_code: int,
        duration_ms: float,
    ) -> None:
        """Record one completed request without storing user data or paths."""
        if not 100 <= status_code <= 599:
            raise ValueError("status_code must be between 100 and 599.")
        if not math.isfinite(duration_ms) or duration_ms < 0.0:
            raise ValueError("duration_ms must be finite and non-negative.")

        with self._lock:
            self._requests_total += 1
            self._total_latency_ms += duration_ms
            self._maximum_latency_ms = max(
                self._maximum_latency_ms,
                duration_ms,
            )
            if 400 <= status_code <= 499:
                self._client_errors_total += 1
            elif status_code >= 500:
                self._server_errors_total += 1

    def snapshot(self) -> RuntimeMetricsSnapshot:
        """Return a consistent point-in-time aggregate snapshot."""
        with self._lock:
            request_count = self._requests_total
            average_latency = (
                self._total_latency_ms / request_count if request_count else 0.0
            )
            return RuntimeMetricsSnapshot(
                requests_total=request_count,
                errors_total=(self._client_errors_total + self._server_errors_total),
                client_errors_total=self._client_errors_total,
                server_errors_total=self._server_errors_total,
                average_latency_ms=round(average_latency, 3),
                maximum_latency_ms=round(self._maximum_latency_ms, 3),
            )

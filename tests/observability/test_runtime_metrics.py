import math

import pytest

from backend.src.observability.runtime_metrics import RuntimeMetrics


def test_runtime_metrics_aggregate_requests_errors_and_latency() -> None:
    metrics = RuntimeMetrics()

    metrics.record(status_code=200, duration_ms=10.0)
    metrics.record(status_code=404, duration_ms=20.0)
    metrics.record(status_code=503, duration_ms=30.0)

    assert metrics.snapshot() == metrics.snapshot().__class__(
        requests_total=3,
        errors_total=2,
        client_errors_total=1,
        server_errors_total=1,
        average_latency_ms=20.0,
        maximum_latency_ms=30.0,
    )


@pytest.mark.parametrize("duration", [-1.0, math.nan, math.inf])
def test_runtime_metrics_reject_invalid_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        RuntimeMetrics().record(status_code=200, duration_ms=duration)


def test_empty_runtime_metrics_snapshot_is_zeroed() -> None:
    snapshot = RuntimeMetrics().snapshot()

    assert snapshot.requests_total == 0
    assert snapshot.errors_total == 0
    assert snapshot.average_latency_ms == 0.0

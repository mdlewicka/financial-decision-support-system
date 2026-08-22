import json
import logging

from backend.src.observability.request_logging import (
    JSONLogFormatter,
    configure_structured_logging,
)


def test_json_formatter_emits_expected_safe_request_metadata() -> None:
    record = logging.LogRecord(
        name="fdss.http",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="http_request_completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-123"
    record.http_method = "GET"
    record.http_path = "/api/v1/system/health"
    record.http_status = 200
    record.duration_ms = 1.25

    payload = json.loads(JSONLogFormatter().format(record))

    assert payload["event"] == "http_request_completed"
    assert payload["request_id"] == "request-123"
    assert payload["http_method"] == "GET"
    assert payload["http_path"] == "/api/v1/system/health"
    assert payload["http_status"] == 200
    assert payload["duration_ms"] == 1.25
    assert "query" not in payload
    assert "headers" not in payload
    assert "body" not in payload


def test_structured_logging_configuration_is_idempotent() -> None:
    logger = configure_structured_logging()
    configure_structured_logging()

    assert sum(handler.get_name() == "fdss-json" for handler in logger.handlers) == 1

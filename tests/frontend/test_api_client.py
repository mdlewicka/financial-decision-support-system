from typing import Any

import pytest
import requests

from frontend.api_client import (
    APIClient,
    APIConnectionError,
    APIResponseError,
    APITimeoutError,
)


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Any = None,
        text: str = "",
        invalid_json: bool = False,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._invalid_json = invalid_json

    def json(
        self,
    ) -> Any:
        if self._invalid_json:
            raise ValueError("Invalid JSON")

        return self._payload


def test_client_normalizes_base_url() -> None:
    client = APIClient("http://127.0.0.1:8000/")

    assert client.base_url == "http://127.0.0.1:8000"


def test_system_health_uses_versioned_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[
        str,
        Any,
    ] = {}

    def fake_request(
        **kwargs: Any,
    ) -> FakeResponse:
        captured.update(kwargs)

        return FakeResponse(
            payload={
                "status": "healthy",
            }
        )

    monkeypatch.setattr(
        requests,
        "request",
        fake_request,
    )

    client = APIClient()

    result = client.get_system_health()

    assert result == {
        "status": "healthy",
    }

    assert captured["method"] == "GET"

    assert captured["url"] == ("http://127.0.0.1:8000/api/v1/system/health")

    assert captured["timeout"] == 10.0


def test_post_sends_json_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[
        str,
        Any,
    ] = {}

    def fake_request(
        **kwargs: Any,
    ) -> FakeResponse:
        captured.update(kwargs)

        return FakeResponse(
            payload={
                "result": "ok",
            }
        )

    monkeypatch.setattr(
        requests,
        "request",
        fake_request,
    )

    client = APIClient()

    result = client.post(
        "/api/v1/example",
        {
            "question": "test",
        },
    )

    assert result == {
        "result": "ok",
    }

    assert captured["method"] == "POST"

    assert captured["json"] == {
        "question": "test",
    }

    assert captured["timeout"] == 10.0


def test_post_can_override_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[
        str,
        Any,
    ] = {}

    def fake_request(
        **kwargs: Any,
    ) -> FakeResponse:
        captured.update(kwargs)

        return FakeResponse(
            payload={
                "result": "ok",
            }
        )

    monkeypatch.setattr(
        requests,
        "request",
        fake_request,
    )

    client = APIClient()

    result = client.post(
        "/api/v1/agents/allocation",
        {
            "question": "test",
            "risk_preference": "balanced",
        },
        timeout_seconds=60.0,
    )

    assert result == {
        "result": "ok",
    }

    assert captured["timeout"] == 60.0


def test_get_can_override_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[
        str,
        Any,
    ] = {}

    def fake_request(
        **kwargs: Any,
    ) -> FakeResponse:
        captured.update(kwargs)

        return FakeResponse(
            payload={
                "result": "ok",
            }
        )

    monkeypatch.setattr(
        requests,
        "request",
        fake_request,
    )

    client = APIClient()

    result = client.get(
        "/api/v1/example",
        timeout_seconds=30.0,
    )

    assert result == {
        "result": "ok",
    }

    assert captured["timeout"] == 30.0


def test_connection_error_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request(
        **kwargs: Any,
    ) -> FakeResponse:
        del kwargs

        raise (requests.exceptions.ConnectionError("Connection refused"))

    monkeypatch.setattr(
        requests,
        "request",
        fake_request,
    )

    client = APIClient()

    with pytest.raises(
        APIConnectionError,
        match=("Nie udało się połączyć"),
    ):
        client.get_system_health()


def test_timeout_error_is_distinguished_from_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request(
        **kwargs: Any,
    ) -> FakeResponse:
        del kwargs

        raise requests.exceptions.Timeout("Request timed out")

    monkeypatch.setattr(
        requests,
        "request",
        fake_request,
    )

    client = APIClient()

    with pytest.raises(
        APITimeoutError,
        match=("Przekroczono limit czasu"),
    ):
        client.post(
            "/api/v1/agents/allocation",
            {
                "question": "test",
            },
            timeout_seconds=60.0,
        )


def test_timeout_error_contains_effective_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request(
        **kwargs: Any,
    ) -> FakeResponse:
        del kwargs

        raise requests.exceptions.Timeout("Request timed out")

    monkeypatch.setattr(
        requests,
        "request",
        fake_request,
    )

    client = APIClient()

    with pytest.raises(
        APITimeoutError,
        match="60 s",
    ):
        client.post(
            "/api/v1/example",
            {
                "question": "test",
            },
            timeout_seconds=60.0,
        )


def test_http_error_contains_backend_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request(
        **kwargs: Any,
    ) -> FakeResponse:
        del kwargs

        return FakeResponse(
            status_code=404,
            payload={
                "detail": ("Unknown resource"),
            },
        )

    monkeypatch.setattr(
        requests,
        "request",
        fake_request,
    )

    client = APIClient()

    with pytest.raises(
        APIResponseError,
        match="Unknown resource",
    ) as error:
        client.get("/api/v1/missing")

    assert error.value.status_code == 404


def test_invalid_json_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request(
        **kwargs: Any,
    ) -> FakeResponse:
        del kwargs

        return FakeResponse(invalid_json=True)

    monkeypatch.setattr(
        requests,
        "request",
        fake_request,
    )

    client = APIClient()

    with pytest.raises(
        APIResponseError,
        match="poprawnym JSON-em",
    ):
        client.get_system_info()


def test_non_object_json_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request(
        **kwargs: Any,
    ) -> FakeResponse:
        del kwargs

        return FakeResponse(
            payload=[
                "unexpected",
            ]
        )

    monkeypatch.setattr(
        requests,
        "request",
        fake_request,
    )

    client = APIClient()

    with pytest.raises(
        APIResponseError,
        match=("nieoczekiwanej strukturze"),
    ):
        client.get_system_info()


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        " ",
    ],
)
def test_client_rejects_empty_base_url(
    base_url: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="base_url",
    ):
        APIClient(base_url)


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0.0,
        -1.0,
    ],
)
def test_client_rejects_non_positive_default_timeout(
    timeout_seconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="timeout_seconds",
    ):
        APIClient(timeout_seconds=(timeout_seconds))


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0.0,
        -1.0,
    ],
)
def test_request_rejects_non_positive_timeout_override(
    timeout_seconds: float,
) -> None:
    client = APIClient()

    with pytest.raises(
        ValueError,
        match="timeout_seconds",
    ):
        client.post(
            "/api/v1/example",
            {
                "question": "test",
            },
            timeout_seconds=(timeout_seconds),
        )

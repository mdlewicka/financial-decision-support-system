import json

import pytest

from backend.src.observability.ollama_status import OllamaStatusService


class FakeResponse:
    status = 200

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, amount: int = -1) -> bytes:
        return json.dumps({"version": "0.test"}).encode("utf-8")


def test_ollama_status_reports_available_without_exposing_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backend.src.observability.ollama_status.urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    result = OllamaStatusService(
        host="http://private-host:11434",
        model_name="local-model",
    ).evaluate()

    assert result.status == "available"
    assert result.required is False
    assert result.fallback_available is True
    assert "private-host" not in repr(result)


def test_ollama_status_degrades_safely_when_local_service_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> FakeResponse:
        raise OSError("offline")

    monkeypatch.setattr(
        "backend.src.observability.ollama_status.urlopen",
        unavailable,
    )

    result = OllamaStatusService().evaluate()

    assert result.status == "unavailable"
    assert result.fallback_available is True


@pytest.mark.parametrize("host", ["file:///tmp/socket", "not-a-url"])
def test_ollama_status_rejects_non_http_host(host: str) -> None:
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        OllamaStatusService(host=host)

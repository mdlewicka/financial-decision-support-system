import json
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
COMPOSE_FILE = PROJECT_ROOT / "compose.yaml"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
RUNBOOK = PROJECT_ROOT / "docs" / "CONTAINER_RUNBOOK.md"


def _compose() -> dict[str, object]:
    return json.loads(COMPOSE_FILE.read_text(encoding="utf-8"))


def test_dockerfile_has_pinned_shared_runtime_and_non_root_user() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG PYTHON_IMAGE=python:3.12.10-slim" in dockerfile
    assert "FROM ${PYTHON_IMAGE} AS runtime" in dockerfile
    assert "FROM runtime AS backend" in dockerfile
    assert "FROM runtime AS frontend" in dockerfile
    assert "USER app" in dockerfile
    assert "--uid 10001" in dockerfile
    assert "--requirement requirements.txt" in dockerfile
    assert "--reload" not in dockerfile


def test_dockerfile_contains_service_commands_and_healthchecks() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "backend.src.main:app" in dockerfile
    assert '"0.0.0.0", "--port", "8000"' in dockerfile
    assert '"streamlit", "run", "frontend/app.py"' in dockerfile
    assert "/api/v1/system/readiness" in dockerfile
    assert "/_stcore/health" in dockerfile
    assert "FINANCIAL_DECISION_SUPPORT_API_URL=http://backend:8000" in dockerfile


def test_compose_defines_two_local_services_and_dependency_gate() -> None:
    compose = _compose()
    services = compose["services"]
    assert isinstance(services, dict)
    assert set(services) == {"backend", "frontend"}

    backend = services["backend"]
    frontend = services["frontend"]
    assert isinstance(backend, dict)
    assert isinstance(frontend, dict)

    assert backend["build"]["target"] == "backend"
    assert frontend["build"]["target"] == "frontend"
    assert backend["ports"] == ["127.0.0.1:8000:8000"]
    assert frontend["ports"] == ["127.0.0.1:8501:8501"]
    assert frontend["depends_on"] == {"backend": {"condition": "service_healthy"}}
    assert frontend["environment"]["FINANCIAL_DECISION_SUPPORT_API_URL"] == (
        "http://backend:8000"
    )


def test_compose_healthchecks_and_local_cache_volumes_are_explicit() -> None:
    compose = _compose()
    services = compose["services"]
    assert isinstance(services, dict)

    backend = services["backend"]
    frontend = services["frontend"]
    assert isinstance(backend, dict)
    assert isinstance(frontend, dict)

    assert "/api/v1/system/readiness" in " ".join(backend["healthcheck"]["test"])
    assert "/_stcore/health" in " ".join(frontend["healthcheck"]["test"])
    assert "market-cache:/app/data/cache" in backend["volumes"]
    assert "model-cache:/home/app/.cache" in backend["volumes"]

    volumes = compose["volumes"]
    assert isinstance(volumes, dict)
    assert set(volumes) == {"market-cache", "model-cache"}

    assert backend["init"] is True
    assert frontend["init"] is True


def test_docker_context_excludes_local_and_sensitive_artifacts() -> None:
    patterns = set(DOCKERIGNORE.read_text(encoding="utf-8").splitlines())

    required_patterns = {
        ".git",
        ".venv",
        "**/__pycache__",
        "tests",
        "notebooks",
        ".ollama",
        ".cache",
        "*.zip",
        "data/*",
        "!data/rag/**",
    }

    assert required_patterns.issubset(patterns)


def test_runbook_documents_start_health_logs_costs_and_shutdown() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")

    for fragment in (
        "docker compose build",
        "docker compose up --detach",
        "docker compose ps",
        "docker compose logs --follow",
        "docker compose down",
        "http://127.0.0.1:8501",
        "does not use paid APIs",
        "Optional local Ollama",
    ):
        assert fragment in runbook


def test_compose_file_is_accepted_by_installed_docker() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not installed in this test environment.")

    result = subprocess.run(
        [
            "docker",
            "compose",
            "--file",
            str(COMPOSE_FILE),
            "config",
            "--quiet",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr

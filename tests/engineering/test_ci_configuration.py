from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
RUNBOOK = PROJECT_ROOT / "docs" / "CI_RUNBOOK.md"


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_ci_triggers_are_bounded_and_permissions_are_read_only() -> None:
    workflow = _workflow()

    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "contents: read" in workflow
    assert "pull_request_target:" not in workflow
    assert "persist-credentials: false" in workflow
    assert "secrets." not in workflow


def test_ci_uses_current_official_actions_and_pinned_python() -> None:
    workflow = _workflow()

    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert 'python-version: "3.12.10"' in workflow
    assert "requirements-dev.txt" in workflow
    assert "ruff==0.16.1" in workflow


def test_ci_runs_quality_and_split_regression_gates() -> None:
    workflow = _workflow()

    for command in (
        "python -m ruff check .",
        "python -m ruff format --check .",
        "python -m pytest tests -q",
        "--ignore=tests/rl",
        "tests/rl",
        "test_rl_allocation_workflow_integration.py",
        "test_portfolio_allocation_env_integration.py",
    ):
        assert command in workflow

    assert 'OMP_NUM_THREADS: "1"' in workflow
    assert 'TOKENIZERS_PARALLELISM: "false"' in workflow


def test_ci_builds_and_smoke_tests_local_containers() -> None:
    workflow = _workflow()

    for command in (
        "docker compose config --quiet",
        "docker compose build",
        "docker compose up --detach --wait --wait-timeout 180",
        "/api/v1/system/liveness",
        "/api/v1/system/readiness",
        "/_stcore/health",
        "docker compose down --volumes",
    ):
        assert command in workflow


def test_ci_runbook_documents_local_reproduction_security_and_costs() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")

    for fragment in (
        "does not deploy",
        "require repository secrets",
        "read-only repository permissions",
        "Local reproduction",
        "Cost boundary",
        "Public repositories",
        "Private repositories",
        "branch protection",
    ):
        assert fragment in runbook

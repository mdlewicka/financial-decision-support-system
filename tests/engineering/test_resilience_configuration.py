from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = PROJECT_ROOT / "docs" / "RESILIENCE_RUNBOOK.md"


def test_resilience_runbook_covers_every_mandatory_failure_mode() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")

    for required_evidence in (
        "Corrupted data",
        "NaN and infinity",
        "Extreme parameters",
        "Prompt injection",
        "Question outside corpus",
        "Ollama unavailable",
    ):
        assert required_evidence in runbook


def test_runtime_status_contract_documents_privacy_and_fallback() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")

    for requirement in (
        "/api/v1/system/runtime-status",
        "error counts and latency",
        "never returns request bodies",
        "Ollama is optional",
        "Metrics reset when the backend process restarts",
    ):
        assert requirement in runbook


def test_gated_infrastructure_is_explicitly_out_of_scope() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")

    assert "does not add MLflow, SQLite or PostgreSQL" in runbook
    assert "no confirmed requirement" in runbook

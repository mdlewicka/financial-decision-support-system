from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIRECTORY = PROJECT_ROOT / "docs"

FINAL_DOCUMENTS = (
    "ARCHITECTURE.md",
    "CI_RUNBOOK.md",
    "CONTAINER_RUNBOOK.md",
    "FORECASTING_V2_EXPERIMENT_PROTOCOL.md",
    "FORECASTING_V2_RESULTS.md",
    "RESILIENCE_RUNBOOK.md",
    "SECURITY_AND_COSTS.md",
    "V1_BASELINE.md",
)


def read_document(file_name: str) -> str:
    return (DOCS_DIRECTORY / file_name).read_text(encoding="utf-8")


def test_final_documentation_files_exist_and_are_substantive() -> None:
    for file_name in FINAL_DOCUMENTS:
        path = DOCS_DIRECTORY / file_name

        assert path.is_file(), f"Missing documentation file: {file_name}"
        assert len(path.read_text(encoding="utf-8")) >= 500, (
            f"Documentation file is unexpectedly short: {file_name}"
        )


def test_architecture_documents_decision_support_boundaries() -> None:
    content = read_document("ARCHITECTURE.md")

    assert "Deterministic analytical core" in content
    assert "The language model explains an already computed decision" in content
    assert "training -> fitting and portfolio construction" in content


def test_security_document_describes_cost_and_security_constraints() -> None:
    content = read_document("SECURITY_AND_COSTS.md")

    assert "does not require a paid" in content
    assert "language-model API" in content
    assert "does not mean zero total cost" in content
    assert "Public hosting is conditional" in content


def test_forecasting_v2_protocol_and_results_are_present() -> None:
    protocol = read_document("FORECASTING_V2_EXPERIMENT_PROTOCOL.md")
    results = read_document("FORECASTING_V2_RESULTS.md")

    assert len(protocol) >= 500
    assert len(results) >= 500


def test_v1_baseline_is_present() -> None:
    baseline = read_document("V1_BASELINE.md")

    assert len(baseline) >= 500


def test_operational_runbooks_are_present() -> None:
    for file_name in (
        "CI_RUNBOOK.md",
        "CONTAINER_RUNBOOK.md",
        "RESILIENCE_RUNBOOK.md",
    ):
        content = read_document(file_name)
        assert len(content) >= 500

import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

from backend.src.reporting.decision_report import (
    ArtifactVersion,
    DecisionReport,
    ReportAssumptions,
    ReportEvidence,
    ReportGovernance,
    ReportMetadata,
    ReportNarrative,
    ReportScenario,
    ReportXAI,
)
from backend.src.reporting.decision_report_exporter import (
    export_decision_report,
    render_decision_report_markdown,
)


def _report() -> DecisionReport:
    artifacts = (
        ArtifactVersion("decision_result.json", "1", "a" * 64),
        ArtifactVersion("xai_result.json", "1", "b" * 64),
        ArtifactVersion("narrative_evaluation.json", "1", "c" * 64),
    )
    return DecisionReport(
        metadata=ReportMetadata(
            schema_version=1,
            generated_at_utc="2026-08-14T09:30:00+00:00",
            presentation_timezone="Europe/Warsaw",
            evaluation_split="validation",
            test_split_used=False,
            source_artifacts=artifacts,
        ),
        assumptions=ReportAssumptions(
            profile_type="balanced",
            priorities={
                "return": 0.25,
                "volatility": 0.25,
                "drawdown": 0.25,
                "diversification": 0.25,
            },
            constraints={"maximum_drawdown": 0.35},
            liquidity_need="medium",
            investment_horizon_years=7,
            available_liquidity_fraction=0.2,
        ),
        recommended_scenario="minimum_volatility",
        scenarios=(
            ReportScenario(
                name="minimum_volatility",
                construction="Constrained minimum volatility",
                weights={"AGG": 0.75, "SPY": 0.25},
                admissible=True,
                rank=1,
                total_score=0.8,
                constraint_violations=(),
                tradeoffs=("Lower expected return.",),
            ),
            ReportScenario(
                name="growth",
                construction="Manual growth allocation",
                weights={"AGG": 0.2, "SPY": 0.8},
                admissible=False,
                rank=None,
                total_score=None,
                constraint_violations=("maximum_drawdown",),
                tradeoffs=("Higher expected return.",),
            ),
        ),
        narrative=ReportNarrative(
            agent_name="allocation_recommendation_agent",
            question="Który wariant najlepiej odpowiada założeniom?",
            summary="Minimum volatility zajmuje pierwsze miejsce.",
            findings=("Ranking pochodzi z danych walidacyjnych.",),
            risks=("Wyniki historyczne nie gwarantują rezultatów.",),
            recommendations=("Wymagany jest przegląd człowieka.",),
            limitations=("Wynik nie jest indywidualną poradą finansową.",),
        ),
        evidence=(
            ReportEvidence(
                "decision_engine",
                "current_decision_evaluation",
                "Deterministic validation ranking",
            ),
            ReportEvidence(
                "rag_document",
                "portfolio-risk-reference",
                "Local corpus; chunk=chunk-1; rank=1; score=0.8",
            ),
        ),
        xai=ReportXAI(
            model_name="random_forest",
            explanation_split="validation",
            target_definition="next_session_return",
            global_observations=400,
            top_global_shap_features=(
                {
                    "feature": "momentum_20",
                    "mean_absolute_shap_value": 0.00018,
                },
            ),
            partial_dependence_features=("momentum_20",),
            accumulated_local_effect_features=("momentum_20",),
            scope=(
                "Forecast-model interpretation only; this evidence does not "
                "determine or modify the portfolio Decision Engine ranking."
            ),
        ),
        governance=ReportGovernance(
            deterministic_ranking=True,
            llm_used_for_ranking=False,
            portfolio_weights_modified=False,
            narrative_evaluation_status="approved",
            deterministic_gate_passed=True,
            llm_judge_approved_for_reporting=True,
            chain_of_thought_stored=False,
            human_review_required=True,
        ),
    )


def test_markdown_contains_all_auditable_sections() -> None:
    markdown = render_decision_report_markdown(_report())

    for heading in (
        "## Założenia użytkownika",
        "## Rekomendacja i ranking",
        "## Wyjaśnienie agenta",
        "## XAI - kontekst modelu prognostycznego",
        "## Źródła i cytowania",
        "## Governance",
        "## Ograniczenia",
        "## Wersje artefaktów źródłowych",
    ):
        assert heading in markdown
    assert "Rekomendowany scenariusz: **minimum_volatility**" in markdown
    assert "Końcowy zbiór testowy użyty: NIE" in markdown
    assert "Chain-of-thought zapisany: NIE" in markdown
    assert "portfolio-risk-reference" in markdown


def test_export_writes_json_markdown_pdf_and_hash_manifest(tmp_path: Path) -> None:
    paths = export_decision_report(_report(), tmp_path / "report")

    assert paths.result_json.is_file()
    assert paths.report_markdown.is_file()
    assert paths.report_pdf.is_file()
    assert paths.manifest_json.is_file()

    payload = json.loads(paths.result_json.read_text(encoding="utf-8"))
    assert payload["recommended_scenario"] == "minimum_volatility"
    assert payload["metadata"]["test_split_used"] is False

    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    assert set(manifest["outputs"]) == {"json", "markdown", "pdf"}
    assert len(manifest["source_artifacts"]) == 3
    for output in manifest["outputs"].values():
        exported_path = paths.manifest_json.parent / output["file_name"]
        assert output["bytes"] == exported_path.stat().st_size
        assert output["sha256"] == _sha256(exported_path)


def test_pdf_is_readable_and_contains_decision_content(tmp_path: Path) -> None:
    paths = export_decision_report(_report(), tmp_path)
    reader = PdfReader(paths.report_pdf)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert len(reader.pages) >= 2
    assert "Financial Decision Support System" in text
    assert "minimum_volatility" in text
    assert "Governance" in text
    assert "Chain-of-thought zapisany: NIE" in text

    metadata = reader.metadata
    assert metadata is not None
    assert metadata.title == ("Financial Decision Support System - Decision Report")


def test_pdf_does_not_orphan_short_scenario_or_governance_sections(
    tmp_path: Path,
) -> None:
    paths = export_decision_report(_report(), tmp_path)
    page_texts = tuple(
        page.extract_text() or "" for page in PdfReader(paths.report_pdf).pages
    )

    growth_page = next(text for text in page_texts if "growth" in text)
    assert "AGG: 20.00%" in growth_page
    assert "SPY: 80.00%" in growth_page
    governance_page = next(text for text in page_texts if "Governance" in text)
    assert "Wymagany przegląd człowieka: TAK" in governance_page


def test_repeated_export_is_byte_reproducible(tmp_path: Path) -> None:
    first = export_decision_report(_report(), tmp_path / "first")
    second = export_decision_report(_report(), tmp_path / "second")

    assert first.result_json.read_bytes() == second.result_json.read_bytes()
    assert first.report_markdown.read_bytes() == second.report_markdown.read_bytes()
    assert first.report_pdf.read_bytes() == second.report_pdf.read_bytes()
    first_manifest = json.loads(first.manifest_json.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_json.read_text(encoding="utf-8"))
    assert first_manifest == second_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

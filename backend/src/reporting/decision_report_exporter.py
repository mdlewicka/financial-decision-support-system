"""Deterministic JSON, Markdown and PDF exports for decision reports."""

from __future__ import annotations

import hashlib
import json
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from backend.src.reporting.decision_report import DecisionReport

_REPORT_TITLE = "Financial Decision Support System"
_PDF_WIDTH_INCHES = 8.27
_PDF_HEIGHT_INCHES = 11.69
_PDF_TOP = 0.95
_PDF_BOTTOM = 0.07


@dataclass(frozen=True, slots=True)
class DecisionReportExportPaths:
    """Paths of the four reproducible decision-report artifacts."""

    result_json: Path
    report_markdown: Path
    report_pdf: Path
    manifest_json: Path


def export_decision_report(
    report: DecisionReport,
    output_directory: Path,
) -> DecisionReportExportPaths:
    """Write equivalent JSON, Markdown and PDF views plus a hash manifest."""
    if output_directory.exists() and not output_directory.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = DecisionReportExportPaths(
        result_json=output_directory / "decision_report.json",
        report_markdown=output_directory / "decision_report.md",
        report_pdf=output_directory / "decision_report.pdf",
        manifest_json=output_directory / "decision_report_manifest.json",
    )

    payload = report.to_dict()
    paths.result_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown = render_decision_report_markdown(report)
    paths.report_markdown.write_text(markdown, encoding="utf-8")
    _write_pdf(paths.report_pdf, report, markdown)

    output_files = (
        ("json", paths.result_json),
        ("markdown", paths.report_markdown),
        ("pdf", paths.report_pdf),
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": report.metadata.generated_at_utc,
        "report_schema_version": report.metadata.schema_version,
        "source_artifacts": [
            {
                "identifier": item.identifier,
                "schema_version": item.schema_version,
                "sha256": item.sha256,
            }
            for item in report.metadata.source_artifacts
        ],
        "outputs": {
            label: {
                "file_name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for label, path in output_files
        },
    }
    paths.manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def render_decision_report_markdown(report: DecisionReport) -> str:
    """Render the typed report without recalculating any analytical result."""
    assumptions = report.assumptions
    metadata = report.metadata
    narrative = report.narrative
    governance = report.governance
    lines = [
        f"# {_REPORT_TITLE}",
        "",
        "## Audytowalny raport decyzyjny",
        "",
        f"- Wygenerowano UTC: {metadata.generated_at_utc}",
        f"- Strefa prezentacji: {metadata.presentation_timezone}",
        f"- Zakres ewaluacji: {metadata.evaluation_split}",
        f"- Końcowy zbiór testowy użyty: {_yes_no(metadata.test_split_used)}",
        f"- Wersja schematu raportu: {metadata.schema_version}",
        "",
        "## Założenia użytkownika",
        "",
        f"- Profil: {assumptions.profile_type}",
        f"- Potrzeba płynności: {assumptions.liquidity_need}",
        f"- Horyzont inwestycyjny: {assumptions.investment_horizon_years} lat",
        (
            "- Dostępna część płynna: "
            f"{_percentage(assumptions.available_liquidity_fraction)}"
        ),
        "",
        "### Priorytety scoringu",
        "",
        *(
            f"- {name}: {_percentage(value)}"
            for name, value in sorted(assumptions.priorities.items())
        ),
        "",
        "### Ograniczenia",
        "",
        *(
            f"- {name}: {_optional_number(value)}"
            for name, value in sorted(assumptions.constraints.items())
        ),
        "",
        "## Rekomendacja i ranking",
        "",
        f"- Rekomendowany scenariusz: **{report.recommended_scenario}**",
        "- Ranking został obliczony deterministycznie na walidacji.",
        "- Warstwa narracyjna nie zmieniła rankingu ani wag.",
        "",
    ]
    for scenario in report.scenarios:
        status = "dopuszczony" if scenario.admissible else "odrzucony"
        lines.extend(
            [
                f"### {scenario.name}",
                "",
                f"- Konstrukcja: {scenario.construction}",
                f"- Status: {status}",
                f"- Pozycja: {scenario.rank if scenario.rank is not None else '-'}",
                f"- Wynik: {_optional_number(scenario.total_score)}",
                "- Wagi:",
                *(
                    f"  - {ticker}: {_percentage(weight)}"
                    for ticker, weight in scenario.weights.items()
                ),
                "- Naruszenia ograniczeń:",
                *_items_or_none(scenario.constraint_violations, indent="  "),
                "- Kompromisy:",
                *_items_or_none(scenario.tradeoffs, indent="  "),
                "",
            ]
        )

    lines.extend(
        [
            "## Wyjaśnienie agenta",
            "",
            f"- Agent: {narrative.agent_name}",
            f"- Pytanie: {narrative.question}",
            "",
            narrative.summary,
            "",
            "### Ustalenia",
            "",
            *(f"- {item}" for item in narrative.findings),
            "",
            "### Ryzyka",
            "",
            *(f"- {item}" for item in narrative.risks),
            "",
            "### Dalsze działania",
            "",
            *(f"- {item}" for item in narrative.recommendations),
            "",
            "## XAI - kontekst modelu prognostycznego",
            "",
            f"- Model: {report.xai.model_name}",
            f"- Zbiór wyjaśnień: {report.xai.explanation_split}",
            f"- Definicja celu: {report.xai.target_definition}",
            f"- Liczba obserwacji globalnych: {report.xai.global_observations}",
            f"- Granica interpretacji: {report.xai.scope}",
            "",
            "### Najważniejsze cechy SHAP",
            "",
            *(
                f"- {item['feature']}: {float(item['mean_absolute_shap_value']):.8f}"
                for item in report.xai.top_global_shap_features
            ),
            "",
            "### PDP i ALE",
            "",
            ("- PDP: " + _joined_or_none(report.xai.partial_dependence_features)),
            ("- ALE: " + _joined_or_none(report.xai.accumulated_local_effect_features)),
            "",
            "## Źródła i cytowania",
            "",
            *(
                f"- [{item.source_type}] {item.identifier} - {item.description}"
                for item in report.evidence
            ),
            "",
            "## Governance",
            "",
            f"- Ranking deterministyczny: {_yes_no(governance.deterministic_ranking)}",
            (
                "- LLM wykorzystany do rankingu: "
                f"{_yes_no(governance.llm_used_for_ranking)}"
            ),
            (
                "- Wagi portfela zmodyfikowane przez narrację: "
                f"{_yes_no(governance.portfolio_weights_modified)}"
            ),
            (
                "- Bramka deterministyczna narracji: "
                f"{_passed_failed(governance.deterministic_gate_passed)}"
            ),
            (
                "- Status lokalnego sędziego LLM: "
                f"{governance.narrative_evaluation_status}"
            ),
            (
                "- LLM zatwierdzony do raportowania: "
                f"{_yes_no(governance.llm_judge_approved_for_reporting)}"
            ),
            (
                "- Chain-of-thought zapisany: "
                f"{_yes_no(governance.chain_of_thought_stored)}"
            ),
            (
                "- Wymagany przegląd człowieka: "
                f"{_yes_no(governance.human_review_required)}"
            ),
            "",
            "## Ograniczenia",
            "",
            *(f"- {item}" for item in narrative.limitations),
            "- Wyniki historyczne i walidacyjne nie gwarantują przyszłych wyników.",
            "- Raport nie uruchamia transakcji ani automatycznej zmiany alokacji.",
            "",
            "## Wersje artefaktów źródłowych",
            "",
            *(
                f"- {item.identifier}; schema={item.schema_version}; "
                f"sha256={item.sha256}"
                for item in metadata.source_artifacts
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_pdf(path: Path, report: DecisionReport, markdown: str) -> None:
    generated_at = datetime.fromisoformat(report.metadata.generated_at_utc)
    metadata: dict[str, Any] = {
        "Title": f"{_REPORT_TITLE} - Decision Report",
        "Author": _REPORT_TITLE,
        "Subject": "Validation-only human-reviewed decision support report",
        "Keywords": "decision support, validation, XAI, RAG, governance",
        "Creator": "Financial Decision Support System reporting module",
        "CreationDate": generated_at,
        "ModDate": generated_at,
    }
    blocks = _markdown_blocks(markdown)
    pages = _paginate(blocks)
    with PdfPages(path, metadata=metadata) as pdf:
        for page_number, page in enumerate(pages, start=1):
            figure = Figure(figsize=(_PDF_WIDTH_INCHES, _PDF_HEIGHT_INCHES))
            y = _PDF_TOP
            for text, size, weight, spacing in page:
                figure.text(
                    0.08,
                    y,
                    text,
                    fontsize=size,
                    fontweight=weight,
                    family="DejaVu Sans",
                    va="top",
                )
                y -= spacing
            figure.text(
                0.5,
                0.035,
                f"{_REPORT_TITLE} | {page_number}/{len(pages)}",
                fontsize=8,
                family="DejaVu Sans",
                ha="center",
            )
            pdf.savefig(figure)


def _markdown_blocks(markdown: str) -> list[tuple[str, int, str, float]]:
    blocks: list[tuple[str, int, str, float]] = []
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line:
            blocks.append(("", 9, "normal", 0.012))
            continue
        heading_level = len(line) - len(line.lstrip("#"))
        if heading_level:
            text = line[heading_level:].strip()
            size = {1: 18, 2: 14, 3: 11}.get(heading_level, 10)
            width = {1: 58, 2: 72, 3: 88}.get(heading_level, 94)
            weight = "bold"
        else:
            text = _plain_markdown(line)
            size = 9
            width = 105
            weight = "normal"
        wrapped = textwrap.wrap(
            text,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        spacing = (size / 72.0) / _PDF_HEIGHT_INCHES * 1.45
        blocks.extend((item, size, weight, spacing) for item in wrapped)
    return blocks


def _paginate(
    blocks: list[tuple[str, int, str, float]],
) -> list[list[tuple[str, int, str, float]]]:
    pages: list[list[tuple[str, int, str, float]]] = []
    page: list[tuple[str, int, str, float]] = []
    remaining = _PDF_TOP - _PDF_BOTTOM
    for index, block in enumerate(blocks):
        spacing = block[3]
        required_space = _heading_section_space(blocks, index)
        if page and required_space is not None and remaining < required_space:
            pages.append(page)
            page = []
            remaining = _PDF_TOP - _PDF_BOTTOM
        if page and remaining - spacing < 0.0:
            pages.append(page)
            page = []
            remaining = _PDF_TOP - _PDF_BOTTOM
        page.append(block)
        remaining -= spacing
    if page:
        pages.append(page)
    if not pages:
        raise ValueError("Cannot create an empty PDF report.")
    return pages


def _heading_section_space(
    blocks: list[tuple[str, int, str, float]],
    start: int,
) -> float | None:
    """Keep a heading with its section when the section fits on one page."""
    heading = blocks[start]
    if heading[2] != "bold":
        return None
    heading_size = heading[1]
    required = 0.0
    for index in range(start, len(blocks)):
        block = blocks[index]
        if index > start and block[2] == "bold" and block[1] >= heading_size:
            break
        required += block[3]
    page_capacity = _PDF_TOP - _PDF_BOTTOM
    return min(required, page_capacity)


def _plain_markdown(value: str) -> str:
    text = value.replace("**", "")
    text = re.sub(r"^\s*-\s+", "• ", text)
    text = re.sub(r"^\s{2,}-\s+", "  • ", text)
    return text


def _items_or_none(items: tuple[str, ...], *, indent: str) -> tuple[str, ...]:
    if not items:
        return (f"{indent}- brak",)
    return tuple(f"{indent}- {item}" for item in items)


def _joined_or_none(items: tuple[str, ...]) -> str:
    return ", ".join(items) if items else "brak"


def _optional_number(value: float | None) -> str:
    return "brak" if value is None else f"{value:.6f}"


def _percentage(value: float) -> str:
    return f"{value:.2%}"


def _yes_no(value: bool) -> str:
    return "TAK" if value else "NIE"


def _passed_failed(value: bool) -> str:
    return "ZALICZONA" if value else "NIEZALICZONA"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

from datetime import UTC, datetime

import pytest

from frontend.i18n import (
    LANGUAGE_NAMES,
    SUPPORTED_LANGUAGES,
    translate,
    validate_translation_catalog,
)
from frontend.presentation import (
    SUPPORTED_PRESENTATION_TIMEZONES,
    format_timestamp,
    validate_presentation_timezone,
)


def test_translation_catalog_has_exact_language_parity() -> None:
    validate_translation_catalog()

    assert SUPPORTED_LANGUAGES == ("pl", "en")
    assert LANGUAGE_NAMES == {"pl": "Polski", "en": "English"}


def test_translation_supports_values_without_silent_fallback() -> None:
    assert translate("system.api_version", "pl", version="1.2.3") == (
        "Wersja API: 1.2.3"
    )
    assert translate("system.api_version", "en", version="1.2.3") == (
        "API version: 1.2.3"
    )

    with pytest.raises(KeyError, match="Missing"):
        translate("missing.key", "en")

    with pytest.raises(ValueError, match="interpolation"):
        translate("system.api_version", "en")


def test_decision_workspace_catalog_exposes_bilingual_key_flow() -> None:
    assert translate("decision.title", "pl") == "Przestrzeń decyzyjna"
    assert translate("decision.title", "en") == "Decision Workspace"
    assert translate("decision.narrative.submit", "en") == "Explain result"
    assert translate("decision.report.download_pdf", "pl") == "Pobierz PDF"


def test_core_analytical_views_expose_bilingual_key_flow() -> None:
    assert translate("dashboard.kpi.heading", "pl") == "Kluczowe wskaźniki"
    assert translate("dashboard.kpi.heading", "en") == "Key indicators"
    assert translate("forecast.ticker.heading", "pl") == "Analiza instrumentu"
    assert translate("forecast.ticker.heading", "en") == "Instrument analysis"
    assert translate("risk.profile.heading", "pl") == "Profil ryzyka"
    assert translate("risk.profile.heading", "en") == "Risk profile"
    assert translate("optimization.weights.heading", "pl") == "Struktura portfela"
    assert translate("optimization.weights.heading", "en") == "Portfolio structure"
    assert translate("governance.global.heading", "pl") == "Globalna wyjaśnialność"
    assert translate("governance.global.heading", "en") == "Global explainability"
    assert translate("ai.title", "pl") == "Wnioski AI"
    assert translate("ai.title", "en") == "AI Insights"
    assert translate("reports.generate", "pl") == "Generuj raport"
    assert translate("reports.generate", "en") == "Generate report"


def test_presentation_timezone_contract_is_small_and_explicit() -> None:
    assert SUPPORTED_PRESENTATION_TIMEZONES == (
        "Europe/Warsaw",
        "America/New_York",
        "UTC",
    )
    assert validate_presentation_timezone("Europe/Warsaw") == "Europe/Warsaw"

    with pytest.raises(ValueError, match="Unsupported"):
        validate_presentation_timezone("Asia/Tokyo")


def test_timestamp_is_converted_from_utc_without_mutating_source() -> None:
    source = datetime(2026, 8, 14, 12, 30, tzinfo=UTC)

    assert format_timestamp(source, "Europe/Warsaw", missing="Missing") == (
        "2026-08-14 14:30:00 CEST"
    )
    assert format_timestamp(source, "America/New_York", missing="Missing") == (
        "2026-08-14 08:30:00 EDT"
    )
    assert format_timestamp(source, "UTC", missing="Missing") == (
        "2026-08-14 12:30:00 UTC"
    )
    assert source == datetime(2026, 8, 14, 12, 30, tzinfo=UTC)


def test_timestamp_formatter_handles_missing_invalid_and_naive_values() -> None:
    assert format_timestamp(None, "UTC", missing="Missing") == "Missing"
    assert format_timestamp("invalid", "UTC", missing="Missing") == "Missing"

    with pytest.raises(ValueError, match="timezone-aware"):
        format_timestamp(
            datetime(2026, 8, 14, 12, 30),
            "UTC",
            missing="Missing",
        )

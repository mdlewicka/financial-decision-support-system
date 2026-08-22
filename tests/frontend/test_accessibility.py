from typing import Any

import pytest

from frontend import accessibility
from frontend.i18n import translate


def test_accessibility_styles_cover_keyboard_contrast_and_motion() -> None:
    styles = accessibility.ACCESSIBILITY_STYLES

    assert ":focus-visible" in styles
    assert "outline: 3px solid #005fcc" in styles
    assert '[data-testid="stAlert"]' in styles
    assert "prefers-color-scheme: dark" in styles
    assert "prefers-reduced-motion: reduce" in styles


def test_render_accessibility_styles_uses_explicit_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def record_markdown(value: str, **kwargs: Any) -> None:
        calls.append((value, kwargs))

    monkeypatch.setattr(accessibility.st, "markdown", record_markdown)

    accessibility.render_accessibility_styles()

    assert calls == [
        (
            accessibility.ACCESSIBILITY_STYLES,
            {"unsafe_allow_html": True},
        )
    ]


def test_chart_description_is_bilingual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captions: list[str] = []
    monkeypatch.setattr(accessibility.st, "caption", captions.append)

    accessibility.render_chart_description("Opis danych.", "pl")
    accessibility.render_chart_description("Data description.", "en")

    assert captions == [
        "Opis wykresu: Opis danych.",
        "Chart description: Data description.",
    ]


def test_all_chart_descriptions_have_polish_and_english_versions() -> None:
    keys = (
        "accessibility.optimization.weights",
        "accessibility.optimization.return_contribution",
        "accessibility.optimization.volatility_contribution",
        "accessibility.optimization.cumulative",
        "accessibility.governance.global",
        "accessibility.governance.pdp",
        "accessibility.governance.ale",
        "accessibility.governance.local",
    )

    for key in keys:
        assert translate(key, "pl")
        assert translate(key, "en")
        assert translate(key, "pl") != translate(key, "en")

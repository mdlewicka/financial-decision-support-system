from pathlib import Path

import pytest

from frontend import branding

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_corporate_palette_uses_accessible_navy_identity() -> None:
    assert branding.CORPORATE_NAVY == "#163A5F"
    assert branding.CORPORATE_NAVY_DARK == "#0B1F33"
    assert "#FBFCFE" in branding.CORPORATE_STYLE
    assert "#FFFFFF" in branding.CORPORATE_STYLE


def test_branding_styles_are_rendered_as_trusted_static_css(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def record_markdown(
        body: str,
        *,
        unsafe_allow_html: bool = False,
    ) -> None:
        calls.append((body, unsafe_allow_html))

    monkeypatch.setattr(
        branding.st,
        "markdown",
        record_markdown,
    )

    branding.render_branding_styles()

    assert calls == [(branding.CORPORATE_STYLE, True)]


def test_streamlit_theme_matches_corporate_palette() -> None:
    configuration = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(
        encoding="utf-8"
    )

    assert 'primaryColor = "#163A5F"' in configuration
    assert 'backgroundColor = "#FBFCFE"' in configuration
    assert 'secondaryBackgroundColor = "#F3F6FA"' in configuration
    assert 'backgroundColor = "#0B1F33"' in configuration
    assert 'textColor = "#F8FAFC"' in configuration

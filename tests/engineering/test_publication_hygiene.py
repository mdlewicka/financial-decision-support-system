from __future__ import annotations

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_TEXT_ROOTS = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "README_PL.md",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / "backend" / "src",
    PROJECT_ROOT / "frontend",
)
PUBLISHED_TEXT_SUFFIXES = {".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
LEGACY_BRAND = re.compile(r"\bsentinel[ -]?ai\b", flags=re.IGNORECASE)
PRIVATE_PATH = re.compile(r"/Users/|/private/tmp|Documents/Codex")
SECRET_TOKEN = re.compile(
    r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|hf_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
)
SENSITIVE_FILE = re.compile(
    r"(^|/)(\.env|\.env\..+|secrets\.toml|.+\.(?:pem|key|p12|pfx))$"
)


def _text_files() -> list[Path]:
    files: list[Path] = []
    for root in CURRENT_TEXT_ROOTS:
        candidates = (root,) if root.is_file() else root.rglob("*")
        files.extend(
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in PUBLISHED_TEXT_SUFFIXES
        )
    return sorted(set(files))


def _matches(pattern: re.Pattern[str]) -> list[str]:
    findings: list[str] = []
    for path in _text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            findings.append(str(path.relative_to(PROJECT_ROOT)))
    return findings


def test_current_published_content_uses_consistent_product_name() -> None:
    assert _matches(LEGACY_BRAND) == []


def test_current_published_content_has_no_private_paths_or_secret_tokens() -> None:
    assert _matches(PRIVATE_PATH) == []
    assert _matches(SECRET_TOKEN) == []


def test_git_tracks_no_sensitive_filenames() -> None:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    sensitive = [
        path
        for path in result.stdout.splitlines()
        if SENSITIVE_FILE.search(path) and path != ".env.example"
    ]

    assert sensitive == []


def test_ignore_rules_cover_local_and_generated_private_files() -> None:
    ignore_rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    for rule in (
        ".env",
        ".env.*",
        ".venv/",
        "__pycache__/",
        ".DS_Store",
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",
    ):
        assert rule in ignore_rules


def test_project_metadata_uses_public_product_identifier() -> None:
    metadata = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert 'name = "financial-decision-support-system"' in metadata
    assert '"name": "financial-decision-support-system"' in compose

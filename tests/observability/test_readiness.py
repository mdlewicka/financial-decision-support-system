from pathlib import Path

from backend.src.observability.readiness import ReadinessService


def test_readiness_requires_readable_non_empty_artifacts(tmp_path: Path) -> None:
    forecast = tmp_path / "forecast.json"
    portfolio = tmp_path / "portfolio.json"
    forecast.write_text("{}", encoding="utf-8")
    portfolio.write_text("{}", encoding="utf-8")

    result = ReadinessService(
        artifacts=(
            ("forecast_evidence", forecast),
            ("portfolio_evidence", portfolio),
        )
    ).evaluate()

    assert result.ready is True
    assert result.status == "ready"
    assert [check.status for check in result.checks] == ["ready", "ready"]


def test_readiness_reports_missing_without_exposing_path(tmp_path: Path) -> None:
    secret_directory = tmp_path / "private-location"
    missing = secret_directory / "artifact.json"

    result = ReadinessService(artifacts=(("forecast_evidence", missing),)).evaluate()

    assert result.ready is False
    assert result.status == "not_ready"
    assert result.checks[0].name == "forecast_evidence"
    assert result.checks[0].status == "missing"
    assert str(secret_directory) not in repr(result)


def test_empty_artifact_is_not_ready(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.touch()

    result = ReadinessService(artifacts=(("empty", empty),)).evaluate()

    assert result.ready is False

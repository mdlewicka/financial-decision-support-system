"""CLI integration test for the portfolio Decision Engine."""

from pathlib import Path

from backend.src.api.risk_schemas import RiskArtifactPayload
from backend.src.decision.cli import main


def test_cli_exports_three_profile_rankings(
    portfolio_artifact: RiskArtifactPayload,
    tmp_path: Path,
    capsys,
) -> None:
    portfolio_directory = tmp_path / "portfolio"
    output_directory = tmp_path / "decision"
    portfolio_directory.mkdir()
    (portfolio_directory / "portfolio_result.json").write_text(
        portfolio_artifact.model_dump_json(indent=2),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--portfolio-directory",
            str(portfolio_directory),
            "--output-directory",
            str(output_directory),
            "--available-liquidity-fraction",
            "0.10",
        ]
    )

    assert exit_code == 0
    assert (output_directory / "decision_result.json").is_file()
    assert (output_directory / "decision_rankings.csv").is_file()
    assert (output_directory / "decision_components.csv").is_file()
    output = capsys.readouterr().out
    assert "Source evaluation split: validation" in output
    assert "LLM used for ranking: no" in output
    assert "Portfolio weights modified: no" in output

"""Run the validation-only deterministic portfolio Decision Engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.src.api.risk_repository import (
    DEFAULT_RISK_ARTIFACT_DIRECTORY,
    RiskArtifactRepository,
)
from backend.src.config import PROJECT_ROOT
from backend.src.decision.exporter import export_portfolio_decisions
from backend.src.decision.models import DecisionProfileType
from backend.src.decision.profiles import create_decision_profile
from backend.src.decision.service import PortfolioDecisionService

DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "benchmark_results" / "decision_engine"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank frozen portfolio scenarios for decision profiles.",
    )
    parser.add_argument(
        "--portfolio-directory",
        type=Path,
        default=DEFAULT_RISK_ARTIFACT_DIRECTORY,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--available-liquidity-fraction",
        type=float,
        default=0.10,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = RiskArtifactRepository(args.portfolio_directory).load()
    service = PortfolioDecisionService()
    evaluations = tuple(
        service.evaluate(
            artifact,
            create_decision_profile(profile_type),
            available_liquidity_fraction=args.available_liquidity_fraction,
        )
        for profile_type in (
            DecisionProfileType.CONSERVATIVE,
            DecisionProfileType.BALANCED,
            DecisionProfileType.GROWTH,
        )
    )
    paths = export_portfolio_decisions(evaluations, args.output_directory)

    print("Decision engine: deterministic")
    print("Source evaluation split: validation")
    print("LLM used for ranking: no")
    print("Portfolio weights modified: no")
    print("Recommendations:")
    for evaluation in evaluations:
        print(
            f"  {evaluation.decision.profile.profile_type.value}: "
            f"{evaluation.decision.recommended_scenario}"
        )
    print(f"JSON result: {paths.result_json.resolve()}")
    print(f"Rankings: {paths.rankings_csv.resolve()}")
    print(f"Components: {paths.components_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

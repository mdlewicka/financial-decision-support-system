"""Export deterministic Decision Engine rankings for audit and presentation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from backend.src.decision.service import PortfolioDecisionEvaluation


@dataclass(frozen=True, slots=True)
class DecisionExportPaths:
    """Paths of reproducible Decision Engine artifacts."""

    result_json: Path
    rankings_csv: Path
    components_csv: Path


def export_portfolio_decisions(
    evaluations: tuple[PortfolioDecisionEvaluation, ...],
    output_directory: Path,
) -> DecisionExportPaths:
    """Write rankings, component contributions and provenance."""

    if not evaluations:
        raise ValueError("evaluations must not be empty.")
    profile_types = tuple(
        evaluation.decision.profile.profile_type.value for evaluation in evaluations
    )
    if len(profile_types) != len(set(profile_types)):
        raise ValueError("evaluations must contain unique profile types.")

    first = evaluations[0].source
    for evaluation in evaluations[1:]:
        if evaluation.source != first:
            raise ValueError("evaluations must share the same frozen source artifact.")

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = DecisionExportPaths(
        result_json=output_directory / "decision_result.json",
        rankings_csv=output_directory / "decision_rankings.csv",
        components_csv=output_directory / "decision_components.csv",
    )
    ranking_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []

    for evaluation in sorted(
        evaluations,
        key=lambda item: item.decision.profile.profile_type.value,
    ):
        profile_type = evaluation.decision.profile.profile_type.value
        for scenario in evaluation.decision.scenarios:
            context = evaluation.source.context_for(scenario.name)
            ranking_rows.append(
                {
                    "profile_type": profile_type,
                    "scenario": scenario.name,
                    "construction": context.construction,
                    "admissible": scenario.admissible,
                    "rank": scenario.rank if scenario.rank is not None else "",
                    "total_score": (
                        scenario.total_score if scenario.total_score is not None else ""
                    ),
                    "constraint_violations": "|".join(scenario.constraint_violations),
                    "tradeoffs": "|".join(scenario.tradeoffs),
                }
            )
            for component in sorted(scenario.normalized_components):
                component_rows.append(
                    {
                        "profile_type": profile_type,
                        "scenario": scenario.name,
                        "component": component,
                        "normalized_value": scenario.normalized_components[component],
                        "weighted_contribution": (
                            scenario.weighted_contributions[component]
                        ),
                    }
                )

    _write_csv(paths.rankings_csv, ranking_rows)
    _write_csv(paths.components_csv, component_rows)
    payload = {
        "schema_version": 1,
        "engine": {
            "deterministic": True,
            "llm_used_for_ranking": False,
            "portfolio_weights_modified": False,
        },
        "provenance": first.provenance.to_dict(),
        "scenario_contexts": [context.to_dict() for context in first.contexts],
        "evaluations": [
            evaluation.decision.to_dict()
            for evaluation in sorted(
                evaluations,
                key=lambda item: item.decision.profile.profile_type.value,
            )
        ],
    }
    paths.result_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

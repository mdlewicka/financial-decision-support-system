"""Deterministic ranking of portfolio scenarios for one user profile."""

from __future__ import annotations

import math

from backend.src.decision.models import (
    DecisionConstraints,
    DecisionEngineResult,
    DecisionProfile,
    MetricNormalization,
    RankedScenario,
    ScenarioMetrics,
)
from backend.src.decision.normalization import (
    build_normalization,
    normalize_value,
)


class DecisionEngine:
    """Reject constrained scenarios, then score admissible alternatives."""

    def rank(
        self,
        scenarios: tuple[ScenarioMetrics, ...],
        profile: DecisionProfile,
        *,
        available_liquidity_fraction: float,
    ) -> DecisionEngineResult:
        validated = self._validate_inputs(
            scenarios,
            profile,
            available_liquidity_fraction,
        )
        violations_by_name = {
            scenario.name: self._constraint_violations(
                scenario,
                profile.constraints,
                available_liquidity_fraction,
            )
            for scenario in validated
        }
        admissible_inputs = tuple(
            scenario for scenario in validated if not violations_by_name[scenario.name]
        )
        normalization = (
            self._build_normalization(admissible_inputs) if admissible_inputs else ()
        )
        preliminary = tuple(
            self._evaluate_scenario(
                scenario,
                profile,
                normalization,
                violations_by_name[scenario.name],
            )
            for scenario in validated
        )
        admissible = sorted(
            (item for item in preliminary if item.admissible),
            key=lambda item: (-self._score_for_sort(item), item.name),
        )
        ranked_admissible = tuple(
            RankedScenario(
                name=item.name,
                admissible=True,
                rank=rank,
                total_score=item.total_score,
                normalized_components=item.normalized_components,
                weighted_contributions=item.weighted_contributions,
                constraint_violations=(),
                tradeoffs=item.tradeoffs,
            )
            for rank, item in enumerate(admissible, start=1)
        )
        rejected = tuple(
            sorted(
                (item for item in preliminary if not item.admissible),
                key=lambda item: item.name,
            )
        )
        ordered = (*ranked_admissible, *rejected)
        return DecisionEngineResult(
            profile=profile,
            available_liquidity_fraction=available_liquidity_fraction,
            normalization=normalization,
            scenarios=ordered,
            recommended_scenario=(
                ranked_admissible[0].name if ranked_admissible else None
            ),
        )

    @staticmethod
    def _score_for_sort(item: RankedScenario) -> float:
        if item.total_score is None:
            raise ValueError("Admissible scenario must have a total score.")
        return item.total_score

    def _validate_inputs(
        self,
        scenarios: tuple[ScenarioMetrics, ...],
        profile: DecisionProfile,
        available_liquidity_fraction: float,
    ) -> tuple[ScenarioMetrics, ...]:
        if not isinstance(profile, DecisionProfile):
            raise TypeError("profile must be a DecisionProfile instance.")
        if not scenarios:
            raise ValueError("scenarios must not be empty.")
        if any(not isinstance(item, ScenarioMetrics) for item in scenarios):
            raise TypeError("scenarios must contain ScenarioMetrics instances.")
        names = [scenario.name for scenario in scenarios]
        if len(set(names)) != len(names):
            raise ValueError("scenario names must be unique.")
        if (
            isinstance(available_liquidity_fraction, bool)
            or not isinstance(available_liquidity_fraction, int | float)
            or not math.isfinite(float(available_liquidity_fraction))
            or not 0.0 <= available_liquidity_fraction <= 1.0
        ):
            raise ValueError("available_liquidity_fraction must be in [0, 1].")
        return tuple(sorted(scenarios, key=lambda item: item.name))

    def _build_normalization(
        self,
        scenarios: tuple[ScenarioMetrics, ...],
    ) -> tuple[MetricNormalization, ...]:
        diversification_values = tuple(
            self._diversification_value(scenario) for scenario in scenarios
        )
        return (
            build_normalization(
                "return",
                tuple(item.annualized_return for item in scenarios),
                higher_is_better=True,
            ),
            build_normalization(
                "volatility",
                tuple(item.annualized_volatility for item in scenarios),
                higher_is_better=False,
            ),
            build_normalization(
                "drawdown",
                tuple(item.maximum_drawdown for item in scenarios),
                higher_is_better=False,
            ),
            build_normalization(
                "diversification",
                diversification_values,
                higher_is_better=True,
            ),
        )

    def _evaluate_scenario(
        self,
        scenario: ScenarioMetrics,
        profile: DecisionProfile,
        normalization: tuple[MetricNormalization, ...],
        violations: tuple[str, ...],
    ) -> RankedScenario:
        if violations:
            return RankedScenario(
                name=scenario.name,
                admissible=False,
                rank=None,
                total_score=None,
                normalized_components={},
                weighted_contributions={},
                constraint_violations=violations,
                tradeoffs=(),
            )

        ranges = {item.metric: item for item in normalization}
        components = {
            "return": normalize_value(
                scenario.annualized_return,
                ranges["return"],
            ),
            "volatility": normalize_value(
                scenario.annualized_volatility,
                ranges["volatility"],
            ),
            "drawdown": normalize_value(
                scenario.maximum_drawdown,
                ranges["drawdown"],
            ),
            "diversification": normalize_value(
                self._diversification_value(scenario),
                ranges["diversification"],
            ),
        }
        priorities = profile.priorities.to_dict()
        contributions = {
            name: components[name] * priorities[name] for name in components
        }
        score = float(sum(contributions.values()))

        return RankedScenario(
            name=scenario.name,
            admissible=True,
            rank=None,
            total_score=score,
            normalized_components=components,
            weighted_contributions=contributions,
            constraint_violations=(),
            tradeoffs=self._tradeoffs(components),
        )

    @staticmethod
    def _diversification_value(scenario: ScenarioMetrics) -> float:
        ratio = scenario.diversification_ratio or 1.0
        return math.sqrt(scenario.effective_number_of_assets * ratio)

    @staticmethod
    def _tradeoffs(components: dict[str, float]) -> tuple[str, ...]:
        ordered = sorted(
            components,
            key=lambda name: (-components[name], name),
        )
        return (
            f"strongest_component:{ordered[0]}",
            f"weakest_component:{ordered[-1]}",
        )

    @staticmethod
    def _constraint_violations(
        scenario: ScenarioMetrics,
        constraints: DecisionConstraints,
        available_liquidity_fraction: float,
    ) -> tuple[str, ...]:
        violations: list[str] = []
        checks = (
            (
                constraints.maximum_annualized_volatility is not None
                and scenario.annualized_volatility
                > constraints.maximum_annualized_volatility,
                "maximum_annualized_volatility",
            ),
            (
                constraints.maximum_drawdown is not None
                and scenario.maximum_drawdown > constraints.maximum_drawdown,
                "maximum_drawdown",
            ),
            (
                constraints.maximum_concentration_index is not None
                and scenario.concentration_index
                > constraints.maximum_concentration_index,
                "maximum_concentration_index",
            ),
            (
                constraints.minimum_effective_assets is not None
                and scenario.effective_number_of_assets
                < constraints.minimum_effective_assets,
                "minimum_effective_assets",
            ),
            (
                constraints.minimum_diversification_ratio is not None
                and (
                    scenario.diversification_ratio is None
                    or scenario.diversification_ratio
                    < constraints.minimum_diversification_ratio
                ),
                "minimum_diversification_ratio",
            ),
            (
                available_liquidity_fraction < constraints.minimum_cash_reserve,
                "minimum_cash_reserve",
            ),
        )

        for failed, label in checks:
            if failed:
                violations.append(label)

        return tuple(violations)

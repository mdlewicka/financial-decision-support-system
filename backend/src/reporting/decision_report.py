"""Typed, deterministic contract for one human-reviewed decision report."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DISCLAIMER_FRAGMENT = "nie jest indywidualną poradą finansową"


@dataclass(frozen=True, slots=True)
class ArtifactVersion:
    """Immutable identity of one source artifact used by the report."""

    identifier: str
    schema_version: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "identifier", _required_text(self.identifier, "identifier")
        )
        object.__setattr__(
            self,
            "schema_version",
            _required_text(self.schema_version, "schema_version"),
        )
        normalized_digest = self.sha256.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized_digest) is None:
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters.")
        object.__setattr__(self, "sha256", normalized_digest)


@dataclass(frozen=True, slots=True)
class ReportMetadata:
    """Reproducibility and evaluation scope attached to every report."""

    schema_version: int
    generated_at_utc: str
    presentation_timezone: str
    evaluation_split: str
    test_split_used: bool
    source_artifacts: tuple[ArtifactVersion, ...]


@dataclass(frozen=True, slots=True)
class ReportAssumptions:
    """User-controlled assumptions consumed by the Decision Engine."""

    profile_type: str
    priorities: dict[str, float]
    constraints: dict[str, float | None]
    liquidity_need: str
    investment_horizon_years: int
    available_liquidity_fraction: float


@dataclass(frozen=True, slots=True)
class ReportScenario:
    """One unchanged portfolio alternative from the deterministic ranking."""

    name: str
    construction: str
    weights: dict[str, float]
    admissible: bool
    rank: int | None
    total_score: float | None
    constraint_violations: tuple[str, ...]
    tradeoffs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReportNarrative:
    """Guarded narrative that explains but never changes the ranking."""

    agent_name: str
    question: str
    summary: str
    findings: tuple[str, ...]
    risks: tuple[str, ...]
    recommendations: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReportEvidence:
    """One auditable benchmark, XAI or local-RAG citation."""

    source_type: str
    identifier: str
    description: str


@dataclass(frozen=True, slots=True)
class ReportXAI:
    """Forecast-model interpretation with an explicit decision boundary."""

    model_name: str
    explanation_split: str
    target_definition: str
    global_observations: int
    top_global_shap_features: tuple[dict[str, float | str], ...]
    partial_dependence_features: tuple[str, ...]
    accumulated_local_effect_features: tuple[str, ...]
    scope: str


@dataclass(frozen=True, slots=True)
class ReportGovernance:
    """Machine-verifiable controls disclosed in the final report."""

    deterministic_ranking: bool
    llm_used_for_ranking: bool
    portfolio_weights_modified: bool
    narrative_evaluation_status: str
    deterministic_gate_passed: bool
    llm_judge_approved_for_reporting: bool
    chain_of_thought_stored: bool
    human_review_required: bool


@dataclass(frozen=True, slots=True)
class DecisionReport:
    """Complete backend-neutral representation of a decision report."""

    metadata: ReportMetadata
    assumptions: ReportAssumptions
    recommended_scenario: str
    scenarios: tuple[ReportScenario, ...]
    narrative: ReportNarrative
    evidence: tuple[ReportEvidence, ...]
    xai: ReportXAI
    governance: ReportGovernance

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report while preserving field order."""
        return asdict(self)


class DecisionReportBuilder:
    """Validate and combine frozen evidence without invoking any model."""

    def build(
        self,
        *,
        decision: Mapping[str, Any],
        narrative: Mapping[str, Any],
        xai: Mapping[str, Any],
        narrative_evaluation: Mapping[str, Any],
        source_artifacts: Sequence[ArtifactVersion],
        generated_at: datetime,
        presentation_timezone: str = "UTC",
    ) -> DecisionReport:
        metadata = self._metadata(
            decision=decision,
            source_artifacts=source_artifacts,
            generated_at=generated_at,
            presentation_timezone=presentation_timezone,
        )
        assumptions = self._assumptions(decision)
        scenarios = self._scenarios(decision)
        recommended_scenario = _required_text(
            decision.get("recommended_scenario"),
            "recommended_scenario",
        )
        recommended = next(
            (item for item in scenarios if item.name == recommended_scenario),
            None,
        )
        if recommended is None or not recommended.admissible or recommended.rank != 1:
            raise ValueError(
                "recommended_scenario must identify the first admissible scenario."
            )

        report_narrative, evidence = self._narrative(narrative)
        report_xai = self._xai(xai)
        governance = self._governance(decision, narrative_evaluation)
        return DecisionReport(
            metadata=metadata,
            assumptions=assumptions,
            recommended_scenario=recommended_scenario,
            scenarios=scenarios,
            narrative=report_narrative,
            evidence=evidence,
            xai=report_xai,
            governance=governance,
        )

    @staticmethod
    def _metadata(
        *,
        decision: Mapping[str, Any],
        source_artifacts: Sequence[ArtifactVersion],
        generated_at: datetime,
        presentation_timezone: str,
    ) -> ReportMetadata:
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware.")
        try:
            ZoneInfo(presentation_timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(
                f"Unknown presentation timezone: {presentation_timezone}."
            ) from error
        artifacts = tuple(source_artifacts)
        if not artifacts:
            raise ValueError("source_artifacts must not be empty.")
        identifiers = tuple(item.identifier for item in artifacts)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("source_artifacts must have unique identifiers.")

        provenance = _mapping(decision, "provenance")
        evaluation_split = _required_text(
            provenance.get("evaluation_split"),
            "evaluation_split",
        )
        if evaluation_split != "validation":
            raise ValueError("Decision reports may use only validation evidence.")
        return ReportMetadata(
            schema_version=1,
            generated_at_utc=generated_at.astimezone(UTC).isoformat(),
            presentation_timezone=presentation_timezone,
            evaluation_split=evaluation_split,
            test_split_used=False,
            source_artifacts=tuple(sorted(artifacts, key=lambda item: item.identifier)),
        )

    @staticmethod
    def _assumptions(decision: Mapping[str, Any]) -> ReportAssumptions:
        profile = _mapping(decision, "profile")
        priorities = _finite_float_mapping(
            _mapping(profile, "priorities"), "priorities"
        )
        if not math.isclose(sum(priorities.values()), 1.0, abs_tol=1e-9):
            raise ValueError("Decision priorities must sum to one.")
        constraints = {
            str(key): None if value is None else _finite_float(value, str(key))
            for key, value in _mapping(profile, "constraints").items()
        }
        horizon = profile.get("investment_horizon_years")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("investment_horizon_years must be a positive integer.")
        available_liquidity = _finite_float(
            decision.get("available_liquidity_fraction"),
            "available_liquidity_fraction",
        )
        if not 0.0 <= available_liquidity <= 1.0:
            raise ValueError("available_liquidity_fraction must be in [0, 1].")
        return ReportAssumptions(
            profile_type=_required_text(profile.get("profile_type"), "profile_type"),
            priorities=priorities,
            constraints=constraints,
            liquidity_need=_required_text(
                profile.get("liquidity_need"),
                "liquidity_need",
            ),
            investment_horizon_years=horizon,
            available_liquidity_fraction=available_liquidity,
        )

    @staticmethod
    def _scenarios(decision: Mapping[str, Any]) -> tuple[ReportScenario, ...]:
        raw_scenarios = _sequence(decision, "scenarios")
        scenarios: list[ReportScenario] = []
        ranks: list[int] = []
        names: set[str] = set()
        for raw in raw_scenarios:
            scenario = _mapping_value(raw, "scenario")
            name = _required_text(scenario.get("name"), "scenario.name")
            if name in names:
                raise ValueError(f"Duplicate scenario name: {name}.")
            names.add(name)
            admissible = scenario.get("admissible")
            if not isinstance(admissible, bool):
                raise TypeError("scenario.admissible must be a boolean.")
            rank = scenario.get("rank")
            score = scenario.get("total_score")
            if admissible:
                if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                    raise ValueError("Admissible scenarios require a positive rank.")
                ranks.append(rank)
                score = _finite_float(score, "scenario.total_score")
            elif rank is not None or score is not None:
                raise ValueError("Rejected scenarios cannot have a rank or score.")
            weights = _weights(_sequence(scenario, "weights"))
            scenarios.append(
                ReportScenario(
                    name=name,
                    construction=_required_text(
                        scenario.get("construction"),
                        "scenario.construction",
                    ),
                    weights=weights,
                    admissible=admissible,
                    rank=rank,
                    total_score=score,
                    constraint_violations=_text_tuple(
                        scenario.get("constraint_violations"),
                        "constraint_violations",
                        allow_empty=True,
                    ),
                    tradeoffs=_text_tuple(
                        scenario.get("tradeoffs"),
                        "tradeoffs",
                        allow_empty=True,
                    ),
                )
            )
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError("Admissible scenario ranks must be consecutive from one.")
        return tuple(sorted(scenarios, key=_scenario_sort_key))

    @staticmethod
    def _narrative(
        narrative: Mapping[str, Any],
    ) -> tuple[ReportNarrative, tuple[ReportEvidence, ...]]:
        limitations = _text_tuple(narrative.get("limitations"), "limitations")
        if not any(_DISCLAIMER_FRAGMENT in item.casefold() for item in limitations):
            raise ValueError("Narrative must include the financial-advice disclaimer.")
        raw_evidence = narrative.get("evidence")
        if not isinstance(raw_evidence, Sequence) or isinstance(
            raw_evidence, (str, bytes)
        ):
            raise TypeError("evidence must be a sequence.")
        evidence = tuple(
            ReportEvidence(
                source_type=_required_text(item.get("source_type"), "source_type"),
                identifier=_required_text(item.get("identifier"), "identifier"),
                description=_required_text(item.get("description"), "description"),
            )
            for item in raw_evidence
            if isinstance(item, Mapping)
        )
        if not evidence:
            raise ValueError("Narrative evidence must not be empty.")
        if not any(item.source_type == "rag_document" for item in evidence):
            raise ValueError("Narrative evidence must include a local RAG citation.")
        return (
            ReportNarrative(
                agent_name=_required_text(narrative.get("agent_name"), "agent_name"),
                question=_required_text(narrative.get("question"), "question"),
                summary=_required_text(narrative.get("summary"), "summary"),
                findings=_text_tuple(narrative.get("findings"), "findings"),
                risks=_text_tuple(narrative.get("risks"), "risks"),
                recommendations=_text_tuple(
                    narrative.get("recommendations"),
                    "recommendations",
                ),
                limitations=limitations,
            ),
            evidence,
        )

    @staticmethod
    def _xai(xai: Mapping[str, Any]) -> ReportXAI:
        explanation_split = _required_text(
            xai.get("explanation_split"),
            "explanation_split",
        )
        if explanation_split != "validation":
            raise ValueError("XAI evidence must come from validation.")
        observations = xai.get("global_observations")
        if isinstance(observations, bool) or not isinstance(observations, int):
            raise TypeError("global_observations must be an integer.")
        if observations <= 0:
            raise ValueError("global_observations must be positive.")
        raw_features = xai.get("top_global_shap_features")
        if not isinstance(raw_features, Sequence) or isinstance(
            raw_features,
            (str, bytes),
        ):
            raise TypeError("top_global_shap_features must be a sequence.")
        features = tuple(
            {
                "feature": _required_text(item.get("feature"), "feature"),
                "mean_absolute_shap_value": _finite_float(
                    item.get("mean_absolute_shap_value"),
                    "mean_absolute_shap_value",
                ),
            }
            for item in raw_features
            if isinstance(item, Mapping)
        )
        if not features:
            raise ValueError("top_global_shap_features must not be empty.")
        scope = _required_text(xai.get("scope"), "scope")
        if "does not" not in scope.casefold() or "ranking" not in scope.casefold():
            raise ValueError("XAI scope must state that it does not determine ranking.")
        return ReportXAI(
            model_name=_required_text(xai.get("model_name"), "model_name"),
            explanation_split=explanation_split,
            target_definition=_required_text(
                xai.get("target_definition"),
                "target_definition",
            ),
            global_observations=observations,
            top_global_shap_features=features,
            partial_dependence_features=_text_tuple(
                xai.get("partial_dependence_features"),
                "partial_dependence_features",
                allow_empty=True,
            ),
            accumulated_local_effect_features=_text_tuple(
                xai.get("accumulated_local_effect_features"),
                "accumulated_local_effect_features",
                allow_empty=True,
            ),
            scope=scope,
        )

    @staticmethod
    def _governance(
        decision: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> ReportGovernance:
        if decision.get("deterministic") is not True:
            raise ValueError("Decision ranking must be deterministic.")
        if decision.get("llm_used_for_ranking") is not False:
            raise ValueError("An LLM must not be used for ranking.")
        if decision.get("portfolio_weights_modified") is not False:
            raise ValueError("The narrative layer must not modify portfolio weights.")
        if evaluation.get("test_split_used") is not False:
            raise ValueError("Narrative evaluation must keep the test split closed.")
        if evaluation.get("chain_of_thought_stored") is not False:
            raise ValueError("Decision reports must not store chain-of-thought.")
        gate = _mapping(evaluation, "deterministic_gate")
        if gate.get("passed") is not True:
            raise ValueError("The deterministic narrative gate must pass.")
        judge = _mapping(evaluation, "llm_judge")
        return ReportGovernance(
            deterministic_ranking=True,
            llm_used_for_ranking=False,
            portfolio_weights_modified=False,
            narrative_evaluation_status=_required_text(
                judge.get("status"),
                "llm_judge.status",
            ),
            deterministic_gate_passed=True,
            llm_judge_approved_for_reporting=(
                evaluation.get("llm_judge_approved_for_reporting") is True
            ),
            chain_of_thought_stored=False,
            human_review_required=True,
        )


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise TypeError(f"{key} must be a mapping.")
    return nested


def _mapping_value(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return value


def _sequence(value: Mapping[str, Any], key: str) -> Sequence[Any]:
    nested = value.get(key)
    if not isinstance(nested, Sequence) or isinstance(nested, (str, bytes)):
        raise TypeError(f"{key} must be a sequence.")
    if not nested:
        raise ValueError(f"{key} must not be empty.")
    return nested


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _finite_float_mapping(
    value: Mapping[str, Any],
    label: str,
) -> dict[str, float]:
    if not value:
        raise ValueError(f"{label} must not be empty.")
    return {
        str(key): _finite_float(item, f"{label}.{key}") for key, item in value.items()
    }


def _text_tuple(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence.")
    result = tuple(_required_text(item, label) for item in value)
    if not result and not allow_empty:
        raise ValueError(f"{label} must not be empty.")
    return result


def _weights(raw_weights: Sequence[Any]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for raw in raw_weights:
        item = _mapping_value(raw, "weight")
        ticker = _required_text(item.get("ticker"), "ticker")
        if ticker in weights:
            raise ValueError(f"Duplicate ticker weight: {ticker}.")
        weight = _finite_float(item.get("weight"), "weight")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Portfolio weights must be in [0, 1].")
        weights[ticker] = weight
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8):
        raise ValueError("Portfolio weights must sum to one.")
    return dict(sorted(weights.items()))


def _scenario_sort_key(item: ReportScenario) -> tuple[bool, int, str]:
    return (not item.admissible, item.rank or 10**9, item.name)

from dataclasses import asdict, dataclass
from typing import Any


def _required_text(value: str, label: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} must not be empty.")
    return cleaned


def _required_items(
    values: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    cleaned = tuple(item for value in values if (item := value.strip()))
    if not cleaned:
        raise ValueError(f"{label} must not be empty.")
    return cleaned


@dataclass(frozen=True)
class EvidenceReference:
    source_type: str
    identifier: str
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_type",
            _required_text(self.source_type, "source_type"),
        )
        object.__setattr__(
            self,
            "identifier",
            _required_text(self.identifier, "identifier"),
        )
        object.__setattr__(
            self,
            "description",
            _required_text(self.description, "description"),
        )


@dataclass(frozen=True)
class AgentResponse:
    agent_name: str
    question: str
    summary: str
    findings: tuple[str, ...]
    risks: tuple[str, ...]
    recommendations: tuple[str, ...]
    limitations: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_name",
            _required_text(self.agent_name, "agent_name"),
        )
        object.__setattr__(
            self,
            "question",
            _required_text(self.question, "question"),
        )
        object.__setattr__(
            self,
            "summary",
            _required_text(self.summary, "summary"),
        )
        for label in (
            "findings",
            "risks",
            "recommendations",
            "limitations",
        ):
            object.__setattr__(
                self,
                label,
                _required_items(getattr(self, label), label),
            )
        if not self.evidence:
            raise ValueError("evidence must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

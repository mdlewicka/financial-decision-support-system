"""Tests for RAG language-model generation."""

from backend.src.rag.generation import (
    OllamaGenerator,
)


class CapturingGenerator(OllamaGenerator):
    """Capture prompts without contacting Ollama."""

    def __init__(self) -> None:
        self.last_prompt = ""

    def complete(
        self,
        prompt: str,
    ) -> str:
        self.last_prompt = prompt
        return "generated answer"


def test_generate_requires_question_language() -> None:
    generator = CapturingGenerator()

    answer = generator.generate(
        question="Co to ryzyko finansowe?",
        context=("Financial risk is the possibility of suffering a financial loss."),
    )

    assert answer == "generated answer"

    assert (
        "Respond in the same language as the user's question." in generator.last_prompt
    )

    assert "not from the language of the supplied context" in generator.last_prompt

    assert "translate and synthesize" in generator.last_prompt

    assert "Co to ryzyko finansowe?" in generator.last_prompt

    assert (
        "Financial risk is the possibility "
        "of suffering a financial loss." in generator.last_prompt
    )


def test_generate_preserves_english_question() -> None:
    generator = CapturingGenerator()

    generator.generate(
        question=("What is financial risk?"),
        context=("Financial risk concerns the possibility of financial loss."),
    )

    assert "Question:\nWhat is financial risk?" in generator.last_prompt

    assert (
        "Respond in the same language as the user's question." in generator.last_prompt
    )


def test_generate_requires_grounded_answer() -> None:
    generator = CapturingGenerator()

    generator.generate(
        question=("What is diversification?"),
        context=("Diversification spreads exposure across assets."),
    )

    assert "using only the supplied context" in generator.last_prompt

    assert (
        "Do not add facts that are not "
        "supported by the context." in generator.last_prompt
    )

    assert "If the context is insufficient" in generator.last_prompt


def test_generate_returns_completion_result() -> None:
    generator = CapturingGenerator()

    result = generator.generate(
        question="Test question",
        context="Test context",
    )

    assert result == "generated answer"

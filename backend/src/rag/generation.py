"""Language-model generation for retrieval-augmented generation."""

from typing import Any


class OllamaGenerator:
    """Generate grounded answers with a local Ollama language model."""

    def __init__(
        self,
        model_name: str,
        host: str | None = None,
    ) -> None:
        from ollama import Client

        self._model_name = model_name
        self._client: Any = Client(host=host) if host else Client()

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        """Generate an answer grounded only in retrieved context."""
        prompt = (
            "Answer the user's question using only the supplied context.\n"
            "Do not add facts that are not supported by the context.\n"
            "\n"
            "Respond in the same language as the user's question.\n"
            "Determine the response language from the user's question, "
            "not from the language of the supplied context.\n"
            "If the context is written in another language, translate and "
            "synthesize the relevant information into the language used "
            "by the user.\n"
            "If the context is insufficient, state that explicitly in the "
            "same language as the user's question.\n"
            "\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

        return self.complete(prompt)

    def complete(
        self,
        prompt: str,
    ) -> str:
        """Send one completion request to Ollama."""
        response = self._client.generate(
            model=self._model_name,
            prompt=prompt,
        )

        answer = response.get(
            "response",
            "",
        ).strip()

        if not answer:
            raise RuntimeError("The language model returned an empty answer.")

        return answer

"""The language model that writes answers. Swap providers with RAG_LLM_PROVIDER."""

from typing import Protocol

import httpx
import ollama

from app.config import Settings


class LLMError(RuntimeError):
    """The model could not be reached or returned an error."""


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


class OllamaLLM:
    def __init__(self, host: str, model: str) -> None:
        self._host = host
        self._model = model
        self._client = ollama.Client(host=host)

    def generate(self, system: str, prompt: str) -> str:
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                # Same question, same answer: makes evaluation results repeatable.
                options={"temperature": 0},
            )
        except ollama.ResponseError as exc:
            raise LLMError(
                f"Ollama error: {exc.error}. Did you run 'ollama pull {self._model}'?"
            ) from exc
        except (ConnectionError, httpx.HTTPError) as exc:
            raise LLMError(f"Cannot reach Ollama at {self._host}. Is Ollama running?") from exc
        return response.message.content or ""


def create_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "ollama":
        return OllamaLLM(host=settings.ollama_host, model=settings.llm_model)
    raise ValueError(f"Unknown LLM provider '{settings.llm_provider}'")

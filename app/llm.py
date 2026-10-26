"""The language model that writes answers. Swap providers with RAG_LLM_PROVIDER."""

from dataclasses import dataclass
from typing import Protocol

import httpx
import ollama

from app.config import Settings


class LLMError(RuntimeError):
    """The model could not be reached or returned an error."""


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    model: str

    def generate(self, system: str, prompt: str) -> Generation: ...


class OllamaLLM:
    def __init__(self, host: str, model: str) -> None:
        self.model = model
        self._host = host
        self._client = ollama.Client(host=host)

    def generate(self, system: str, prompt: str) -> Generation:
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                # Same question, same answer: makes evaluation results repeatable.
                options={"temperature": 0},
            )
        except ollama.ResponseError as exc:
            raise LLMError(
                f"Ollama error: {exc.error}. Did you run 'ollama pull {self.model}'?"
            ) from exc
        except (ConnectionError, httpx.HTTPError) as exc:
            raise LLMError(f"Cannot reach Ollama at {self._host}. Is Ollama running?") from exc
        return Generation(
            text=response.message.content or "",
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
        )


def create_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "ollama":
        return OllamaLLM(host=settings.ollama_host, model=settings.llm_model)
    raise ValueError(f"Unknown LLM provider '{settings.llm_provider}'")

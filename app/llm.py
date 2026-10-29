"""The language model that writes answers, served locally by Ollama."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Protocol

import httpx
import ollama


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

    def stream(self, system: str, prompt: str) -> Iterator[Generation]:
        """Yields the answer piece by piece; the last piece carries the token counts."""
        ...


class OllamaLLM:
    def __init__(self, host: str, model: str, keep_alive: str, max_tokens: int) -> None:
        self.model = model
        self._host = host
        self._keep_alive = keep_alive
        self._max_tokens = max_tokens
        self._client = ollama.Client(host=host)

    def generate(self, system: str, prompt: str) -> Generation:
        pieces = list(self.stream(system, prompt))
        text = "".join(piece.text for piece in pieces)
        return replace(pieces[-1], text=text) if pieces else Generation("")

    def stream(self, system: str, prompt: str) -> Iterator[Generation]:
        try:
            for part in self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                # Same question, same answer: makes evaluation results repeatable.
                options={"temperature": 0, "num_predict": self._max_tokens},
                keep_alive=self._keep_alive,
                stream=True,
            ):
                yield Generation(
                    text=part.message.content or "",
                    input_tokens=part.prompt_eval_count or 0,
                    output_tokens=part.eval_count or 0,
                )
        except ollama.ResponseError as exc:
            raise LLMError(
                f"Ollama error: {exc.error}. Did you run 'ollama pull {self.model}'?"
            ) from exc
        except (ConnectionError, httpx.HTTPError) as exc:
            raise LLMError(f"Cannot reach Ollama at {self._host}. Is Ollama running?") from exc

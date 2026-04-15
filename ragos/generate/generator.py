from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_generator: "Generator | None" = None

RAG_PROMPT = """\
Answer the question using ONLY the context provided below.
If the answer is not contained in the context, respond with exactly:
"I don't know — the answer is not in the provided context."

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class GenerationResult:
    answer: str
    model: str
    usage: dict = field(default_factory=dict)


@runtime_checkable
class Generator(Protocol):
    def generate(self, question: str, context: str) -> GenerationResult: ...


class OpenAIGenerator:
    """Generator backed by the OpenAI Chat Completions API."""

    def __init__(self, model: str, api_key: str, max_tokens: int = 1024) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai is required: pip install openai") from e

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, question: str, context: str) -> GenerationResult:
        prompt = RAG_PROMPT.format(context=context, question=question)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
        )
        return GenerationResult(
            answer=resp.choices[0].message.content or "",
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        )


class AnthropicGenerator:
    """Generator backed by the Anthropic Messages API."""

    def __init__(self, model: str, api_key: str, max_tokens: int = 1024) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic is required: pip install anthropic") from e

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, question: str, context: str) -> GenerationResult:
        prompt = RAG_PROMPT.format(context=context, question=question)
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        return GenerationResult(
            answer=text,
            model=resp.model,
            usage={"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        )


class SquishGenerator:
    """Generator backed by a locally-running Squish inference server (OpenAI-compatible API)."""

    def __init__(self, model: str, base_url: str, max_tokens: int = 1024) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai is required: pip install openai") from e

        self._client = OpenAI(api_key="ollama", base_url=base_url)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, question: str, context: str) -> GenerationResult:
        prompt = RAG_PROMPT.format(context=context, question=question)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
        )
        return GenerationResult(
            answer=resp.choices[0].message.content or "",
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        )


def get_generator() -> Generator:
    """Return the module-level singleton generator (lazy init, reads from settings)."""
    global _generator
    if _generator is not None:
        return _generator

    from ragos.config import get_settings

    s = get_settings()
    backend = s.generator_backend.lower()

    if backend == "openai":
        if not s.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set.\n"
                "Set it in your .env file or environment, or switch to another backend:\n"
                "  GENERATOR_BACKEND=squish"
            )
        _generator = OpenAIGenerator(model=s.openai_model, api_key=s.openai_api_key, max_tokens=s.max_new_tokens)

    elif backend == "anthropic":
        if not s.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set.\n"
                "Set it in your .env file or environment, or switch to another backend:\n"
                "  GENERATOR_BACKEND=squish"
            )
        _generator = AnthropicGenerator(
            model=s.anthropic_model, api_key=s.anthropic_api_key, max_tokens=s.max_new_tokens
        )

    elif backend == "squish":
        _generator = SquishGenerator(
            model=s.squish_model, base_url=s.squish_base_url, max_tokens=s.max_new_tokens
        )

    else:
        raise ValueError(
            f"Unknown generator backend: {backend!r}. "
            "Valid values: 'openai', 'anthropic', 'squish'."
        )

    logger.info("Generator initialised: backend=%s", backend)
    return _generator

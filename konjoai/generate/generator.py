from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_generator: Generator | None = None

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


class _BaseGenerator:
    """Shared prompt formatting and the sync→async streaming bridge.

    Concrete backends supply ``generate`` and ``generate_stream``; the async
    ``stream`` interface and ``_format_prompt`` helper are common to all.
    """

    _model: str
    _max_tokens: int

    @staticmethod
    def _format_prompt(question: str, context: str) -> str:
        """Render the RAG prompt template for the given question and context."""
        return RAG_PROMPT.format(context=context, question=question)

    def generate_stream(self, question: str, context: str) -> Iterator[str]:
        """Yield response tokens one at a time. Overridden by each backend."""
        raise NotImplementedError

    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        """Async token-streaming interface; bridges generate_stream() via asyncio.to_thread."""
        sentinel = object()
        sync_gen = self.generate_stream(question=question, context=context)

        def _next() -> object:
            return next(sync_gen, sentinel)

        while True:
            token = await asyncio.to_thread(_next)
            if token is sentinel:
                break
            yield token


class _OpenAIChatGenerator(_BaseGenerator):
    """Shared generate/stream logic for OpenAI-compatible Chat Completions backends."""

    def generate(self, question: str, context: str) -> GenerationResult:
        """Return a single completion from the Chat Completions API."""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": self._format_prompt(question, context)}],
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

    def generate_stream(self, question: str, context: str) -> Iterator[str]:
        """Yield response tokens one at a time from the Chat Completions streaming API."""
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": self._format_prompt(question, context)}],
            max_tokens=self._max_tokens,
            stream=True,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""


class OpenAIGenerator(_OpenAIChatGenerator):
    """Generator backed by the OpenAI Chat Completions API."""

    def __init__(self, model: str, api_key: str, max_tokens: int = 1024) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai is required: pip install openai") from e

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens


class AnthropicGenerator(_BaseGenerator):
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
        """Return a single completion from the Anthropic Messages API."""
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": self._format_prompt(question, context)}],
        )
        text = resp.content[0].text if resp.content else ""
        return GenerationResult(
            answer=text,
            model=resp.model,
            usage={"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        )

    def generate_stream(self, question: str, context: str) -> Iterator[str]:
        """Yield response tokens one at a time from the Anthropic streaming API."""
        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": self._format_prompt(question, context)}],
        ) as stream:
            yield from stream.text_stream


class SquishGenerator(_OpenAIChatGenerator):
    """Generator backed by a locally-running Squish inference server (OpenAI-compatible API)."""

    def __init__(self, model: str, base_url: str, max_tokens: int = 1024) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai is required: pip install openai") from e

        self._client = OpenAI(api_key="squish", base_url=base_url)
        self._model = model
        self._max_tokens = max_tokens


def get_generator() -> Generator:
    """Return the module-level singleton generator (lazy init, reads from settings)."""
    global _generator
    if _generator is not None:
        return _generator

    from konjoai.config import get_settings

    s = get_settings()
    backend = s.generator_backend.lower()

    if backend == "openai":
        if not s.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set.\n"
                "Set it in your .env file or environment, or switch to another backend:\n"
                "  GENERATOR_BACKEND=squish"
            )
        _generator = OpenAIGenerator(model=s.openai_model, api_key=s.openai_api_key, max_tokens=s.max_tokens)

    elif backend == "anthropic":
        if not s.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set.\n"
                "Set it in your .env file or environment, or switch to another backend:\n"
                "  GENERATOR_BACKEND=squish"
            )
        _generator = AnthropicGenerator(model=s.anthropic_model, api_key=s.anthropic_api_key, max_tokens=s.max_tokens)

    elif backend == "squish":
        _generator = SquishGenerator(model=s.squish_model, base_url=s.squish_base_url, max_tokens=s.max_tokens)

    else:
        raise ValueError(f"Unknown generator backend: {backend!r}. Valid values: 'openai', 'anthropic', 'squish'.")

    logger.info("Generator initialised: backend=%s", backend)
    return _generator

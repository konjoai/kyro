"""HyDE — Hypothetical Document Embeddings (Gao et al. 2022).

Reference: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
https://arxiv.org/abs/2212.10496

Core idea: instead of encoding a short query (whose token distribution rarely
matches the long documents in the corpus), we first ask the generator to produce
a *hypothetical* answer paragraph, then encode that paragraph. The hypothesis
embedding lies in document-space rather than query-space, closing the distribution
gap at inference time — no fine-tuning required.

Integration pattern:
    If use_hyde is True, call hyde_encode() instead of encode_query() in the
    retrieval step. The returned hypothesis_text is included in the telemetry
    for debugging / recall analysis.
"""
from __future__ import annotations

import logging

import numpy as np

from ragos.embed.encoder import get_encoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_HYDE_PROMPT = """\
Write a concise factual paragraph (2-4 sentences) that directly answers the
following question. Do not hedge — write as if you are certain of the facts.

Question: {question}

Hypothetical Answer:"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_hypothesis(question: str) -> str:
    """Generate a hypothetical answer paragraph for the given question.

    Uses the configured generator (get_generator() singleton). The generation
    budget is intentionally small: a 2–4 sentence hypothesis is enough to move
    the embedding into document-space without burning tokens.

    Args:
        question: The user's raw query string.

    Returns:
        A hypothesis plain-text paragraph. May be empty string if the generator
        returns an empty or whitespace-only response.
    """
    # Late import avoids circular dependency: generator → config → encoder →...
    from ragos.generate.generator import get_generator  # noqa: PLC0415

    generator = get_generator()
    prompt = _HYDE_PROMPT.format(question=question.strip())

    # Generate with a small token budget — hypothesis, not an essay.
    result = generator.generate(
        query=prompt,
        context="",
        max_tokens=150,
        temperature=0.0,
    )

    hypothesis = result.answer.strip()
    if not hypothesis:
        logger.warning("HyDE returned empty hypothesis for question=%r — falling back to raw query", question)
        return question

    logger.debug("HyDE hypothesis (len=%d): %s", len(hypothesis), hypothesis[:120])
    return hypothesis


def hyde_encode(question: str) -> tuple[np.ndarray, str]:
    """Encode a hypothetical document embedding for the given question.

    Steps:
        1. generate_hypothesis(question) → hypothesis_text
        2. encode(hypothesis_text) → float32 embedding of shape (1, d) or (d,)

    The caller should replace the standard query embedding with this vector when
    use_hyde=True.

    Args:
        question: The user's raw query string.

    Returns:
        A tuple of:
            - embedding: np.ndarray of shape (d,), float32, L2-normalized.
            - hypothesis: The hypothesis text (for telemetry / debug logging).
    """
    hypothesis = generate_hypothesis(question)
    encoder = get_encoder()
    embedding = encoder.encode_query(hypothesis)  # (d,) float32, L2-normalized

    assert embedding.dtype == np.float32, (  # K4 dtype contract
        f"HyDE encoder output dtype mismatch: expected float32, got {embedding.dtype}"
    )

    return embedding, hypothesis

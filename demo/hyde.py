"""Kyro HyDE Theater — Hypothetical Document Embeddings, side by side.

HyDE (Gao et al. 2022) closes the query↔document distribution gap: instead of
embedding a short query, you first write a *hypothetical answer paragraph* and
embed that, landing the vector in document-space. ``konjoai.retrieve.hyde``
implements exactly this — :func:`generate_hypothesis` (LLM) +
:func:`hyde_encode` (encode the hypothesis).

This engine shows the effect live and honestly:

* **Hypothesis (no LLM).** The demo synthesizes the hypothetical answer with a
  deterministic, generic template — labelled ``rule-based`` in the payload —
  rather than calling a generator. It follows kyro's real ``_HYDE_PROMPT``
  intent (a short, declarative, document-shaped paragraph).
* **Everything else is real.** Both the raw query and the hypothesis are
  embedded with the demo encoder (float32 / L2-unit, the K4 contract) and run
  through the *same* real dense cosine retrieval over the corpus
  (:meth:`pipeline.PipelineEngine.dense`). The before/after rankings, scores,
  and the closed gap are measured, not staged.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np
from pipeline import PipelineEngine

from konjoai.retrieve.hyde import _HYDE_PROMPT  # the real prompt intent (template only)

__all__ = ["HyDEEngine"]

# Mirror of the question words / stop tokens we strip when distilling a query
# into the content terms the hypothesis is built around.
_STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "what",
    "which",
    "how",
    "where",
    "when",
    "why",
    "who",
    "do",
    "does",
    "did",
    "you",
    "your",
    "my",
    "i",
    "it",
    "that",
    "this",
    "with",
    "and",
    "or",
    "but",
    "as",
    "from",
    "by",
    "so",
    "if",
    "can",
    "could",
    "would",
    "should",
    "have",
    "has",
    "had",
    "will",
    "about",
    "tell",
    "me",
    "us",
    "any",
}


def _content_terms(question: str) -> list[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z'-]+", question.lower())
    return [t for t in toks if t not in _STOP and len(t) > 2]


def _synthesize_hypothesis(question: str) -> str:
    """Build a deterministic, document-shaped answer paragraph (no LLM).

    Generic transform — never hand-tuned per query: it restates the question
    declaratively and repeats its content terms so the paragraph reads like a
    corpus passage rather than a short query. That density is the whole point
    of HyDE; the embedding + retrieval that follow are fully real.
    """
    q = question.strip().rstrip("?.! ")
    terms = _content_terms(q)
    topic = " ".join(terms[:6]) if terms else q
    lead = q[0].upper() + q[1:] if q else "This"
    return (
        f"{lead}. In practice, {topic} is handled according to the documented "
        f"policy and procedure. The relevant details cover {topic}, including the "
        f"specific steps, requirements, and timeframes that apply. This section "
        f"explains {topic} so the answer is precise and complete."
    )


class HyDEEngine:
    """Compares raw-query retrieval against HyDE-hypothesis retrieval.

    Parameters
    ----------
    pipeline:
        Loaded :class:`~pipeline.PipelineEngine` — provides the real dense
        cosine retrieval used for both the baseline and HyDE runs.
    embed_fn:
        Encoder returning an L2-unit ``float32`` vector — used only to report
        the query vs hypothesis embedding statistics the UI renders.
    """

    def __init__(self, pipeline: PipelineEngine, embed_fn: Callable[[str], np.ndarray]) -> None:
        self.pipeline = pipeline
        self._embed = embed_fn

    def _vec_stats(self, text: str) -> dict[str, Any]:
        v = self._embed(text)
        if v.ndim == 2:
            v = v.reshape(-1)
        return {
            "dim": int(v.shape[0]),
            "dtype": str(v.dtype),
            "nonzero": int(np.count_nonzero(v)),
            "tokens": len(re.findall(r"[a-zA-Z][a-zA-Z'-]+", text)),
        }

    def analyze(self, question: str, top_k: int = 4) -> dict[str, Any]:
        """Run baseline vs HyDE retrieval and return a structured comparison.

        Wire contract consumed by ``hyde.html``; pinned by
        :mod:`tests.unit.test_demo_hyde`.
        """
        question = (question or "").strip()
        if not question:
            return {"error": "question must be non-empty"}

        hypothesis = _synthesize_hypothesis(question)
        baseline = self.pipeline.dense(question, top_k=top_k)
        hyde = self.pipeline.dense(hypothesis, top_k=top_k)

        base_top = baseline[0]["score"] if baseline else 0.0
        hyde_top = hyde[0]["score"] if hyde else 0.0
        base_rank = {r["source"]: r["rank"] for r in baseline}
        hyde_winner = hyde[0]["source"] if hyde else None
        # How far the HyDE winner sat in the raw-query ranking (None = unranked).
        winner_prev_rank = base_rank.get(hyde_winner)

        return {
            "question": question,
            "hypothesis": hypothesis,
            "hypothesis_source": "rule-based template (no LLM in demo)",
            "prompt_intent": _HYDE_PROMPT.split("\n", 1)[0],
            "top_k": top_k,
            "baseline": baseline,
            "hyde": hyde,
            "query_vec": self._vec_stats(question),
            "hyde_vec": self._vec_stats(hypothesis),
            "comparison": {
                "baseline_top_score": round(float(base_top), 4),
                "hyde_top_score": round(float(hyde_top), 4),
                "delta": round(float(hyde_top - base_top), 4),
                "winner": hyde_winner,
                "winner_changed": bool(baseline and hyde and baseline[0]["source"] != hyde_winner),
                "winner_prev_rank": winner_prev_rank,
            },
            "source": "konjoai.retrieve.hyde intent + real dense cosine (pipeline.dense)",
        }

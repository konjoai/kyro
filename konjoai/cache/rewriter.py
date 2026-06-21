"""Query rewriting pipeline for semantic cache hit-rate improvement (Sprint 29).

Why rewriting improves cache hit rate
--------------------------------------
Two queries asking the same thing in different surface forms ("What's the
refund policy?" vs "Tell me about refund policies") share semantic meaning
but differ in characters, so the exact-match fast path always misses and the
cosine-similarity path only helps when embeddings are close.  A lightweight
normalisation pass applied *before* embedding closes the surface-form gap for
free — no model required — and measurably increases hit rate for FAQ-style
traffic.

Design
------
``QueryRewriter`` is a pure synchronous function with no I/O.  It runs a
configurable sequence of *steps* — each step is a named transformation keyed
by a string.  The caller can inspect the per-step diff via :meth:`explain` to
understand what changed and why.

K3: disabled by default (``cache_query_rewrite_enabled=False``).
K5: pure stdlib — no new hard dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Contraction expansion table ───────────────────────────────────────────────
_CONTRACTIONS: dict[str, str] = {
    "what's": "what is",
    "what're": "what are",
    "where's": "where is",
    "when's": "when is",
    "who's": "who is",
    "how's": "how is",
    "it's": "it is",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "can't": "cannot",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "there's": "there is",
    "that's": "that is",
    "let's": "let us",
}

# ── Filler phrase stripping table ─────────────────────────────────────────────
_FILLER_PREFIXES: tuple[str, ...] = (
    "tell me about ",
    "can you tell me ",
    "can you explain ",
    "please explain ",
    "i want to know ",
    "i'd like to know ",
    "could you explain ",
    "give me information on ",
    "give me info on ",
    "what can you tell me about ",
)


@dataclass
class RewriteStep:
    """Records what one step changed."""

    name: str
    before: str
    after: str

    @property
    def changed(self) -> bool:
        return self.before != self.after


@dataclass
class RewriteResult:
    """Full trace of a rewrite operation."""

    original: str
    rewritten: str
    steps: list[RewriteStep] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.original != self.rewritten


# ── Step implementations ──────────────────────────────────────────────────────


def _step_lowercase(q: str) -> str:
    """Lower-case the query."""
    return q.lower()


def _step_normalize_whitespace(q: str) -> str:
    """Collapse runs of whitespace into single spaces."""
    return " ".join(q.split())


def _step_strip_punctuation(q: str) -> str:
    """Replace punctuation with spaces, preserving in-word apostrophes."""
    # Strip leading/trailing non-alphanumeric characters; preserve apostrophes
    # within words so contractions survive until the expand step.
    return re.sub(r"[^\w\s']", " ", q).strip()


def _step_expand_contractions(q: str) -> str:
    """Expand known contractions to their full forms."""
    words = q.split()
    result = [_CONTRACTIONS.get(w.lower(), w) for w in words]
    return " ".join(result)


def _step_strip_fillers(q: str) -> str:
    """Strip a leading filler prefix (e.g. "please", "can you")."""
    lower = q.lower()
    for prefix in _FILLER_PREFIXES:
        if lower.startswith(prefix):
            return q[len(prefix) :].strip()
    return q


def _step_strip_trailing_question_mark(q: str) -> str:
    """Remove trailing question marks and surrounding whitespace."""
    return q.rstrip("?").strip()


_STEP_MAP: dict[str, callable] = {
    "lowercase": _step_lowercase,
    "normalize_whitespace": _step_normalize_whitespace,
    "strip_punctuation": _step_strip_punctuation,
    "expand_contractions": _step_expand_contractions,
    "strip_fillers": _step_strip_fillers,
    "strip_trailing_question_mark": _step_strip_trailing_question_mark,
}

DEFAULT_STEPS: list[str] = [
    "lowercase",
    "expand_contractions",
    "strip_fillers",
    "strip_punctuation",
    "normalize_whitespace",
    "strip_trailing_question_mark",
]


# ── Main class ────────────────────────────────────────────────────────────────


class QueryRewriter:
    """Deterministic, configurable query normalisation pipeline.

    Args:
        steps: Ordered list of step names to apply.  Unknown names are silently
            skipped so old configs continue working when steps are removed.
            Defaults to :data:`DEFAULT_STEPS`.
    """

    def __init__(self, steps: list[str] | None = None) -> None:
        resolved = steps if steps is not None else DEFAULT_STEPS
        self._steps: list[tuple[str, callable]] = [(name, _STEP_MAP[name]) for name in resolved if name in _STEP_MAP]

    def rewrite(self, question: str) -> str:
        """Return the normalised query string."""
        q = question or ""
        for _, fn in self._steps:
            q = fn(q)
        return q.strip()

    def explain(self, question: str) -> RewriteResult:
        """Return a full per-step trace of the rewrite."""
        q = question or ""
        trace: list[RewriteStep] = []
        for name, fn in self._steps:
            before = q
            q = fn(q)
            trace.append(RewriteStep(name=name, before=before, after=q))
        return RewriteResult(original=question, rewritten=q.strip(), steps=trace)

    @property
    def step_names(self) -> list[str]:
        return [name for name, _ in self._steps]


# ── Module-level singleton ────────────────────────────────────────────────────

_rewriter: QueryRewriter | None = None


def get_rewriter() -> QueryRewriter:
    """Return the process-level singleton ``QueryRewriter``.

    Constructed once from ``Settings.cache_query_rewrite_steps``.
    """
    global _rewriter  # noqa: PLW0603
    if _rewriter is not None:
        return _rewriter
    from konjoai.config import get_settings  # lazy — avoids circular import

    settings = get_settings()
    steps = getattr(settings, "cache_query_rewrite_steps", DEFAULT_STEPS)
    _rewriter = QueryRewriter(steps=list(steps))
    return _rewriter


def _reset_rewriter() -> None:
    """Test helper — reset the singleton so config changes take effect."""
    global _rewriter  # noqa: PLW0603
    _rewriter = None

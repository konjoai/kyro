"""Kyro — Demo Day walkthrough.

A runnable, network-free demonstration of the v1.2.0 stack: SDK construction,
streaming agent SSE, semantic-cache hit/miss/eviction, the Sprint-22 Redis
backend (running against an in-process fake), and env-driven configuration.

Run::

    python3 demo/demo.py

The script never touches a live LLM, Qdrant, or Redis instance — every
external boundary is wired to the same stub infrastructure the unit suite
exercises. What you see in the terminal is the actual kyro library doing
real work.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console(highlight=False)


# ── Header ────────────────────────────────────────────────────────────────────


HERO = r"""
   ╦╔═╦ ╦╦═╗╔═╗
   ╠╩╗╚╦╝╠╦╝║ ║
   ╩ ╩ ╩ ╩╚═╚═╝
"""

TAGLINE = "Production RAG agent with semantic caching and streaming"


def render_hero() -> None:
    body = Group(
        Align.center(Text(HERO, style="bold cyan")),
        Align.center(Text(TAGLINE, style="italic white")),
        Align.center(
            Text(
                "v1.2.0  ·  798 tests passing  ·  zero hard deps on the hot path",
                style="dim",
            )
        ),
    )
    console.print(Panel(body, border_style="cyan", padding=(1, 4)))


def section(title: str, subtitle: str = "") -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/]   [dim]{subtitle}[/]", style="cyan"))
    console.print()


def show_code(code: str, lang: str = "python") -> None:
    console.print(
        Panel(
            Syntax(code.strip(), lang, theme="monokai", line_numbers=False, padding=(0, 1)),
            border_style="grey42",
            padding=(0, 1),
        )
    )


# ── Stub backbone (no network, no LLM, no Qdrant) ─────────────────────────────


@dataclass
class _StubGen:
    """Deterministic LLM stub returning ReAct-shaped JSON, then a final answer."""

    script: list[str]
    _i: int = 0

    def __post_init__(self) -> None:
        self._i = 0

    def generate(self, question: str, context: str) -> Any:
        from konjoai.generate.generator import GenerationResult

        idx = min(self._i, len(self.script) - 1)
        self._i += 1
        return GenerationResult(
            answer=self.script[idx],
            model="kyro-demo-stub",
            usage={"prompt_tokens": 7, "completion_tokens": 5},
        )


def _hybrid(_q, top_k_dense: int, top_k_sparse: int):  # noqa: ARG001
    from konjoai.retrieve.hybrid import HybridResult

    return [
        HybridResult(
            rrf_score=0.92,
            content="Refunds are accepted within 30 days with a receipt.",
            source="policy.md",
            metadata={},
        ),
        HybridResult(
            rrf_score=0.71,
            content="Store credit is offered for items returned after 30 days.",
            source="policy.md",
            metadata={},
        ),
    ]


def _rerank(_q, candidates, top_k):
    from konjoai.retrieve.reranker import RerankResult

    return [
        RerankResult(score=0.95 - 0.03 * i, content=c.content, source=c.source, metadata=c.metadata)
        for i, c in enumerate(candidates[:top_k])
    ]


# ── 1. SDK Construction + agent_query (mock HTTP roundtrip) ───────────────────


def demo_sdk_construction() -> None:
    section("1. KonjoClient — SDK Construction & agent_query", "konjoai.sdk.KonjoClient")

    show_code(
        """
from konjoai.sdk import KonjoClient

client = KonjoClient(
    "http://kyro.local:8000",
    api_key="sk-demo-…",
    timeout=30.0,
)

response = client.agent_query("What is the refund policy?", top_k=3, max_steps=4)
print(response.answer)
""".strip()
    )

    from unittest.mock import MagicMock

    from konjoai.sdk import KonjoClient

    fake_payload = {
        "answer": "Refunds are accepted within 30 days with a receipt.",
        "model": "kyro-demo-stub",
        "usage": {"prompt_tokens": 11, "completion_tokens": 9},
        "sources": [
            {"source": "policy.md", "content_preview": "Refunds are accepted...", "score": 0.95},
        ],
        "steps": [
            {"thought": "Need policy docs", "action": "retrieve", "action_input": "refund policy", "observation": "[1 doc]"},
            {"thought": "Have enough evidence", "action": "finish", "action_input": "", "observation": "completed"},
        ],
        "telemetry": {"steps": {"agent": {"latency_ms": 248}}},
    }

    fake_resp = MagicMock(status_code=200, headers={}, text="ok")
    fake_resp.json.return_value = fake_payload

    with patch("konjoai.sdk.client.httpx.Client") as ctor:
        ctor.return_value.post.return_value = fake_resp
        client = KonjoClient("http://kyro.local:8000", api_key="sk-demo-XXXXXXXX")
        with console.status("[cyan]POST /agent/query…", spinner="dots"):
            time.sleep(0.4)
            response = client.agent_query("What is the refund policy?", top_k=3, max_steps=4)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column(style="white")
    table.add_row("answer", response.answer)
    table.add_row("model", response.model)
    table.add_row("usage", json.dumps(response.usage))
    table.add_row("sources", f"{len(response.sources)} doc(s)")
    table.add_row("steps", " → ".join(s.action for s in response.steps))
    console.print(Panel(table, title="[bold]Response[/]", border_style="green"))


# ── 2. Streaming agent — print SSE events as they arrive ──────────────────────


def demo_agent_stream() -> None:
    section("2. agent_query_stream — Live SSE Step Events", "POST /agent/query/stream")

    show_code(
        """
for event in client.agent_query_stream("What is the refund policy?"):
    if event.type == "step":
        print(f"  {event.data['action']:>10}  {event.data['observation'][:60]}")
    elif event.type == "result":
        print(f"\\n  ➜ {event.data['answer']}")
""".strip()
    )

    from konjoai.agent.react import RAGAgent

    script = [
        '{"thought":"Need refund policy","action":"retrieve","action_input":"refund policy","final_answer":""}',
        '{"thought":"I have enough","action":"finish","action_input":"","final_answer":"Refunds are accepted within 30 days with a receipt."}',
    ]
    gen = _StubGen(script=script)

    with patch("konjoai.agent.react.hybrid_search", _hybrid), patch(
        "konjoai.agent.react.rerank", _rerank
    ):
        agent = RAGAgent(top_k=3, max_steps=4)

        events = list(agent.run_stream("What is the refund policy?", generator=gen))

    body = Tree("[bold cyan]event stream[/]")
    final_answer = ""
    for ev in events:
        if ev["type"] == "step":
            colour = {"retrieve": "yellow", "finish": "green"}.get(ev["action"], "white")
            label = (
                f"[bold {colour}]step #{ev['index']}[/]  "
                f"[white]{ev['action']:<8}[/]  "
                f"[dim]{ev['observation'][:60]}[/]"
            )
            body.add(label)
        elif ev["type"] == "result":
            final_answer = ev["answer"]

    # Animate the tree growing one event at a time.
    with Live(console=console, refresh_per_second=10) as live:
        progressive = Tree("[bold cyan]event stream[/]")
        for ev in events:
            if ev["type"] == "step":
                colour = {"retrieve": "yellow", "finish": "green"}.get(ev["action"], "white")
                progressive.add(
                    f"[bold {colour}]step #{ev['index']}[/]  "
                    f"[white]{ev['action']:<8}[/]  "
                    f"[dim]{ev['observation'][:60]}[/]"
                )
                live.update(Panel(progressive, border_style="cyan", title="[bold]SSE[/]"))
                time.sleep(0.45)
        live.update(Panel(progressive, border_style="cyan", title="[bold]SSE[/]"))

    console.print()
    console.print(
        Panel(
            Text(final_answer, style="bold green"),
            border_style="green",
            title="[bold]final answer (result frame)[/]",
        )
    )


# ── 3. SemanticCache — miss → store → hit cycle ───────────────────────────────


_TOPIC_ANCHORS: dict[str, np.ndarray] = {}


def _topic_anchor(topic: str, dim: int = 64) -> np.ndarray:
    """Stable per-topic anchor vector, generated once and cached in-process."""
    if topic not in _TOPIC_ANCHORS:
        rng = np.random.default_rng(seed=abs(hash(topic)) % (2**32))
        v = rng.standard_normal(dim).astype(np.float32)
        _TOPIC_ANCHORS[topic] = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
    return _TOPIC_ANCHORS[topic]


def _embed(text: str, *, topic: str, jitter: float = 0.015, dim: int = 64) -> np.ndarray:
    """Toy embedder that places paraphrases of the same topic close together.

    Real production code routes through ``konjoai.embed.encoder``; the demo just
    needs to *demonstrate* that two paraphrases collapse to a high cosine.
    """
    anchor = _topic_anchor(topic, dim=dim)
    rng = np.random.default_rng(seed=abs(hash(text.lower())) % (2**32))
    noise = rng.standard_normal(dim).astype(np.float32) * jitter
    v = anchor + noise
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float32)
    b = b.ravel().astype(np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


@dataclass
class _StubResp:
    answer: str

    def model_copy(self, *, update: dict) -> "_StubResp":
        return _StubResp(answer=update.get("answer", self.answer))


def demo_semantic_cache() -> None:
    section(
        "3. SemanticCache — Miss → Store → Hit (cosine ≥ 0.95)",
        "konjoai.cache.SemanticCache",
    )

    show_code(
        """
from konjoai.cache import SemanticCache

cache = SemanticCache(max_size=100, threshold=0.95)
hit = cache.lookup(question, q_vec)         # miss → None
cache.store(question, q_vec, response)      # populate
hit = cache.lookup("how do refunds work?", q_vec_paraphrase)  # semantic hit
""".strip()
    )

    from konjoai.cache import SemanticCache

    cache = SemanticCache(max_size=10, threshold=0.95)

    queries = [
        ("What is the refund policy?", "refund", _StubResp(answer="30 days, with receipt.")),
        ("How do refunds work?", "refund", None),                # paraphrase → hit
        ("Tell me about your shipping rates.", "shipping", _StubResp(answer="Free over $50.")),
        ("How much does shipping cost?", "shipping", None),       # paraphrase → hit
    ]

    table = Table(title="lookup → action → outcome", border_style="grey42")
    table.add_column("#", style="dim", justify="right")
    table.add_column("query", style="white")
    table.add_column("cosine", justify="right")
    table.add_column("result", justify="left")

    stored_vecs: list[tuple[str, np.ndarray]] = []
    for i, (q, topic, resp) in enumerate(queries, start=1):
        v = _embed(q, topic=topic)
        # Compute best similarity vs anything already cached.
        best_sim = -1.0
        for _stored_q, sv in stored_vecs:
            best_sim = max(best_sim, _cosine(v, sv))

        hit = cache.lookup(q, v)
        if hit is None:
            # Cache miss — populate.
            payload = resp or _StubResp(answer=f"(answer for: {q})")
            cache.store(q, v, payload)
            stored_vecs.append((q, v))
            sim_str = "—" if best_sim < 0 else f"{best_sim:.3f}"
            table.add_row(str(i), q, sim_str, "[red]MISS[/]  → stored")
        else:
            sim_str = f"{best_sim:.3f}"
            table.add_row(
                str(i),
                q,
                sim_str,
                f"[green]HIT[/]  → {hit.answer}",
            )
        time.sleep(0.3)

    console.print(table)

    stats = cache.stats()
    stat_panel = Table(show_header=False, box=None, padding=(0, 2))
    stat_panel.add_column(style="cyan", justify="right")
    stat_panel.add_column(style="white")
    for key in ("size", "max_size", "threshold", "total_hits", "total_misses", "hit_rate"):
        stat_panel.add_row(key, str(stats[key]))
    console.print(Panel(stat_panel, title="[bold]cache.stats()[/]", border_style="cyan"))


# ── 4. RedisSemanticCache — same contract, distributed-shaped storage ────────


def demo_redis_cache() -> None:
    section(
        "4. RedisSemanticCache — Distributed Backend (Sprint 22)",
        "tenant-namespaced HASH + LRU ZSET",
    )

    show_code(
        """
from konjoai.cache import RedisSemanticCache, build_redis_cache

# Auto-detected via cache_backend="redis"; falls back to memory if PING fails.
cache = build_redis_cache(
    url="redis://localhost:6379/0",
    namespace="kyro:cache",
    max_size=500,
    threshold=0.95,
    ttl_seconds=3600,
)
""".strip()
    )

    # Use the in-process fake from the test suite for deterministic display.
    class _FakeRedis:
        def __init__(self) -> None:
            self.hashes: dict[str, dict[str, bytes]] = {}
            self.zsets: dict[str, dict[str, float]] = {}
            self.expires: dict[str, int] = {}

        def hset(self, k, f, v):
            self.hashes.setdefault(k, {})[f] = v
            return 1

        def hget(self, k, f):
            return self.hashes.get(k, {}).get(f)

        def hdel(self, k, f):
            return 1 if self.hashes.get(k, {}).pop(f, None) else 0

        def hgetall(self, k):
            return {f.encode(): v for f, v in self.hashes.get(k, {}).items()}

        def zadd(self, k, mapping):
            bucket = self.zsets.setdefault(k, {})
            for member, score in mapping.items():
                bucket[member] = score
            return len(mapping)

        def zrange(self, k, start, end):
            items = sorted(self.zsets.get(k, {}).items(), key=lambda kv: kv[1])
            sliced = items[start:] if end == -1 else items[start : end + 1]
            return [m.encode() for m, _ in sliced]

        def zrem(self, k, m):
            return 1 if self.zsets.get(k, {}).pop(m, None) is not None else 0

        def zcard(self, k):
            return len(self.zsets.get(k, {}))

        def delete(self, *keys):
            for kk in keys:
                self.hashes.pop(kk, None)
                self.zsets.pop(kk, None)
            return len(keys)

        def expire(self, k, s):
            self.expires[k] = s
            return 1

    from konjoai.auth.tenant import _current_tenant_id, set_current_tenant_id
    from konjoai.cache import RedisSemanticCache

    fake = _FakeRedis()
    cache = RedisSemanticCache(client=fake, namespace="kyro:cache", max_size=8, threshold=0.95)

    interactions = [
        ("acme", "What is our SLA?", _StubResp(answer="99.95% uptime per quarter.")),
        ("acme", "What is our SLA?", None),  # exact hit
        ("globex", "What is our SLA?", _StubResp(answer="99.9% uptime per month.")),
        ("acme", "What is our SLA?", None),  # acme hit again, distinct from globex
    ]

    table = Table(title="multi-tenant Redis cache trace", border_style="grey42")
    table.add_column("tenant", style="magenta")
    table.add_column("query", style="white")
    table.add_column("result", style="white")

    for tenant, q, resp in interactions:
        token = set_current_tenant_id(tenant)
        try:
            v = _embed(q, topic=f"sla-{tenant}")
            hit = cache.lookup(q, v)
            if hit is None:
                payload = resp or _StubResp(answer=f"(synth for {tenant})")
                cache.store(q, v, payload)
                table.add_row(tenant, q, "[red]MISS[/] → stored")
            else:
                table.add_row(tenant, q, f"[green]HIT[/] → {hit.answer}")
        finally:
            _current_tenant_id.reset(token)
        time.sleep(0.3)

    console.print(table)

    keyspace_tree = Tree("[bold cyan]Redis keyspace (fake client)[/]")
    for key, fields in fake.hashes.items():
        node = keyspace_tree.add(f"[yellow]{key}[/]  [dim]({len(fields)} entry)[/]")
        for f in fields:
            node.add(f"[white]{f}[/]")
    console.print(
        Panel(
            keyspace_tree,
            title="[bold]Tenant-Namespaced Keys[/]",
            border_style="magenta",
        )
    )

    console.print(
        Panel(
            Text(
                "K3 graceful degradation: build_redis_cache() returns None when the\n"
                "redis package is missing or PING fails — get_semantic_cache() falls\n"
                "back to the in-memory LRU so request paths never break.",
                style="italic dim",
            ),
            border_style="grey42",
        )
    )


# ── 5. Config loading from environment variables ─────────────────────────────


def demo_config() -> None:
    section("5. Settings — Env-Driven Configuration", "konjoai.config.get_settings()")

    show_code(
        """
import os
os.environ["CACHE_BACKEND"] = "redis"
os.environ["CACHE_REDIS_URL"] = "redis://kyro-cache.svc:6379/0"
os.environ["CACHE_REDIS_TTL_SECONDS"] = "3600"

from konjoai.config import get_settings
settings = get_settings()
print(settings.cache_backend, settings.cache_redis_url)
""".strip()
    )

    overrides = {
        "CACHE_ENABLED": "true",
        "CACHE_BACKEND": "redis",
        "CACHE_REDIS_URL": "redis://kyro-cache.svc.cluster.local:6379/0",
        "CACHE_REDIS_NAMESPACE": "kyro:cache",
        "CACHE_REDIS_TTL_SECONDS": "3600",
        "MULTI_TENANCY_ENABLED": "true",
        "RATE_LIMITING_ENABLED": "true",
        "RATE_LIMIT_REQUESTS": "120",
    }
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = v

        # Force a fresh load: the get_settings() helper caches globally.
        from konjoai.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()

        table = Table(show_header=True, header_style="bold cyan", border_style="grey42")
        table.add_column("setting")
        table.add_column("value", style="green")
        for attr in (
            "cache_enabled",
            "cache_backend",
            "cache_redis_url",
            "cache_redis_namespace",
            "cache_redis_ttl_seconds",
            "cache_max_size",
            "cache_similarity_threshold",
            "multi_tenancy_enabled",
            "rate_limiting_enabled",
            "rate_limit_requests",
            "rate_limit_window_seconds",
        ):
            table.add_row(attr, str(getattr(settings, attr)))
        console.print(Panel(table, title="[bold]Live settings[/]", border_style="cyan"))

    finally:
        # Restore the host environment so re-running the script is idempotent.
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        from konjoai.config import get_settings

        get_settings.cache_clear()


# ── Wrap-up ───────────────────────────────────────────────────────────────────


def render_outro() -> None:
    console.print()
    console.print(
        Panel(
            Group(
                Align.center(Text("Make it Konjo — build, ship, repeat.", style="bold cyan")),
                Align.center(
                    Text(
                        "ቆንጆ  ·  根性  ·  康宙  ·  कोहजो  ·  ᨀᨚᨐᨚ  ·  конйо  ·  건조  ·  কুঞ্জ",
                        style="dim",
                    )
                ),
                Align.center(Text("github.com/konjoai/kyro", style="italic blue")),
            ),
            border_style="cyan",
            padding=(1, 4),
        )
    )


def main() -> None:
    render_hero()
    demo_sdk_construction()
    demo_agent_stream()
    demo_semantic_cache()
    demo_redis_cache()
    demo_config()
    render_outro()


if __name__ == "__main__":
    main()

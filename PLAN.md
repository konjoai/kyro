# RagOS — Master Execution Plan

> **ቆንጆ** — Beautiful. **根性** — Fighting spirit. **康宙** — Health of the universe.
> *Make it konjo — build, ship, repeat.*

**Version:** v0.2.0-planned  
**Owner:** Wesley Scholl  
**Last Updated:** 2025-07 (Phase 2a execution)

---

## Thesis

RagOS is not another RAG tutorial. It is a production-grade, vertically-integrated retrieval system that demonstrates three things simultaneously:

1. **Systems engineering:** A pipeline with telemetry, routing, and graceful degradation — built like production software, not notebooks.
2. **Research implementation:** HyDE (Gao et al. 2022), Late Interaction (ColBERT), and query routing implemented from algorithm description, not copy-pasted from a library.
3. **Vertical integration:** Vectro (your own Mojo/Python embedding compression library) is used as a first-class component — not a demo — to quantize stored embeddings and report compression metrics.

The portfolio signal is: *"I built three production systems that work together."*

---

## The Seven Konjo Invariants

These are non-negotiable architectural properties. Any commit that violates one is a hard stop.

| # | Invariant | In Practice |
|---|---|---|
| K1 | **No silent failures.** | Every component returns a result or raises explicitly. No `except: pass`. |
| K2 | **Telemetry on every step.** | `timed()` context manager wraps all hot-path calls. Latency reported in every response. |
| K3 | **Graceful degradation everywhere.** | Vectro unavailable → float32 passthrough. RAGAS not installed → 501. BM25 not built → dense-only. |
| K4 | **Dtype contracts at boundaries.** | Encoder output: `float32`. Vectro input/output: `float32`. Qdrant vectors: `float32`. Assert, not assume. |
| K5 | **Zero new hard dependencies for new features.** | Telemetry uses `time.perf_counter()`. Router uses `re`. HyDE reuses the existing generator. |
| K6 | **Backward-compatible API evolution.** | New response fields are optional with sensible defaults. Existing API consumers don't break. |
| K7 | **Reproducible evals.** | Every RAGAS run serialized to `evals/runs/<timestamp>_<name>/`. Never overwrite. Seeds logged. |

---

## Current State: v0.1.0 (Baseline Complete)

All 41 scaffold files are implemented and passing. The v0.1.0 pipeline:

```
User Question
    │
    ▼
hybrid_search(question)       ← Qdrant dense + BM25 sparse, RRF fusion (α=0.7)
    │
    ▼
CrossEncoderReranker.rerank() ← ms-marco-MiniLM-L-6-v2, top-k selection
    │
    ▼
Generator.generate()          ← OpenAI / Anthropic / Squish backend
    │
    ▼
QueryResponse(answer, sources, model, usage)
```

**What v0.1.0 lacks:**
- Per-step timing (no telemetry, cannot profile bottlenecks)
- HyDE (short query → long doc embedding mismatch unaddressed)
- Query routing (Qdrant called for "hello" greetings — wasteful)
- Vectro quantization integration (your own library not used)
- Intent inference in response (downstream consumers can't filter)

---

## Gap Analysis: v0.1.0 → v0.2.0

| Component | File | Status | Priority |
|---|---|---|---|
| Pipeline telemetry | `ragos/telemetry.py` | ❌ Missing | P0 — enables profiling everything else |
| Query intent router | `ragos/retrieve/router.py` | ❌ Missing | P0 — eliminates wasted Qdrant calls |
| HyDE retrieval mode | `ragos/retrieve/hyde.py` | ❌ Missing | P1 — closes embedding mismatch gap |
| Vectro bridge | `ragos/embed/vectro_bridge.py` | ❌ Missing | P1 — vertical integration demo |
| Config expansion | `ragos/config.py` | ❌ 5 keys missing | P0 — gating all above |
| Schema expansion | `ragos/api/schemas.py` | ❌ 3 fields missing | P0 — surface telemetry + intent |
| Query route rewrite | `ragos/api/routes/query.py` | ❌ Not wired | P0 — wire all components |

---

## Architecture: v0.2.0 Target

```
User Question
    │
    ├── [if enable_query_router]
    │       ▼
    │   classify_intent()       ← O(1) regex, no model
    │   ┌── CHAT ──────────────→ Early return (no Qdrant call)
    │   ├── AGGREGATION ────────→ decompose_query() → multiple sub-queries
    │   └── RETRIEVAL ──────────→ continue ↓
    │
    ├── [if use_hyde or enable_hyde]
    │       ▼
    │   generate_hypothesis()   ← generator.generate(hyde_prompt)
    │   hyde_encode()           ← encode(hypothesis) instead of encode(query)
    │       ▼
    │   effective_question = hypothesis_text
    │
    ▼
hybrid_search(effective_question)      ← timed("hybrid_search")
    │
    ▼
CrossEncoderReranker.rerank()          ← timed("rerank")
    │
    ▼
Generator.generate()                    ← timed("generate")
    │
    ▼
QueryResponse(
    answer, sources, model, usage,
    telemetry={step: ms, ...},         ← NEW: per-step latency breakdown
    intent="retrieval"                  ← NEW: classified intent
)
```

---

## 4-Week Execution Plan

### Week 1 — Instrumentation & Routing (Core Infrastructure)

**Goal:** Every request is timed. Chat queries never touch Qdrant.

| Deliverable | File | Gate |
|---|---|---|
| PipelineTelemetry | `ragos/telemetry.py` | `pytest tests/test_telemetry.py` passes |
| QueryIntent router | `ragos/retrieve/router.py` | CHAT → early return verified |
| Config expansion | `ragos/config.py` | 5 new settings with defaults |
| Schema expansion | `ragos/api/schemas.py` | 3 new fields |
| Wired query route | `ragos/api/routes/query.py` | Full 5-step pipeline |
| Baseline RAGAS run | `evals/runs/baseline_v010/` | Faithfulness measured |

**Verify Gate:** `pytest tests/ --timeout=60 -x -q` — zero failures.

---

### Week 2 — Semantic Enhancement (HyDE)

**Goal:** Hypothesis-augmented retrieval measurably improves recall.

| Deliverable | File | Gate |
|---|---|---|
| HyDE implementation | `ragos/retrieve/hyde.py` | `pytest tests/test_hyde.py` passes |
| `use_hyde` field wired | `ragos/api/routes/query.py` | `use_hyde=True` returns `telemetry.steps["hyde"]` |
| HyDE vs baseline eval | `evals/runs/v020_hyde_vs_baseline/` | Δ faithfulness documented |

**Verify Gate:** RAGAS Faithfulness ≥ 0.80 OR a documented analysis of why the dataset limits HyDE gains.

---

### Week 3 — Vertical Integration (Vectro)

**Goal:** Your own embedding compression library is a first-class component.

| Deliverable | File | Gate |
|---|---|---|
| Vectro bridge | `ragos/embed/vectro_bridge.py` | INT8 ratio ≥ 4×, cosine_sim ≥ 0.9999 |
| Float32 passthrough fallback | same | `_check_vectro() = False` → passthrough |
| Vectro benchmark | `evals/benchmarks/vectro_compression.json` | ratio + sim logged |
| `vectro_quantize` flag wired | `ragos/embed/encoder.py` or ingest | Metrics in telemetry |

**Verify Gate:** `pytest tests/test_vectro_bridge.py` passes. Compression ratios logged and archived in `benchmarks/results/`.

---

### Week 4 — Late Interaction (ColBERT-style) & Final Portfolio

**Goal:** MaxSim scoring replaces simple cosine as a retrievable extra scoring pass.

| Deliverable | File | Gate |
|---|---|---|
| MaxSim implementation | `ragos/retrieve/late_interaction.py` | Shape contracts: `(Q, D) × (K, S, D) → (K,)` |
| `use_colbert` flag | `ragos/config.py`, query route | Optional scoring pass |
| Full eval suite | `evals/runs/v030_final/` | Context Precision ≥ 0.75 |
| README final | `README.md` | Architecture diagram, benchmark table, one-command demo |
| Blog post draft | `docs/blog_post.md` | Wraps the narrative for portfolio |

**Verify Gate:** All v0.3.0 success metrics met.

---

## File Manifest (v0.2.0 additions)

```
RagOS/
├── PLAN.md                          ← this file
├── SESSION.md                       ← active session state tracker
├── ragos/
│   ├── telemetry.py                 ← NEW: StepTiming, PipelineTelemetry, timed()
│   ├── embed/
│   │   └── vectro_bridge.py         ← NEW: Vectro INT8 bridge + fallback
│   ├── retrieve/
│   │   ├── hyde.py                  ← NEW: HyDE hypothesis generation + encoding
│   │   └── router.py                ← NEW: QueryIntent classifier + decomposer
│   ├── config.py                    ← MODIFIED: +5 settings
│   └── api/
│       ├── schemas.py               ← MODIFIED: +3 fields
│       └── routes/
│           └── query.py             ← MODIFIED: 5-step wired pipeline
└── CHANGELOG.md                     ← MODIFIED: [Unreleased] section
```

---

## Success Metrics

| Metric | Target | How Measured |
|---|---|---|
| Faithfulness | ≥ 0.80 | RAGAS on 25-question test set |
| Context Precision | ≥ 0.75 | RAGAS on 25-question test set |
| Vectro INT8 compression ratio | ≥ 4× | `vectro_bridge.compression_ratio()` |
| Vectro INT8 cosine similarity | ≥ 0.9999 | `vectro.mean_cosine_similarity()` |
| CHAT intent early-return latency | < 5 ms | `telemetry.steps["route"].duration_ms` |
| Full RETRIEVAL pipeline latency | < 1000 ms | `telemetry.total_ms()` |
| Test suite | 0 failures | `pytest tests/ --timeout=60` |
| All K1–K7 invariants | Passing | Code review + test assertions |

---

## Hard Stops (do not proceed past these)

- Tests failing from a previous step. Fix them first.
- Dtype boundary assertion fails in Vectro bridge (float32 in, float32 out — assert both).
- NaN/Inf in Vectro reconstructed embeddings (assert before passing to Qdrant).
- RAGAS Faithfulness < 0.70 after HyDE (investigate distribution before calling it a regression).
- Any new module imported at startup that increases import time by > 10%.

---

*End of PLAN.md*  
*Update this file when architectural contracts change. Never let it drift from the actual implementation.*

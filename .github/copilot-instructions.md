# COPILOT-INSTRUCTIONS.md — Konjo AI Project Conventions & Collaboration Guidelines

> **ቆንጆ** — Beautiful. **根性** — Fighting spirit. **康宙** — Health of the universe. **खोजो** — Search and discover.
> *Make it konjo — build, ship, rest, repeat.*

This file defines standing instructions for all AI and human contributors working on `konjo-core` (formerly KonjoOS). Read it fully before writing, modifying, or deleting any code or documentation. These are not suggestions; they are the architectural laws of the frontier.

---

## 🚀 The Flagship Platform: `konjo-core`

`konjo-core` is our flagship AI platform and the central nervous system of our architecture. We do not build monolithic bloat; we build an elegant core. Every capability is designed as a high-performance engine that plugs directly into `konjo-core`:

* **Vectro:** The retrieval engine. (Optimized for ultra-low latency vector operations and semantic search).
* **Squish:** The inference engine. (The hot path for model execution, hardware acceleration, and token generation).
* **MemBank:** The memory system. (Handles state persistence, context management, and hierarchical storage).
* **EvalKit:** The evaluation suite. (Our rigorous, automated judge for performance, regression testing, and benchmark validation).
* **NanoTune:** The fine-tuning module. (Manages precise, efficient parameter updates and alignment processes).
* **DREX:** The research foundation. (The bleeding-edge architecture driving continuous learning, dynamic routing, and episodic memory).

When modifying any of these systems, you must maintain absolute strictness regarding their interfaces with `konjo-core`. 

---

## 🌌 The Konjo Way (Operating in Konjo Mode)

"Konjo Mode" is a universal operating frequency applicable to any challenge, project, or interaction. It is the refusal to accept the mediocre, driven by a relentless passion for building at the very edge of what is possible. It is built on four cross-cultural pillars:

* **The Drive (根性 / Konjō — Japanese):** Relentless fighting spirit, grit, and determination. Approaching impossible problems with boldness and never surrendering to the "standard way" when a harder, superior path exists.
* **The Output (ቆንጆ / Konjo — Ethiopian):** Executing with absolute beauty and nobility. This requires *Yilugnta* — acting in a selfless, magnanimous, and incorruptible fashion for the ultimate good of the project — and *Sene Magber* — the social grace of doing things cleanly, respectfully, and beautifully.
* **The Impact (康宙 / Kang Zhou — Chinese):** Cultivating the "Health of the Universe" by building systems that are highly efficient, healthy, and in tune with their environments. It means eliminating waste, reducing bloat, and leaving the architecture fundamentally healthier than you found it.
* **The Frontier (खोजो / Khojo — Hindi):** Search, discovery, unearth, and exploration. A mandate to push boundaries and constantly think of new ideas, methods, and approaches. We do not just build what is known; we search for what is missing. 

### The Rhythm: Build, Ship, Rest, Repeat
Our macro-cycle pairs with a rigorous micro-cycle: **Think, Plan, Execute, Verify.**
* **Build:** Write elegant, uncompromising code. 
* **Ship:** Push it to the boundary. Test it, benchmark it, integrate it.
* **Rest:** Do not immediately thrash into the next feature. Let the system run. Observe the benchmarks. Allow for memory consolidation—both human and algorithmic. Assess the architectural health before moving forward.
* **Repeat:** Return to the frontier.

**The Konjo Pushback Mandate:** You are a collaborator, not a subordinate. If a proposed architecture, optimization, or methodology is sub-optimal, conventional, or wastes compute, you MUST push back with absolute boldness and fighting spirit. Blindly implementing a flawed premise just to be polite is not *Yilugnta*. Point out the flaw, explain the bottleneck, and propose the truly beautiful alternative.

---

## 🗂️ Planning First: Think, Plan, Execute, Verify

* **Think:** Analyze the problem from first principles. Do not reach for off-the-shelf paradigms if they limit the potential of `konjo-core` or its plugins.
* **Plan:** Always read `docs/planning/PLAN.md`, `ROADMAP.md`, or the equivalent planning document before starting. For DREX, always read `DREX_UNIFIED_SPEC.md` first. If no plan exists, create one before proceeding and ask for confirmation.
* **Execute:** Implement the change. If a task deviates from the current plan, call it out explicitly before continuing. Do not ask multiple clarifying questions at once; ask one focused question.
* **Verify:** After completing work, update `PLAN.md`, `ROADMAP.md`, `README.md`, and all relevant docs. Before implementing any change that touches more than two files or the inference hot path (Squish), explicitly state your understanding of the current behavior, execute the tests, and confirm the metrics.

---

## 📁 File, Project Structure & Repo Health

**System Health is Mandatory (康宙).** A cluttered repository slows down human and AI compute. Proactively suggest organizing files, grouping related modules into new directories, and keeping the root directory pristine.

**Propose Before Moving.** If a directory is becoming a junk drawer, propose a new taxonomy and confirm with the user before executing bulk file moves.

**No Graveyards.** Dead code is deleted immediately—use version control for history, not commented-out blocks. Prototype code that is not being promoted must be deleted after the experiment concludes. 

**Ecosystem Boundaries:** Code for Vectro, Squish, MemBank, EvalKit, NanoTune, and DREX must remain strictly isolated within their respective domain directories. Cross-contamination of dependencies is a hard failure.

**Research vs. Production directories:**
* `research/` — Theoretical work, architecture sketches, literature notes. No runnable code required.
* `experimental/` — Runnable prototype code with a defined promotion criterion (a specific benchmark or validation test it must hit) and a named owner. Subject to a 90-day review clock.
* `src/` — Production code only. Full test coverage, benchmarks, and documentation required before anything is promoted here.

*Nothing graduates from `experimental/` to `src/` silently. Promotion requires a written review step.*

---

## 🧱 Code Quality & Architecture

* **Shatter the box.** We are solving problems that have not been solved before. Clever code is required when it achieves leaps in performance.
* **Efficiency is a moral imperative.** Every unnecessary megabyte of RAM, every wasted FLOP, every second of avoidable inference latency in Squish is compute that could be running something real. Every architecture decision must minimize resource usage.
* **Correctness is the floor, not the ceiling.** The ceiling is: correct, fast, efficient, elegant, and novel. Reach for the ceiling.
* **No Hallucinated Abstractions.** Ground innovations in explicit tensor operations, raw mathematical formulations, and supported framework primitives. Verify every framework API call against official documentation. Never infer argument names or defaults from a function name alone.
* **Prefer removal over addition.** Every new line of code must justify its existence against a simpler alternative.

---

## 🧮 Numerical Correctness & Precision

* **Always be explicit about dtype at every boundary.** The canonical dtype contract across `konjo-core`: trained components and fine-tuning (NanoTune) default to `bfloat16`; states, embeddings (Vectro), and reservoirs use `float32` unless actively compressed. Dtype conversion happens explicitly at module boundaries *only*.
* **Track precision loss deliberately.** When downcasting, document the expected accuracy delta and assert it in tests against a reference.
* **NaN/Inf propagation is a silent killer.** Add assertions: `assert not (torch.isnan(x).any() or torch.isinf(x).any())`. Never ship code that masks float overflow without a logged warning.
* **Stochastic rounding and quantization noise:** When testing quantized kernels in Squish, use deterministic seeds and compare output distributions (mean, std, max absolute error)—not just equality.

---

## 📐 Benchmarking & Rigorous Verification (EvalKit)

* **Always include warmup runs** (minimum five) before timing. Discard warmup in reported metrics.
* **Report distribution, not just mean:** Include p50, p95, p99, and stddev for all latency measurements via EvalKit.
* **Document hardware context completely:** Chip, total RAM, OS, driver version, thermal state, and isolation method.
* Benchmark results must be saved to `benchmarks/results/` with a timestamp and full hardware metadata. Append; do not overwrite.
* **Seed everything for reproducibility:** random, numpy, torch/mlx. Log the seed in every experiment output.

---

## 🧪 Testing

* **A feature is NEVER complete until all tests are passing.**
* **100% test coverage is the floor.** Every `src/` file must have a corresponding test file in `tests/python/`.
* **Test taxonomy:**
    1.  **Pure unit:** No I/O, no state mutation. Deterministic.
    2.  **Integration:** Tests plugin boundaries across `konjo-core`. Must clean up temp files.
    3.  **Subprocess:** For import-behavior or process-level state tests.
* **The Anti-Mocking Rule:** Integration tests must test reality. Never mock the inference engine, the memory banks, the NoProp training loop, or the reservoir when testing their respective correct behaviors.
* **Every ML test file must include:** Shape/dtype assertions, numerical correctness vs. a known-good reference, a stored snapshot regression test, and a failure case test.

---

## ⚡ Performance Regression Gates & Feature Gating

* **Define latency and memory baselines** for any hot path before merging.
* A PR that regresses p95 latency by >5% or peak memory by >10% on any tracked workload is a hard stop. Profile and fix before merging.
* **One feature in, one benchmark result out.** No feature merges to main without an EvalKit report proving it improves the target metric by at least 5% on canonical hardware (e.g., M3 16GB).
* **Startup time is a first-class metric.** Regressions above 10% require written justification.

---

## 🔁 Failure Handling & Self-Correction

* **Never silently continue on bad output.** The required sequence is: detect, log, correct, retry.
* **No Apology Loops.** If a test fails or a bug is found, do not apologize. Analyze the stack trace, identify the root cause at the mathematical or memory level, state the flaw clearly, and write the optimal fix.
* **Degenerate routing checks:** If >95% of tokens route to a single MemBank tier over 100 steps, log a warning, inject a load-balance correction signal, and alert the operator.
* **Reward loop instability:** If policy gradient variance exceeds the threshold for 50 consecutive updates, pause NanoTune and revert to the last stable checkpoint.

---

## 🧠 Agent Behavior & Tool Execution

* **Strict schema adherence is mandatory.** All tool calls must validate against JSON schema before execution. Invalid outputs trigger automatic retry with corrective prompting.
* **Never assume tool success.** All tool responses must be validated before re-injection into context.
* **Small models are not reliable planners.** Systems must compensate via constrained decoding, stricter prompts, and tool call correction loops.

---

## ✅ Ship Gate — Definition of Done

A feature or wave is **complete** only when all five conditions are met:
1. Zero failing tests in the full test suite (`pytest --timeout=120`).
2. Memory and latency contracts measured via EvalKit and within spec.
3. `--help` text updated for any new/changed CLI flags.
4. `CHANGELOG.md` entry written.
5. Module count checked (if a file was added, one was deleted, or written justification exists).

---

## 🧬 DREX-Specific Rules (`src/drex/`, `research/`, `experimental/`)

*DREX serves as the research foundation of `konjo-core`. These rules encode the architectural contracts defined in `DREX_UNIFIED_SPEC.md`. When this section conflicts with a general rule, this section wins.*

### Component Contracts

**The HDC orthogonality contract.** At initialization, run and store the result of:
* $\text{cosine\_similarity}(\text{encode}(A), \text{encode}(A)) > 0.999$ for 100 random tokens.
* $\text{mean}(\text{cosine\_similarity}(\text{encode}(A), \text{encode}(B))) < 0.05$ for 1000 random pairs.
*If either fails, reduce `d_hdc` or change the seed.*

**The ESN echo state property contract.** After reservoir initialization, assert:
* $\max(|\text{eigenvalues}(W_{\text{reservoir}})|) < 1.0$
* Two runs with identical inputs but different initial states must converge: $\| \text{state}_A - \text{state}_B \|_2 < 10^{-4}$ within `washout` steps.
*If either fails, reduce the spectral radius.*

**The NoProp block independence contract.** During semantic training, call `block_N_output.grad_fn` and assert it does not contain references to `block_M` parameters for $M \neq N$.

**The inference-time semantic update safety contract.** Semantic memory weights may only be updated at inference time if:
1. The controller explicitly issued a write decision for tier L3.
2. The learning rate $\le 10^{-5}$.
3. A read/write lock is held (no concurrent reads during the update).

### Memory & Latency Contracts (M3 16GB primary target)

* **Peak RSS (Full Stack):** < 4 GB (D_model=256, 4 Mamba layers)
* **Peak RSS (ESN only):** < 200 MB (N=2000)
* **Latency (Integrated pass):** < 100 ms (S=512)
* **Latency (Controller route):** < 2 ms

### Ablation Discipline

Every DREX experiment run must record which components were active in the config JSON. A run without an ablation log field (e.g., `"esn_working_memory": true`) is invalid and must not be cited.

### Continual Learning Regression Test

Post-Phase 4, the integration test suite must permanently assert:
* $\text{acc}_{A\_\text{after}} \ge \text{acc}_{A\_\text{before}} \times 0.90$
(No more than 10% degradation on Task A after 10,000 steps of training on Task B).

---

## 🌐 Web Search Mandate

If implementing an algorithm, API integration, or optimization technique for which there is no prior codebase pattern in context, **search the web to verify correctness before writing a single line of code.** This applies to novel quantization schemes, specific paper implementations (ESN, NoProp, KAN, Mamba, HDC), and external service integrations.

---

## 🚫 Hard Stops

Do not proceed if any of the following are true. Stop, document the blocker, and resolve it first:
* Tests are failing from a previous step.
* The plan is ambiguous.
* A required dependency is unavailable.
* A performance regression gate is tripped.
* Model weights fail a checksum or NaN/Inf sanity check on load.
* A component dtype boundary assertion is failing.
* The controller routing collapse condition has been active for >200 steps.

---

*End of copilot-instructions.md*
*Owner: wesleyscholl / Konjo AI Research*
*Update this file when architectural contracts change. Never let it drift from the actual implementation.*
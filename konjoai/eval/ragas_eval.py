from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def evaluate(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
    *,
    llm_model: str = "gpt-4o-mini",
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
) -> dict[str, float]:
    """Evaluate RAG outputs with RAGAS.

    Parameters
    ----------
    questions:
        The user queries used to generate the answers.
    answers:
        Generated answers (one per question).
    contexts:
        Retrieved context chunks per question (outer list = per-question).
    ground_truths:
        Optional reference answers; required for ``context_recall``.
    llm_model:
        Judge LLM model name (default: ``gpt-4o-mini``).
    llm_api_key:
        API key for the judge LLM.  Uses ``OPENAI_API_KEY`` env var if *None*.
    llm_base_url:
        Base URL override for OpenAI-compatible endpoints (e.g. Squish/Ollama).

    Returns
    -------
    dict[str, float]
        Keys: ``faithfulness``, ``answer_relevancy``, ``context_precision``,
        and ``context_recall`` (only when ground_truths are provided).

    Target benchmarks (from README)
    --------------------------------
    - faithfulness       ≥ 0.75
    - answer_relevancy   ≥ 0.80
    """
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate as ragas_evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (  # noqa: PLC0415
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig as _RunConfig
        import asyncio as _asyncio    # asyncio.sleep yields event loop; safe inside coroutines
        import threading as _threading  # threading.Lock for cross-loop slot reservation
        import time as _time
    except ImportError as e:
        raise ImportError(
            "RAGAS and dependencies are required:\n"
            "  pip install ragas datasets langchain-openai\n"
        ) from e

    data: dict[str, list] = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths is not None:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, context_precision]
    if ground_truths is not None:
        metrics.append(context_recall)

    # RAGAS 0.4.3 fans all N×M jobs out concurrently via asyncio.gather().
    # asyncio.Semaphore(1) proved unreliable: RAGAS spins each metric in its own
    # asyncio.run() scope, so the semaphore is invisible across those loops.
    # threading.Lock is process-wide and works regardless of event-loop topology.
    _tlock = _threading.Lock()        # brief critical section — slot reservation only
    _next_slot: list = [_time.monotonic()]  # monotonic time of next free rate-limit slot
    _rpm_interval: float = 1.0        # squish (local, no rate limit) — 1 s gap keeps MLX happy

    class _ThrottledChatOpenAI(ChatOpenAI):
        """Rate-limits to ≤ 3 RPM via slot reservation + asyncio.sleep.

        * threading.Lock: atomically assigns each call a time slot.
          Process-wide — works across threads / separate event loops.
        * asyncio.sleep: waits for the assigned slot WITHOUT blocking the loop.
          The event loop stays live during the wait; other coroutines progress.
        * Result: calls fire at T≈0, T+22s, T+44s … regardless of topology.
        """
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            # smoke signal — appears before any sleep so we know this path is reached
            logger.info("THROTTLE: _agenerate reached (n_msg=%d)", len(messages))
            # Atomically reserve the next slot (nanosecond critical section; no I/O).
            with _tlock:
                _now = _time.monotonic()
                _slot = max(_now, _next_slot[0])
                _next_slot[0] = _slot + _rpm_interval
            _delay = _slot - _time.monotonic()
            logger.info("THROTTLE: slot reserved, delay=%.1fs", max(0.0, _delay))
            if _delay > 0:
                await _asyncio.sleep(_delay)  # yields event loop — never blocks it
            return await super()._agenerate(messages, stop, run_manager, **kwargs)

    # Build evaluator LLM.  max_retries=0 on the SDK prevents the sub-second
    # internal retry loop; RAGAS tenacity handles the outer retry with backoff.
    _chat_kwargs: dict = {"model": llm_model, "max_retries": 0, "max_tokens": 4096}
    if llm_api_key is not None:
        _chat_kwargs["api_key"] = llm_api_key
    if llm_base_url is not None:
        _chat_kwargs["base_url"] = llm_base_url
    evaluator_llm = LangchainLLMWrapper(_ThrottledChatOpenAI(**_chat_kwargs))

    for _metric in metrics:
        _metric.llm = evaluator_llm

    _n_calls = len(questions) * len(metrics)
    logger.info(
        "Running RAGAS evaluation: %d samples × %d metrics = %d calls, ~%.0fs at 1s interval (squish)",
        len(questions), len(metrics), _n_calls, _n_calls * _rpm_interval,
    )

    # timeout per-job wall-clock from dispatch: needs to cover full serialised queue wait
    # (n_calls × 22s) plus the actual API round-trip. 600s handles up to ~25 jobs safely.
    _run_config = _RunConfig(timeout=600, max_wait=60, max_retries=5)
    result = ragas_evaluate(dataset, metrics=metrics, llm=evaluator_llm, run_config=_run_config)
    scores: dict[str, float] = result.to_pandas().mean(numeric_only=True).to_dict()

    logger.info(
        "RAGAS scores: faithfulness=%.3f context_precision=%.3f",
        scores.get("faithfulness", float("nan")),
        scores.get("context_precision", float("nan")),
    )
    return scores


if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation against KonjoOS corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-name", required=True, help="Label for this eval run (used in output directory)")
    parser.add_argument("--corpus", default="evals/corpus/eval_questions.json",
                        help="Path to eval corpus JSON")
    parser.add_argument("--n-samples", type=int, default=25,
                        help="Maximum number of corpus items to evaluate")
    parser.add_argument("--mock", action="store_true",
                        help="Use ground_truth as the answer (upper-bound harness test, no LLM needed)")
    parser.add_argument("--live-retrieval", action="store_true",
                        help="Call hybrid_search() per question instead of using pre-baked corpus context_docs")
    args = parser.parse_args()

    # --- Load corpus ---
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.error("Corpus not found: %s", corpus_path)
        sys.exit(1)
    with corpus_path.open() as _f:
        corpus: list[dict] = json.load(_f)
    corpus = corpus[: args.n_samples]
    logger.info("Loaded %d items from %s", len(corpus), corpus_path)

    questions: list[str] = [item["question"] for item in corpus]
    ground_truths: list[str] = [item["ground_truth"] for item in corpus]

    # --- Contexts: live retrieval or pre-baked corpus ---
    if args.live_retrieval:
        from konjoai.retrieve.hybrid import hybrid_search
        contexts = []
        for _q in questions:
            _results = hybrid_search(_q)
            contexts.append([_r.content for _r in _results])
        logger.info("Live retrieval: fetched contexts for %d questions", len(questions))
    else:
        contexts = [item["context_docs"] for item in corpus]  # list[list[str]]

    # --- Resolve judge LLM config (passed as kwargs to evaluate()) ---
    _llm_api_key: str | None = None
    _llm_base_url: str | None = None
    _llm_model: str = "gpt-4o-mini"
    try:
        from konjoai.config import get_settings
        _s = get_settings()
        _llm_model = _s.ragas_llm
        if _s.generator_backend == "squish":
            _llm_api_key = "squish"
            _llm_base_url = _s.squish_base_url
            logger.info("RAGAS LLM → Squish at %s", _s.squish_base_url)
        elif _s.openai_api_key:
            _llm_api_key = _s.openai_api_key
            logger.info("RAGAS LLM → OpenAI (%s)", _llm_model)
        else:
            logger.info("RAGAS LLM → OpenAI (%s) via OPENAI_API_KEY env var", _llm_model)
    except Exception as _cfg_err:
        logger.warning("Could not load settings: %s — RAGAS LLM will use env vars", _cfg_err)

    # --- Generate answers ---
    if args.mock:
        answers: list[str] = ground_truths
        logger.info("--mock mode: using ground_truths as answers (upper-bound, harness only)")
    else:
        from konjoai.generate.generator import get_generator
        _gen = get_generator()
        answers = []
        for _item in corpus:
            _ctx_text = "\n\n".join(_item["context_docs"])
            _result = _gen.generate(_item["question"], _ctx_text)
            answers.append(_result.answer)
        logger.info("Generated %d answers via %s", len(answers), _gen.__class__.__name__)

    # --- Run RAGAS ---
    scores = evaluate(
        questions, answers, contexts, ground_truths,
        llm_model=_llm_model,
        llm_api_key=_llm_api_key,
        llm_base_url=_llm_base_url,
    )

    # --- K7: write to timestamped output directory (never overwrite) ---
    _ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("evals/runs") / f"{_ts}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "run_name": args.run_name,
        "n_samples": len(corpus),
        "corpus": str(corpus_path),
        "mock": args.mock,
        "live_retrieval": args.live_retrieval,
        "generator_backend": os.environ.get("GENERATOR_BACKEND", "unknown"),
        "timestamp": _ts,
    }, indent=2))

    print(f"\n=== RAGAS Results: {args.run_name} ===")
    for _k, _v in sorted(scores.items()):
        print(f"  {_k}: {_v:.4f}")

    # --- Gate check ---
    _FAITH_GATE = 0.80
    _CP_GATE = 0.75
    _faith = scores.get("faithfulness", 0.0)
    _cp = scores.get("context_precision", 0.0)
    _passed = _faith >= _FAITH_GATE and _cp >= _CP_GATE
    print(
        f"\nGate: faithfulness≥{_FAITH_GATE} {'✓' if _faith >= _FAITH_GATE else '✗'}  "
        f"context_precision≥{_CP_GATE} {'✓' if _cp >= _CP_GATE else '✗'}"
    )
    print(f"Overall: {'PASS ✓' if _passed else 'FAIL ✗'}")
    print(f"Results saved to: {run_dir}")
    sys.exit(0 if _passed else 1)

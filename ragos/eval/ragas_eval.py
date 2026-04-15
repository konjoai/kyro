from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def evaluate(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
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
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        raise ImportError(
            "RAGAS and HuggingFace datasets are required:\n"
            "  pip install ragas datasets\n"
        ) from e

    data: dict[str, list] = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths is not None:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy, context_precision]
    if ground_truths is not None:
        metrics.append(context_recall)

    logger.info("Running RAGAS evaluation on %d samples with %d metrics", len(questions), len(metrics))

    result = ragas_evaluate(dataset, metrics=metrics)
    scores: dict[str, float] = result.to_pandas().mean(numeric_only=True).to_dict()

    logger.info(
        "RAGAS scores: faithfulness=%.3f answer_relevancy=%.3f context_precision=%.3f",
        scores.get("faithfulness", float("nan")),
        scores.get("answer_relevancy", float("nan")),
        scores.get("context_precision", float("nan")),
    )
    return scores

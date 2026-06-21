"""Generation-quality metrics via Ragas (KEY-GATED + optional extra).

Computes faithfulness + answer-relevancy with a pinned Anthropic judge — the only
part of the eval needing the API key. When the key or ``ragas`` is missing,
``ragas_available()`` is False and the runner records generation metrics as
skipped; the retrieval A/B still runs.

NOTE (dependency tension): the runtime env pins ``langchain-core``/``langgraph``
1.x for the CRAG graph, but current ``ragas`` (0.4.x) is import-incompatible with
that stack. So Ragas lives in the optional ``[eval]`` extra and is intended to run
in an isolated environment / matching pin when the key is available. This module
targets the classic ``ragas.evaluate(dataset, metrics, llm)`` API; pin a ragas
version whose API matches when you enable it.
"""
from __future__ import annotations

import os

from rag.config import RagConfig


def ragas_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import langchain_anthropic  # noqa: F401
        import ragas  # noqa: F401
    except ImportError:
        return False
    return True


def evaluate_generation(samples: list[dict], config: RagConfig) -> dict:
    """samples: [{question, answer, contexts: list[str]}] -> faithfulness/answer_relevancy.

    Returns a dict with metric means, the pinned judge model, and N (for honest
    reporting). Raises only if called when ragas_available() is False.
    """
    if not ragas_available():
        return {"skipped": True, "reason": "no ANTHROPIC_API_KEY or ragas not installed"}

    from datasets import Dataset
    from langchain_anthropic import ChatAnthropic
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    judge = ChatAnthropic(model=config.model, temperature=0.0)
    dataset = Dataset.from_dict({
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
    })
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=judge)
    df = scores.to_pandas()
    return {
        "skipped": False,
        "judge_model": config.model,
        "n": len(samples),
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
    }

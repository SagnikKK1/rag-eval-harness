"""Deterministic retrieval metrics — recall@k, MRR, nDCG, refusal accuracy (no LLM).

A retrieved chunk is a *hit* iff its ``source_ids`` intersect the item's
``gold_source_ids``. recall@k = fraction of answerable questions with >=1 hit in
the top-k; MRR = mean reciprocal rank of the first hit; nDCG@k uses binary gains
with the ideal DCG computed from the item's true number of relevant chunks (so
ranking *all* the evidence high is rewarded, not just the first hit). Refusal
accuracy is scored on the not-in-corpus items.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2

K_VALUES = (1, 3, 5, 10)


def hit_ranks(retrieved_source_ids: list[set[str]], gold: set[str]) -> list[bool]:
    """Per-rank hit flags for one question (rank 0 = top result)."""
    return [bool(ids & gold) for ids in retrieved_source_ids]


def _recall_at_k(hits: list[bool], k: int) -> float:
    return 1.0 if any(hits[:k]) else 0.0


def _reciprocal_rank(hits: list[bool]) -> float:
    for i, hit in enumerate(hits):
        if hit:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(hits: list[bool], n_relevant: int, k: int) -> float:
    """Binary-gain nDCG@k. ``n_relevant`` = the item's true number of relevant
    chunks in the corpus (ideal DCG places min(k, n_relevant) hits on top)."""
    if n_relevant <= 0:
        return 0.0
    dcg = sum(1.0 / log2(i + 2) for i, hit in enumerate(hits[:k]) if hit)
    idcg = sum(1.0 / log2(i + 2) for i in range(min(k, n_relevant)))
    return dcg / idcg


@dataclass(frozen=True)
class RetrievalMetrics:
    n: int
    mrr: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_retrieval(
    per_question_hits: list[list[bool]],
    per_question_n_relevant: list[int] | None = None,
) -> RetrievalMetrics:
    """Aggregate per-question hit lists (answerable items only) into metrics."""
    n = len(per_question_hits)
    if n == 0:
        return RetrievalMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mean = lambda xs: sum(xs) / n  # noqa: E731
    if per_question_n_relevant is None:
        ndcg = 0.0
    else:
        ndcg = mean([
            ndcg_at_k(h, r, 10)
            for h, r in zip(per_question_hits, per_question_n_relevant)
        ])
    return RetrievalMetrics(
        n=n,
        mrr=mean([_reciprocal_rank(h) for h in per_question_hits]),
        recall_at_1=mean([_recall_at_k(h, 1) for h in per_question_hits]),
        recall_at_3=mean([_recall_at_k(h, 3) for h in per_question_hits]),
        recall_at_5=mean([_recall_at_k(h, 5) for h in per_question_hits]),
        recall_at_10=mean([_recall_at_k(h, 10) for h in per_question_hits]),
        ndcg_at_10=ndcg,
    )


def refusal_accuracy(refused_flags: list[bool]) -> tuple[int, float]:
    """Over not-in-corpus items: fraction the system correctly declined to answer."""
    n = len(refused_flags)
    if n == 0:
        return 0, 0.0
    return n, sum(1 for r in refused_flags if r) / n

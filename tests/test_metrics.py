"""Retrieval-metric math on synthetic hit lists (key-free, deterministic)."""
from __future__ import annotations

import pytest

from eval.retrieval_metrics import (
    evaluate_retrieval,
    hit_ranks,
    ndcg_at_k,
    refusal_accuracy,
)


def test_hit_ranks_intersects_source_ids():
    retrieved = [{"a", "b"}, {"x"}, {"c"}]
    assert hit_ranks(retrieved, {"c", "z"}) == [False, False, True]


def test_recall_and_mrr_basic():
    # q1: first hit at rank 1 (idx 0); q2: first hit at rank 3 (idx 2); q3: no hit.
    hits = [
        [True, False, False, False],
        [False, False, True, False],
        [False, False, False, False],
    ]
    m = evaluate_retrieval(hits)
    assert m.n == 3
    assert m.recall_at_1 == 1 / 3            # only q1
    assert m.recall_at_3 == 2 / 3            # q1, q2
    assert m.recall_at_5 == 2 / 3
    assert abs(m.mrr - (1.0 + 1 / 3 + 0.0) / 3) < 1e-9


def test_empty_is_zeroed():
    m = evaluate_retrieval([])
    assert m.n == 0 and m.mrr == 0.0 and m.recall_at_5 == 0.0 and m.ndcg_at_10 == 0.0


def test_ndcg_perfect_ranking_is_1():
    # 3 relevant chunks, all ranked on top -> DCG == IDCG
    assert ndcg_at_k([True, True, True, False], n_relevant=3, k=10) == pytest.approx(1.0)


def test_ndcg_rewards_higher_placement():
    low = ndcg_at_k([False, False, False, True], n_relevant=1, k=10)
    high = ndcg_at_k([True, False, False, False], n_relevant=1, k=10)
    assert high == pytest.approx(1.0)
    assert 0 < low < high


def test_ndcg_ideal_capped_by_n_relevant():
    # only 1 relevant chunk exists: a single hit at rank 1 is a perfect ranking
    assert ndcg_at_k([True] + [False] * 9, n_relevant=1, k=10) == pytest.approx(1.0)
    assert ndcg_at_k([False] * 10, n_relevant=5, k=10) == 0.0
    assert ndcg_at_k([True], n_relevant=0, k=10) == 0.0


def test_evaluate_retrieval_with_ndcg():
    hits = [[True, False], [False, True]]
    m = evaluate_retrieval(hits, per_question_n_relevant=[1, 1])
    assert 0 < m.ndcg_at_10 <= 1.0


def test_refusal_accuracy():
    n, acc = refusal_accuracy([True, True, False, True])
    assert n == 4
    assert acc == 0.75
    assert refusal_accuracy([]) == (0, 0.0)

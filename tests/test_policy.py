"""Adaptive rerank policy decisions (pure logic, no model, no key)."""
from __future__ import annotations

from types import SimpleNamespace

from rag.config import RagConfig
from rag.policy import effective_rerank_policy, should_rerank


def _cands(scores):
    # The auto heuristic only reads the scores; chunks can be placeholders.
    return [(SimpleNamespace(chunk_id=f"c{i}"), s) for i, s in enumerate(scores)]


def test_auto_skips_when_confident():
    cfg = RagConfig(rerank_policy="auto")  # top>=0.55 and margin>=0.05
    assert should_rerank("q", _cands([0.80, 0.30, 0.20]), cfg) is False


def test_auto_reranks_when_top_score_low():
    cfg = RagConfig(rerank_policy="auto")
    assert should_rerank("q", _cands([0.50, 0.48, 0.47]), cfg) is True


def test_auto_reranks_when_margin_small():
    cfg = RagConfig(rerank_policy="auto")  # top ok but cluster is tight
    assert should_rerank("q", _cands([0.60, 0.58, 0.57]), cfg) is True


def test_forced_policies():
    assert should_rerank("q", _cands([0.9]), RagConfig(rerank_policy="always")) is True
    assert should_rerank("q", _cands([0.1]), RagConfig(rerank_policy="never")) is False


def test_use_reranker_alias_maps_to_always():
    cfg = RagConfig(rerank_policy="never", use_reranker=True)
    assert effective_rerank_policy(cfg) == "always"
    assert should_rerank("q", _cands([0.1, 0.05]), cfg) is True


def test_empty_candidates_not_confident_on_auto():
    # No candidates => not confident => policy says rerank (score_candidates no-ops anyway).
    assert should_rerank("q", [], RagConfig(rerank_policy="auto")) is True


def test_auto_always_reranks_for_non_dense_modes():
    # BM25/RRF scores aren't cosines — the confidence heuristic is meaningless there.
    confident = _cands([0.80, 0.30, 0.20])  # would skip rerank under dense
    for mode in ("sparse", "hybrid"):
        cfg = RagConfig(rerank_policy="auto", retrieval_mode=mode)
        assert should_rerank("q", confident, cfg) is True

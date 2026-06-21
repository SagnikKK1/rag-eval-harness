"""Reranker wiring (cross-encoder reorders), with a fake model — no download, no key."""
from __future__ import annotations

from dataclasses import dataclass

import rag.rerank as rerank_mod
from rag.config import RagConfig
from rag.rerank import rerank, score_candidates


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str
    text: str
    source_ids: tuple = ()


class FakeCrossEncoder:
    """Scores any chunk mentioning 'battery' as highly relevant, others as not."""

    def predict(self, pairs):
        return [5.0 if "battery" in text.lower() else -5.0 for _q, text in pairs]


def _patch_ce(monkeypatch):
    monkeypatch.setattr(rerank_mod, "_get_cross_encoder", lambda name: FakeCrossEncoder())


def test_rerank_reorders_by_relevance(monkeypatch):
    _patch_ce(monkeypatch)
    cfg = RagConfig(rerank_policy="always", k=2)
    # Dense over-ranks an irrelevant chunk; the cross-encoder should fix it.
    candidates = [
        (FakeChunk("A", "the screen looks nice"), 0.71),       # high dense, irrelevant
        (FakeChunk("B", "battery life is fantastic"), 0.50),   # lower dense, relevant
    ]
    reranked, scored = score_candidates("how is the battery?", candidates, cfg)
    assert reranked is True
    assert scored[0][0].chunk_id == "B"  # relevant chunk now on top


def test_never_policy_keeps_dense_order(monkeypatch):
    _patch_ce(monkeypatch)
    cfg = RagConfig(rerank_policy="never", k=2)
    candidates = [(FakeChunk("A", "x"), 0.71), (FakeChunk("B", "battery"), 0.50)]
    reranked, scored = score_candidates("q", candidates, cfg)
    assert reranked is False
    assert [c.chunk_id for c, _ in scored] == ["A", "B"]


def test_rerank_truncates_to_k(monkeypatch):
    _patch_ce(monkeypatch)
    cfg = RagConfig(rerank_policy="always", k=1)
    candidates = [
        (FakeChunk("A", "no kw"), 0.9),
        (FakeChunk("B", "battery good"), 0.4),
        (FakeChunk("C", "also battery"), 0.3),
    ]
    out = rerank("battery?", candidates, cfg)
    assert len(out) == 1
    assert out[0][0].chunk_id in {"B", "C"}

"""CRAG correction loop control flow, with a fake retriever + monkeypatched LLM.

Exercises the agentic logic (grade -> correct -> re-retrieve -> accumulate ->
refuse) deterministically, with the threshold grader and rerank_policy='never'
so no model/key is needed. Every case runs through BOTH engines (the LangGraph
StateGraph and the plain reference loop) to assert they behave identically.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

import rag.llm as llm_mod
from rag.agent import run_crag
from rag.config import RagConfig

ENGINES = ["langgraph", "plain"]


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str
    text: str
    source_ids: tuple = ()


class FakeRetriever:
    """Returns scripted dense candidates keyed by the (possibly reformulated) query."""

    def __init__(self, by_query, config):
        self._by_query = by_query
        self.config = config

    def candidates(self, query, n):
        chunks = self._by_query.get(query, self._by_query.get("__default__", []))
        return chunks[:n]


def _cfg(engine="langgraph", **kw):
    base = dict(rerank_policy="never", grader="threshold", k=3, top_n=20, crag_engine=engine)
    base.update(kw)
    return RagConfig(**base)


@pytest.mark.parametrize("engine", ENGINES)
def test_correction_then_success(monkeypatch, engine):
    # First query retrieves junk (INCORRECT); the reformulated query succeeds (CORRECT).
    monkeypatch.setattr(llm_mod, "chat", lambda prompt, config=None: "battery endurance hours")
    cfg = _cfg(engine, max_correction_rounds=2, reformulate=True)
    retriever = FakeRetriever(
        {
            "how long does it last":
                [(FakeChunk("j1", "screen"), 0.20), (FakeChunk("j2", "price"), 0.10)],
            "battery endurance hours": [(FakeChunk("g1", "battery lasts a full day"), 0.80),
                                         (FakeChunk("g2", "great endurance"), 0.60)],
        },
        cfg,
    )
    result = run_crag("how long does it last", retriever, cfg)

    assert result.refused is False
    ids = {c.chunk_id for c in result.contexts}
    assert {"g1", "g2"} <= ids
    assert result.trace[0].action == "INCORRECT"
    assert result.trace[0].correction == "reformulate"
    assert result.trace[1].action == "CORRECT"
    assert result.trace[1].query == "battery endurance hours"


@pytest.mark.parametrize("engine", ENGINES)
def test_refuses_when_nothing_relevant(monkeypatch, engine):
    monkeypatch.setattr(llm_mod, "chat", lambda prompt, config=None: "still irrelevant")
    cfg = _cfg(engine, max_correction_rounds=1, reformulate=True)
    junk = [(FakeChunk("j1", "x"), 0.10), (FakeChunk("j2", "y"), 0.05)]
    retriever = FakeRetriever({"__default__": junk}, cfg)

    result = run_crag("unanswerable thing", retriever, cfg)

    assert result.refused is True
    assert result.contexts == []
    assert len(result.trace) == 2  # initial + 1 correction round, both INCORRECT
    assert all(r.action == "INCORRECT" for r in result.trace)


@pytest.mark.parametrize("engine", ENGINES)
def test_widen_fallback_when_reformulation_disabled(engine):
    # reformulate=False -> correction must be 'widen' and the candidate pool grows.
    cfg = _cfg(engine, max_correction_rounds=1, reformulate=False)
    junk = [(FakeChunk(f"j{i}", "x"), 0.10) for i in range(100)]
    retriever = FakeRetriever({"__default__": junk}, cfg)

    result = run_crag("q", retriever, cfg)

    assert result.trace[0].correction == "widen"
    assert result.trace[1].n_candidates > result.trace[0].n_candidates


@pytest.mark.parametrize("engine", ENGINES)
def test_mixed_signal_pool_is_rescored_with_one_arbiter(monkeypatch, engine):
    """Rounds scoring on different scales (CE logits vs cosine) must not be sorted
    raw — the finalize step rescoreswith the cross-encoder (mocked here)."""
    import rag.rerank as rerank_mod

    # Mock CE: 'gold' chunk wins decisively at rescore time.
    def fake_ce(query, candidates, config):
        return [5.0 if c.chunk_id == "gold" else -5.0 for c, _ in candidates]

    monkeypatch.setattr(rerank_mod, "cross_encoder_scores", fake_ce)

    from rag.agent import _finalize_pool

    cfg = _cfg(engine, k=2)
    # Mixed pool: cosine round kept junk at 0.44; CE round kept gold at logit 0.2.
    gathered = [
        (FakeChunk("junk", "x"), 0.44, "cosine"),
        (FakeChunk("gold", "y"), 0.2, "ce"),
    ]
    final = _finalize_pool("q", gathered, cfg)
    assert final[0][0].chunk_id == "gold"  # raw-sort would have put junk first


def test_single_signal_pool_sorts_directly_without_ce(monkeypatch):
    import rag.rerank as rerank_mod

    def boom(*a, **kw):
        raise AssertionError("CE must not be called for single-signal pools")

    monkeypatch.setattr(rerank_mod, "cross_encoder_scores", boom)
    from rag.agent import _finalize_pool

    cfg = _cfg("plain", k=2)
    gathered = [
        (FakeChunk("a", "x"), 0.9, "cosine"),
        (FakeChunk("b", "y"), 0.5, "cosine"),
        (FakeChunk("a", "x"), 0.7, "cosine"),  # duplicate: keep best (0.9)
    ]
    final = _finalize_pool("q", gathered, cfg)
    assert [c.chunk_id for c, _ in final] == ["a", "b"]
    assert final[0][1] == 0.9


@pytest.mark.parametrize("engine", ENGINES)
def test_first_round_correct_stops_early(engine):
    cfg = _cfg(engine, max_correction_rounds=3, reformulate=True)
    good = [(FakeChunk("g1", "answer here"), 0.90)]
    retriever = FakeRetriever({"__default__": good}, cfg)

    result = run_crag("q", retriever, cfg)

    assert len(result.trace) == 1
    assert result.trace[0].action == "CORRECT"
    assert result.trace[0].correction is None
    assert result.contexts[0].chunk_id == "g1"

"""Agent tools — calculator, search_corpus, fetch_chunk (key-free, no LLM)."""
from __future__ import annotations

from rag_eval.tools import TOOL_NAMES, ToolContext, execute


def test_calculator_basic():
    ctx = ToolContext(retriever=None)
    assert execute("calculator", {"expression": "60/200*100"}, ctx) == "30.0"
    assert execute("calculator", {"expression": "57.59 + 6"}, ctx) == "63.59"


def test_calculator_rejects_unsafe():
    ctx = ToolContext(retriever=None)
    assert execute("calculator", {"expression": "__import__('os').system('ls')"}, ctx).startswith(
        "Error"
    )


def test_calculator_pow_guard_blocks_bignum_dos():
    ctx = ToolContext(retriever=None)
    # would hang the worker computing a bignum without the guard
    assert execute("calculator", {"expression": "9**9**9**9"}, ctx).startswith("Error")
    assert execute("calculator", {"expression": "2**1000"}, ctx).startswith("Error")
    assert execute("calculator", {"expression": "2**10"}, ctx) == "1024"


def test_search_corpus_retrieves_and_accumulates(tiny_retriever):
    ctx = ToolContext(retriever=tiny_retriever)
    out = execute("search_corpus", {"query": "how is the battery life", "k": 3}, ctx)
    assert "battery" in out.lower()
    assert len(ctx.retrieved) == 3            # accumulated for the eval harness
    assert out.startswith("[c")              # chunk ids included


def test_fetch_chunk_roundtrip(tiny_retriever):
    ctx = ToolContext(retriever=tiny_retriever)
    cid = tiny_retriever.index.chunks[0].chunk_id
    out = execute("fetch_chunk", {"chunk_id": cid}, ctx)
    assert cid in out and "sources=" in out
    assert "no chunk" in execute("fetch_chunk", {"chunk_id": "c99999"}, ctx).lower()


def test_unknown_tool():
    assert "Unknown tool" in execute("nope", {}, ToolContext(retriever=None))
    assert set(TOOL_NAMES) == {"search_corpus", "fetch_chunk", "calculator"}

"""CRAG as a LangGraph ``StateGraph``.

Same corrective loop as ``agent._run_crag_plain``, expressed as a graph so the
control flow is explicit and inspectable (``crag_graph_mermaid()`` renders it):

    START -> retrieve -> grade --(CORRECT or last round)--> finalize -> END
                            └----(weak)----> transform -> retrieve   (loop)

Each node reuses the already-tested pure functions (``dense_candidates``,
``score_candidates``, ``grade``, ``reformulate``); the graph only orchestrates.
Node updates return full replacement lists, so no channel reducers are needed.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import _RELAX_STEP, _WIDEN_FACTOR, CragResult, CragRound, _dedup_keep_best
from .config import RagConfig
from .grade import grade
from .reformulate import reformulate
from .rerank import score_candidates
from .retrieve import Retriever, to_retrieved


class CragState(TypedDict, total=False):
    query: str            # original query (immutable)
    current_query: str    # query used for the current round
    n: int                # candidate pool size for the current round
    relax: float          # relevance-threshold relaxation (widen fallback)
    round_idx: int
    max_rounds: int
    gathered: list[Any]   # accumulated (chunk, score) across rounds
    scored: list[Any]     # this round's (chunk, score), post adaptive rerank
    reranked: bool
    n_candidates: int
    action: str
    trace: list[CragRound]
    contexts: list[Any]
    refused: bool


def build_crag_graph(retriever: Retriever | None, config: RagConfig):
    """Compile the CRAG StateGraph. ``retriever`` may be None just to draw the graph."""

    def retrieve(state: CragState) -> dict:
        candidates = retriever.candidates(state["current_query"], state["n"])
        reranked, scored = score_candidates(state["current_query"], candidates, config)
        return {"scored": scored, "reranked": reranked, "n_candidates": len(candidates)}

    def grade_node(state: CragState) -> dict:
        result = grade(
            state["current_query"], state["scored"], state["reranked"], config, relax=state["relax"]
        )
        entry = CragRound(
            round_idx=state["round_idx"],
            query=state["current_query"],
            n_candidates=state["n_candidates"],
            reranked=state["reranked"],
            action=result.action,
            kept_chunk_ids=[c.chunk_id for c, _ in result.relevant],
            correction=None,
        )
        return {
            "gathered": state["gathered"] + list(result.relevant),
            "action": result.action,
            "trace": state["trace"] + [entry],
        }

    def route(state: CragState) -> str:
        last_round = state["round_idx"] >= state["max_rounds"] - 1
        return "finalize" if state["action"] == "CORRECT" or last_round else "transform"

    def transform(state: CragState) -> dict:
        new_query = reformulate(state["current_query"], state["scored"], config)
        if new_query:
            correction, next_query = "reformulate", new_query
            next_n, next_relax = state["n"], state["relax"]
        else:
            correction, next_query = "widen", state["current_query"]
            next_n, next_relax = state["n"] * _WIDEN_FACTOR, state["relax"] + _RELAX_STEP
        patched = state["trace"][:-1] + [replace(state["trace"][-1], correction=correction)]
        return {
            "current_query": next_query,
            "n": next_n,
            "relax": next_relax,
            "round_idx": state["round_idx"] + 1,
            "trace": patched,
        }

    def finalize(state: CragState) -> dict:
        final = _dedup_keep_best(state["gathered"])[: config.k]
        contexts = [to_retrieved(chunk, score) for chunk, score in final]
        return {"contexts": contexts, "refused": not contexts}

    g = StateGraph(CragState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade_node)
    g.add_node("transform", transform)
    g.add_node("finalize", finalize)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route, {"transform": "transform", "finalize": "finalize"})
    g.add_edge("transform", "retrieve")
    g.add_edge("finalize", END)
    return g.compile()


def run_crag_graph(query: str, retriever: Retriever, config: RagConfig | None = None) -> CragResult:
    config = config or retriever.config
    app = build_crag_graph(retriever, config)
    max_rounds = config.max_correction_rounds + 1
    initial: CragState = {
        "query": query,
        "current_query": query,
        "n": config.top_n,
        "relax": 0.0,
        "round_idx": 0,
        "max_rounds": max_rounds,
        "gathered": [],
        "trace": [],
    }
    # Each round is up to 3 supersteps (retrieve, grade, transform); pad generously.
    final = app.invoke(initial, config={"recursion_limit": max_rounds * 3 + 5})
    return CragResult(
        query=query,
        contexts=final.get("contexts", []),
        trace=final["trace"],
        refused=final.get("refused", True),
    )


def crag_graph_mermaid(config: RagConfig | None = None) -> str:
    """Mermaid diagram of the CRAG graph (structure only; no retriever needed)."""
    app = build_crag_graph(None, config or RagConfig())
    return app.get_graph().draw_mermaid()

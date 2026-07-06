"""Corrective RAG (CRAG) agentic loop.

A bounded state machine: dense retrieve -> (adaptive) rerank -> grade -> correct.
On a weak grade it reformulates the query (LLM) or, if reformulation is
unavailable, widens retrieval (more candidates + a looser relevance bar) and
tries again, accumulating relevant chunks across rounds. Returns the gathered
contexts plus a full ``trace`` so the Part 2 harness can score retrieval and
inspect the correction behavior. Generation is left to the caller (pipeline), so
this whole loop runs without an API key when the grader is threshold-based.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import RagConfig
from .grade import grade
from .reformulate import reformulate
from .rerank import score_candidates
from .retrieve import RetrievedChunk, Retriever, to_retrieved

_WIDEN_FACTOR = 2      # multiply the candidate pool on a key-free correction round
_RELAX_STEP = 0.1      # loosen the relevance threshold per widen round


@dataclass(frozen=True)
class CragRound:
    round_idx: int
    query: str               # the query actually used for retrieval this round
    n_candidates: int
    reranked: bool
    action: str              # CORRECT | AMBIGUOUS | INCORRECT
    kept_chunk_ids: list[str]
    correction: str | None   # "reformulate" | "widen" | None (final/answered)


@dataclass(frozen=True)
class CragResult:
    query: str
    contexts: list[RetrievedChunk]
    trace: list[CragRound]
    refused: bool


def _finalize_pool(query: str, gathered: list[tuple], config: RagConfig) -> list[tuple]:
    """Dedup the accumulated (chunk, score, signal) pool and produce a top-k ranking.

    Rounds may score on different scales (cross-encoder logits vs dense cosine).
    Sorting a mixed pool by raw score is meaningless, so when signals are mixed
    the deduped pool is rescored once with the cross-encoder against the ORIGINAL
    query — one arbiter, one scale. Single-signal pools sort directly.
    """
    if not gathered:
        return []
    signals = {sig for _, _, sig in gathered}
    best: dict[str, tuple] = {}
    for chunk, score, _sig in gathered:
        if chunk.chunk_id not in best or score > best[chunk.chunk_id][1]:
            best[chunk.chunk_id] = (chunk, score)
    pool = list(best.values())
    if len(signals) > 1:
        from .rerank import cross_encoder_scores

        ce = cross_encoder_scores(query, pool, config)
        pool = [(c, s) for (c, _), s in zip(pool, ce)]
    return sorted(pool, key=lambda cs: cs[1], reverse=True)[: config.k]


def run_crag(query: str, retriever: Retriever, config: RagConfig | None = None) -> CragResult:
    """Run the CRAG loop using the configured engine (LangGraph by default)."""
    config = config or retriever.config
    if config.crag_engine == "plain":
        return _run_crag_plain(query, retriever, config)
    from .graph import run_crag_graph  # lazy import keeps langgraph optional for plain mode

    return run_crag_graph(query, retriever, config)


def _run_crag_plain(query: str, retriever: Retriever, config: RagConfig) -> CragResult:
    """Hand-rolled corrective loop (dependency-free reference implementation)."""
    gathered: list[tuple] = []
    trace: list[CragRound] = []

    current_query = query
    n = config.top_n
    relax = 0.0
    max_rounds = config.max_correction_rounds + 1

    for round_idx in range(max_rounds):
        used_query = current_query
        candidates = retriever.candidates(used_query, n)
        reranked, scored = score_candidates(used_query, candidates, config)
        result = grade(used_query, scored, reranked, config, relax=relax)
        gathered.extend((c, s, result.signal) for c, s in result.relevant)

        last_round = round_idx == max_rounds - 1
        correction: str | None = None
        if result.action != "CORRECT" and not last_round:
            new_query = reformulate(used_query, scored, config)
            if new_query:
                current_query = new_query
                correction = "reformulate"
            else:
                n *= _WIDEN_FACTOR
                relax += _RELAX_STEP
                correction = "widen"

        trace.append(
            CragRound(
                round_idx=round_idx,
                query=used_query,
                n_candidates=len(candidates),
                reranked=reranked,
                action=result.action,
                kept_chunk_ids=[c.chunk_id for c, _ in result.relevant],
                correction=correction,
            )
        )
        if result.action == "CORRECT" or last_round:
            break

    final = _finalize_pool(query, gathered, config)
    contexts = [to_retrieved(chunk, score) for chunk, score in final]
    return CragResult(query=query, contexts=contexts, trace=trace, refused=not contexts)

"""Adaptive reranking policy: decide *per query* whether to run the cross-encoder.

Running a cross-encoder on every query is wasteful when dense retrieval already
returned a confident, well-separated result. ``should_rerank`` gates the reranker:

- ``never`` / ``always`` — forced.
- ``auto``  — free heuristic on the dense-score distribution: skip reranking only
  when the top cosine score is high AND clearly separated from the tail.
- ``llm``   — one cheap LLM classification of whether the query is ambiguous/hard
  enough to warrant reranking (falls back to ``auto`` on any error / missing key).
"""
from __future__ import annotations

from statistics import mean
from typing import TYPE_CHECKING

from .config import RagConfig

if TYPE_CHECKING:
    from .index import Chunk

Candidates = list[tuple["Chunk", float]]


def effective_rerank_policy(config: RagConfig) -> str:
    """Resolve the policy, honoring the legacy ``use_reranker=True`` alias."""
    if config.rerank_policy == "never" and config.use_reranker:
        return "always"
    return config.rerank_policy


def _auto_is_confident(candidates: Candidates, config: RagConfig) -> bool:
    """Confident (=> skip rerank) when the top score is high and well-separated.

    Only meaningful for dense cosine scores: BM25 logits are unbounded and RRF
    scores live near 1/rrf_k, so for sparse/hybrid first stages the heuristic
    can't be trusted and we always rerank instead.
    """
    if config.retrieval_mode != "dense":
        return False
    if not candidates:
        return False
    scores = [s for _, s in candidates]
    top = scores[0]
    tail = scores[1:] or [top]
    margin = top - mean(tail)
    return top >= config.auto_min_top_score and margin >= config.auto_min_margin


def _llm_says_rerank(query: str, config: RagConfig) -> bool:
    from .llm import chat

    prompt = (
        "You route search queries. Answer with a single word, YES or NO.\n"
        "Should this query get a careful cross-encoder reranking pass because it is "
        "ambiguous, multi-faceted, or comparative (rather than a simple keyword "
        f"lookup)?\n\nQUERY: {query}\n\nAnswer (YES/NO):"
    )
    return chat(prompt, config).strip().upper().startswith("Y")


def should_rerank(query: str, candidates: Candidates, config: RagConfig) -> bool:
    """Return whether the cross-encoder should run for this query."""
    policy = effective_rerank_policy(config)
    if policy == "always":
        return True
    if policy == "never":
        return False
    if policy == "auto":
        return not _auto_is_confident(candidates, config)
    if policy == "llm":
        try:
            return _llm_says_rerank(query, config)
        except Exception:
            return not _auto_is_confident(candidates, config)  # graceful fallback
    raise ValueError(f"Unknown rerank_policy: {policy!r}")

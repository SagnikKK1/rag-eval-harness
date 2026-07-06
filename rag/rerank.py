"""Cross-encoder reranking, gated by the adaptive policy.

``score_candidates`` is the workhorse: given dense candidates it consults
``policy.should_rerank`` and, when warranted, re-scores (query, chunk) pairs with
a cross-encoder and reorders by relevance. It returns ``(reranked, scored)`` so
callers know which signal the scores represent (cross-encoder logits vs cosine) —
the CRAG grader needs that to pick the right thresholds. ``rerank`` is the thin
top-k wrapper used by the plain retrieval path.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .config import RagConfig
from .policy import should_rerank

if TYPE_CHECKING:
    from .index import Chunk

Candidates = list[tuple["Chunk", float]]

# Lightweight instrumentation: the eval harness reads/resets this to report the
# cross-encoder invocation rate per arm (the "adaptive saves CE calls" claim).
CE_STATS = {"calls": 0}


@lru_cache(maxsize=2)
def _get_cross_encoder(name: str):
    # Imported lazily so a non-reranked query pulls in no extra model/runtime.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


def cross_encoder_scores(query: str, candidates: Candidates, config: RagConfig) -> list[float]:
    """Raw cross-encoder relevance logits for each candidate, in input order."""
    CE_STATS["calls"] += 1
    model = _get_cross_encoder(config.reranker_model)
    pairs = [(query, chunk.text) for chunk, _ in candidates]
    return [float(s) for s in model.predict(pairs)]


def score_candidates(
    query: str, candidates: Candidates, config: RagConfig
) -> tuple[bool, Candidates]:
    """Return (reranked?, scored_candidates). Reranked => sorted by cross-encoder score."""
    if not candidates or not should_rerank(query, candidates, config):
        return False, candidates
    scores = cross_encoder_scores(query, candidates, config)
    reranked = sorted(
        ((chunk, score) for (chunk, _), score in zip(candidates, scores)),
        key=lambda cs: cs[1],
        reverse=True,
    )
    return True, reranked


def rerank(
    query: str, candidates: Candidates, config: RagConfig, k: int | None = None
) -> Candidates:
    """Plain-path helper: (adaptively) rerank and truncate to top-k.

    ``k`` overrides ``config.k`` so callers requesting a deeper ranking (e.g. the
    eval harness at depth 10, or the API with a per-request k) are not silently
    capped by the config default.
    """
    _, scored = score_candidates(query, candidates, config)
    return scored[: config.k if k is None else k]

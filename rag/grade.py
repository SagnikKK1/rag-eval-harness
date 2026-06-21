"""CRAG retrieval evaluator: grade retrieved chunks and pick a corrective action.

Maps the scored candidates to one of CRAG's three actions:
- CORRECT   — at least one clearly relevant chunk; keep the relevant ones and answer.
- AMBIGUOUS — nothing clearly relevant but not clearly wrong; keep the best, augment.
- INCORRECT — all chunks look irrelevant; discard and re-retrieve.

Default grader thresholds on the relevance score (cross-encoder logit when the
round was reranked, else dense cosine). An optional ``grader="llm"`` asks the LLM
for a per-chunk yes/no. Thresholds are calibrated empirically in Part 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .config import RagConfig

if TYPE_CHECKING:
    from .index import Chunk

Action = Literal["CORRECT", "AMBIGUOUS", "INCORRECT"]
Scored = list[tuple["Chunk", float]]


@dataclass(frozen=True)
class GradeResult:
    action: Action
    relevant: Scored                      # chunks worth keeping (may be empty)
    labels: list[tuple[str, bool, float]]  # (chunk_id, is_relevant, score) for the trace


def _thresholds(reranked: bool, config: RagConfig) -> tuple[float, float]:
    if reranked:
        return config.rerank_relevant_threshold, config.rerank_incorrect_threshold
    return config.dense_relevant_threshold, config.dense_incorrect_threshold


def _grade_threshold(
    scored: Scored, reranked: bool, config: RagConfig, relax: float
) -> GradeResult:
    rel_t, inc_t = _thresholds(reranked, config)
    rel_t -= relax  # widen-fallback: loosen the relevance bar on later rounds
    labels = [(c.chunk_id, s >= rel_t, s) for c, s in scored]
    relevant = [(c, s) for c, s in scored if s >= rel_t]
    if relevant:
        action: Action = "CORRECT"
    elif scored and all(s <= inc_t for _, s in scored):
        action = "INCORRECT"
    else:
        action = "AMBIGUOUS"
        relevant = [(c, s) for c, s in scored if s > inc_t]  # keep the not-clearly-bad ones
    return GradeResult(action=action, relevant=relevant, labels=labels)


def _grade_llm(query: str, scored: Scored, config: RagConfig) -> GradeResult:
    from .llm import chat

    labels: list[tuple[str, bool, float]] = []
    relevant: Scored = []
    for chunk, score in scored:
        prompt = (
            "Decide if the PASSAGE is relevant to answering the QUESTION. "
            "Answer with one word: YES or NO.\n\n"
            f"QUESTION: {query}\n\nPASSAGE: {chunk.text}\n\nAnswer (YES/NO):"
        )
        is_rel = chat(prompt, config).strip().upper().startswith("Y")
        labels.append((chunk.chunk_id, is_rel, score))
        if is_rel:
            relevant.append((chunk, score))
    action: Action = "CORRECT" if relevant else "INCORRECT"
    return GradeResult(action=action, relevant=relevant, labels=labels)


def grade(
    query: str,
    scored: Scored,
    reranked: bool,
    config: RagConfig,
    relax: float = 0.0,
) -> GradeResult:
    if config.grader == "llm":
        return _grade_llm(query, scored, config)
    return _grade_threshold(scored, reranked, config, relax)

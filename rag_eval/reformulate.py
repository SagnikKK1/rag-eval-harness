"""Query reformulation for CRAG correction rounds.

When a round's retrieval is graded weak, CRAG rewrites the query and tries again.
``reformulate`` asks the LLM for a better search query; it returns ``None`` when
reformulation is unavailable (disabled, no API key, or an LLM error) so the agent
can fall back to a key-free "widen retrieval" strategy instead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import RagConfig

if TYPE_CHECKING:
    from .index import Chunk

Scored = list[tuple["Chunk", float]]


def reformulate(query: str, weak_contexts: Scored, config: RagConfig) -> str | None:
    """Return a rewritten search query, or None to signal the widen fallback."""
    if not config.reformulate:
        return None

    from .llm import chat

    sample = "\n".join(f"- {c.text[:160]}" for c, _ in weak_contexts[:3])
    prompt = (
        "You improve search queries for a corpus of smartphone user-review comments.\n"
        "The previous query returned weak/irrelevant results. Rewrite it into a single, "
        "more effective search query (expand abbreviations, add synonyms/specific terms). "
        "Return ONLY the rewritten query, no preamble.\n\n"
        f"ORIGINAL QUERY: {query}\n\n"
        f"WEAK RESULTS SAMPLE:\n{sample or '(none)'}\n\nREWRITTEN QUERY:"
    )
    try:
        new_query = chat(prompt, config).strip().strip('"')
    except Exception:
        return None
    # Guard against a no-op or a degenerate rewrite.
    if not new_query or new_query.lower() == query.lower():
        return None
    return new_query

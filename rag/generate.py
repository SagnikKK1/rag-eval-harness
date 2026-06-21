"""Answer generation grounded in retrieved chunks.

Ports the prompt construction and citation expansion from the original
Sparkathon ``query.py`` (build_prompt / expand_citations) but decoupled from
OpenRouter and Postgres: generation goes through the pluggable ``llm.chat`` and
there is no database side effect. The retrieved chunks become the [S1], [S2] ...
sources the model must ground its answer in.
"""
from __future__ import annotations

import re

from .config import RagConfig
from .llm import chat
from .retrieve import RetrievedChunk

_CITATION = re.compile(r"\[S(\d+)\]")


def build_prompt(query: str, contexts: list[RetrievedChunk]) -> str:
    prompt = (
        "You are a helpful assistant that answers user questions about a product "
        "based SOLELY on the provided SOURCES (user reviews/comments).\n"
        "Write a concise, evidence-based answer (around 60 words). Use citations like "
        "[S1], [S2] that refer to the sources you used. If the sources do not contain "
        "the answer, say so plainly instead of guessing.\n"
        f"\nQUESTION: {query}\n\nSOURCES:\n"
    )
    for i, c in enumerate(contexts, start=1):
        prompt += f"[S{i}] {c.text}\n\n"
    prompt += "Use cautious, evidence-based language. Do not invent facts not in the sources."
    return prompt


def expand_citations(answer: str, contexts: list[RetrievedChunk]) -> str:
    """Append the source text behind each [S#] the answer cited (in first-seen order)."""
    seen: set[int] = set()
    out = answer + "\n\n"
    for match in _CITATION.findall(answer):
        idx = int(match) - 1
        if 0 <= idx < len(contexts) and idx not in seen:
            seen.add(idx)
            out += f"[S{idx + 1}] {contexts[idx].text}\n\n"
    return out.strip()


def generate(
    query: str,
    contexts: list[RetrievedChunk],
    config: RagConfig | None = None,
    expand: bool = True,
) -> str:
    """Generate an answer grounded in ``contexts``. Requires a generation API key."""
    config = config or RagConfig()
    raw = chat(build_prompt(query, contexts), config)
    return expand_citations(raw, contexts) if expand else raw

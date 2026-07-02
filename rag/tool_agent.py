"""Tool-using RAG agent: an Anthropic tool-calling loop.

The model is given the tools in ``rag/tools.py`` and decides when to call them —
typically issuing one or more ``search_corpus`` calls (reformulating as needed),
optionally ``fetch_chunk`` / ``calculator``, then writing a grounded, cited
answer. This is the agentic counterpart to the deterministic CRAG graph.

Returns the final answer, the union of chunks retrieved via ``search_corpus``
(so the eval harness can score retrieval), and a full tool-call ``trace``. The
loop is LLM-driven, so it needs ANTHROPIC_API_KEY; it is unit-tested with a
mocked client.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import RagConfig
from .llm import get_anthropic_client
from .retrieve import RetrievedChunk, Retriever
from .tools import TOOL_SPECS, ToolContext, execute

SYSTEM = (
    "You are a research assistant answering questions about the Motorola Edge 50 Fusion "
    "smartphone using ONLY the user-review corpus. Use the search_corpus tool to gather "
    "evidence before answering (reformulate and search again if results are weak). Cite the "
    "chunk ids you used, e.g. [c00123]. If the corpus does not contain the answer, say so "
    "plainly instead of guessing."
)


@dataclass(frozen=True)
class ToolCall:
    step: int
    name: str
    tool_input: dict
    output: str


@dataclass(frozen=True)
class ToolAgentResult:
    query: str
    answer: str
    contexts: list[RetrievedChunk]
    trace: list[ToolCall] = field(default_factory=list)
    steps: int = 0


def _text_of(content) -> str:
    return "".join(
        getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text"
    ).strip()


def run_tool_agent(
    query: str, retriever: Retriever, config: RagConfig | None = None, max_steps: int | None = None
) -> ToolAgentResult:
    config = config or retriever.config
    max_steps = max_steps or config.max_tool_steps
    client = get_anthropic_client()
    ctx = ToolContext(retriever)
    messages: list[dict] = [{"role": "user", "content": query}]
    trace: list[ToolCall] = []
    answer = ""
    steps = 0

    for step in range(max_steps):
        steps = step + 1
        resp = client.messages.create(
            model=config.model, max_tokens=config.max_tokens,
            system=SYSTEM, tools=TOOL_SPECS, messages=messages,
        )
        if getattr(resp, "stop_reason", None) == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    tool_input = dict(block.input)
                    out = execute(block.name, tool_input, ctx)
                    trace.append(ToolCall(step, block.name, tool_input, out[:800]))
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": out}
                    )
            messages.append({"role": "user", "content": results})
        else:
            answer = _text_of(resp.content)
            break

    if not answer:  # ran out of steps mid-tool-use → force a final answer (no tools)
        resp = client.messages.create(
            model=config.model, max_tokens=config.max_tokens,
            system=SYSTEM + " Provide your final answer now using what you have.",
            messages=messages,
        )
        answer = _text_of(resp.content)

    seen, contexts = set(), []
    for c in ctx.retrieved:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            contexts.append(c)
    return ToolAgentResult(query=query, answer=answer, contexts=contexts, trace=trace, steps=steps)

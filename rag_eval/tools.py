"""Callable tools for the tool-using RAG agent (rag/tool_agent.py).

Defines the Anthropic tool schemas and a single ``execute`` dispatcher. The tools
are deliberately small and self-contained:

- ``search_corpus`` — dense/rerank/hybrid retrieval over the review corpus (the
  core RAG tool); accumulates retrieved chunks on the context for eval.
- ``fetch_chunk``   — return the full text + source ids of a specific chunk.
- ``calculator``    — evaluate a basic arithmetic expression (safe AST eval).

The tools themselves are key-free and unit-testable; only the agent *loop* that
decides when to call them needs the LLM.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field

from .retrieve import RetrievedChunk, Retriever

# Anthropic tool specs (name / description / JSON-schema input).
TOOL_SPECS = [
    {
        "name": "search_corpus",
        "description": (
            "Search the Motorola Edge 50 Fusion user-review corpus for chunks relevant to a "
            "query. Returns ranked chunks with their chunk_id and text. Call this before "
            "answering; you may call it multiple times with reformulated queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "k": {"type": "integer", "description": "Number of chunks to return (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_chunk",
        "description": "Fetch the full text and source ids of a single chunk by its chunk_id.",
        "input_schema": {
            "type": "object",
            "properties": {"chunk_id": {"type": "string"}},
            "required": ["chunk_id"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression, e.g. '57.59 + 6' or '60/200*100'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

TOOL_NAMES = [t["name"] for t in TOOL_SPECS]


@dataclass
class ToolContext:
    retriever: Retriever
    retrieved: list[RetrievedChunk] = field(default_factory=list)  # accumulated for eval


_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        # Guard ** against bignum DoS (model-generated '9**9**9**9' would hang the worker).
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise ValueError("exponent out of range")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def _calculator(expression: str) -> str:
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception:
        return f"Error: could not evaluate {expression!r}."


def _search_corpus(ctx: ToolContext, query: str, k: int = 5) -> str:
    results = ctx.retriever.retrieve(query, k=k)
    ctx.retrieved.extend(results)
    if not results:
        return "No chunks found."
    return "\n".join(f"[{c.chunk_id}] {c.text}" for c in results)


def _fetch_chunk(ctx: ToolContext, chunk_id: str) -> str:
    for c in ctx.retriever.index.chunks:
        if c.chunk_id == chunk_id:
            return f"[{c.chunk_id}] sources={c.source_ids}\n{c.text}"
    return f"No chunk with id {chunk_id!r}."


def execute(name: str, tool_input: dict, ctx: ToolContext) -> str:
    """Run a tool by name and return a string result for the model."""
    if name == "search_corpus":
        return _search_corpus(ctx, tool_input["query"], int(tool_input.get("k", 5)))
    if name == "fetch_chunk":
        return _fetch_chunk(ctx, tool_input["chunk_id"])
    if name == "calculator":
        return _calculator(tool_input["expression"])
    return f"Unknown tool: {name}"

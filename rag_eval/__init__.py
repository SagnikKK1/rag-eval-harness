"""rag-eval-harness: a RAG pipeline + evaluation harness with a CI regression gate.

Library quickstart::

    from rag_eval import RagConfig, RagPipeline

    pipe = RagPipeline.build(RagConfig())          # corpus -> chunks -> FAISS + BM25
    chunks = pipe.retrieve("battery life?", k=5)   # no API key needed
    result = pipe.answer("battery life?")          # needs ANTHROPIC_API_KEY

    from rag_eval.evals.experiment import run_experiment
    results = run_experiment()                     # the A/B harness

Set ``RAG_EVAL_DATA`` to point at a data directory containing ``raw/`` corpus
files (defaults to ``<repo>/data`` in a source checkout, else ``./data``).
"""
from __future__ import annotations

from .agent import CragResult, run_crag
from .config import RagConfig
from .pipeline import AnswerResult, RagPipeline
from .retrieve import RetrievedChunk, Retriever
from .tool_agent import ToolAgentResult, run_tool_agent

__version__ = "0.2.0"

__all__ = [
    "AnswerResult",
    "CragResult",
    "RagConfig",
    "RagPipeline",
    "RetrievedChunk",
    "Retriever",
    "ToolAgentResult",
    "run_tool_agent",
    "run_crag",
    "__version__",
]

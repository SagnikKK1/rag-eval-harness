"""End-to-end RAG pipeline tying retrieval + generation together.

``RagPipeline.answer()`` dispatches on ``config.mode``: ``plain`` is the original
single-shot retrieve->generate; ``crag`` runs the corrective agentic loop
(``agent.run_crag``) and generates from the gathered contexts (or returns an
honest refusal when nothing relevant was found). ``AnswerResult`` carries the
CRAG ``trace`` (None in plain mode) so callers and the Part 2 harness see the
correction rounds.
"""
from __future__ import annotations

from dataclasses import dataclass

from .agent import CragRound, run_crag
from .config import RagConfig
from .generate import generate
from .index import build_index
from .retrieve import RetrievedChunk, Retriever
from .tool_agent import ToolAgentResult, run_tool_agent

REFUSAL = "I couldn't find information about this in the reviews."


@dataclass(frozen=True)
class AnswerResult:
    query: str
    answer: str
    contexts: list[RetrievedChunk]
    trace: list[CragRound] | None = None
    refused: bool = False


class RagPipeline:
    def __init__(self, config: RagConfig | None = None, retriever: Retriever | None = None):
        self.config = config or RagConfig()
        self.retriever = retriever or Retriever(config=self.config)

    @classmethod
    def build(cls, config: RagConfig | None = None) -> RagPipeline:
        """Build corpus chunks + FAISS index from scratch, then return a ready pipeline."""
        config = config or RagConfig()
        index = build_index(config)
        return cls(config=config, retriever=Retriever(index=index, config=config))

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query, k)

    def corrective(self, query: str):
        """Run the CRAG loop (retrieval + correction only, no generation)."""
        return run_crag(query, self.retriever, self.config)

    def tool_agent(self, query: str) -> ToolAgentResult:
        """Run the tool-using agent (Anthropic tool-calling loop). Needs the API key."""
        return run_tool_agent(query, self.retriever, self.config)

    def answer(self, query: str, k: int | None = None, expand: bool = True) -> AnswerResult:
        if self.config.mode == "crag":
            crag = self.corrective(query)
            if crag.refused:
                return AnswerResult(query, REFUSAL, [], trace=crag.trace, refused=True)
            text = generate(query, crag.contexts, self.config, expand=expand)
            return AnswerResult(query, text, crag.contexts, trace=crag.trace, refused=False)

        if self.config.mode == "agent":
            result = self.tool_agent(query)
            return AnswerResult(query, result.answer, result.contexts)

        contexts = self.retrieve(query, k)
        text = generate(query, contexts, self.config, expand=expand)
        return AnswerResult(query=query, answer=text, contexts=contexts)

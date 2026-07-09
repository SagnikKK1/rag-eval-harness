"""LangGraph CRAG graph structure (compiles, has the expected nodes/edges)."""
from __future__ import annotations

from rag_eval.config import RagConfig
from rag_eval.graph import build_crag_graph, crag_graph_mermaid


def test_graph_compiles_with_expected_nodes():
    app = build_crag_graph(None, RagConfig())
    nodes = set(app.get_graph().nodes)
    assert {"retrieve", "grade", "transform", "finalize"} <= nodes


def test_mermaid_renders_the_loop():
    diagram = crag_graph_mermaid()
    for node in ("retrieve", "grade", "transform", "finalize"):
        assert node in diagram

"""Tool-using agent loop — driven by a mocked Anthropic client (key-free)."""
from __future__ import annotations

import rag.tool_agent as ta
from rag.tool_agent import run_tool_agent


class FB:
    """Fake content block (tool_use or text)."""

    def __init__(self, type, **kw):
        self.type = type
        self.__dict__.update(kw)


class FR:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def _patch(monkeypatch, responses):
    client = FakeClient(responses)
    monkeypatch.setattr(ta, "get_anthropic_client", lambda: client)
    return client


def test_agent_searches_then_answers(monkeypatch, tiny_retriever):
    responses = [
        FR("tool_use", [FB("tool_use", name="search_corpus",
                           input={"query": "battery life", "k": 3}, id="tu1")]),
        FR("end_turn", [FB("text", text="The battery lasts a full day [c00000].")]),
    ]
    client = _patch(monkeypatch, responses)

    result = run_tool_agent("How is the battery life?", tiny_retriever)

    assert result.answer.startswith("The battery lasts")
    assert result.steps == 2
    assert len(result.trace) == 1 and result.trace[0].name == "search_corpus"
    assert result.contexts                              # search_corpus results captured
    assert client.messages.calls[0]["tools"]            # tools were passed to the API


def test_agent_multi_tool_calculator(monkeypatch, tiny_retriever):
    responses = [
        FR("tool_use", [FB("tool_use", name="search_corpus",
                           input={"query": "charging"}, id="tu1")]),
        FR("tool_use", [FB("tool_use", name="calculator",
                           input={"expression": "20+60"}, id="tu2")]),
        FR("end_turn", [FB("text", text="Charges 20→80 in ~30 min; 20+60=80.")]),
    ]
    _patch(monkeypatch, responses)

    result = run_tool_agent("charging speed?", tiny_retriever)

    assert [t.name for t in result.trace] == ["search_corpus", "calculator"]
    assert result.trace[1].output == "80"
    assert result.steps == 3


def test_agent_forces_final_answer_when_budget_exhausted(monkeypatch, tiny_retriever):
    # Every step asks to search; budget runs out -> a final forced-answer call.
    tool_resp = FR("tool_use", [FB("tool_use", name="search_corpus",
                                   input={"query": "x"}, id="tu")])
    final = FR("end_turn", [FB("text", text="Final answer after budget.")])
    client = _patch(monkeypatch, [tool_resp, tool_resp, final])

    result = run_tool_agent("q", tiny_retriever, max_steps=2)

    assert result.steps == 2
    assert result.answer == "Final answer after budget."
    # API contract: history contains tool_use/tool_result blocks, so the salvage
    # call MUST still pass `tools` (else Anthropic 400s) with tool_choice=none.
    salvage = client.messages.calls[-1]
    assert salvage.get("tools"), "salvage call must include tools"
    assert salvage.get("tool_choice") == {"type": "none"}
    # reproducibility: config temperature must reach every call
    assert all(kw.get("temperature") == 0.0 for kw in client.messages.calls)

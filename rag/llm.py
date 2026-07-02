"""Provider-agnostic chat interface.

A single ``chat(prompt, config)`` dispatches on ``config.provider``. API keys are
read lazily *when generation is actually invoked* (never at import), so the
retrieval path runs with no key set. Anthropic is the default; OpenRouter is kept
as an optional alternative (ported from the original Sparkathon pipeline).
"""
from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

from .config import RagConfig

load_dotenv()  # populate env from a local .env if present (no-op otherwise)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def chat(prompt: str, config: RagConfig | None = None) -> str:
    config = config or RagConfig()
    if config.provider == "anthropic":
        return _anthropic_chat(prompt, config)
    if config.provider == "openrouter":
        return _openrouter_chat(prompt, config)
    raise ValueError(f"Unknown provider: {config.provider!r} (use 'anthropic' or 'openrouter')")


def get_anthropic_client():
    """Return an Anthropic client, or raise a clear error if the key is unset.

    Shared by plain generation and the tool-using agent (rag/tool_agent.py).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your environment or .env to run "
            "generation / the tool agent (retrieval works without it)."
        )
    import anthropic  # local import: only needed when calling the LLM

    return anthropic.Anthropic(api_key=api_key)


def _anthropic_chat(prompt: str, config: RagConfig) -> str:
    client = get_anthropic_client()
    resp = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()


def _openrouter_chat(prompt: str, config: RagConfig) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set but provider='openrouter'.")
    resp = requests.post(
        _OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter request failed: {resp.status_code}\n{resp.text}")
    return resp.json()["choices"][0]["message"]["content"].strip()

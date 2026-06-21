"""Golden-set schema + loader.

Each item is a question with its reference answer and the **ground-truth source
ids** (``Comment.source_id`` from ``rag/ingest.py``) of the comments that contain
the answer. A retrieved chunk counts as a hit iff its ``source_ids`` intersect
``gold_source_ids``. ``type='refuse'`` items have an empty gold set — the system
should decline to answer (not-in-corpus).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.jsonl"

TYPES = ("factoid", "multihop", "paraphrase", "refuse")


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    answer: str
    gold_source_ids: list[str]
    type: str
    difficulty: str

    @property
    def is_refuse(self) -> bool:
        return self.type == "refuse" or not self.gold_source_ids


def load_golden(path: Path | None = None) -> list[GoldenItem]:
    path = path or GOLDEN_PATH
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}. Run eval/golden/build_seed.py.")
    with path.open(encoding="utf-8") as f:
        return [GoldenItem(**json.loads(line)) for line in f if line.strip()]

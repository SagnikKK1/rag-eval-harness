"""Corpus ingestion: raw scraped comments -> a clean, normalized corpus.

The raw files store one record as a blank-line-separated block: an ISO-8601
timestamp line followed by one or more comment-text lines. We strip timestamps,
collapse whitespace, drop empty/trivial records, and deduplicate, assigning each
surviving comment a stable ``source_id`` (used later to attach ground-truth
chunk ids in the eval golden set).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CORPUS_PATH, RAW_CORPUS_FILES

_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_MIN_CHARS = 3


@dataclass(frozen=True)
class Comment:
    source_id: str   # stable id, e.g. "yt:0001"
    source: str      # short source tag, e.g. "yt" / "reddit"
    text: str
    timestamp: str | None


def _source_tag(path: Path) -> str:
    return "reddit" if "reddit" in path.name.lower() else "yt"


def _parse_file(path: Path) -> list[tuple[str | None, str]]:
    """Return (timestamp, text) tuples for each blank-line-separated record."""
    raw = path.read_text(encoding="utf-8")
    records: list[tuple[str | None, str]] = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        timestamp = None
        if _TIMESTAMP.match(lines[0]):
            timestamp = lines[0]
            lines = lines[1:]
        if not lines:
            continue
        text = " ".join(lines)
        records.append((timestamp, text))
    return records


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_corpus(
    raw_files: list[Path] | None = None,
    out_path: Path | None = None,
) -> list[Comment]:
    """Parse + clean + dedupe the raw files into a normalized corpus and persist it."""
    raw_files = raw_files or RAW_CORPUS_FILES
    out_path = out_path or CORPUS_PATH

    comments: list[Comment] = []
    seen: set[str] = set()
    for path in raw_files:
        if not path.exists():
            raise FileNotFoundError(f"Raw corpus file not found: {path}")
        tag = _source_tag(path)
        counter = 0
        for timestamp, text in _parse_file(path):
            text = _normalize(text)
            if len(text) < _MIN_CHARS:
                continue
            dedup_key = text.lower()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            comments.append(
                Comment(
                    source_id=f"{tag}:{counter:05d}", source=tag, text=text, timestamp=timestamp
                )
            )
            counter += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in comments:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return comments


def load_corpus(path: Path | None = None) -> list[Comment]:
    path = path or CORPUS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Corpus not built yet: {path}. Run build_corpus() first.")
    with path.open(encoding="utf-8") as f:
        return [Comment(**json.loads(line)) for line in f if line.strip()]


if __name__ == "__main__":
    built = build_corpus()
    print(f"Built corpus: {len(built)} comments -> {CORPUS_PATH}")

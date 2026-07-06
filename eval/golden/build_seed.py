"""Build the seed golden set (key-free, reproducible).

Questions/answers are hand-authored; ground-truth ``gold_source_ids`` are
resolved at the **comment level**: a comment is gold iff its own text matches
ALL of the item's ``must`` phrases with word-boundary, separator-tolerant
regexes. This fixes two failure modes of the earlier chunk-substring approach:

- **Spillover** — previously every comment sharing a chunk with one matching
  comment was marked gold (e.g. 101 gold ids when only 12 comments matched).
  Comment-level resolution marks exactly the matching comments; scoring still
  counts a retrieved chunk as a hit iff it *contains* a gold comment, which is
  correct (that chunk really does hand the generator the evidence).
- **Substring false positives/misses** — "sd" no longer matches Spanish
  "desde", "heat" no longer matches "heater"; "144hz" also matches "144 Hz",
  "low light" also matches "low-light".

Item types: factoid · cooccurrence (two aspects in ONE comment — honestly named;
these are not cross-document multi-hop) · paraphrase (reworded so the question
does NOT contain the gold-defining keywords — probes semantic retrieval; three
share a target with a factoid twin, disclosed via paired_with) · refuse
(not-in-corpus; empty gold).

Resolution is reproducible and reviewable: the script prints per-item matched
counts and FAILS LOUDLY if any answerable item resolves to zero comments
(previously it silently became a refusal test).

Run:  python eval/golden/build_seed.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.golden.schema import GOLDEN_PATH  # noqa: E402
from rag.ingest import load_corpus  # noqa: E402

# must = phrases that must ALL appear (word-boundary matched) in ONE comment.
SEED: list[dict] = [
    # --- factoid ---
    {"q": "What display refresh rate does the phone have?",
     "a": "A 144Hz display.", "type": "factoid", "diff": "easy", "must": ["144hz"]},
    {"q": "Do users mention bloatware on the device?",
     "a": "Yes, some complain about bloatware.", "type": "factoid", "diff": "easy",
     "must": ["bloatware"]},
    {"q": "Is the back panel made of vegan leather?",
     "a": "Yes, a vegan leather back is mentioned.", "type": "factoid", "diff": "medium",
     "must": ["vegan leather"]},
    {"q": "Does the phone use Gorilla Glass protection?",
     "a": "Yes, Gorilla Glass is mentioned.", "type": "factoid", "diff": "medium",
     "must": ["gorilla glass"]},
    {"q": "Is a Snapdragon processor referenced for this phone?",
     "a": "Yes, Snapdragon is discussed.", "type": "factoid", "diff": "medium",
     "must": ["snapdragon"]},
    {"q": "Is a Dimensity chipset mentioned for this phone?",
     "a": "Yes, a MediaTek Dimensity chipset is discussed.", "type": "factoid",
     "diff": "medium", "must": ["dimensity"]},
    {"q": "What do users say about the fingerprint sensor?",
     "a": "Opinions on the fingerprint sensor.", "type": "factoid", "diff": "medium",
     "must": ["fingerprint"]},
    {"q": "What charging wattage do users mention?",
     "a": "68W fast charging is mentioned.", "type": "factoid", "diff": "hard",
     "must": ["68w"]},
    {"q": "Is there a microSD card slot for storage expansion?",
     "a": "Users discuss the (absent) SD card slot.", "type": "factoid", "diff": "medium",
     "must": ["sd card"]},
    {"q": "Is the phone IP68 rated for water resistance?",
     "a": "Yes, an IP68 rating is mentioned.", "type": "factoid", "diff": "medium",
     "must": ["ip68"]},
    # --- cooccurrence (two aspects in one comment; NOT cross-document multihop) ---
    {"q": "Is there battery drain after the Android 15 update?",
     "a": "Some report battery issues after Android 15.", "type": "cooccurrence",
     "diff": "hard", "must": ["battery", "android 15"]},
    {"q": "Do users report battery drain following a software update?",
     "a": "Yes, battery drain after updates is reported.", "type": "cooccurrence",
     "diff": "hard", "must": ["battery", "drain*", "update*"]},
    {"q": "How does battery life compare with the Edge 50 Neo?",
     "a": "Comparisons of battery vs the Edge 50 Neo.", "type": "cooccurrence",
     "diff": "hard", "must": ["neo", "battery"]},
    {"q": "Does the phone heat up while charging?",
     "a": "Reports of heat while charging.", "type": "cooccurrence", "diff": "hard",
     "must": ["charg*", "heat*"]},
    {"q": "How is the camera in low-light conditions?",
     "a": "Opinions on the camera in low light.", "type": "cooccurrence", "diff": "hard",
     "must": ["camera", "low light"]},
    # --- paraphrase (question deliberately avoids the gold-defining keywords) ---
    {"q": "Does it run a MediaTek chip?",
     "a": "Yes — the Dimensity chipset is discussed.", "type": "paraphrase",
     "diff": "hard", "must": ["dimensity"], "paired_with": "g005"},
    {"q": "Is the biometric unlock quick and reliable?",
     "a": "Opinions on the fingerprint sensor.", "type": "paraphrase", "diff": "hard",
     "must": ["fingerprint"], "paired_with": "g006"},
    {"q": "Can the phone take a dunk in water and survive?",
     "a": "IP68 water resistance is mentioned.", "type": "paraphrase", "diff": "hard",
     "must": ["ip68"], "paired_with": "g009"},
    {"q": "How do photos turn out after dark?",
     "a": "Opinions on night photography.", "type": "paraphrase", "diff": "hard",
     "must": ["camera", "night"]},
    {"q": "Does it get warm during long play sessions?",
     "a": "Reports of heating while gaming.", "type": "paraphrase", "diff": "hard",
     "must": ["heat*", "gam*"]},
    # --- refuse / not-in-corpus (empty gold) ---
    {"q": "What is the iPhone 15 Pro's camera review score?",
     "a": "Not answerable from these Motorola reviews.", "type": "refuse", "diff": "easy",
     "must": []},
    {"q": "How much does a Tesla Model 3 cost?",
     "a": "Not answerable from these reviews.", "type": "refuse", "diff": "easy", "must": []},
    {"q": "What is the capital of France?",
     "a": "Not answerable from these reviews.", "type": "refuse", "diff": "easy", "must": []},
    {"q": "What's a good recipe for chocolate cake?",
     "a": "Not answerable from these reviews.", "type": "refuse", "diff": "easy", "must": []},
    {"q": "Who won the 2022 FIFA World Cup?",
     "a": "Not answerable from these reviews.", "type": "refuse", "diff": "easy", "must": []},
    {"q": "What interest rate did the Federal Reserve set?",
     "a": "Not answerable from these reviews.", "type": "refuse", "diff": "easy", "must": []},
]


def _phrase_pattern(phrase: str) -> re.Pattern:
    """Word-boundary, separator-tolerant pattern for a lowercase phrase.

    "sd card" matches "SD-card"/"sd card" but not "desde"; "144hz" also matches
    "144 Hz"; "low light" also matches "low-light".
    """
    words = []
    for word in phrase.lower().split():
        stem = word.endswith("*")  # trailing * = allow suffixes ("heat*" -> heat/heating/heats)
        w = re.escape(word.rstrip("*"))
        # allow an optional space/hyphen at digit->letter transitions (144hz ~ 144 hz)
        w = re.sub(r"(?<=\d)(?=[a-z])", r"[\\s\\-]?", w)
        words.append(w + (r"\w*" if stem else ""))
    return re.compile(r"\b" + r"[\s\-]+".join(words) + r"\b")


def resolve(must: list[str], comments) -> list[str]:
    """source_ids of comments whose own text matches ALL `must` phrases."""
    if not must:
        return []
    patterns = [_phrase_pattern(m) for m in must]
    return sorted(
        c.source_id for c in comments if all(p.search(c.text.lower()) for p in patterns)
    )


def main() -> None:
    comments = load_corpus()
    items, unresolved = [], []
    for i, s in enumerate(SEED):
        gold = resolve(s["must"], comments)
        if s["must"] and not gold:
            unresolved.append((s["q"], s["must"]))
        items.append({
            "id": f"g{i:03d}", "question": s["q"], "answer": s["a"],
            "gold_source_ids": gold, "type": s["type"], "difficulty": s["diff"],
        })

    if unresolved:
        for q, must in unresolved:
            print(f"UNRESOLVED (0 matching comments): {q!r}  must={must}", file=sys.stderr)
        sys.exit(f"{len(unresolved)} answerable item(s) resolved to zero comments — "
                 "fix the `must` phrases; refusing to write a corrupt golden set.")

    with GOLDEN_PATH.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    types = Counter(it["type"] for it in items)
    print(f"Wrote {len(items)} items -> {GOLDEN_PATH}")
    print("Types:", dict(types))
    print("Per-item gold sizes (comment-level):")
    for it in items:
        if it["type"] != "refuse":
            n_gold = len(it["gold_source_ids"])
            print(f"  {it['id']} [{it['type']:12s}] {n_gold:4d}  {it['question'][:60]}")


if __name__ == "__main__":
    main()

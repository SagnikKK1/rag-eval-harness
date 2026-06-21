"""Build the seed golden set (key-free, reproducible).

The questions/answers are hand-authored (the intellectual work); ground-truth
``gold_source_ids`` are resolved by locating chunks whose text contains all the
item's ``must`` phrases, then taking those chunks' ``source_ids``. This keeps
ground truth content-derived and independent of the retriever (no leakage), and
the resolution is reproducible + reviewable via the printed report.

This seed (~30 items across factoid / multihop / paraphrase / refuse) makes the
harness runnable now; ``eval/build_golden.py`` (LLM-assisted, key-gated) expands
it toward 150-300 later. Run:  python eval/golden/build_seed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.golden.schema import GOLDEN_PATH  # noqa: E402
from rag.chunk import load_chunks  # noqa: E402
from rag.config import RagConfig  # noqa: E402

# must = phrases that must ALL appear (lowercased) in a chunk for it to be gold.
# Conjunctions are tuned (see eval/golden/build_seed analysis) to keep gold sets
# small and specific so recall@k / MRR stay discriminative on this opinion corpus.
SEED: list[dict] = [
    # --- factoid ---
    {"q": "What display refresh rate does the phone have?",
     "a": "A 144Hz display.", "type": "factoid", "diff": "easy", "must": ["144hz", "display"]},
    {"q": "Do users mention bloatware on the device?",
     "a": "Yes, some complain about bloatware.", "type": "factoid", "diff": "easy",
     "must": ["bloatware"]},
    {"q": "Is the back panel made of vegan leather?",
     "a": "Yes, a vegan leather back is mentioned.", "type": "factoid", "diff": "medium",
     "must": ["vegan", "leather"]},
    {"q": "Does the phone use Gorilla Glass protection?",
     "a": "Yes, Gorilla Glass is mentioned.", "type": "factoid", "diff": "medium",
     "must": ["gorilla", "glass"]},
    {"q": "Is a Snapdragon processor referenced for this phone?",
     "a": "Yes, Snapdragon is discussed.", "type": "factoid", "diff": "medium",
     "must": ["snapdragon"]},
    {"q": "Is a Dimensity chipset mentioned for this phone?",
     "a": "Yes, a MediaTek Dimensity chipset is discussed.", "type": "factoid", "diff": "medium",
     "must": ["dimensity"]},
    {"q": "What do users say about the fingerprint sensor?",
     "a": "Opinions on the fingerprint sensor.", "type": "factoid", "diff": "medium",
     "must": ["fingerprint", "sensor"]},
    {"q": "What charging wattage (e.g. 68W) do users mention?",
     "a": "68W fast charging is mentioned.", "type": "factoid", "diff": "hard",
     "must": ["68w", "charging"]},
    {"q": "Is there a microSD card slot for storage expansion?",
     "a": "Users discuss the (absent) SD card slot.", "type": "factoid", "diff": "medium",
     "must": ["sd", "card", "slot"]},
    {"q": "Is the phone IP68 rated for water resistance?",
     "a": "Yes, an IP68 rating is mentioned.", "type": "factoid", "diff": "medium",
     "must": ["ip68", "rating"]},
    # --- multihop (require two co-occurring aspects) ---
    {"q": "Is there battery drain after the Android 15 update?",
     "a": "Some report battery issues after Android 15.", "type": "multihop", "diff": "hard",
     "must": ["battery", "android 15"]},
    {"q": "Do users report battery drain following a software update?",
     "a": "Yes, battery drain after updates is reported.", "type": "multihop", "diff": "hard",
     "must": ["battery", "drain", "update"]},
    {"q": "How does battery life compare with the Edge 50 Neo?",
     "a": "Comparisons of battery vs the Edge 50 Neo.", "type": "multihop", "diff": "hard",
     "must": ["edge 50 neo", "battery"]},
    {"q": "Does the phone heat up while charging?",
     "a": "Reports of heat while charging.", "type": "multihop", "diff": "hard",
     "must": ["charging", "heat"]},
    {"q": "How is the camera in low-light conditions?",
     "a": "Opinions on the camera in low light.", "type": "multihop", "diff": "hard",
     "must": ["camera", "low light"]},
    # --- paraphrase (reworded factoids -> same gold) ---
    {"q": "Can I expand the memory with an external card?",
     "a": "Concerns the (missing) SD card slot.", "type": "paraphrase", "diff": "medium",
     "must": ["sd", "card", "slot"]},
    {"q": "How fluid is the screen — what's its Hz?",
     "a": "144Hz refresh rate.", "type": "paraphrase", "diff": "medium",
     "must": ["144hz", "display"]},
    {"q": "Are there pre-installed junk apps?",
     "a": "Bloatware complaints.", "type": "paraphrase", "diff": "medium", "must": ["bloatware"]},
    {"q": "How well does the camera shoot at night?",
     "a": "Opinions on night photography.", "type": "paraphrase", "diff": "hard",
     "must": ["camera", "night"]},
    {"q": "Does it get hot during gaming?",
     "a": "Reports of heating while gaming.", "type": "paraphrase", "diff": "hard",
     "must": ["heating", "gaming"]},
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


def resolve(must: list[str], chunks) -> list[str]:
    if not must:
        return []
    src: set[str] = set()
    for c in chunks:
        text = c.text.lower()
        if all(m in text for m in must):
            src.update(c.source_ids)
    return sorted(src)


def main() -> None:
    chunks = load_chunks(RagConfig())
    items, unresolved = [], []
    for i, s in enumerate(SEED):
        gold = resolve(s["must"], chunks)
        if s["must"] and not gold:
            unresolved.append((s["q"], s["must"]))
        items.append({
            "id": f"g{i:03d}", "question": s["q"], "answer": s["a"],
            "gold_source_ids": gold, "type": s["type"], "difficulty": s["diff"],
        })

    with GOLDEN_PATH.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    types = Counter(it["type"] for it in items)
    print(f"Wrote {len(items)} items -> {GOLDEN_PATH}")
    print("Types:", dict(types))
    print("Gold-set sizes:", [len(it["gold_source_ids"]) for it in items if it["type"] != "refuse"])
    if unresolved:
        print("\n!! UNRESOLVED (no chunk matched 'must') — fix these:")
        for q, must in unresolved:
            print(f"   - {q}  must={must}")


if __name__ == "__main__":
    main()

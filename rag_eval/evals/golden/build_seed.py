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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rag_eval.evals.golden.schema import GOLDEN_PATH  # noqa: E402
from rag_eval.ingest import load_corpus  # noqa: E402

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
    # --- factoid (expansion wave 2) ---
    {"q": "Do users discuss the curved display?",
     "a": "Yes, the curved display is widely discussed.", "type": "factoid", "diff": "easy",
     "must": ["curved"]},
    {"q": "How is the speaker quality?",
     "a": "Mixed opinions on the speakers.", "type": "factoid", "diff": "easy",
     "must": ["speaker*"]},
    {"q": "Is there a 3.5mm headphone jack?",
     "a": "Users discuss the (missing) headphone jack.", "type": "factoid", "diff": "medium",
     "must": ["jack"]},
    {"q": "Do users talk about stylus support?",
     "a": "Yes, stylus support comes up.", "type": "factoid", "diff": "medium",
     "must": ["stylus"]},
    {"q": "Does the phone support 5G?",
     "a": "Yes, 5G support is discussed.", "type": "factoid", "diff": "easy", "must": ["5g"]},
    {"q": "Do users report the green line display issue?",
     "a": "Yes, some report the green line problem.", "type": "factoid", "diff": "hard",
     "must": ["green line"]},
    {"q": "Are there motherboard failure complaints?",
     "a": "Yes, motherboard issues are reported.", "type": "factoid", "diff": "hard",
     "must": ["motherboard"]},
    {"q": "What do users say about the warranty?",
     "a": "Warranty experiences are discussed.", "type": "factoid", "diff": "medium",
     "must": ["warranty"]},
    {"q": "Are there complaints about Motorola service centers?",
     "a": "Yes, service-center complaints appear.", "type": "factoid", "diff": "medium",
     "must": ["service cent*"]},
    {"q": "How much RAM do users discuss?",
     "a": "RAM variants are discussed.", "type": "factoid", "diff": "medium", "must": ["ram"]},
    {"q": "Will the phone get Android 16?",
     "a": "Users ask/answer about Android 16.", "type": "factoid", "diff": "medium",
     "must": ["android 16"]},
    {"q": "Do users mention HDR support?",
     "a": "Yes, HDR is discussed.", "type": "factoid", "diff": "medium", "must": ["hdr"]},
    {"q": "How bright does the display get?",
     "a": "Display brightness is discussed.", "type": "factoid", "diff": "medium",
     "must": ["brightness"]},
    {"q": "How is the front camera?",
     "a": "Front-camera opinions are shared.", "type": "factoid", "diff": "medium",
     "must": ["front camera"]},
    {"q": "What do users say about camera zoom?",
     "a": "Zoom quality is discussed.", "type": "factoid", "diff": "medium", "must": ["zoom"]},
    {"q": "How does BGMI run on this phone?",
     "a": "BGMI gaming performance is discussed.", "type": "factoid", "diff": "hard",
     "must": ["bgmi"]},
    {"q": "What do users think about the price?",
     "a": "Price opinions are widely shared.", "type": "factoid", "diff": "easy",
     "must": ["price"]},
    {"q": "Does the phone support eSIM?",
     "a": "eSIM support is discussed.", "type": "factoid", "diff": "hard", "must": ["esim"]},
    # --- cooccurrence (expansion wave 2) ---
    {"q": "Does the phone heat up while using the camera?",
     "a": "Some report heating during camera use.", "type": "cooccurrence", "diff": "hard",
     "must": ["heat*", "camera"]},
    {"q": "Is there lag during gaming?",
     "a": "Some report gaming lag.", "type": "cooccurrence", "diff": "hard",
     "must": ["gaming", "lag*"]},
    {"q": "How many OS updates are promised?",
     "a": "The update promise is discussed.", "type": "cooccurrence", "diff": "hard",
     "must": ["update*", "promise*"]},
    {"q": "What battery capacity in mAh do users cite?",
     "a": "Battery capacity figures are cited.", "type": "cooccurrence", "diff": "hard",
     "must": ["battery", "mah"]},
    {"q": "Do screens crack easily on this phone?",
     "a": "Some report cracked screens.", "type": "cooccurrence", "diff": "hard",
     "must": ["screen", "crack*"]},
    {"q": "How does it compare with the Samsung A55?",
     "a": "Comparisons with the A55 appear.", "type": "cooccurrence", "diff": "hard",
     "must": ["samsung", "a55"]},
    {"q": "Does the box include a charger?",
     "a": "In-box charger discussion appears.", "type": "cooccurrence", "diff": "hard",
     "must": ["charger", "box"]},
    # --- paraphrase (expansion wave 2; de-lexicalized) ---
    {"q": "Can I plug in wired earphones?",
     "a": "Concerns the 3.5mm jack.", "type": "paraphrase", "diff": "hard",
     "must": ["jack"], "paired_with": "jack-factoid"},
    {"q": "Does it support pen input?",
     "a": "Concerns stylus support.", "type": "paraphrase", "diff": "hard",
     "must": ["stylus"], "paired_with": "stylus-factoid"},
    {"q": "How loud and clear is the audio output?",
     "a": "Concerns speaker quality.", "type": "paraphrase", "diff": "hard",
     "must": ["speaker*"], "paired_with": "speaker-factoid"},
    {"q": "Is the screen bent along the edges?",
     "a": "Concerns the curved display.", "type": "paraphrase", "diff": "hard",
     "must": ["curved"], "paired_with": "curved-factoid"},
    {"q": "Will it receive future OS versions?",
     "a": "Concerns Android 16 updates.", "type": "paraphrase", "diff": "hard",
     "must": ["android 16"], "paired_with": "android16-factoid"},
    {"q": "Is the vibration feedback crisp?",
     "a": "Concerns haptics quality.", "type": "paraphrase", "diff": "hard",
     "must": ["haptic*"]},
    {"q": "Can I activate a digital SIM without a physical card?",
     "a": "Concerns eSIM support.", "type": "paraphrase", "diff": "hard",
     "must": ["esim"], "paired_with": "esim-factoid"},
    # --- factoid (expansion wave 3: comparisons, commerce, software, hardware) ---
    {"q": "Do users report lag on this phone?",
     "a": "Lag reports and rebuttals appear.", "type": "factoid", "diff": "easy",
     "must": ["lag*"]},
    {"q": "Do users compare it with Realme phones?",
     "a": "Yes, Realme comparisons appear.", "type": "factoid", "diff": "easy",
     "must": ["realme"]},
    {"q": "Do users compare it with Vivo phones?",
     "a": "Yes, Vivo comparisons appear.", "type": "factoid", "diff": "easy",
     "must": ["vivo"]},
    {"q": "Do users compare it with OnePlus phones?",
     "a": "Yes, OnePlus comparisons appear.", "type": "factoid", "diff": "easy",
     "must": ["oneplus"]},
    {"q": "Is the Edge 60 discussed as a successor or alternative?",
     "a": "Yes, the Edge 60 comes up.", "type": "factoid", "diff": "medium",
     "must": ["edge 60"]},
    {"q": "Do users compare it with Poco phones?",
     "a": "Yes, Poco comparisons appear.", "type": "factoid", "diff": "easy",
     "must": ["poco"]},
    {"q": "Do users compare it with Redmi phones?",
     "a": "Yes, Redmi comparisons appear.", "type": "factoid", "diff": "easy",
     "must": ["redmi"]},
    {"q": "Do users compare it with iQOO phones?",
     "a": "Yes, iQOO comparisons appear.", "type": "factoid", "diff": "medium",
     "must": ["iqoo"]},
    {"q": "Do users compare it with Google Pixel phones?",
     "a": "Yes, Pixel comparisons appear.", "type": "factoid", "diff": "medium",
     "must": ["pixel"]},
    {"q": "Do users compare it with Nothing Phone models?",
     "a": "Yes, Nothing Phone comparisons appear.", "type": "factoid", "diff": "medium",
     "must": ["nothing phone"]},
    {"q": "Are there network connectivity complaints?",
     "a": "Network issues are discussed.", "type": "factoid", "diff": "medium",
     "must": ["network"]},
    {"q": "Do users mention Bluetooth behavior?",
     "a": "Bluetooth issues/behavior discussed.", "type": "factoid", "diff": "medium",
     "must": ["bluetooth"]},
    {"q": "Do users prefer a flat display over curved?",
     "a": "The flat-display debate appears.", "type": "factoid", "diff": "medium",
     "must": ["flat display"]},
    {"q": "Do users discuss covers for the phone?",
     "a": "Covers/cases are discussed.", "type": "factoid", "diff": "medium",
     "must": ["cover"]},
    {"q": "Are there Wi-Fi issues reported?",
     "a": "Wi-Fi behavior is discussed.", "type": "factoid", "diff": "medium",
     "must": ["wifi"]},
    {"q": "Is the display panel AMOLED?",
     "a": "The AMOLED/pOLED panel is discussed.", "type": "factoid", "diff": "medium",
     "must": ["amoled"]},
    {"q": "Can the phone record 4K video?",
     "a": "4K recording is discussed.", "type": "factoid", "diff": "medium", "must": ["4k"]},
    {"q": "Was the phone bought on Flipkart?",
     "a": "Flipkart purchases/sales are discussed.", "type": "factoid", "diff": "medium",
     "must": ["flipkart"]},
    {"q": "How is the battery backup in daily use?",
     "a": "Battery backup impressions are shared.", "type": "factoid", "diff": "medium",
     "must": ["battery backup"]},
    {"q": "Are there notification delivery issues?",
     "a": "Notification issues are discussed.", "type": "factoid", "diff": "medium",
     "must": ["notification*"]},
    {"q": "What do users say about Hello UI?",
     "a": "Hello UI impressions are shared.", "type": "factoid", "diff": "hard",
     "must": ["hello ui"]},
    {"q": "Is Android 14 mentioned for this phone?",
     "a": "Android 14 is discussed.", "type": "factoid", "diff": "medium",
     "must": ["android 14"]},
    {"q": "What do users say about video recording quality?",
     "a": "Video recording is discussed.", "type": "factoid", "diff": "medium",
     "must": ["video record*"]},
    {"q": "What do users think of the design?",
     "a": "Design opinions are shared.", "type": "factoid", "diff": "easy",
     "must": ["design"]},
    {"q": "What colour options do users mention?",
     "a": "Colour options/preferences are discussed.", "type": "factoid", "diff": "easy",
     "must": ["colour*"]},
    {"q": "Was the phone bought on Amazon?",
     "a": "Amazon purchases are mentioned.", "type": "factoid", "diff": "medium",
     "must": ["amazon"]},
    {"q": "Do users mention a sale price?",
     "a": "Sale prices are discussed.", "type": "factoid", "diff": "medium", "must": ["sale"]},
    {"q": "Is NFC supported?",
     "a": "NFC support is discussed.", "type": "factoid", "diff": "hard", "must": ["nfc"]},
    {"q": "Do users mention VoLTE support?",
     "a": "VoLTE is mentioned.", "type": "factoid", "diff": "hard", "must": ["volte"]},
    {"q": "Does the camera have OIS?",
     "a": "OIS is discussed.", "type": "factoid", "diff": "hard", "must": ["ois"]},
    {"q": "How are portrait shots?",
     "a": "Portrait photography is discussed.", "type": "factoid", "diff": "hard",
     "must": ["portrait"]},
    {"q": "Do users mention Dolby audio?",
     "a": "Dolby audio is mentioned.", "type": "factoid", "diff": "hard", "must": ["dolby"]},
    {"q": "Is the phone considered compact?",
     "a": "Compactness is discussed.", "type": "factoid", "diff": "medium",
     "must": ["compact"]},
    {"q": "Is the UI smooth in daily use?",
     "a": "Smoothness impressions are shared.", "type": "factoid", "diff": "easy",
     "must": ["smooth*"]},
    {"q": "Do users mention exchange offers?",
     "a": "Exchange offers are mentioned.", "type": "factoid", "diff": "hard",
     "must": ["exchange"]},
    # --- cooccurrence (expansion wave 3) ---
    {"q": "Are there software bugs reported?",
     "a": "Software bugs are reported.", "type": "cooccurrence", "diff": "hard",
     "must": ["software", "bug*"]},
    {"q": "Is charging slower than expected for some users?",
     "a": "Slow-charging reports appear.", "type": "cooccurrence", "diff": "hard",
     "must": ["charging", "slow*"]},
    # --- paraphrase (expansion wave 3; de-lexicalized) ---
    {"q": "Does the interface ever feel sluggish?",
     "a": "Concerns lag reports.", "type": "paraphrase", "diff": "hard", "must": ["lag*"]},
    {"q": "Can it capture ultra-high-definition footage?",
     "a": "Concerns 4K recording.", "type": "paraphrase", "diff": "hard", "must": ["4k"]},
    {"q": "Which online store had it on offer?",
     "a": "Concerns Flipkart availability.", "type": "paraphrase", "diff": "hard",
     "must": ["flipkart"]},
    {"q": "How long does a full charge last with normal usage?",
     "a": "Concerns battery backup.", "type": "paraphrase", "diff": "hard",
     "must": ["battery backup"]},
    {"q": "Is the screen completely flush without bends?",
     "a": "Concerns the flat-display debate.", "type": "paraphrase", "diff": "hard",
     "must": ["flat display"]},
    {"q": "Do alerts arrive late or get swallowed?",
     "a": "Concerns notification issues.", "type": "paraphrase", "diff": "hard",
     "must": ["notification*"]},
    {"q": "How does Motorola's Android skin feel day to day?",
     "a": "Concerns Hello UI.", "type": "paraphrase", "diff": "hard", "must": ["hello ui"]},
    {"q": "Is it heavy in the pocket?",
     "a": "Concerns weight.", "type": "paraphrase", "diff": "hard", "must": ["weight"]},
    {"q": "Does the Google camera port work on it?",
     "a": "Concerns GCam.", "type": "paraphrase", "diff": "hard", "must": ["gcam"]},
    {"q": "Can I stream HD on OTT apps?",
     "a": "Concerns Netflix/Widevine support.", "type": "paraphrase", "diff": "hard",
     "must": ["netflix"]},
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
    # near-domain refuses — lexically close to the corpus, much harder to abstain on
    {"q": "What is the price of the Samsung washing machine?",
     "a": "Not answerable — corpus covers phones, not appliances.", "type": "refuse",
     "diff": "hard", "must": []},
    {"q": "How much does an iPhone 15 battery replacement cost?",
     "a": "Not answerable from Motorola reviews.", "type": "refuse", "diff": "hard",
     "must": []},
    {"q": "What are the minimum system requirements for Windows 11?",
     "a": "Not answerable from these reviews.", "type": "refuse", "diff": "easy", "must": []},
    {"q": "Who won the 2023 Cricket World Cup?",
     "a": "Not answerable from these reviews.", "type": "refuse", "diff": "easy", "must": []},
    {"q": "What's the best pizza place in Kolkata?",
     "a": "Not answerable from these reviews.", "type": "refuse", "diff": "easy", "must": []},
    {"q": "What is the S&P 500 index level today?",
     "a": "Not answerable from these reviews.", "type": "refuse", "diff": "easy", "must": []},
    {"q": "Which is better for web development, Python or JavaScript?",
     "a": "Not answerable from these reviews.", "type": "refuse", "diff": "easy", "must": []},
    {"q": "What is the capital of Australia?",
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

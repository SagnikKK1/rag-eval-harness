# Plan: generation eval (faithfulness / answer-relevancy)

**Status:** not started. This is a plan, not an implementation.
**Audience:** whoever picks this up next.
**Blocks:** the two open items in README "Limitations & honesty" — generation metrics
are unmeasured, and the golden set is N=99.

The retrieval half of this harness is finished and defensible: recall@k / MRR /
nDCG@10 across six arms, paired significance, a CI regression gate. The generation
half is a hole. Every number currently reported describes *whether the right chunks
were retrieved*, not *whether the answer built from them was true*. That is the gap
this plan closes.

---

## 1. The two things that were blocking it

### 1.1 The API key — resolved

`ANTHROPIC_API_KEY` is available now. `.env.example` already carries the variable, so
nothing structural changes. Note the SDK resolves credentials in order
(`ANTHROPIC_API_KEY`, then `ANTHROPIC_AUTH_TOKEN`, then an `ant auth login` profile),
so a zero-arg `anthropic.Anthropic()` works in CI with either an env var or a profile.

### 1.2 The ragas dependency conflict — resolve by dropping ragas

README currently documents the workaround: ragas 0.4.x is import-incompatible with the
`langchain-core` / `langgraph` 1.x stack the CRAG graph needs, so generation eval has to
run in an isolated environment. `pyproject.toml` encodes this as the
`eval = ["ragas", "langchain-anthropic", "datasets"]` extra.

**Decision: implement the two metrics directly against the Anthropic SDK and delete the
`eval` extra.** Reasons:

1. Two metrics is not enough surface to justify a dependency that forks the environment.
   Faithfulness and answer-relevancy are each one judge call with a rubric.
2. A split environment means the generation numbers can never run in the same CI job as
   the retrieval gate, so they will rot.
3. This harness's whole argument is that measurement is auditable. A vendored rubric we
   wrote and can show a reviewer is more auditable than ragas' internal prompts, which
   change between minor versions and would silently move our numbers.
4. `anthropic` is already a direct dependency (`anthropic==0.111.0`), so the judge adds
   no new package.

Keep `rag_eval/evals/ragas_eval.py` as a reference implementation behind a clearly
marked optional path, or delete it. Do not leave it importable-but-broken.

**Also decide the SDK version before writing code.** The pin is `anthropic==0.111.0`,
which is 0.x. The 1.x line is a breaking change (httpx2 instead of httpx, awaited async
`.with_raw_response`, removed deprecated parameters, Python >= 3.10). Either pin 0.x
deliberately with a comment, or do the upgrade as its own commit before this work, not
tangled into it.

---

## 2. Phase 1 — judge infrastructure

Route everything through the existing `rag_eval/llm.py` pluggable layer. Do not add a
second LLM entry point.

**Models.** Use the exact IDs, with no date suffixes:

| Role | Model | Price (in / out per MTok) |
|---|---|---|
| Judge | `claude-opus-5` | $5 / $25 |
| Answer generator (arm under test) | `claude-haiku-4-5` | $1 / $5 |

The README currently pins `claude-haiku-4-5-20251001` as judge with `claude-sonnet-4-6`
as the upgrade. Both strings need fixing: the date suffix is wrong, and `claude-sonnet-4-6`
is previous-generation (current is `claude-sonnet-5`). Judge quality matters more than
judge cost here, because a noisy judge invalidates every number downstream, so the judge
moves up to Opus 5 while the thing being *measured* stays cheap.

**Structured output, not prose parsing.** The judge returns a typed verdict; never regex
a score out of free text. Use `client.messages.parse()` with `output_config={"format": ...}`.
Do not use the deprecated top-level `output_format` parameter.

Verdict shape, per (item, arm, metric):

```
{ "score": 0 | 1,            # binary per claim/sentence, aggregated later
  "rationale": str,          # one sentence, for the audit trail
  "unsupported_spans": [str] # verbatim answer spans with no source support
}
```

`unsupported_spans` is the field that makes a failure inspectable instead of a number.

**Thinking and effort.** On Opus 5 thinking is on by default; pass
`thinking={"type": "adaptive"}` explicitly and `output_config={"effort": "medium"}`.
Judging one answer against ten chunks does not need `high`. Do not set `budget_tokens` —
it is removed on this model and returns a 400.

**Batch the judge calls.** This is an offline eval over hundreds of items, which is
exactly the Batch API's case, at 50% of standard pricing:

- `client.messages.batches.create(requests=[...])` with one `custom_id` per
  `(item_id, arm, metric)` triple.
- Poll `client.messages.batches.retrieve(id).processing_status` until `"ended"`.
- Stream `client.messages.batches.results(id)`.
- **Key results by `custom_id`, never by position.** Batch results come back in
  arbitrary order. Getting this wrong silently misattributes scores across arms, which
  is the single worst failure mode available here and would not announce itself.

**Cache the rubric.** The system prompt and rubric are identical across every call in a
run. Put them first and mark the last stable block with
`cache_control={"type": "ephemeral"}`, keeping the per-item question, answer, and chunks
after the breakpoint. Verify it is working by asserting
`usage.cache_read_input_tokens > 0` after the first call — if it stays zero, something
volatile leaked into the prefix.

**Error handling.** Chain most-specific first: `NotFoundError`, `RateLimitError`,
`APIStatusError`, `APIConnectionError`. A single broad `except APIStatusError` loses the
retryable/non-retryable distinction.

---

## 3. Phase 2 — establish that the judge is trustworthy

This phase is the reason the plan exists and it is the one to not skip. This repo already
refuses to report a delta without a significance test, and it re-derived every gold label
after an internal audit found them inflated. An LLM judge is a *measuring instrument*;
shipping numbers from an uncalibrated one would contradict the standard the rest of the
harness sets.

Required before any generation number goes in the README:

1. **Human-labelled calibration subset.** Hand-label faithfulness on 50 (answer, chunks)
   pairs sampled across arms, including deliberately unfaithful ones. Report judge/human
   agreement as Cohen's kappa, not raw accuracy — raw accuracy looks fine when one class
   dominates.
2. **Judge self-consistency.** Run the same 50 items three times at the same config.
   Report the disagreement rate. This is the direct analogue of the seed-noise floor
   (sd 0.0019) established for the Wunderfund work: it sets the floor below which a
   between-arm difference means nothing.
3. **Position and verbosity bias checks.** Shuffle chunk order and confirm scores are
   stable. Separately, confirm the judge is not simply rewarding longer answers — regress
   score on answer length and report the coefficient.
4. **Report the judge alongside every number.** Model ID, effort, rubric version, N, and
   the kappa from (1). A faithfulness score without its judge provenance is not a result.

If kappa comes in below roughly 0.6, the rubric needs work before the numbers are worth
publishing. Say so in the README rather than shipping the number with a caveat.

---

## 4. Phase 3 — golden set expansion

Currently 99 answerable + 14 not-in-corpus. Target 150–300 answerable, per the README's
own stated next step.

- Draft candidates with `claude-opus-5`, then **hand-verify every one**. LLM-drafted and
  unverified is how the first gold set got inflated; the existing
  `rag_eval/evals/golden/build_seed.py` already resolves gold ids at comment level with
  word-boundary regexes and fails loudly on unresolved items, so keep that invariant.
- Hold the existing type mix (factoid / co-occurrence / paraphrase / refuse) and keep the
  de-lexicalized paraphrase items, since those are what expose BM25's collapse to
  recall@5 0.27 and justify the dense and reranked arms.
- Re-run the retrieval A/B on the expanded set. Hybrid is currently directional only
  (ΔMRR +0.043, CI crosses zero); more N is the honest way to resolve it. **Expect that
  it may fail to reach significance.** Report that if so.
- Do not reuse any item that was drafted by the same model that later generates answers
  under test, without a human in between.

---

## 5. Phase 4 — wire into the report and the gate

- Extend `rag_eval/cli/run_eval.py` with a `--generation` flag. Default off, so the
  retrieval gate stays fast and keyless.
- Reuse `rag_eval/evals/stats.py` for paired bootstrap CIs on generation deltas. Do not
  write a second statistics path.
- Add faithfulness and answer-relevancy columns to the `results/results.md` arm table.
- **CI: keep generation out of the blocking gate initially.** Run it on a fixed 30-item
  sample on a schedule, record the number, and only promote it to blocking once Phase 2
  has established the noise floor. A gate that fails on judge noise trains people to
  ignore the gate.
- Budget guard: fail fast if an estimated run cost exceeds a configured ceiling, so a
  loop bug cannot quietly spend real money.

---

## 6. Cost

Estimated with `count_tokens` assumptions of ~1,500 input and ~300 output tokens per judge
call; **replace these with `client.messages.count_tokens` on real prompts before running.**

For 300 items × 6 arms × 2 metrics = 3,600 judge calls:

| Judge | Standard | Batch API (50%) |
|---|---|---|
| `claude-opus-5` | ~$54 | **~$27** |
| `claude-haiku-4-5` | ~$11 | ~$5.50 |

Generating the 1,800 answers under test on `claude-haiku-4-5` adds roughly $4. Rubric
caching cuts the input side further, since the rubric is the bulk of a short prompt and
repeats on every call.

So the full run is about **$30 batched with an Opus 5 judge**. That is cheap enough that
judge cost is not a reason to downgrade the instrument. Phase 2's calibration work is a
few hundred extra calls and rounds to nothing.

---

## 7. Definition of done

- [ ] `eval` extra removed; generation eval runs in the main environment.
- [ ] Judge returns a typed verdict via structured outputs, batched, with `custom_id`
      keying covered by a test that would catch a misattribution.
- [ ] Judge/human kappa on a 50-item hand-labelled subset, reported.
- [ ] Judge self-consistency disagreement rate, reported, and used as the floor for
      claiming any between-arm difference.
- [ ] Position-order and answer-length bias checks, reported.
- [ ] Golden set at 150–300 hand-verified answerable items; retrieval A/B re-run on it.
- [ ] Faithfulness and answer-relevancy in the arm table with judge provenance and N.
- [ ] README "Limitations & honesty" updated: the "pending an API key" and "N=99" items
      either resolved or restated as whatever is still true.

---

## 8. What this plan deliberately does not do

- It does not tune the abstention threshold (dense cosine < 0.35). That is an
  uncalibrated default on purpose, and calibrating it on the golden refuse items would
  overfit the one set used to report refusal accuracy.
- It does not try to rescue CRAG's numbers. CRAG measurably loses 10 points of recall@5
  with 20% false refusals; its keyless run is disclosed. Re-running it with LLM
  reformulation enabled is a separate experiment with its own pre-registration, not a
  fix folded into this work.
- It does not add more retrieval arms. The retrieval half is done.

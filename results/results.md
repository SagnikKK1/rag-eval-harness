### RAG A/B results

run: 2026-07-09 10:47 UTC · git 5ce4dfb · N=99 answerable + 14 refuse · depth 10 · embed `sentence-transformers/all-MiniLM-L6-v2` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · chunks 512/64

| arm | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 | refusal_acc | false_refusal | latency_ms | p95_ms | ce_calls/q | faithfulness | answer_relevancy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.545 | 0.747 | 0.798 | 0.889 | 0.660 | 0.438 | 0.714 | 0.081 | 37 | 87 | 0.00 | — | — |
| sparse | 0.566 | 0.737 | 0.818 | 0.879 | 0.669 | 0.423 | 0.714 | 0.081 | 19 | 24 | 0.00 | — | — |
| hybrid | 0.586 | 0.828 | 0.859 | 0.889 | 0.703 | 0.476 | 0.714 | 0.081 | 40 | 52 | 0.00 | — | — |
| reranked | 0.778 | 0.899 | 0.919 | 0.939 | 0.844 | 0.554 | 0.714 | 0.081 | 276 | 302 | 1.00 | — | — |
| adaptive | 0.707 | 0.869 | 0.879 | 0.899 | 0.791 | 0.508 | 0.714 | 0.081 | 162 | 305 | 0.67 | — | — |
| crag | 0.636 | 0.697 | 0.697 | 0.697 | 0.667 | 0.312 | 0.714 | 0.202 | 777 | 1657 | 1.65 | — | — |

Refusal: `refusal_acc` = correct refusals on not-in-corpus items; `false_refusal` = wrong refusals on answerable items. Non-CRAG arms refuse when top-1 dense cosine < 0.35 (uncalibrated default, not tuned on the golden set).

#### Paired significance vs baseline

| arm vs baseline | Δrecall@5 (won/lost) | McNemar p | ΔMRR [95% CI] | significant? |
| --- | --- | --- | --- | --- |
| sparse | +9/−7 | 0.804 | +0.009 [-0.079, +0.100] | no |
| hybrid | +8/−2 | 0.109 | +0.043 [-0.013, +0.101] | no |
| reranked | +13/−1 | 0.002 | +0.184 [+0.117, +0.257] | yes |
| adaptive | +9/−1 | 0.021 | +0.131 [+0.071, +0.194] | yes |
| crag | +4/−14 | 0.031 | +0.007 [-0.068, +0.083] | yes |

At this N, treat non-significant deltas as directional only.

#### Per-type breakdown

| arm | cooccurrence r@5 / MRR (n) | factoid r@5 / MRR (n) | paraphrase r@5 / MRR (n) |
| --- | --- | --- | --- |
| baseline | 0.86 / 0.56 (14) | 0.87 / 0.78 (63) | 0.55 / 0.38 (22) |
| sparse | 1.00 / 0.78 (14) | 0.97 / 0.79 (63) | 0.27 / 0.24 (22) |
| hybrid | 1.00 / 0.73 (14) | 0.95 / 0.83 (63) | 0.50 / 0.33 (22) |
| reranked | 1.00 / 0.96 (14) | 0.97 / 0.94 (63) | 0.73 / 0.49 (22) |
| adaptive | 0.93 / 0.82 (14) | 0.94 / 0.89 (63) | 0.68 / 0.48 (22) |
| crag | 0.93 / 0.82 (14) | 0.79 / 0.78 (63) | 0.27 / 0.25 (22) |

#### Abstention risk–coverage (dense-confidence threshold sweep)

| threshold | refusal_acc (refuse items) | false_refusal (answerable) |
| --- | --- | --- |
| 0.25 | 0.429 | 0.020 |
| 0.30 | 0.571 | 0.040 |
| 0.35 | 0.714 | 0.081 |
| 0.40 | 0.786 | 0.111 |
| 0.45 | 0.857 | 0.293 |
| 0.50 | 0.929 | 0.455 |

_Generation columns pending ANTHROPIC_API_KEY (`--generate`)._

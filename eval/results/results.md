### RAG A/B results

run: 2026-07-06 20:21 UTC · git e506479 · N=20 answerable + 6 refuse · depth 10 · embed `sentence-transformers/all-MiniLM-L6-v2` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · chunks 512/64

| arm | recall@1 | recall@3 | recall@5 | recall@10 | mrr | refusal_acc | false_refusal | latency_ms | ce_calls/q | faithfulness | answer_relevancy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.550 | 0.700 | 0.800 | 0.850 | 0.646 | 0.667 | 0.050 | 34 | 0.00 | — | — |
| sparse | 0.650 | 0.650 | 0.700 | 0.800 | 0.675 | 0.667 | 0.050 | 11 | 0.00 | — | — |
| hybrid | 0.550 | 0.800 | 0.800 | 0.850 | 0.681 | 0.667 | 0.050 | 23 | 0.00 | — | — |
| reranked | 0.650 | 0.800 | 0.850 | 0.850 | 0.729 | 0.667 | 0.050 | 494 | 1.00 | — | — |
| adaptive | 0.600 | 0.800 | 0.850 | 0.850 | 0.704 | 0.667 | 0.050 | 157 | 0.77 | — | — |
| crag | 0.650 | 0.700 | 0.700 | 0.700 | 0.675 | 0.667 | 0.150 | 767 | 1.77 | — | — |

Refusal: `refusal_acc` = correct refusals on not-in-corpus items; `false_refusal` = wrong refusals on answerable items. Non-CRAG arms refuse when top-1 dense cosine < 0.35 (uncalibrated default, not tuned on the golden set).

#### Paired significance vs baseline

| arm vs baseline | Δrecall@5 (won/lost) | McNemar p | ΔMRR [95% CI] | significant? |
| --- | --- | --- | --- | --- |
| sparse | +1/−3 | 0.625 | +0.029 [-0.194, +0.242] | no |
| hybrid | +1/−1 | 1.000 | +0.034 [-0.121, +0.196] | no |
| reranked | +1/−0 | 1.000 | +0.083 [-0.061, +0.225] | no |
| adaptive | +1/−0 | 1.000 | +0.058 [-0.079, +0.194] | no |
| crag | +1/−3 | 0.625 | +0.029 [-0.180, +0.233] | no |

At this N, treat non-significant deltas as directional only.

_Generation columns pending ANTHROPIC_API_KEY (`--generate`)._

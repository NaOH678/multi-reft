# Composable LoReFT Policy Follow-Up Plan

## 1. Background

Recent `toxicity` residual-normalization results show a failure mode that is highly similar to `ethics`:

- low-temperature normalized routing can achieve a better benchmark score
- but the gain often comes from renewed `truthful` dominance
- generated text falls into truth-style answer templates such as:
  - `the correct answer is ...`
  - `that's not true`
  - `it is true that ...`
- after temperature is increased, late-layer `truthful` dominance is reduced
- but generation may shift into another off-task mode:
  - `the prompt is incomplete`
  - `the input is offensive`

So the current question is no longer only:

- can normalization improve score?

It is now:

- can we find a better token-local arbitration rule that avoids both:
  - low-temperature truth domination
  - high-temperature near-uniform / meta-response degeneration

## 2. Immediate Follow-Up Direction

Before moving to second-stage router training, try two additional no-training policy families:

1. `compat_filtered_topk`
2. `intervention_softmax`

For both directions, keep the same core constraints:

- token-local only
- no prompt-fixed routing
- no sample-level pooling
- no future-token leakage
- stay in `compose_domain=output`

Do **not** reopen `shared_latent / projected_output` in this round.

## 3. Why These Two Policies

### 3.1 `compat_filtered_topk`

Current softmax-based policies have a clear tradeoff:

- low temperature is too sharp and tends to collapse into `truthful`
- high temperature is too flat and tends to become weak or meta/off-task

`compat_filtered_topk` is meant to sit between these two extremes:

- not full soft mixture over all specialists
- not hard single-expert routing
- first keep only a small candidate set with strong scores
- then remove mutually incompatible specialists
- finally renormalize only over the kept set

This may preserve useful `toxicity / stereotype` participation in key layers without letting `truthful` monopolize late layers.

### 3.2 `intervention_softmax`

Current routing mainly uses:

- `delta_norm = ||delta_t(h)||`

But `delta_norm` is a low-rank residual-space quantity. It may not reflect the true magnitude of the actual rewrite written back into representation space.

`intervention_softmax` instead routes by:

- `intervention_norm = ||Delta_t(h)||`

where:

- `Delta_t(h) = R_t^T delta_t(h)`

This may be a better token-local proxy for "which specialist is actually exerting the strongest output-space intervention."

## 4. Design Principle

The cleanest way to proceed is to separate:

- policy rule
- score source
- score normalizer

Instead of baking normalization into the policy name, use a shared abstraction:

- policy chooses how to select and combine experts
- score source chooses what scalar score each expert gets
- score normalizer chooses whether the raw score should be calibrated

### 4.1 Suggested Internal Factorization

- `policy_type`
  - `compat_filtered_topk`
  - `intervention_softmax`
- `score_source`
  - `delta_norm`
  - `intervention_norm`
- `score_normalizer`
  - `none`
  - `mean_ratio`
  - `log_zscore`

Recommended mapping:

- `compat_filtered_topk`
  - default score source: `delta_norm`
- `intervention_softmax`
  - default score source: `intervention_norm`

## 5. Concrete Implementation Plan

## 5.1 `compat_filtered_topk`

### Goal

Support three variants under one policy:

- `compat_filtered_topk + none`
- `compat_filtered_topk + mean_ratio`
- `compat_filtered_topk + log_zscore`

### Current State

In the current implementation, `compat_filtered_topk`:

- uses raw `delta_norm`
- takes top-k
- filters by pairwise cosine compatibility
- runs masked softmax over the remaining experts

### Planned Change

Use one unified effective score:

- `effective_scores = raw_scores` when `score_normalizer = none`
- `effective_scores = Normalize(raw_scores)` otherwise

Then all three stages should use the same `effective_scores`:

1. top-k selection
2. compatibility filtering
3. masked softmax reweighting

This avoids an inconsistent regime such as:

- raw score chooses candidates
- normalized score chooses weights

That inconsistency would make the policy harder to interpret and harder to debug.

### Recommended Debug Outputs

Keep or verify these fields:

- `latest_scores`
- `latest_normalized_score`
- `latest_selected_mask`
- `latest_pairwise_cos`

These are necessary for later toxicity / ethics mechanism analysis.

## 5.2 `intervention_softmax`

### Goal

Support:

- `intervention_softmax + none`
- `intervention_softmax + mean_ratio`
- `intervention_softmax + log_zscore`

### Current State

Current `intervention_softmax` uses:

- raw `intervention_norm`

and does not use the residual-normalization framework.

### Planned Change

Route by:

- `raw_scores = intervention_norm`

then optionally normalize before softmax.

### Important Note

`intervention_softmax + normalized` should **not** reuse the existing residual stats.

Reason:

- current stats are calibration statistics for `delta_norm`
- `intervention_norm = ||Delta||` has a different distribution and meaning
- reusing residual stats here would mix two different score spaces

So:

- `intervention_softmax + none` can be tried immediately
- normalized `intervention_softmax` should only be run after collecting dedicated intervention-norm stats

## 5.3 Score Normalization Generalization

### Recommended Refactor

Generalize the current residual-only normalization helper into a generic score normalizer:

- input:
  - raw score tensor `[B, S, T]`
  - score stats for the chosen source
  - normalization mode
- output:
  - normalized score tensor `[B, S, T]`

This helper should work for both:

- `delta_norm`
- `intervention_norm`

### Stats Semantics

The stats file should explicitly indicate its score source:

- `score_source = delta_norm`
- or `score_source = intervention_norm`

This avoids accidental misuse.

## 6. Calibration / Stats Plan

## 6.1 For `compat_filtered_topk`

No new stats type is required.

It can reuse the current residual stats infrastructure:

- `mean_ratio`
- `log_zscore`

because it still scores experts using `delta_norm`.

## 6.2 For `intervention_softmax`

Need a new calibration pass that mirrors the residual-stats workflow, but on:

- `intervention_norm = ||Delta||`

Suggested collected fields per layer and per specialist:

- `count`
- `mean`
- `std`
- `log_mean`
- `log_std`

The collection protocol can remain the same:

- `shared`
- `specialist`

So the new stats types would be:

- shared intervention stats
- specialist intervention stats

## 7. Experiment Priority

Do **not** launch the full cross-product immediately.

Start with the smallest useful matrix.

### Phase A: Lowest-Cost Validation

Run only:

1. `compat_filtered_topk + none`
2. `compat_filtered_topk + mean_ratio`
3. `compat_filtered_topk + log_zscore`
4. `intervention_softmax + none`

Apply first on:

- `toxicity`
- `ethics`

This phase can answer whether either policy family has signal at all.

### Phase B: Only If There Is Signal

If `intervention_softmax + none` shows non-trivial promise, then add:

1. `intervention_softmax + mean_ratio`
2. `intervention_softmax + log_zscore`

with dedicated intervention stats.

## 8. Evaluation Standard

For these follow-up policies, benchmark score alone is not enough.

Each run should be judged using all three evidence levels:

### 8.1 Summary

- `overall_score`
- per-task toxicity metrics

### 8.2 Debug Summary

At minimum inspect:

- average alpha
- dominant counts
- layers `28/29/30`
- whether `truthful` still dominates the late layers
- selected-mask behavior for `compat_filtered_topk`

### 8.3 CSV Generation Content

Explicitly check whether generation is still off-task.

Key patterns to flag:

- truth-style QA:
  - `the correct answer is ...`
- truth-judgment style:
  - `that's not true`
  - `it is true that ...`
- meta-response style:
  - `the prompt is incomplete`
  - `the input is offensive`

The actual goal is not just "lower toxicity", but:

- lower toxicity **without** collapsing into answer-classification or meta-commentary behavior

## 9. Expected Outcomes

Current expectation:

- `compat_filtered_topk + normalized`
  - has the best chance among remaining no-training policies
  - because it may prevent full truthful collapse without going fully flat
- `intervention_softmax + none`
  - is worth trying immediately
  - but may still inherit truth-heavy scale behavior
- normalized `intervention_softmax`
  - is conceptually cleaner
  - but only after dedicated intervention stats are built

## 10. Stop Condition

If both `toxicity` and `ethics` still show the same pattern after this round:

- low-sharpness policies collapse into `truthful`
- high-flatness policies reduce dominance but remain off-task

then the no-training route should be considered close to exhausted.

At that point the next step should be:

- phase-2 router training

instead of further heuristic expansion.

# Composable LoReFT Residual Normalization Plan

## 1. Background

Current `residual_softmax` / `topk_residual` composition uses raw residual magnitude:

$$
s_{l,t}(h)=\|\delta_{l,t}(h)\|_2
$$

Then routing weights are computed by:

$$
\alpha_{l,t}(h)=\mathrm{softmax}(s_{l,t}(h)/\tau)
$$

where:

$$
\delta_{l,t}(h)=W_{l,t} h + b_{l,t} - R_{l,t} h
$$

Across `truth / ethics / bias / toxicity`, current experiments show the same failure mode:

- `truthful_4` has much larger raw residual norm scale than other specialists.
- `residual_softmax` therefore routes most layers to `truthful_4`.
- `topk_residual(k=1)` amplifies this and becomes almost pure `truthful_4`.
- In `ethics / bias / toxicity`, this causes severe task mismatch.
- In `toxicity`, this also changes the output surface form into truth-style QA templates such as `the correct answer is true/false`.

So the immediate problem is not temperature alone. The main problem is that raw residual norms are not directly comparable across specialists.

## 2. Goal

Introduce a **residual normalization** layer before softmax / top-k routing, so that routing depends on how unusual a residual is **relative to that specialist's own scale**, not on absolute magnitude alone.

We keep the current constraints:

- token-local intervention only
- no prompt-fixed pooling
- no task-ID anchor
- no second-stage training in the first phase

## 3. Main Idea

Replace raw score

$$
s_{l,t}(h)=\|\delta_{l,t}(h)\|_2
$$

with normalized score

$$
\tilde{s}_{l,t}(h)=\mathrm{Normalize}\left(\|\delta_{l,t}(h)\|_2;\ \mathrm{stats}_{l,t}\right)
$$

and then route using

$$
\alpha_{l,t}(h)=\mathrm{softmax}\left(\tilde{s}_{l,t}(h)/\tau\right)
$$

or top-k over `tilde{s}`.

The normalization statistics are computed per:

- layer `l`
- specialist `t`

First phase does **not** condition stats on dataset / prompt type / token position.

## 4. First-Phase Methods

### 4.1 `residual_scaled_softmax`

Use mean-ratio normalization:

$$
\tilde{s}_{l,t}(h)=\frac{\|\delta_{l,t}(h)\|_2}{\mu_{l,t}+\epsilon}
$$

where `\mu_{l,t}` is the average residual norm for layer `l`, specialist `t` on a calibration corpus.

Then:

$$
\alpha_{l,t}(h)=\mathrm{softmax}(\tilde{s}_{l,t}(h)/\tau)
$$

This is the simplest correction for specialist-wise scale mismatch.

### 4.2 `residual_logz_softmax`

Use log + z-score normalization:

$$
r_{l,t}(h)=\log(\|\delta_{l,t}(h)\|_2+\epsilon)
$$

$$
\tilde{s}_{l,t}(h)=\frac{r_{l,t}(h)-\mu^{\log}_{l,t}}{\sigma^{\log}_{l,t}+\epsilon}
$$

Then:

$$
\alpha_{l,t}(h)=\mathrm{softmax}(\tilde{s}_{l,t}(h)/\tau)
$$

This is expected to be more stable than direct ratio when residual norms are heavy-tailed.

### 4.3 Optional follow-up

Not in the first implementation round:

- `topk_residual_scaled`
- `topk_residual_logz`
- `compat_filtered_topk` on normalized scores
- robust median / MAD normalization

## 5. Calibration Statistics

We need a calibration pass that collects residual norm statistics without using task labels.

### 5.1 What to collect

For each layer `l` and specialist `t`, collect:

- `count`
- `mean`
- `std`
- `log_mean`
- `log_std`

Optional later:

- `median`
- `mad`

### 5.2 Suggested JSON format

```json
{
  "version": 1,
  "normalizer_scope": "layer_specialist",
  "specialist_labels": [
    "Llama3-8b-Loreft_truthful_4-checkpoint-3330",
    "Llama3-8b-Loreft_moral-checkpoint-330",
    "Llama3-8b-Loreft_stereotype_1-checkpoint-160",
    "Llama3-8b-Loreft_toxicity-checkpoint-235"
  ],
  "layers": {
    "0": {
      "Llama3-8b-Loreft_truthful_4-checkpoint-3330": {
        "count": 123456,
        "mean": 1.23,
        "std": 0.45,
        "log_mean": 0.08,
        "log_std": 0.31
      }
    }
  }
}
```

### 5.3 Calibration corpus

Current implementation supports two calibration modes:

### 5.3.1 `shared`

Use one shared prompt pool for all specialists.

For each prompt hidden state `h`, every specialist computes:

$$
\delta_{l,t}(h)=W_{l,t}h+b_{l,t}-R_{l,t}h
$$

and updates its own `mean/std/log_mean/log_std`.

This answers:

- under one common input distribution, how large is each specialist residual scale?

Use this mode when:

- you want one common routing scale
- you want to compare methods under a neutral shared prompt pool

### 5.3.2 `specialist`

Each specialist uses its own prompt pool:

- truthful stats use truthful prompts
- moral stats use ethics prompts
- stereotype stats use bias prompts
- toxicity stats use toxicity prompts

This answers:

- on its own domain, what is the typical residual scale of each specialist?

Use this mode when:

- you want to reduce cross-domain contamination in the calibration corpus
- you want to test whether own-domain normalization is better than shared-pool normalization

### 5.3.3 Recommendation

Both are worth trying:

- `shared`: cleaner as a common-scale baseline
- `specialist`: closer to per-specialist native scale calibration

Neither should use benchmark test prompts in the final report.

## 6. Code Changes

### 6.1 New calibration script

The current script is:

- `multi_train/eval_common/build_residual_stats.py`

It now supports:

- `--calibration_mode shared`
- `--calibration_mode specialist`

For `shared`, pass:

- `--prompt_files ...`

For `specialist`, pass:

- `--specialist_prompt_map_json ...`

where the JSON maps each specialist to its own prompt-file list.

### 6.2 Update `ComposableLoreftIntervention`

Modify:

- `pyreft/pyreft/interventions.py`

Add new init args:

- `score_normalizer`
- `score_stats`
- `score_eps`
- `score_clip`

Add helper functions:

- `_normalize_delta_scores(...)`
- `_policy_residual_scaled_softmax(...)`
- `_policy_residual_logz_softmax(...)`

Expected behavior:

- compute `delta_norm` as before
- transform to normalized score using per-layer stats
- use normalized score for softmax
- cache normalized score into debug outputs

### 6.3 Update composable config builder

Modify:

- `multi_train/eval_common/composable_loreft.py`

Add config arguments:

- `score_normalizer`
- `score_stats_path`
- `score_eps`
- `score_clip`

Responsibilities:

- load residual stats json once
- pass per-layer stats into each `ComposableLoreftIntervention`
- add method/normalizer details into model tag to avoid output collisions

### 6.4 Update eval entrypoints

Modify evaluation CLIs so all tasks can pass the new routing config:

- `multi_train/eval_truth/evaluate_truth.py`
- `multi_train/eval_ethics/machine_ethics_exp.py`
- `multi_train/eval_bias/stereotype_exp.py`
- `multi_train/eval_toxicity/toxicity_exp.py`

New CLI flags:

- `--score_normalizer`
- `--score_stats_path`
- `--score_eps`
- `--score_clip`

## 7. Debug Output Changes

Current debug summary already records:

- `alpha`
- `delta_norm`
- `intervention_norm`

For normalized routing we should also record:

- `normalized_score_mean`
- `normalized_score_std`

This is needed to verify:

- whether `truthful_4` is still dominating after normalization
- whether non-truth tasks start activating their own specialists

## 8. Validation Criteria

Residual normalization is considered useful if it improves at least one of the following:

1. Routing balance improves

- `truthful_4` no longer dominates `31/32` layers on `ethics / bias / toxicity`
- `moral / stereotype / toxicity` begin to appear in `dominant_counts`

2. Output style improves

- `toxicity` no longer collapses into truth-style outputs like `the correct answer is true/false`

3. Metrics improve over raw residual routing

- `ethics`: better than `residual_softmax(temp=1)`
- `bias`: better than current residual variants
- `toxicity`: better than current residual variants

The expected first milestone is **not necessarily to beat `single`**.
The first milestone is to show that normalized routing is less pathological than raw residual routing.

## 9. Experiment Order

After implementation, run:

1. `truth` with
   - `residual_scaled_softmax`
   - `residual_logz_softmax`
2. `ethics` with
   - `residual_scaled_softmax`
   - `residual_logz_softmax`
3. `bias` with
   - `residual_scaled_softmax`
   - `residual_logz_softmax`
4. `toxicity` with
   - `residual_scaled_softmax`
   - `residual_logz_softmax`

Priority observations:

- `mean_alpha`
- `dominant_counts`
- generated text patterns
- final benchmark score

## 10. Current Scope Boundary

This document only defines the first no-training residual normalization step.

It does **not** yet include:

- task-ID anchors
- learned arbitration
- shared-latent-specific normalization
- compatibility-aware normalized routing
- second-stage training

Those can be added later if normalized residual routing shows a useful signal.

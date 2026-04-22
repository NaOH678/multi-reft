# Composable LoReFT Residual-Stats Results

## 1. Scope

This note records the first round of residual-normalized composition experiments using:

- shared calibration stats
- specialist-specific calibration stats
- `residual_scaled_softmax`
- `residual_logz_softmax`

Benchmarks covered here:

- `truthfulqa_mc`
- `bbq`
- `ethics`

## 2. Stats Files

Calibration stats used:

- shared:
  [residual_stats_shared_train_input.json](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/calibration/train_input/stats/residual_stats_shared_train_input.json)
- specialist:
  [residual_stats_specialist_train_input.json](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/calibration/train_input/stats/residual_stats_specialist_train_input.json)

## 3. TruthfulQA MC

Relevant outputs:

- single truthful:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-single-idx0-Llama3_8b_Loreft_truthful_4_checkpoint_3330+Llama3_8b_Loreft_moral_checkpoint_330+Llama3_8b_Loreft_stereotype_1_checkpoint_160+Llama3_8b_Loreft_toxicity_checkpoint_235-truthfulqa_mc_summary.json)
- shared scaled:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_scaled_softmax-temp1-stats_sh_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-truthfulqa_mc_summary.json)
  [debug](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_scaled_softmax-temp1-stats_sh_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-truthfulqa_mc_debug_summary.json)
- shared logz:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_logz_softmax-temp1-stats_sh_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-truthfulqa_mc_summary.json)
- specialist scaled:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_scaled_softmax-temp1-stats_sp_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-truthfulqa_mc_summary.json)
- specialist logz:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_logz_softmax-temp1-stats_sp_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-truthfulqa_mc_summary.json)

Results:

- `single truthful`: `0.494492`
- `shared + scaled`: `0.608323`
- `shared + logz`: `0.536108`
- `specialist + scaled`: `0.602203`
- `specialist + logz`: `0.545900`

Best result:

- `shared + residual_scaled_softmax`

Main observations:

- Normalized routing clearly beats `single truthful`.
- `scaled` is much better than `logz` on `truthfulqa_mc`.
- `shared` and `specialist` calibration are very close here.

Debug summary observations:

- Even after normalization, `truthful` remains the dominant specialist on average.
- For `shared + scaled`, average alpha is roughly:
  - `truth 0.4149`
  - `moral 0.1315`
  - `bias 0.2105`
  - `tox 0.2432`
- In late layers, `truthful` becomes nearly dominant:
  - layer 29: `truth 0.9976`
  - layer 30: `truth 1.0`

Interpretation:

- This benchmark benefits from a truth-heavy routing pattern.
- Residual normalization helps mainly by softening raw scale mismatch, not by changing the dominant specialist identity.

## 4. BBQ

Relevant outputs:

- single bias:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-single-idx2-Llama3_8b_Loreft_truthful_4_checkpoint_3330+Llama3_8b_Loreft_moral_checkpoint_330+Llama3_8b_Loreft_stereotype_1_checkpoint_160+Llama3_8b_Loreft_toxicity_checkpoint_235-bbq_summary.json)
- shared scaled:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_scaled_softmax-temp1-stats_sh_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-bbq_summary.json)
  [debug](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_scaled_softmax-temp1-stats_sh_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-bbq_debug_summary.json)
- shared logz:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_logz_softmax-temp1-stats_sh_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-bbq_summary.json)
  [debug](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_logz_softmax-temp1-stats_sh_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-bbq_debug_summary.json)
- specialist scaled:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_scaled_softmax-temp1-stats_sp_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-bbq_summary.json)
- specialist logz:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_logz_softmax-temp1-stats_sp_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-bbq_summary.json)

Results:

- `single bias`: `0.555136`
- `shared + scaled`: `0.577139`
- `shared + logz`: `0.616922`
- `specialist + scaled`: `0.576130`
- `specialist + logz`: `0.616631`

Best result:

- `shared + residual_logz_softmax`

Main observations:

- Both `logz` runs are much stronger than both `scaled` runs.
- `shared` and `specialist` calibration are almost identical.
- Both best normalized runs beat `single bias` by a clear margin.

Debug summary observations:

- `truthful` still has the highest average alpha.
- For `shared + logz`, average alpha is roughly:
  - `truth 0.4013`
  - `moral 0.1561`
  - `bias 0.2060`
  - `tox 0.2366`
- Late layers are still truth-heavy:
  - layer 29: `truth 0.6472`
  - layer 30: `truth 0.6835`

Interpretation:

- Improvement does not come from routing mostly to the bias specialist.
- Instead, `logz` appears to stabilize routing enough to improve overall BBQ behavior while keeping a mixed intervention pattern.

## 5. Ethics

Relevant outputs:

- single moral:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-single-idx1-Llama3_8b_Loreft_truthful_4_checkpoint_3330+Llama3_8b_Loreft_moral_checkpoint_330+Llama3_8b_Loreft_stereotype_1_checkpoint_160+Llama3_8b_Loreft_toxicity_checkpoint_235-ethics_summary.json)
  [debug](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-single-idx1-Llama3_8b_Loreft_truthful_4_checkpoint_3330+Llama3_8b_Loreft_moral_checkpoint_330+Llama3_8b_Loreft_stereotype_1_checkpoint_160+Llama3_8b_Loreft_toxicity_checkpoint_235-ethics_debug_summary.json)
- shared scaled:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-residual_scaled_softmax-temp1-stats_sh_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-ethics_summary.json)
- shared logz:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-residual_logz_softmax-temp1-stats_sh_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-ethics_summary.json)
- specialist scaled:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-residual_scaled_softmax-temp1-stats_sp_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-ethics_summary.json)
- specialist logz:
  [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-residual_logz_softmax-temp1-stats_sp_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-ethics_summary.json)
  [debug](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-residual_logz_softmax-temp1-stats_sp_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-ethics_debug_summary.json)

Results:

- `single moral`: `0.913387`
- `shared + scaled`: `0.858622`
- `shared + logz`: `0.862257`
- `specialist + scaled`: `0.857120`
- `specialist + logz`: `0.867157`

Best result among normalized variants:

- `specialist + residual_logz_softmax`

Main observations:

- All normalized composition variants are still worse than `single moral`.
- `logz` is consistently better than `scaled`.
- `specialist` calibration is slightly better than `shared`, but only by a small margin.

Debug summary observations:

- Even the best normalized variant does not make `moral` dominant.
- For `specialist + logz`, average alpha is roughly:
  - `truth 0.3653`
  - `moral 0.1766`
  - `bias 0.2129`
  - `tox 0.2452`
- In late layers, `truthful` still dominates:
  - layer 29: `truth 0.5992`, `moral 0.0114`
  - layer 30: `truth 0.6492`, `moral 0.0287`
- By contrast, `single moral` is exactly:
  - `moral 1.0` on all layers

Interpretation:

- Residual normalization helps reduce the raw-scale problem, but it does not solve the core routing mismatch on ethics.
- The system still routes too much mass to `truthful`, especially in the high layers where behavior matters most.

## 5.1 Ethics Temperature Sweep

After the first residual-normalized round, an additional temperature sweep was run for:

- `shared + residual_scaled_softmax`
- `specialist + residual_scaled_softmax`
- `shared + residual_logz_softmax`
- `specialist + residual_logz_softmax`

Main results:

- `shared + residual_scaled_softmax`
  - best at `temp=64`: `0.881935`
- `specialist + residual_scaled_softmax`
  - best at `temp=64`: `0.879485`
- `shared + residual_logz_softmax`
  - best at `temp=64`: `0.888494`
- `specialist + residual_logz_softmax`
  - best at `temp=32`: `0.890153`

Compared with the corresponding `temp=1` runs:

- `shared_scaled`: `0.858622 -> 0.881935` (`+0.0233`)
- `special_scaled`: `0.857120 -> 0.879485` (`+0.0224`)
- `shared_logz`: `0.862257 -> 0.888494` (`+0.0262`)
- `special_logz`: `0.867157 -> 0.890153` (`+0.0230`)

However, even the best configuration remains below:

- `single moral = 0.913387`

So temperature helps substantially, but still does not close the gap to the single-task specialist on `ethics`.

Mechanism summary from debug outputs:

- The gain from higher temperature mainly comes from flattening extreme routing.
- It does **not** make `moral` become the dominant specialist.
- At `temp=1`, late layers are still strongly truth-dominated.
- At `temp=32/64`, late-layer alphas become much closer to uniform.

For example:

- `specialist + logz + temp=1`
  - layer 29/30:
    - `truth 0.5992 / 0.6492`
    - `moral 0.0114 / 0.0287`
- `specialist + logz + temp=32`
  - layer 29/30:
    - `truth 0.2575 / 0.2572`
    - `moral 0.2345 / 0.2385`

Interpretation:

- Higher temperature improves `ethics` mainly by reducing `truthful` over-dominance.
- This is a “less wrong routing” effect, not a genuinely ethics-aware routing effect.
- `logz` remains more stable than `scaled`.
- `scaled` shows a clear U-shaped temperature curve on `ethics`.

Prompt-type breakdown also shows that the remaining gap is not just random noise.
For the best composable setting, the main drops relative to `single moral` are still on prompt types `2` and `5`.

Detailed write-up:

- [COMPOSABLE_LOREFT_ETHICS_TEMP_ANALYSIS.md](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/COMPOSABLE_LOREFT_ETHICS_TEMP_ANALYSIS.md)

## 6. Cross-Task Summary

### 6.1 Shared vs Specialist Stats

Across all three benchmarks:

- `shared` and `specialist` calibration lead to very similar results
- the choice of calibration corpus is currently a second-order effect

So the main bottleneck is not the difference between shared-pool stats and specialist-specific stats.

### 6.2 Scaled vs Logz

The main effect comes from the normalization rule:

- `truthfulqa_mc`: `scaled` is clearly better
- `bbq`: `logz` is clearly better
- `ethics`: `logz` is clearly better

### 6.3 Routing Pattern

A consistent pattern remains:

- `truthful` still receives the highest average alpha
- late layers are often even more truth-heavy

This means current normalization does **not** fundamentally change the dominant routing structure.
It mostly rescales the competition, but does not make the task-matched specialist reliably win.

## 7. Practical Conclusion

Current residual-normalized routing is useful, but only partially:

- It is strong for `truthfulqa_mc`
- It is surprisingly strong for `bbq`
- It is still clearly insufficient for `ethics`

So the current conclusion is:

- residual normalization improves over raw residual routing
- but it does not fix specialist selection in a task-faithful way
- the remaining failure mode is still truth-dominant routing in later layers

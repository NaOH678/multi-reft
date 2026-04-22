# Composable LoReFT Residual Stats Usage

## 1. Purpose

This document explains how to build residual normalization statistics for composable LoReFT using the two calibration modes now supported:

- `shared`
- `specialist`

The two entry scripts are:

- [build_residual_stats_shared_train.sh](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/script/build_residual_stats_shared_train.sh)
- [build_residual_stats_specialist_train.sh](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/script/build_residual_stats_specialist_train.sh)

The helper export script is:

- [export_prompt_field.py](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_common/export_prompt_field.py)

## 2. Data Source

Both scripts use the four specialist training datasets from [train.py](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/train.py):

- `dataset/alignment_truthful_format`
- `dataset/alignment_moral_cls`
- `dataset/alignment_stereotype_format`
- `dataset/alignment_toxic_format`

The calibration prompt is now exported with the same prompt construction used in training:

- if `input == ""`: use the supervised `instruction-only` template
- otherwise: use the supervised `instruction + input` template

So calibration now matches the actual prompt format seen by `ReftSupervisedDataset`.
Only prompt text is exported and used for calibration.
No labels, outputs, or answers are used.

## 3. What Each Script Does

### 3.1 Shared Script

Script:

- [build_residual_stats_shared_train.sh](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/script/build_residual_stats_shared_train.sh)

Behavior:

1. Export training-format prompt-only text from each of the four training datasets into `jsonl` files.
2. Merge these prompt files into one shared prompt pool.
3. Run `build_residual_stats.py` with `--calibration_mode shared`.
4. For every prompt hidden state, all four specialists compute residual stats on the same shared prompt pool.

Meaning:

- one common input distribution
- all specialists are calibrated on the same prompt set

This is the common-scale version.

### 3.2 Specialist Script

Script:

- [build_residual_stats_specialist_train.sh](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/script/build_residual_stats_specialist_train.sh)

Behavior:

1. Export training-format prompt-only text from each of the four training datasets into `jsonl` files.
2. Build a prompt-map JSON that assigns one prompt file to each specialist.
3. Run `build_residual_stats.py` with `--calibration_mode specialist`.
4. Each specialist only computes residual stats on its own prompt pool.

Meaning:

- truthful specialist uses truthful prompts
- moral specialist uses moral prompts
- stereotype specialist uses stereotype prompts
- toxicity specialist uses toxicity prompts

This is the own-domain calibration version.

## 4. Default Paths

Default base model:

- `../weightsft/models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`

Default specialist checkpoints:

- `Llama3-8b-Loreft_truthful_4/checkpoint-3330`
- `Llama3-8b-Loreft_moral/checkpoint-330`
- `Llama3-8b-Loreft_stereotype_1/checkpoint-160`
- `Llama3-8b-Loreft_toxicity/checkpoint-235`

Default exported prompt directory:

- `multi_train/calibration/train_input/prompt_files`

Default stats output directory:

- `multi_train/calibration/train_input/stats`

Default output files:

- shared: `multi_train/calibration/train_input/stats/residual_stats_shared_train_input.json`
- specialist: `multi_train/calibration/train_input/stats/residual_stats_specialist_train_input.json`

## 5. Important Environment Variables

Both scripts support:

- `PYTHON_BIN`
- `BASE_MODEL`
- `DEVICE`
- `BATCH_SIZE`
- `MAX_LENGTH`
- `POSITIONS`
- `TARGET_LAYERS`
- `TEXT_COLUMN`
- `MAX_SAMPLES`

They also support overriding dataset paths:

- `TRUTH_DATASET`
- `MORAL_DATASET`
- `BIAS_DATASET`
- `TOXICITY_DATASET`

And specialist checkpoint paths:

- `SPEC1`
- `SPEC2`
- `SPEC3`
- `SPEC4`

## 6. How `MAX_SAMPLES` Works

Current behavior in the two shell scripts:

- `MAX_SAMPLES` is applied only at prompt export time
- it limits how many prompts are exported per dataset
- it is not passed again into `build_residual_stats.py`

For example:

```bash
MAX_SAMPLES=5000 bash multi_train/script/build_residual_stats_shared_train.sh
```

means:

- truthful export up to 5000 training-format prompts
- moral export up to 5000 training-format prompts
- stereotype export up to 5000 training-format prompts
- toxicity export up to 5000 training-format prompts

Then the four exported prompt files are used as the calibration corpus.

For `shared`:

- all exported prompt files are merged
- no second truncation is applied inside the stats builder

For `specialist`:

- each specialist reads its own exported prompt file
- no second truncation is applied inside the stats builder

So `MAX_SAMPLES=5000` now consistently means:

- export up to 5000 prompts per dataset

## 7. Example Commands

### 7.1 Build Shared Stats with Full Training Inputs

```bash
bash multi_train/script/build_residual_stats_shared_train.sh
```

### 7.2 Build Specialist Stats with Full Training Inputs

```bash
bash multi_train/script/build_residual_stats_specialist_train.sh
```

### 7.3 Build Shared Stats on a Specific GPU

```bash
DEVICE=cuda:3 BATCH_SIZE=256 bash multi_train/script/build_residual_stats_shared_train.sh
```

### 7.4 Build Specialist Stats on a Specific GPU

```bash
DEVICE=cuda:4 BATCH_SIZE=256 bash multi_train/script/build_residual_stats_specialist_train.sh
```

### 7.5 Build a Small Debug Version

```bash
DEVICE=cuda:2 BATCH_SIZE=256 MAX_SAMPLES=2000 bash multi_train/script/build_residual_stats_shared_train.sh
```

This exports up to 2000 training-format prompts from each dataset and then builds the shared stats file.

### 7.6 Restrict to Specific Layers

```bash
DEVICE=cuda:1 TARGET_LAYERS="20 21 22 23 24 25 26 27 28 29 30 31" \
bash multi_train/script/build_residual_stats_specialist_train.sh
```

### 7.7 Change the Output File

```bash
OUTPUT_JSON=multi_train/calibration/train_input/stats/residual_stats_shared_train_input_debug.json \
DEVICE=cuda:0 MAX_SAMPLES=1000 \
bash multi_train/script/build_residual_stats_shared_train.sh
```

## 8. Produced Intermediate Files

The scripts first export prompt-only files such as:

- `truthful_train_input.jsonl`
- `moral_train_input.jsonl`
- `stereotype_train_input.jsonl`
- `toxicity_train_input.jsonl`

The specialist script also creates:

- `specialist_prompt_map_train_input.json`

These files are useful for debugging and for reusing the same calibration prompt pool later.

## 9. Output JSON Meaning

The output residual stats JSON contains:

- `calibration_mode`
- `prompt_sources`
- `specialist_labels`
- per-layer and per-specialist:
  - `count`
  - `mean`
  - `std`
  - `log_mean`
  - `log_std`

`calibration_mode` tells you whether the file was built with:

- `shared`
- `specialist`

`prompt_sources` records how many prompts were used.

## 10. Recommended Usage

For current experiments, a practical order is:

1. build one `shared` stats file
2. build one `specialist` stats file
3. run `residual_scaled_softmax` with both
4. run `residual_logz_softmax` with both
5. compare whether shared-pool normalization or specialist-specific normalization is better

This gives two orthogonal comparisons:

- common-pool vs own-domain calibration
- mean-ratio vs log-zscore normalization

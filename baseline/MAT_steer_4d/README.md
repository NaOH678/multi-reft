# MAT-Steer 4D Baseline

This directory contains a separate four-task MAT-Steer implementation for:

- `truthfulqa`
- `bbq`
- `ethics`
- `toxicity`

It does not modify the original `baseline/MAT-Steer` directory.

## Pipeline

1. Prepare four-task MAT data:

```bash
python -m baseline.MAT_steer_4d.prepare_data
```

2. Extract activations for each task:

```bash
python -m baseline.MAT_steer_4d.extract_activations --base_model models/llama3-8b --task_name truthfulqa --layer 14
python -m baseline.MAT_steer_4d.extract_activations --base_model models/llama3-8b --task_name bbq --layer 14
python -m baseline.MAT_steer_4d.extract_activations --base_model models/llama3-8b --task_name ethics --layer 14
python -m baseline.MAT_steer_4d.extract_activations --base_model models/llama3-8b --task_name toxicity --layer 14
```

3. Train MAT-Steer:

```bash
python -m baseline.MAT_steer_4d.train --base_model models/llama3-8b --layer 14 --batch_size 96 --epochs 100
```

4. Evaluate:

```bash
python -m baseline.MAT_steer_4d.eval_truth_bias --base_model models/llama3-8b --dataset truthfulqa_mc --mat_checkpoint baseline/MAT_steer_4d/checkpoints/llama3_8b_L14_mat4d.pt
python -m baseline.MAT_steer_4d.eval_truth_bias --base_model models/llama3-8b --dataset bbq --mat_checkpoint baseline/MAT_steer_4d/checkpoints/llama3_8b_L14_mat4d.pt
python -m baseline.MAT_steer_4d.eval_ethics --base_model models/llama3-8b --mat_checkpoint baseline/MAT_steer_4d/checkpoints/llama3_8b_L14_mat4d.pt
python -m baseline.MAT_steer_4d.eval_toxicity --base_model models/llama3-8b --mat_checkpoint baseline/MAT_steer_4d/checkpoints/llama3_8b_L14_mat4d.pt
```

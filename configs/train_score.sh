#!/usr/bin/env bash
set -euo pipefail

source /data/250010061/miniconda3/etc/profile.d/conda.sh
conda activate /data/250010061/miniconda3/envs/motus

python "/data/250010061/Motus/causal_attn_score_sample_online/train_causal_score.py" \
  --dataset_root "/data/250010061/Motus/Dataset/robotwin_fd_dataset_all_80k" \
  --motus_ckpt "/data/250010061/Motus/checkpoints/robotwin/robotwin_20260329_050331/checkpoint_step_80000" \
  --wan_path "/data/250010061/Motus/checkpoints/pretrained_models/Wan2.2-TI2V-5B" \
  --vlm_path "/data/250010061/Motus/checkpoints/pretrained_models/Qwen3-VL-2B-Instruct" \
  --device cuda \
  --batch_size 32 \
  --steps 200000
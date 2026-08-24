## causal_attn_score

This directory contains a **completely independent** binary classification training/data loading module, separate from Motus's existing training code (it does not modify any existing files).

### Data Input (Your Current Export Format)

Positive sample root directory: e.g., `Motus/Dataset/robotwin_fd_dataset_clean10_random100_pos`  
Negative sample root directory: e.g., `Motus/Dataset/robotwin_fd_dataset_clean10_random100_neg`

Both contain:

- `manifest.jsonl` (one sample's relative `path` per line)
- `samples/.../*.pt`

Each `.pt` (sliced sample) is expected to contain:

- `video_frames_4`: `[4,3,H,W]` (GT)
- `action_sequence_17`: `[17,14]`
- `predicted_token_block`: `[120,3072]` (history)
- `predicted_token_block_future`: `[120,3072]` (future)
- `und_tokens_last`: `[Lu,512]`
- `und_attention_mask`: `[Lu]` (optional, but recommended)
- `label`: 0/1

### Loader

See `dataset_fd_binary.py`:

- `build_fd_binary_index(pos_root=..., neg_root=...)`
- `RobotWinFDBinaryDataset(index=...)`
- `collate_fd_binary(batch)`

### Training (Minimal Runnable Version)

See `train_causal_score.py`: a minimal single-node, single-GPU training loop (does not depend on Motus's original training entry point).

It will:
- Load samples from pos/neg manifests
- Use Motus's VAE encoder + `prepare_input` to encode `video_frames_4` into GT tokens
- Feed `und_tokens_last` + GT tokens + predicted history/future tokens + action tokens + CLS into a lightweight Transformer, applying attention with a **grouped causal mask**
- Use the CLS token for binary classification BCE loss

Example command (adjust paths to your actual checkpoints):

```bash
python "/data/250010061/Motus/causal_attn_score/train_causal_score.py" \
  --pos_root "/data/250010061/Motus/Dataset/robotwin_fd_dataset_clean10_random100_pos" \
  --neg_root "/data/250010061/Motus/Dataset/robotwin_fd_dataset_clean10_random100_neg" \
  --results_root "/data/250010061/Motus/causal_attn_score/results" \
  --motus_ckpt "/data/250010061/Motus/checkpoints/robotwin/robotwin_20260329_050331/checkpoint_step_40000" \
  --wan_path "/data/250010061/Motus/checkpoints/pretrained_models/Wan2.2-TI2V-5B" \
  --vlm_path "/data/250010061/Motus/checkpoints/pretrained_models/Qwen3-VL-2B-Instruct" \
  --device cuda \
  --batch_size 8 \
  --steps 2000



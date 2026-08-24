## causal_attn_score

This directory contains a **fully independent** binary‑classification training and data‑loading module that does not modify any existing Motus training code.

### Data Input (Your Current Export Format)

- Positive sample root: e.g., `Motus/Dataset/robotwin_fd_dataset_clean10_random100_pos`  
- Negative sample root: e.g., `Motus/Dataset/robotwin_fd_dataset_clean10_random100_neg`

Both directories contain:

- `manifest.jsonl` – one relative `path` per line for each sample
- `samples/.../*.pt` – the actual sample files

Each `.pt` (a sliced sample) is expected to have the following fields:

- `video_frames_4`: `[4, 3, H, W]` (ground‑truth frames)
- `action_sequence_17`: `[17, 14]`
- `predicted_token_block`: `[120, 3072]` (history tokens)
- `predicted_token_block_future`: `[120, 3072]` (future tokens)
- `und_tokens_last`: `[Lu, 512]`
- `und_attention_mask`: `[Lu]` (optional but recommended)
- `label`: 0 or 1

### Loader

See `dataset_fd_binary.py` – it provides:

- `build_fd_binary_index(pos_root=..., neg_root=...)`
- `RobotWinFDBinaryDataset(index=...)`
- `collate_fd_binary(batch)`

### Training (Minimal Runnable Version)

See `train_causal_score.py` – a minimal single‑node, single‑GPU training loop that does not depend on Motus’s original training entry point.

It performs the following steps:

- Loads samples from the positive and negative manifests.
- Uses Motus’s VAE encoder and `prepare_input` to encode `video_frames_4` into ground‑truth tokens.
- Feeds `und_tokens_last`, GT tokens, predicted history/future tokens, action tokens, and a CLS token into a lightweight Transformer, applying attention with a **grouped causal mask**.
- Uses the CLS token for binary‑classification BCE loss.

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

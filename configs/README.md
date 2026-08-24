## causal_attn_score

这里放一套**完全独立**于 Motus 现有训练代码的二分类训练/数据加载模块（不修改任何已有文件）。

### 数据输入（你现在的导出格式）

正样本根目录：例如 `Motus/Dataset/robotwin_fd_dataset_clean10_random100_pos`  
负样本根目录：例如 `Motus/Dataset/robotwin_fd_dataset_clean10_random100_neg`

两者均包含：

- `manifest.jsonl`（每行一个样本的相对路径 `path`）
- `samples/.../*.pt`

每个 `.pt`（切片后的样本）预计包含：

- `video_frames_4`: `[4,3,H,W]`（GT）
- `action_sequence_17`: `[17,14]`
- `predicted_token_block`: `[120,3072]`（历史）
- `predicted_token_block_future`: `[120,3072]`（未来）
- `und_tokens_last`: `[Lu,512]`
- `und_attention_mask`: `[Lu]`（可选，但推荐）
- `label`: 0/1

### Loader

见 `dataset_fd_binary.py`：

- `build_fd_binary_index(pos_root=..., neg_root=...)`
- `RobotWinFDBinaryDataset(index=...)`
- `collate_fd_binary(batch)`

### 训练（最小可跑通版本）

见 `train_causal_score.py`：单机单卡最小训练 loop（不依赖 Motus 原训练入口）。

它会：
- 从 pos/neg manifest 加载样本
- 使用 Motus 的 VAE encoder + `prepare_input` 把 `video_frames_4` 编成 GT tokens
- 将 `und_tokens_last` + GT tokens + 预测历史/未来 tokens + action tokens + CLS 输入轻量 Transformer，并按**分组因果 mask**做 attention
- 用 CLS 做二分类 BCE loss

示例命令（路径按你实际 checkpoint 调整）：

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
```



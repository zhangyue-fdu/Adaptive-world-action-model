#!/usr/bin/env python3
"""
Build negative (label=0) samples from an existing *positive* RobotWin FD dataset.

Input: a directory produced by build_robotwin_fd_augmented_dataset.py (pos samples, label=1),
       containing:
         - manifest.jsonl with sample paths
         - samples/.../*.pt where each pt contains:
             video_frames: [16,3,H,W] (GT targets; condition frame is separate and should be ignored)
             action_sequence: [64,14]
             predicted_video_tokens: [600,3072] (optional but required for this script)

User-specified slicing rule (per source sample) — kept for reuse:
  - ignore condition frame (do not use first_frame)
  - take 4 GT video frames (asynchronous with actions, ratio 1:4)
  - for the 4th GT video frame, take the corresponding action and the next 16 actions:
      => 17 actions total
  - sliding window with overlap on actions:
      next window starts at previous window's last action (i.e. stride 16)
  - for predicted tokens:
      tokens length is 600 = 5 * 12 * 10 = (condition + 4 predicted time slices) * spatial grid
      take two FULL blocks [120,3072]: historical (GT 4-frame slice) + future (next 16-action / next 4-frame quartet; clamped to slice 4).

Negative sample construction:
  - In this file we keep the slicing utility and then generate negatives by action corruptions
    (swap/shuffle/noise/cross-sample exchange).

Output:
  out_dir/manifest.jsonl
  out_dir/samples/{split}/{task}/{episode}/sample_{k:06d}.pt

python "/work/sme-wangr/Motus/data/robotwin2/build_robotwin_fd_negative_dataset.py" \
  --pos_dir "/work/sme-wangr/Motus/Dataset/robotwin_fd_dataset_clean10_random100" \
  --out_dir "/work/sme-wangr/Motus/Dataset/robotwin_fd_dataset_clean10_random100_neg" \
  --seed 0 \
  --max_source_samples 3
  --max_source_samples：调试用，只取前 N 个正样本（可选）
  --gripper_indices：夹爪维度索引（默认 "6,13"）
  --half_noise_sigma：action_half_noise 的噪声强度（默认 0.5）
  --enable_action_corruptions（默认 1）
  --enable_cross_sample_swap（默认 1，第二遍两两不放回配对生成）

生成的负样本类型：
action_time_swap
action_joint_swap
action_flip_gripper
action_half_noise
action_swap_between_samples（两阶段两两不放回配对）

"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch


def _ensure_motus_on_path() -> None:
    motus_root = Path(__file__).resolve().parents[2]  # .../Motus
    if str(motus_root) not in sys.path:
        sys.path.insert(0, str(motus_root))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _token_block_for_time(tokens: torch.Tensor, time_idx_overall: int) -> torch.Tensor:
    """Return the full spatial block [120, D] for a given overall time index (0..4)."""
    start = int(time_idx_overall) * 120
    end = start + 120
    return tokens[start:end]

def _has_tail_padding_by_action_indices(action_indices_window: List[int]) -> bool:
    """
    Detect strict-fill tail padding/clamp by repeated indices (typically repeats of the last frame index).
    If any consecutive indices are equal, we treat it as padded and skip.
    """
    if not action_indices_window or len(action_indices_window) < 2:
        return False
    for i in range(len(action_indices_window) - 1):
        if int(action_indices_window[i]) == int(action_indices_window[i + 1]):
            return True
    return False

def _parse_gripper_indices(s: str) -> List[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [int(p) for p in parts]

def _neg_action_time_swap(actions: torch.Tensor, rng: random.Random, num_swaps: int = 2) -> torch.Tensor:
    """Randomly swap actions between two random timesteps (keeps shape)."""
    a = actions.clone()
    T = a.shape[0]
    for _ in range(max(1, int(num_swaps))):
        i = rng.randrange(0, T)
        j = rng.randrange(0, T)
        if i == j:
            continue
        tmp = a[i].clone()
        a[i] = a[j]
        a[j] = tmp
    return a

def _neg_action_joint_swap(
    actions: torch.Tensor,
    rng: random.Random,
    num_timesteps: int = 3,
    num_swaps_per_t: int = 1,
    avoid_dims: Optional[List[int]] = None,
) -> torch.Tensor:
    """At random timesteps, swap values between two joint dims."""
    a = actions.clone()
    T, D = a.shape
    avoid = set(avoid_dims or [])
    dims = [d for d in range(D) if d not in avoid]
    if len(dims) < 2:
        return a
    for _ in range(max(1, int(num_timesteps))):
        t = rng.randrange(0, T)
        for _ in range(max(1, int(num_swaps_per_t))):
            d1, d2 = rng.sample(dims, 2)
            v = a[t, d1].clone()
            a[t, d1] = a[t, d2]
            a[t, d2] = v
    return a

def _neg_action_flip_gripper(actions: torch.Tensor, gripper_indices: List[int]) -> torch.Tensor:
    """
    Flip gripper direction by negating selected dims.
    Note: action is absolute qpos in this dataset; negation is a strong corruption.
    """
    a = actions.clone()
    for gi in gripper_indices:
        if 0 <= gi < a.shape[1]:
            a[:, gi] = -a[:, gi]
    return a

def _neg_action_half_noise(actions: torch.Tensor, rng: random.Random, sigma: float = 0.5) -> torch.Tensor:
    """Keep first half unchanged, add larger Gaussian noise to second half."""
    a = actions.clone()
    T = a.shape[0]
    half = T // 2
    noise = torch.randn_like(a[half:]) * float(sigma)
    a[half:] = a[half:] + noise
    return a


def _slice_one_source(
    d: Dict[str, Any],
    *,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Returns a list of sliced negative samples (dicts ready to torch.save).
    """
    video_frames: torch.Tensor = d["video_frames"]  # [16,3,H,W]
    action_seq: torch.Tensor = d["action_sequence"]  # [64,14]
    tokens: torch.Tensor = d["predicted_video_tokens"]  # [600,3072]
    und_tokens_last = d.get("und_tokens_last")
    und_attention_mask = d.get("und_attention_mask")

    if video_frames.dim() != 4 or video_frames.shape[0] < 16:
        raise ValueError(f"Unexpected video_frames shape: {tuple(video_frames.shape)}")
    if action_seq.dim() != 2 or action_seq.shape[0] < 64:
        raise ValueError(f"Unexpected action_sequence shape: {tuple(action_seq.shape)}")
    if tokens.dim() != 2 or tokens.shape[0] != 600:
        raise ValueError(f"Unexpected predicted_video_tokens shape: {tuple(tokens.shape)}")

    # Sliding windows over actions:
    # first window uses 16th action (1-based) => index 15 (0-based) as the 1st action of the 17-action slice
    # stride 16 so next window starts at previous last action.
    start0 = 15
    stride = 16
    win_len = 17
    max_start = action_seq.shape[0] - win_len

    out: List[Dict[str, Any]] = []

    s = start0
    while s <= max_start:
        # Determine which 4 GT video frames correspond to this action slice.
        # With ratio 1:4, 4th video frame index v4 satisfies: s == (v4+1)*4 - 1  => v4=(s+1)/4 - 1
        if (s + 1) % 4 != 0:
            # Shouldn't happen given stride/start, but guard anyway.
            s += stride
            continue
        v4 = (s + 1) // 4 - 1
        v_start = v4 - 3
        if v_start < 0 or (v_start + 4) > video_frames.shape[0]:
            s += stride
            continue

        # Only keep the 4th frame (the one aligned with the first action in this 17-action slice).
        gt_video_1 = video_frames[v_start + 3].contiguous()  # [3,H,W]
        gt_actions_17 = action_seq[s : s + win_len].contiguous()  # [17,14]

        # Historical predicted tokens: aligned to the 4 GT frames [v_start : v_start+4].
        # Block mapping by frames: [0..3]->slice1, [4..7]->slice2, [8..11]->slice3, [12..15]->slice4.
        correct_pred_slice = 1 + (v_start // 4)
        correct_pred_slice = max(1, min(4, int(correct_pred_slice)))

        token_block_correct = _token_block_for_time(tokens, correct_pred_slice).contiguous()  # [120,3072]

        # Future predicted tokens: next 16 actions / next 4-frame quartet; clamp slice to 4 at last quartet.
        future_pred_slice = min(correct_pred_slice + 1, 4)
        token_block_future_correct = _token_block_for_time(tokens, future_pred_slice).contiguous()  # [120,3072]

        # Skip if action indices indicate tail clamp/padding in this slice.
        src_action_indices = d.get("source_action_indices") or d.get("action_indices") or []
        try:
            idx_window = [int(x) for x in src_action_indices[s : s + win_len]]
        except Exception:
            idx_window = []
        if _has_tail_padding_by_action_indices(idx_window):
            s += stride
            continue

        out.append(
            {
                # sliced features
                "video_frame_1": gt_video_1,
                "action_sequence_17": gt_actions_17,
                "predicted_token_block_correct": token_block_correct,
                "predicted_token_time_slice_correct": int(correct_pred_slice),
                "correct_time_slice": int(correct_pred_slice),
                "predicted_token_block_future_correct": token_block_future_correct,
                "predicted_token_time_slice_future_correct": int(future_pred_slice),
                # understanding expert outputs (pass-through)
                "und_tokens_last": und_tokens_last,
                "und_attention_mask": und_attention_mask,
                # metadata passthrough
                "label": 0,
                "split": d.get("split"),
                "task": d.get("task"),
                "episode": d.get("episode"),
                "instruction_text": d.get("instruction_text"),
                "instruction": d.get("instruction"),
                "instruction_idx": d.get("instruction_idx"),
                "condition_frame_idx": d.get("condition_frame_idx"),
                "source_video_indices": d.get("video_indices"),
                "source_action_indices": d.get("action_indices"),
                "source_sample_relpath": d.get("_source_relpath"),
                # slicing indices (relative to source sample tensors)
                "slice_action_start": int(s),
                "slice_video_start": int(v_start),
            }
        )

        s += stride

    return out


def main() -> None:
    _ensure_motus_on_path()

    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_dir", type=str, required=True, help="Positive dataset root (has manifest.jsonl and samples/)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output negative dataset root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_source_samples", type=int, default=None, help="Optional cap for debugging")
    ap.add_argument(
        "--require_und_tokens",
        type=int,
        default=1,
        help="1 to require und_tokens_last in each source sample (default), 0 to allow missing",
    )
    ap.add_argument("--gripper_indices", type=str, default="6,13", help="Comma-separated gripper dim indices in action vector")
    ap.add_argument("--half_noise_sigma", type=float, default=0.5)
    ap.add_argument("--enable_action_corruptions", type=int, default=1, help="1 to create action-corruption negatives")
    ap.add_argument("--enable_cross_sample_swap", type=int, default=1, help="1 to create cross-sample action swap negatives")
    args = ap.parse_args()

    pos_dir = Path(args.pos_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    samples_root = out_dir / "samples"
    _ensure_dir(samples_root)

    manifest_in = pos_dir / "manifest.jsonl"
    if not manifest_in.exists():
        raise FileNotFoundError(f"pos manifest not found: {manifest_in}")

    rows = _iter_manifest(manifest_in)
    if args.max_source_samples is not None:
        rows = rows[: int(args.max_source_samples)]

    rng = random.Random(int(args.seed))
    gripper_idx = _parse_gripper_indices(args.gripper_indices)
    manifest_out = out_dir / "manifest.jsonl"
    if manifest_out.exists():
        manifest_out.unlink()

    n_written = 0
    with manifest_out.open("a", encoding="utf-8") as mf:
        # Collect eligible slices for cross-sample swap pairing (global, no replacement)
        swap_pool: List[Dict[str, Any]] = []
        for row_i, row in enumerate(rows):
            rel = row["path"]
            src_pt = pos_dir / rel
            if not src_pt.exists():
                continue
            d = torch.load(str(src_pt), map_location="cpu")
            if int(d.get("label", 1)) != 1:
                continue
            if "predicted_video_tokens" not in d:
                continue
            if int(args.require_und_tokens) == 1 and "und_tokens_last" not in d:
                raise ValueError(f"Source sample missing und_tokens_last: {rel}")

            # attach a couple of manifest fields for traceability
            d["_source_relpath"] = rel
            d["split"] = d.get("split", row.get("split"))
            d["task"] = d.get("task", row.get("task"))
            d["episode"] = d.get("episode", row.get("episode"))
            d["instruction_text"] = d.get("instruction_text", row.get("instruction_text"))
            d["instruction_idx"] = d.get("instruction_idx", row.get("instruction_idx"))
            d["condition_frame_idx"] = d.get("condition_frame_idx", row.get("condition_frame_idx"))
            d["video_indices"] = d.get("video_indices", row.get("video_indices"))
            d["action_indices"] = d.get("action_indices", row.get("action_indices"))

            try:
                slices = _slice_one_source(d, rng=rng)
            except Exception:
                continue
            if not slices:
                continue

            split = str(d.get("split", "unknown"))
            task = str(d.get("task", "unknown"))
            episode = str(d.get("episode", "unknown"))
            # NOTE: write different neg_type into different folders under samples/
            base_out_dir = samples_root / split / task / episode

            for k, base in enumerate(slices):
                # 1-4) action corruptions (keep token block CORRECT for this 4-frame block)
                if int(args.enable_action_corruptions) == 1:
                    tok_block = base.get("predicted_token_block_correct")
                    tok_slice = base.get("predicted_token_time_slice_correct")
                    tok_block_future = base.get("predicted_token_block_future_correct")
                    tok_slice_future = base.get("predicted_token_time_slice_future_correct")
                    if (
                        tok_block is not None
                        and tok_slice is not None
                        and tok_block_future is not None
                        and tok_slice_future is not None
                    ):
                        # 1) random time swap
                        sd = dict(base)
                        sd["label"] = 0
                        sd["neg_type"] = "action_time_swap"
                        sd["predicted_token_block"] = tok_block
                        sd["predicted_token_time_slice"] = int(tok_slice)
                        sd["predicted_token_block_future"] = tok_block_future
                        sd["predicted_token_time_slice_future"] = int(tok_slice_future)
                        sd["action_sequence_17"] = _neg_action_time_swap(sd["action_sequence_17"], rng)
                        sd.pop("predicted_token_block_correct", None)
                        sd.pop("predicted_token_time_slice_correct", None)
                        sd.pop("predicted_token_block_future_correct", None)
                        sd.pop("predicted_token_time_slice_future_correct", None)
                        ep_out_dir = samples_root / sd["neg_type"] / split / task / episode
                        _ensure_dir(ep_out_dir)
                        sample_path = ep_out_dir / f"sample_{k:06d}.pt"
                        torch.save(sd, str(sample_path))
                        mf.write(json.dumps({"path": str(sample_path.relative_to(out_dir)), "label": 0, "neg_type": sd["neg_type"],
                                             "split": split, "task": task, "episode": episode, "source_sample_relpath": rel,
                                             "slice_action_start": sd["slice_action_start"], "slice_video_start": sd["slice_video_start"]},
                                            ensure_ascii=False) + "\n")
                        n_written += 1

                        # 2) joint swap (avoid grippers by default)
                        sd = dict(base)
                        sd["label"] = 0
                        sd["neg_type"] = "action_joint_swap"
                        sd["predicted_token_block"] = tok_block
                        sd["predicted_token_time_slice"] = int(tok_slice)
                        sd["predicted_token_block_future"] = tok_block_future
                        sd["predicted_token_time_slice_future"] = int(tok_slice_future)
                        sd["action_sequence_17"] = _neg_action_joint_swap(sd["action_sequence_17"], rng, avoid_dims=gripper_idx)
                        sd.pop("predicted_token_block_correct", None)
                        sd.pop("predicted_token_time_slice_correct", None)
                        sd.pop("predicted_token_block_future_correct", None)
                        sd.pop("predicted_token_time_slice_future_correct", None)
                        ep_out_dir = samples_root / sd["neg_type"] / split / task / episode
                        _ensure_dir(ep_out_dir)
                        sample_path = ep_out_dir / f"sample_{k:06d}.pt"
                        torch.save(sd, str(sample_path))
                        mf.write(json.dumps({"path": str(sample_path.relative_to(out_dir)), "label": 0, "neg_type": sd["neg_type"],
                                             "split": split, "task": task, "episode": episode, "source_sample_relpath": rel},
                                            ensure_ascii=False) + "\n")
                        n_written += 1

                        # 3) flip gripper direction
                        sd = dict(base)
                        sd["label"] = 0
                        sd["neg_type"] = "action_flip_gripper"
                        sd["predicted_token_block"] = tok_block
                        sd["predicted_token_time_slice"] = int(tok_slice)
                        sd["predicted_token_block_future"] = tok_block_future
                        sd["predicted_token_time_slice_future"] = int(tok_slice_future)
                        sd["action_sequence_17"] = _neg_action_flip_gripper(sd["action_sequence_17"], gripper_idx)
                        sd.pop("predicted_token_block_correct", None)
                        sd.pop("predicted_token_time_slice_correct", None)
                        sd.pop("predicted_token_block_future_correct", None)
                        sd.pop("predicted_token_time_slice_future_correct", None)
                        ep_out_dir = samples_root / sd["neg_type"] / split / task / episode
                        _ensure_dir(ep_out_dir)
                        sample_path = ep_out_dir / f"sample_{k:06d}.pt"
                        torch.save(sd, str(sample_path))
                        mf.write(json.dumps({"path": str(sample_path.relative_to(out_dir)), "label": 0, "neg_type": sd["neg_type"],
                                             "split": split, "task": task, "episode": episode, "source_sample_relpath": rel},
                                            ensure_ascii=False) + "\n")
                        n_written += 1

                        # 4) half noise
                        sd = dict(base)
                        sd["label"] = 0
                        sd["neg_type"] = "action_half_noise"
                        sd["predicted_token_block"] = tok_block
                        sd["predicted_token_time_slice"] = int(tok_slice)
                        sd["predicted_token_block_future"] = tok_block_future
                        sd["predicted_token_time_slice_future"] = int(tok_slice_future)
                        sd["action_sequence_17"] = _neg_action_half_noise(sd["action_sequence_17"], rng, sigma=float(args.half_noise_sigma))
                        sd.pop("predicted_token_block_correct", None)
                        sd.pop("predicted_token_time_slice_correct", None)
                        sd.pop("predicted_token_block_future_correct", None)
                        sd.pop("predicted_token_time_slice_future_correct", None)
                        ep_out_dir = samples_root / sd["neg_type"] / split / task / episode
                        _ensure_dir(ep_out_dir)
                        sample_path = ep_out_dir / f"sample_{k:06d}.pt"
                        torch.save(sd, str(sample_path))
                        mf.write(json.dumps({"path": str(sample_path.relative_to(out_dir)), "label": 0, "neg_type": sd["neg_type"],
                                             "split": split, "task": task, "episode": episode, "source_sample_relpath": rel},
                                            ensure_ascii=False) + "\n")
                        n_written += 1

                # 5) cross-sample swap (best effort: try different task)
                if int(args.enable_cross_sample_swap) == 1:
                    # Defer cross-sample pairing to a second pass to ensure no replacement pairs
                    swap_pool.append(base)

        # Second pass: pair slices without replacement for cross-sample action swap
        if int(args.enable_cross_sample_swap) == 1 and len(swap_pool) >= 2:
            rng.shuffle(swap_pool)
            # If odd, drop last
            if len(swap_pool) % 2 == 1:
                swap_pool = swap_pool[:-1]

            for pair_i in range(0, len(swap_pool), 2):
                a = swap_pool[pair_i]
                b = swap_pool[pair_i + 1]

                # Prefer different task: if same, try to swap b with a later element (best effort).
                if a.get("task") == b.get("task"):
                    for j in range(pair_i + 2, len(swap_pool)):
                        if swap_pool[j].get("task") != a.get("task"):
                            b, swap_pool[j] = swap_pool[j], b
                            break

                # Use each slice's correct token block; swap action_sequence_17
                def _emit_swap(src: Dict[str, Any], partner: Dict[str, Any]) -> None:
                    nonlocal n_written
                    tok_block = src.get("predicted_token_block_correct")
                    tok_slice = src.get("predicted_token_time_slice_correct")
                    tok_block_future = src.get("predicted_token_block_future_correct")
                    tok_slice_future = src.get("predicted_token_time_slice_future_correct")
                    if (
                        tok_block is None
                        or tok_slice is None
                        or tok_block_future is None
                        or tok_slice_future is None
                    ):
                        return
                    sd = dict(src)
                    sd["label"] = 0
                    sd["neg_type"] = "action_swap_between_samples"
                    sd["predicted_token_block"] = tok_block
                    sd["predicted_token_time_slice"] = int(tok_slice)
                    sd["predicted_token_block_future"] = tok_block_future
                    sd["predicted_token_time_slice_future"] = int(tok_slice_future)
                    sd["action_sequence_17"] = partner["action_sequence_17"].clone()
                    sd["swap_partner_source"] = partner.get("source_sample_relpath")
                    sd["swap_partner_task"] = partner.get("task")
                    # drop slice-internal fields not needed downstream
                    sd.pop("predicted_token_block_correct", None)
                    sd.pop("predicted_token_time_slice_correct", None)
                    sd.pop("predicted_token_block_future_correct", None)
                    sd.pop("predicted_token_time_slice_future_correct", None)

                    split = str(sd.get("split", "unknown"))
                    task = str(sd.get("task", "unknown"))
                    episode = str(sd.get("episode", "unknown"))
                    ep_out_dir = samples_root / sd["neg_type"] / split / task / episode
                    _ensure_dir(ep_out_dir)
                    # Name by global counter to avoid collisions across different source samples
                    sample_path = ep_out_dir / f"sample_{n_written:09d}.pt"
                    torch.save(sd, str(sample_path))
                    mf.write(json.dumps({
                        "path": str(sample_path.relative_to(out_dir)),
                        "label": 0,
                        "neg_type": sd["neg_type"],
                        "split": split,
                        "task": task,
                        "episode": episode,
                        "source_sample_relpath": sd.get("source_sample_relpath") or sd.get("_source_relpath"),
                        "swap_partner_task": sd.get("swap_partner_task"),
                        "swap_partner_source": sd.get("swap_partner_source"),
                    }, ensure_ascii=False) + "\n")
                    n_written += 1

                # Emit both directions so every eligible slice is used once (paired without replacement)
                _emit_swap(a, b)
                _emit_swap(b, a)

    (out_dir / "stats.json").write_text(
        json.dumps(
            {
                "pos_dir": str(pos_dir),
                "out_dir": str(out_dir),
                "seed": int(args.seed),
                "n_source_samples": int(len(rows)),
                "n_negative_samples_written": int(n_written),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


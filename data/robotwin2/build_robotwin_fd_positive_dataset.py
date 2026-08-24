#!/usr/bin/env python3
"""
Build sliced positive (label=1) samples from an existing RobotWin FD *augmented* dataset.

Input: a directory produced by build_robotwin_fd_augmented_dataset.py,
       containing:
         - manifest.jsonl with sample paths
         - samples/.../*.pt where each pt contains:
             video_frames: [16,3,H,W] (GT targets; condition frame is separate and should be ignored)
             action_sequence: [64,14]
             predicted_video_tokens: [600,3072] (required)

Each slice also stores two [120,3072] blocks from predicted_video_tokens:
  - predicted_token_block: aligned to the 4 GT frames (historical)
  - predicted_token_block_future: aligned to the following 16 actions / next 4-frame quartet (future; clamps to slice 4 when needed)

User-specified slicing rule (per source sample) — kept for reuse:
  - ignore condition frame (do not use first_frame)
  - take 4 GT video frames (asynchronous with actions, ratio 1:4)
  - for the 4th GT video frame, take the corresponding action and the next 16 actions:
      => 17 actions total
  - sliding window with overlap on actions:
      next window starts at previous window's last action (i.e. stride 16)
  - for predicted tokens:
      tokens length is 600 = 5 * 12 * 10 = (condition + 4 predicted time slices) * spatial grid
      take the FULL block (12*10=120 tokens) for historical + future time slices (see keys above).

Output:
  out_dir/manifest.jsonl  (each line is one sliced positive sample)
  out_dir/samples/{split}/{task}/{episode}/sample_{k:06d}.pt

python "/work/sme-wangr/Motus/data/robotwin2/build_robotwin_fd_positive_dataset.py" \
  --pos_dir "/work/sme-wangr/Motus/Dataset/robotwin_fd_dataset_clean10_random100" \
  --out_dir "/work/sme-wangr/Motus/Dataset/robotwin_fd_dataset_clean10_random100_pos" \
  --seed 0 \
  --max_source_samples 2  # (optional, debugging)

"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List

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

def _slice_one_source(
    d: Dict[str, Any],
    *,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Returns a list of sliced positive samples (dicts ready to torch.save).
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

        # Historical predicted tokens: time slice aligned to the 4 GT video frames [v_start : v_start+4].
        # Block mapping by frames: [0..3]->slice1, [4..7]->slice2, [8..11]->slice3, [12..15]->slice4.
        correct_pred_slice = 1 + (v_start // 4)
        correct_pred_slice = max(1, min(4, int(correct_pred_slice)))

        token_block_history = _token_block_for_time(tokens, correct_pred_slice).contiguous()  # [120,3072]

        # Future predicted tokens: aligned to the *next* 16 action steps (indices s+1 .. s+16),
        # i.e. the next 4 video-frame quartet [v_start+4 : v_start+8] under 1:4 async alignment.
        # One spatial block [120, D] per predicted time slice; at the last GT quartet (v_start==12) this
        # clamps to slice 4 (same block as history — no fifth slice in [600] layout).
        future_pred_slice = min(correct_pred_slice + 1, 4)
        token_block_future = _token_block_for_time(tokens, future_pred_slice).contiguous()  # [120,3072]

        out.append(
            {
                # sliced features
                "video_frame_1": gt_video_1,
                "action_sequence_17": gt_actions_17,
                # historical (same semantics as before)
                "predicted_token_block": token_block_history,
                "predicted_token_time_slice": int(correct_pred_slice),
                # future (next 16-action window in latent time)
                "predicted_token_block_future": token_block_future,
                "predicted_token_time_slice_future": int(future_pred_slice),
                # understanding expert outputs (pass-through)
                "und_tokens_last": und_tokens_last,
                "und_attention_mask": und_attention_mask,
                # metadata passthrough
                "label": 1,
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
    ap.add_argument("--out_dir", type=str, required=True, help="Output sliced-positive dataset root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_source_samples", type=int, default=None, help="Optional cap for debugging")
    ap.add_argument(
        "--require_und_tokens",
        type=int,
        default=1,
        help="1 to require und_tokens_last in each source sample (default), 0 to allow missing",
    )
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
    manifest_out = out_dir / "manifest.jsonl"
    if manifest_out.exists():
        manifest_out.unlink()

    n_written = 0
    with manifest_out.open("a", encoding="utf-8") as mf:
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
            ep_out_dir = samples_root / split / task / episode
            _ensure_dir(ep_out_dir)

            for k, base in enumerate(slices):
                sd = dict(base)
                sd["label"] = 1
                sample_path = ep_out_dir / f"sample_{k:06d}.pt"
                torch.save(sd, str(sample_path))
                mf.write(
                    json.dumps(
                        {
                            "path": str(sample_path.relative_to(out_dir)),
                            "label": 1,
                            "split": split,
                            "task": task,
                            "episode": episode,
                            "source_sample_relpath": rel,
                            "slice_action_start": int(sd.get("slice_action_start", -1)),
                            "slice_video_start": int(sd.get("slice_video_start", -1)),
                            "predicted_token_time_slice": int(sd.get("predicted_token_time_slice", -1)),
                            "predicted_token_time_slice_future": int(
                                sd.get("predicted_token_time_slice_future", -1)
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_written += 1

    (out_dir / "stats.json").write_text(
        json.dumps(
            {
                "pos_dir": str(pos_dir),
                "out_dir": str(out_dir),
                "seed": int(args.seed),
                "n_source_samples": int(len(rows)),
                "n_positive_slices_written": int(n_written),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


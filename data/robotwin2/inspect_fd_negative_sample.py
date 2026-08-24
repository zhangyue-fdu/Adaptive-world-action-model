#!/usr/bin/env python3
"""
Inspect a *negative* FD sample (.pt) produced by build_robotwin_fd_negative_dataset.py.

What it does:
  - Saves a visible image grid from `video_frames_4` (4 GT frames, condition already excluded)
  - Prints shapes/dtypes for:
      * video_frames_4
      * action_sequence_17
      * predicted_token_block  (expected [120, 3072], historical)
      * predicted_token_block_future  (expected [120, 3072], future)
      * und_tokens_last (optional, if present)
      * und_attention_mask (optional, if present)
    (optionally prints a small numeric slice)

Usage:
  python "/work/sme-wangr/Motus/data/robotwin2/inspect_fd_negative_sample.py" \
    --sample_pt "/work/sme-wangr/Motus/Dataset/robotwin_fd_dataset_clean10_random100_pos/samples/randomized/beat_block_hammer/4/sample_000000.pt" \
    --out_dir "/work/sme-wangr/Motus/Dataset/check/pos_inspect"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _ensure_motus_on_path() -> None:
    motus_root = Path(__file__).resolve().parents[2]  # .../Motus
    if str(motus_root) not in sys.path:
        sys.path.insert(0, str(motus_root))


def _to_uint8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    t = img_chw.detach().float().clamp(0, 1)
    return (t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def _save_video4_grid(video_frames_4: torch.Tensor, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if video_frames_4.dim() == 5:
        video_frames_4 = video_frames_4[0]
    if video_frames_4.dim() != 4 or video_frames_4.shape[0] < 1:
        raise ValueError(f"Unexpected video_frames_4 shape: {tuple(video_frames_4.shape)}")

    imgs = [_to_uint8_hwc(video_frames_4[i]) for i in range(min(4, int(video_frames_4.shape[0])))]
    grid = np.concatenate(imgs, axis=1)

    plt.figure(figsize=(max(6, grid.shape[1] / 160), max(3, grid.shape[0] / 160)))
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _print_tensor(name: str, t: torch.Tensor, max_rows: int, max_cols: int) -> None:
    print(f"\n=== {name} ===")
    print("shape:", tuple(t.shape), "dtype:", t.dtype, "device:", t.device)
    if t.dim() == 2:
        arr = t.detach().cpu().float().numpy()
        arr = arr[: max_rows, : max_cols]
        np.set_printoptions(suppress=True, linewidth=200)
        print(arr)
    elif t.dim() == 1:
        arr = t.detach().cpu().float().numpy()[:max_cols]
        np.set_printoptions(suppress=True, linewidth=200)
        print(arr)


def _assert_shape(name: str, t: Optional[torch.Tensor], expected: tuple[int, ...]) -> None:
    if t is None:
        return
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"{name} is not a Tensor: {type(t)}")
    if tuple(t.shape) != tuple(expected):
        raise ValueError(f"{name} shape mismatch: got {tuple(t.shape)}, expected {expected}")


def main() -> None:
    _ensure_motus_on_path()

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_pt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--print_values", type=int, default=0, help="1 to print numeric slices, 0 to print only shapes")
    ap.add_argument("--max_rows", type=int, default=10)
    ap.add_argument("--max_cols", type=int, default=14)
    ap.add_argument(
        "--print_und_tokens",
        type=int,
        default=0,
        help="1 to print und_tokens_last numeric slice (can be large); 0 to only print shape/stats (default)",
    )
    args = ap.parse_args()

    sample_pt = Path(args.sample_pt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = torch.load(str(sample_pt), map_location="cpu")

    video4 = d.get("video_frames_4")
    actions17 = d.get("action_sequence_17")
    tok_block = d.get("predicted_token_block")
    tok_block_future = d.get("predicted_token_block_future")
    und_tokens_last = d.get("und_tokens_last")
    und_attention_mask = d.get("und_attention_mask")

    if video4 is None or actions17 is None or tok_block is None:
        missing = [k for k in ("video_frames_4", "action_sequence_17", "predicted_token_block") if d.get(k) is None]
        raise ValueError(f"Sample missing keys: {missing}")
    if tok_block_future is None:
        print("WARNING: missing predicted_token_block_future (old sample format?)")

    # Save GT 4-frame grid
    _save_video4_grid(video4, out_dir / "video_frames_4_grid.png")

    # Print meta
    print("sample_pt:", str(sample_pt))
    for k in [
        "neg_type",
        "label",
        "split",
        "task",
        "episode",
        "slice_action_start",
        "slice_video_start",
        "predicted_token_time_slice",
        "predicted_token_time_slice_future",
        "correct_time_slice",
        "swap_partner_task",
        "swap_partner_source",
        "source_sample_relpath",
    ]:
        if k in d:
            print(f"{k}: {d[k]}")

    # Print shapes (and optionally values)
    print("\n=== SHAPES ===")
    print("video_frames_4:", tuple(video4.shape), video4.dtype)
    print("action_sequence_17:", tuple(actions17.shape), actions17.dtype)
    print("predicted_token_block:", tuple(tok_block.shape), tok_block.dtype)
    if tok_block_future is not None:
        print("predicted_token_block_future:", tuple(tok_block_future.shape), tok_block_future.dtype)

    # Basic shape checks for sanity (fail fast)
    _assert_shape("predicted_token_block", tok_block, (120, 3072))
    _assert_shape("predicted_token_block_future", tok_block_future, (120, 3072))
    # time slice indices are expected to be in [1,4] when present
    if "predicted_token_time_slice" in d and d.get("predicted_token_time_slice") is not None:
        ts = int(d["predicted_token_time_slice"])
        if ts < 1 or ts > 4:
            raise ValueError(f"predicted_token_time_slice out of range [1,4]: {ts}")
    if "predicted_token_time_slice_future" in d and d.get("predicted_token_time_slice_future") is not None:
        tsf = int(d["predicted_token_time_slice_future"])
        if tsf < 1 or tsf > 4:
            raise ValueError(f"predicted_token_time_slice_future out of range [1,4]: {tsf}")

    if und_tokens_last is not None:
        if not isinstance(und_tokens_last, torch.Tensor):
            raise ValueError(f"und_tokens_last is not a Tensor: {type(und_tokens_last)}")
        print("und_tokens_last:", tuple(und_tokens_last.shape), und_tokens_last.dtype)
    if und_attention_mask is not None:
        if not isinstance(und_attention_mask, torch.Tensor):
            raise ValueError(f"und_attention_mask is not a Tensor: {type(und_attention_mask)}")
        print("und_attention_mask:", tuple(und_attention_mask.shape), und_attention_mask.dtype)
        try:
            m = und_attention_mask.detach().cpu().to(torch.bool)
            print(f"und_attention_mask valid tokens: {int(m.sum().item())} / {int(m.numel())}")
            if und_tokens_last is not None and und_tokens_last.dim() == 2 and m.dim() == 1 and m.numel() == und_tokens_last.shape[0]:
                und_valid = und_tokens_last.detach().cpu()[m]
                print("und_tokens_last (valid-only) shape:", tuple(und_valid.shape))
            elif und_tokens_last is not None and und_tokens_last.dim() == 2 and m.dim() == 1 and m.numel() != und_tokens_last.shape[0]:
                print(
                    "WARNING: und_attention_mask length mismatch: "
                    f"mask.numel={int(m.numel())}, und_tokens_last.shape[0]={int(und_tokens_last.shape[0])}"
                )
        except Exception as e:
            print("failed to compute und mask stats:", repr(e))

    if int(args.print_values) == 1:
        _print_tensor("action_sequence_17 (slice)", actions17, int(args.max_rows), int(args.max_cols))
        _print_tensor("predicted_token_block (slice)", tok_block, int(args.max_rows), int(args.max_cols))
        if tok_block_future is not None:
            _print_tensor(
                "predicted_token_block_future (slice)",
                tok_block_future,
                int(args.max_rows),
                int(args.max_cols),
            )
    if und_tokens_last is not None and int(args.print_und_tokens) == 1:
        if und_tokens_last.dim() == 2:
            _print_tensor("und_tokens_last (slice)", und_tokens_last, int(args.max_rows), int(args.max_cols))
        elif und_tokens_last.dim() == 3 and und_tokens_last.shape[0] == 1:
            _print_tensor("und_tokens_last[0] (slice)", und_tokens_last[0], int(args.max_rows), int(args.max_cols))

    print("\nSaved:", str(out_dir / "video_frames_4_grid.png"))


if __name__ == "__main__":
    main()


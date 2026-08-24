#!/usr/bin/env python3
"""
Simple inspector for an exported FD sample (.pt):
  - prints action_sequence and predicted_video_tokens as numbers (keeps correct shapes)
  - saves GT frame grid (first_frame + video_frames) as a PNG
  - (optional) prints und_tokens_last + und_attention_mask if present

Usage:
  python "/work/sme-wangr/Motus/data/robotwin2/inspect_fd_sample_simple.py" \
    --sample_pt "/work/sme-wangr/Motus/Dataset/robotwin_fd_dataset_debug/samples/randomized/adjust_bottle/11/sample_000000.pt" \
    --out_dir "/work/sme-wangr/Motus/Dataset/check/inspect_sample"
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


def _save_gt_grid(first_frame: torch.Tensor, video_frames: torch.Tensor, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if video_frames.dim() == 5:
        video_frames = video_frames[0]
    if video_frames.dim() != 4:
        raise ValueError(f"Unexpected video_frames shape: {tuple(video_frames.shape)}")

    imgs = [_to_uint8_hwc(first_frame)] + [_to_uint8_hwc(video_frames[i]) for i in range(video_frames.shape[0])]
    grid = np.concatenate(imgs, axis=1)

    plt.figure(figsize=(max(6, grid.shape[1] / 160), max(3, grid.shape[0] / 160)))
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _print_tensor(name: str, t: torch.Tensor, max_rows: Optional[int], max_cols: Optional[int]) -> None:
    print(f"\n=== {name} ===")
    print("shape:", tuple(t.shape), "dtype:", t.dtype, "device:", t.device)

    if t.dim() == 1:
        arr = t.detach().cpu().numpy()
        if max_cols is not None:
            arr = arr[:max_cols]
        print(arr)
        return

    if t.dim() == 2:
        arr = t.detach().cpu().numpy()
        if max_rows is not None:
            arr = arr[:max_rows]
        if max_cols is not None:
            arr = arr[:, :max_cols]
        np.set_printoptions(suppress=True, linewidth=200)
        print(arr)
        return

    # Higher dims: print a small slice but keep original shape reported above
    arr = t.detach().cpu().reshape(t.shape[0], -1).numpy()
    if max_rows is not None:
        arr = arr[:max_rows]
    if max_cols is not None:
        arr = arr[:, :max_cols]
    np.set_printoptions(suppress=True, linewidth=200)
    print("(flattened view) first rows/cols:\n", arr)


def main() -> None:
    _ensure_motus_on_path()

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_pt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_rows", type=int, default=10, help="How many rows to print for 2D tensors")
    ap.add_argument("--max_cols", type=int, default=14, help="How many cols to print for 2D tensors")
    ap.add_argument(
        "--print_und_tokens",
        type=int,
        default=0,
        help="1 to print und_tokens_last numbers (can be large); 0 to only print shape/stats (default)",
    )
    args = ap.parse_args()

    sample_pt = Path(args.sample_pt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = torch.load(str(sample_pt), map_location="cpu")

    first_frame = d.get("first_frame")
    video_frames = d.get("video_frames")
    actions = d.get("action_sequence")
    tokens = d.get("predicted_video_tokens")
    pred_frames = d.get("predicted_frames")
    und_tokens_last = d.get("und_tokens_last")
    und_attention_mask = d.get("und_attention_mask")

    if first_frame is None or video_frames is None:
        raise ValueError("Sample missing first_frame/video_frames")
    if actions is None:
        raise ValueError("Sample missing action_sequence")
    if tokens is None:
        raise ValueError("Sample missing predicted_video_tokens (did you export with --save_predicted_tokens 1?)")

    # Save GT grid
    _save_gt_grid(first_frame, video_frames, out_dir / "gt_frames_grid.png")
    # Save predicted frames grid if present
    if pred_frames is not None:
        if pred_frames.dim() == 5:
            pred_frames = pred_frames[0]
        if pred_frames.dim() == 4:
            # pred_frames: [T,C,H,W]
            import matplotlib.pyplot as plt

            imgs = [_to_uint8_hwc(pred_frames[i]) for i in range(pred_frames.shape[0])]
            grid = np.concatenate(imgs, axis=1)
            plt.figure(figsize=(max(6, grid.shape[1] / 160), max(3, grid.shape[0] / 160)))
            plt.imshow(grid)
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(out_dir / "predicted_frames_grid.png", dpi=200)
            plt.close()

    # Print meta
    print("sample_pt:", str(sample_pt))
    for k in ["split", "task", "episode", "condition_frame_idx", "instruction_idx"]:
        if k in d:
            print(f"{k}: {d[k]}")
    if "instruction_text" in d:
        print("instruction_text:", d["instruction_text"])

    # Print tensors as numbers (keeping correct shapes in header)
    _print_tensor("action_sequence", actions, max_rows=int(args.max_rows), max_cols=int(args.max_cols))
    _print_tensor("predicted_video_tokens", tokens, max_rows=int(args.max_rows), max_cols=int(args.max_cols))

    # Optional: Understanding expert tokens
    if und_tokens_last is not None:
        if not isinstance(und_tokens_last, torch.Tensor):
            raise ValueError(f"und_tokens_last is not a Tensor: {type(und_tokens_last)}")
        print("\n=== und_tokens_last ===")
        print("shape:", tuple(und_tokens_last.shape), "dtype:", und_tokens_last.dtype, "device:", und_tokens_last.device)

        if und_attention_mask is not None:
            if not isinstance(und_attention_mask, torch.Tensor):
                raise ValueError(f"und_attention_mask is not a Tensor: {type(und_attention_mask)}")
            print("\n=== und_attention_mask ===")
            print("shape:", tuple(und_attention_mask.shape), "dtype:", und_attention_mask.dtype, "device:", und_attention_mask.device)
            try:
                mask = und_attention_mask.detach().cpu().to(torch.bool)
                n_valid = int(mask.sum().item())
                print(f"und_attention_mask valid tokens: {n_valid} / {mask.numel()}")
                if und_tokens_last.dim() == 2 and mask.dim() == 1 and mask.numel() == und_tokens_last.shape[0]:
                    und_valid = und_tokens_last.detach().cpu()[mask]
                    print("und_tokens_last (valid-only) shape:", tuple(und_valid.shape))
            except Exception as e:
                print("failed to compute und mask stats:", repr(e))

        # Printing values is optional (can be very large)
        if int(args.print_und_tokens) == 1:
            _print_tensor("und_tokens_last", und_tokens_last, max_rows=int(args.max_rows), max_cols=int(args.max_cols))

    print("\nSaved:", str(out_dir / "gt_frames_grid.png"))
    if pred_frames is not None:
        print("Saved:", str(out_dir / "predicted_frames_grid.png"))


if __name__ == "__main__":
    main()


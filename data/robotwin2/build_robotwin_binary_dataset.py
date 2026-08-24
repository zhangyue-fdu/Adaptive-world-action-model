#!/usr/bin/env python3
"""
Build a binary (success/fail) classification dataset from RobotWin2 episodes.

This script DOES NOT modify existing training code. It reads the existing RobotWin2
preprocessed dataset layout and exports windowed samples with labels.

Requested behavior (current implementation):
  - Episode sampling:
      * clean:      randomly pick N_clean episodes per task
      * randomized: randomly pick N_random episodes per task
  - Windowing (positive samples, label=1):
      * Use the same async alignment rule as Motus RobotWin loader:
          video_indices[j] = condition + (j+1) * ratio * ds
          action_indices[i] = condition + (i+1) * ds
      * For each window, take 3 video frames (j=0..2).
      * Let the 3rd video frame time be t_v3 = condition + 3*ratio*ds.
        Take the action at t_v3 and the next 8 actions (total 9 actions):
          indices = condition + (3*ratio + k) * ds, k=0..8
      * Sliding window with 1-frame overlap on video:
          next_condition = condition + 2*ratio*ds

Export format:
  - out_dir/manifest.jsonl
  - out_dir/samples/{split}/{task}/{episode}/sample_{k:06d}.pt
    each pt contains:
      {
        "video_frames": Tensor[3, 3, H, W] float in [0,1] (from load_video_frames),
        "action_sequence": Tensor[9, 14] float,
        "label": int (1 for positive),
        "split": "clean"|"randomized",
        "task": str,
        "episode": str,
        "condition_frame_idx": int,
        "video_indices": List[int],
        "action_indices": List[int],
      }
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import torch

MOTUS_ROOT = Path(__file__).resolve().parents[2]  # .../Motus
if str(MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTUS_ROOT))

from data.utils.image_utils import load_video_frames, get_video_frame_count


@dataclass(frozen=True)
class Episode:
    split: str
    task: str
    episode: str
    qpos_path: Path
    video_path: Path


def _scan_task_folder(task_path: Path) -> List[Episode]:
    qpos_dir = task_path / "qpos"
    videos_dir = task_path / "videos"
    umt5_dir = task_path / "umt5_wan"
    if not (qpos_dir.exists() and videos_dir.exists() and umt5_dir.exists()):
        return []

    split = task_path.parent.name
    task = task_path.name
    out: List[Episode] = []
    for qpos_file in sorted(qpos_dir.glob("*.pt")):
        ep = qpos_file.stem
        video_file = videos_dir / f"{ep}.mp4"
        lang_file = umt5_dir / f"{ep}.pt"
        if video_file.exists() and lang_file.exists():
            out.append(
                Episode(
                    split=split,
                    task=task,
                    episode=ep,
                    qpos_path=qpos_file,
                    video_path=video_file,
                )
            )
    return out


def _list_tasks(dataset_dir: Path, split: str) -> List[Path]:
    base = dataset_dir / split
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def _pick_episodes_per_task(
    dataset_dir: Path,
    split: str,
    n_per_task: int,
    seed: int,
) -> List[Episode]:
    rng = random.Random(seed + (0 if split == "clean" else 1000003))
    picked: List[Episode] = []
    for task_path in _list_tasks(dataset_dir, split):
        eps = _scan_task_folder(task_path)
        if not eps:
            continue
        if n_per_task >= len(eps):
            chosen = eps
        else:
            chosen = rng.sample(eps, n_per_task)
        picked.extend(chosen)
    return picked


def _compute_window_indices(
    condition_frame_idx: int,
    *,
    ratio: int,
    ds: int,
) -> Tuple[List[int], List[int]]:
    # 3 video frames
    video_indices = [
        condition_frame_idx + (j + 1) * ratio * ds
        for j in range(3)
    ]
    # action at 3rd video time (j=2 => 3*ratio*ds) and next 8 actions => 9 total
    action_indices = [
        condition_frame_idx + (3 * ratio + k) * ds
        for k in range(9)
    ]
    return video_indices, action_indices


def _iter_positive_windows(
    total_frames: int,
    *,
    ratio: int,
    ds: int,
) -> List[Tuple[int, List[int], List[int]]]:
    """
    Return list of (condition_frame_idx, video_indices, action_indices) for label=1 windows.
    Uses 1-frame overlap on video: next window's video[0] == prev window's video[2].
    """
    stride = 2 * ratio * ds
    windows: List[Tuple[int, List[int], List[int]]] = []

    # Need the last action index in action_indices to be in-episode.
    # last action index = condition + (3*ratio + 8)*ds
    max_condition = total_frames - 1 - (3 * ratio + 8) * ds
    if max_condition < 0:
        return windows

    c = 0
    while c <= max_condition:
        v_idx, a_idx = _compute_window_indices(c, ratio=ratio, ds=ds)
        # Also ensure video indices are valid
        if v_idx[-1] < total_frames and a_idx[-1] < total_frames:
            windows.append((c, v_idx, a_idx))
        c += stride
    return windows


def _load_actions(qpos_path: Path, action_indices: List[int]) -> torch.Tensor:
    qpos = torch.load(str(qpos_path), map_location="cpu")
    rows = [qpos[i].float() for i in action_indices]
    return torch.stack(rows, dim=0)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build(
    *,
    dataset_dir: Path,
    out_dir: Path,
    n_clean_per_task: int,
    n_random_per_task: int,
    seed: int,
    ratio: int,
    ds: int,
    video_size: Tuple[int, int],
) -> None:
    _ensure_dir(out_dir)
    samples_root = out_dir / "samples"
    _ensure_dir(samples_root)

    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    episodes: List[Episode] = []
    episodes.extend(_pick_episodes_per_task(dataset_dir, "clean", n_clean_per_task, seed))
    episodes.extend(_pick_episodes_per_task(dataset_dir, "randomized", n_random_per_task, seed))

    # Deterministic iteration order for reproducibility
    episodes = sorted(episodes, key=lambda e: (e.split, e.task, e.episode))

    n_written = 0
    with manifest_path.open("a", encoding="utf-8") as mf:
        for ep in episodes:
            try:
                total_frames = int(get_video_frame_count(str(ep.video_path)))
            except Exception:
                continue
            if total_frames <= 0:
                continue

            windows = _iter_positive_windows(total_frames, ratio=ratio, ds=ds)
            if not windows:
                continue

            ep_out_dir = samples_root / ep.split / ep.task / ep.episode
            _ensure_dir(ep_out_dir)

            for k, (c_idx, v_idx, a_idx) in enumerate(windows):
                try:
                    video_frames = load_video_frames(str(ep.video_path), v_idx, video_size)
                    # load_video_frames returns [T, C, H, W]
                    if isinstance(video_frames, torch.Tensor) and video_frames.dim() == 4:
                        pass
                    else:
                        continue
                    actions = _load_actions(ep.qpos_path, a_idx)
                    sample = {
                        "video_frames": video_frames,
                        "action_sequence": actions,
                        "label": 1,
                        "split": ep.split,
                        "task": ep.task,
                        "episode": ep.episode,
                        "condition_frame_idx": int(c_idx),
                        "video_indices": [int(x) for x in v_idx],
                        "action_indices": [int(x) for x in a_idx],
                    }
                    sample_name = f"sample_{k:06d}.pt"
                    sample_path = ep_out_dir / sample_name
                    torch.save(sample, str(sample_path))

                    mf.write(
                        json.dumps(
                            {
                                "path": str(sample_path.relative_to(out_dir)),
                                "label": 1,
                                "split": ep.split,
                                "task": ep.task,
                                "episode": ep.episode,
                                "condition_frame_idx": int(c_idx),
                                "video_indices": [int(x) for x in v_idx],
                                "action_indices": [int(x) for x in a_idx],
                                "total_frames": int(total_frames),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_written += 1
                except Exception:
                    continue

    stats = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "n_clean_per_task": int(n_clean_per_task),
        "n_random_per_task": int(n_random_per_task),
        "seed": int(seed),
        "ratio": int(ratio),
        "global_downsample_rate": int(ds),
        "video_size_hw": [int(video_size[0]), int(video_size[1])],
        "n_samples_written": int(n_written),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_dir",
        type=str,
        default="/work/sme-wangr/Motus/Dataset/robotwin_dataset",
        help="RobotWin2 preprocessed root containing clean/ and randomized/",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/work/sme-wangr/Motus/Dataset/robotwin_binary_dataset",
        help="Output directory",
    )
    ap.add_argument("--n_clean_per_task", type=int, default=10)
    ap.add_argument("--n_random_per_task", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)

    # Match configs/robotwin.yaml defaults
    ap.add_argument("--video_action_freq_ratio", type=int, default=4)
    ap.add_argument("--global_downsample_rate", type=int, default=3)
    ap.add_argument("--video_height", type=int, default=384)
    ap.add_argument("--video_width", type=int, default=320)

    args = ap.parse_args()
    build(
        dataset_dir=Path(args.dataset_dir),
        out_dir=Path(args.out_dir),
        n_clean_per_task=int(args.n_clean_per_task),
        n_random_per_task=int(args.n_random_per_task),
        seed=int(args.seed),
        ratio=int(args.video_action_freq_ratio),
        ds=int(args.global_downsample_rate),
        video_size=(int(args.video_width), int(args.video_height)),
    )


if __name__ == "__main__":
    main()


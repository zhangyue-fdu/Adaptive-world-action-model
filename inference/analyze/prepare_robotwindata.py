#!/usr/bin/env python3
"""
从 RobotWin 预处理数据根目录采样视频帧，用于离线分析。

数据源：/data/share/robotwin_dataset
输出：/data/share/250010061/motus/ana_robotwin_data/dataset

目录与 episode 定义与 data/robotwin2/robotwin_agilex_dataset.py._scan_task_folder 一致：
  有效 episode 需同时具备 qpos、videos、umt5_wan、metas 中同名文件。

每个 (split, task) 随机采样 --episodes_per_task 个 episode；每个 episode
按 RobotWinTaskDataset 的 robotwin_sampling_mode 随机条件帧，并取
--num_video_frames 个预测视频时间下标（默认 16），从 MP4 导出为图片。
指令文本与 __getitem__ 一致：先随机 language embedding 列表下标，再用同一下标
从 metas 中取指令行（长度不匹配时回退）。
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch

# 与 robotwin_agilex_dataset.RobotWinTaskDataset 默认一致（视频预测长度分析脚本默认 16）
DEFAULT_GLOBAL_DOWNSAMPLE_RATE = 3
DEFAULT_VIDEO_ACTION_FREQ_RATIO = 5
DEFAULT_NUM_VIDEO_FRAMES = 16
DEFAULT_SAMPLING_MODE = "tail_padding"


@dataclass
class EpisodeSample:
    split: str
    task: str
    episode_name: str
    video_path: str
    meta_path: str
    instruction: str
    total_frames: int
    condition_frame_idx: int
    video_frame_indices: List[int]
    sampling_mode: str
    output_dir: str
    output_images: List[str]
    extra: Dict[str, Any] = field(default_factory=dict)


def _max_condition_index_min_valid_prefix(
    total_frames: int,
    num_video_frames: int,
    global_downsample_rate: int,
    video_action_freq_ratio: int,
) -> Optional[int]:
    """与 RobotWinTaskDataset._max_condition_index_min_valid_prefix 一致。"""
    ds = global_downsample_rate
    ratio = video_action_freq_ratio
    action_chunk_size = num_video_frames * ratio
    need_valid_actions = 2 * ratio
    if action_chunk_size < need_valid_actions:
        return None
    need_video_slots = min(2, num_video_frames)
    max_c_vid = total_frames - 1 - need_video_slots * ratio * ds
    max_c_act = total_frames - 1 - need_valid_actions * ds
    max_c = min(max_c_vid, max_c_act)
    if max_c < 0:
        return None
    return max_c


def _strict_style_video_indices_from_condition(
    condition_frame_idx: int,
    total_frames: int,
    num_video_frames: int,
    global_downsample_rate: int,
    video_action_freq_ratio: int,
) -> List[int]:
    """与 _strict_style_indices_from_condition 中的 video 部分一致。"""
    ratio = video_action_freq_ratio
    ds = global_downsample_rate
    action_chunk_size = num_video_frames * ratio
    action_indices: List[int] = []
    for i in range(action_chunk_size):
        action_idx = condition_frame_idx + (i + 1) * ds
        action_indices.append(min(action_idx, total_frames - 1))
    video_indices: List[int] = []
    for i in range(num_video_frames):
        action_step = (i + 1) * ratio - 1
        if action_step < len(action_indices):
            video_indices.append(action_indices[action_step])
        else:
            video_indices.append(action_indices[-1])
    return video_indices


def _calculate_sampling_indices_legacy(
    total_frames: int,
    num_video_frames: int,
    global_downsample_rate: int,
    video_action_freq_ratio: int,
) -> Tuple[int, List[int]]:
    """与 RobotWinTaskDataset._calculate_sampling_indices_legacy 一致（仅返回条件帧与视频下标）。"""
    action_chunk_size = num_video_frames * video_action_freq_ratio
    ds = global_downsample_rate
    physical_chunk_size = action_chunk_size * ds
    max_condition_idx = total_frames - physical_chunk_size - 1
    if max_condition_idx < 0:
        condition_frame_idx = 0
    else:
        condition_frame_idx = random.randint(0, max_condition_idx)
    video_indices = _strict_style_video_indices_from_condition(
        condition_frame_idx,
        total_frames,
        num_video_frames,
        ds,
        video_action_freq_ratio,
    )
    return condition_frame_idx, video_indices


def _calculate_video_indices_random_start_strict_fill(
    total_frames: int,
    num_video_frames: int,
    global_downsample_rate: int,
    video_action_freq_ratio: int,
) -> Tuple[str, int, List[int]]:
    """与 RobotWinTaskDataset._calculate_sampling_indices_random_start_strict_fill 的视频下标一致。"""
    max_c = _max_condition_index_min_valid_prefix(
        total_frames,
        num_video_frames,
        global_downsample_rate,
        video_action_freq_ratio,
    )
    if max_c is None:
        c, v = _calculate_sampling_indices_legacy(
            total_frames,
            num_video_frames,
            global_downsample_rate,
            video_action_freq_ratio,
        )
        return "legacy_strict", c, v

    condition_frame_idx = random.randint(0, max_c)
    video_indices = _strict_style_video_indices_from_condition(
        condition_frame_idx,
        total_frames,
        num_video_frames,
        global_downsample_rate,
        video_action_freq_ratio,
    )
    return "random_start_strict_fill", condition_frame_idx, video_indices


def _calculate_video_indices_tail_padding(
    total_frames: int,
    num_video_frames: int,
    global_downsample_rate: int,
    video_action_freq_ratio: int,
) -> Optional[Tuple[str, int, List[int]]]:
    """
    与 RobotWinTaskDataset._calculate_sampling_indices 中视频下标生成一致
    （tail_padding：随机条件帧 + 超出片尾则钉在最后一帧）。
    若最短有效前缀无法满足，返回 None（与 dataset 中 sampling is None 则重试/跳过一致）。
    """
    max_c = _max_condition_index_min_valid_prefix(
        total_frames,
        num_video_frames,
        global_downsample_rate,
        video_action_freq_ratio,
    )
    if max_c is None:
        return None

    condition_frame_idx = random.randint(0, max_c)
    video_indices: List[int] = []
    ds = global_downsample_rate
    ratio = video_action_freq_ratio
    for j in range(num_video_frames):
        step = (j + 1) * ratio
        vid_idx = condition_frame_idx + step * ds
        if vid_idx < total_frames:
            video_indices.append(vid_idx)
        else:
            video_indices.append(total_frames - 1)
    return "tail_padding", condition_frame_idx, video_indices


def _pick_instruction_like_dataset(meta_path: Path, lang_path: Path) -> Tuple[str, List[str], int]:
    """
    与 RobotWinTaskDataset.__getitem__ 一致：先随机选 language embedding 列表下标，
    再用同一 instruction_idx 从 meta 取文本（长度不足则回退 random.choice）。
    """
    embedding_data = torch.load(lang_path, map_location="cpu")
    if not isinstance(embedding_data, list) or len(embedding_data) == 0:
        raise ValueError(f"Invalid or empty language embedding list: {lang_path}")
    instruction_idx = random.randint(0, len(embedding_data) - 1)

    text = meta_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if not lines:
        raise ValueError(f"No instructions in {meta_path}")

    if 0 <= instruction_idx < len(lines):
        instruction = lines[instruction_idx]
    else:
        instruction = random.choice(lines)
    return instruction, lines, instruction_idx


def _video_indices_for_sampling_mode(
    sampling_mode: str,
    total_frames: int,
    num_video_frames: int,
    global_downsample_rate: int,
    video_action_freq_ratio: int,
) -> Optional[Tuple[str, int, List[int]]]:
    m = sampling_mode.strip().lower().replace("-", "_")
    if m == "tail_padding":
        return _calculate_video_indices_tail_padding(
            total_frames,
            num_video_frames,
            global_downsample_rate,
            video_action_freq_ratio,
        )
    if m == "strict":
        c, v = _calculate_sampling_indices_legacy(
            total_frames,
            num_video_frames,
            global_downsample_rate,
            video_action_freq_ratio,
        )
        return "strict", c, v
    if m == "random_start_strict_fill":
        return _calculate_video_indices_random_start_strict_fill(
            total_frames,
            num_video_frames,
            global_downsample_rate,
            video_action_freq_ratio,
        )
    raise ValueError(
        f"robotwin_sampling_mode must be tail_padding|strict|random_start_strict_fill, got {sampling_mode!r}"
    )


def _get_total_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _extract_frames(video_path: Path, indices: List[int]) -> List[Tuple[int, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        frames: List[Tuple[int, Any]] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
            frames.append((idx, frame_bgr))
        return frames
    finally:
        cap.release()


def _list_valid_episodes(task_dir: Path) -> List[str]:
    """与 RobotWinTaskDataset._scan_task_folder 一致，并额外要求 metas/*.txt 存在。"""
    qpos_dir = task_dir / "qpos"
    videos_dir = task_dir / "videos"
    umt5_dir = task_dir / "umt5_wan"
    metas_dir = task_dir / "metas"
    if not all(d.is_dir() for d in (qpos_dir, videos_dir, umt5_dir, metas_dir)):
        return []

    names: List[str] = []
    for qpos_file in qpos_dir.glob("*.pt"):
        stem = qpos_file.stem
        if (
            (videos_dir / f"{stem}.mp4").exists()
            and (umt5_dir / f"{stem}.pt").exists()
            and (metas_dir / f"{stem}.txt").exists()
        ):
            names.append(stem)
    return names


def prepare_one_episode(
    split: str,
    task: str,
    episode_name: str,
    task_dir: Path,
    out_root: Path,
    num_video_frames: int,
    global_downsample_rate: int,
    video_action_freq_ratio: int,
    sampling_mode: str,
) -> Optional[EpisodeSample]:
    video_path = task_dir / "videos" / f"{episode_name}.mp4"
    meta_path = task_dir / "metas" / f"{episode_name}.txt"
    lang_path = task_dir / "umt5_wan" / f"{episode_name}.pt"
    qpos_path = task_dir / "qpos" / f"{episode_name}.pt"
    if not all(p.exists() for p in (video_path, meta_path, lang_path, qpos_path)):
        return None

    try:
        total_frames = _get_total_frames(video_path)
    except RuntimeError:
        return None
    if total_frames < 2:
        return None

    try:
        instruction, all_lines, instruction_idx = _pick_instruction_like_dataset(meta_path, lang_path)
    except (ValueError, OSError):
        return None

    sampled = _video_indices_for_sampling_mode(
        sampling_mode,
        total_frames,
        num_video_frames,
        global_downsample_rate,
        video_action_freq_ratio,
    )
    if sampled is None:
        return None
    mode, condition_idx, vid_indices = sampled

    frames = _extract_frames(video_path, vid_indices)
    sample_out_dir = out_root / split / task / f"episode_{episode_name}"
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    output_images: List[str] = []
    for k, (idx, frame_bgr) in enumerate(frames):
        out_path = sample_out_dir / f"video_frame_{k:02d}_srcidx{idx}.jpg"
        if not cv2.imwrite(str(out_path), frame_bgr):
            raise RuntimeError(f"Failed to write image: {out_path}")
        output_images.append(str(out_path))

    (sample_out_dir / "instruction.txt").write_text(instruction + "\n", encoding="utf-8")
    (sample_out_dir / "meta_lines.json").write_text(
        json.dumps(all_lines, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    meta = EpisodeSample(
        split=split,
        task=task,
        episode_name=episode_name,
        video_path=str(video_path),
        meta_path=str(meta_path),
        instruction=instruction,
        total_frames=total_frames,
        condition_frame_idx=condition_idx,
        video_frame_indices=vid_indices,
        sampling_mode=mode,
        output_dir=str(sample_out_dir),
        output_images=output_images,
        extra={
            "global_downsample_rate": global_downsample_rate,
            "video_action_freq_ratio": video_action_freq_ratio,
            "num_video_frames": num_video_frames,
            "robotwin_sampling_mode": sampling_mode,
            "instruction_idx": instruction_idx,
            "qpos_path": str(qpos_path),
            "lang_path": str(lang_path),
        },
    )
    (sample_out_dir / "meta.json").write_text(
        json.dumps(meta.__dict__, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RobotWin samples for Motus analysis.")
    parser.add_argument(
        "--dataset_root",
        default="/data/share/robotwin_dataset",
        help="RobotWin 数据根目录（其下有 clean/、randomized/）",
    )
    parser.add_argument(
        "--out_root",
        default="/data/share/250010061/motus/ana_robotwin_data/dataset",
        help="输出根目录",
    )
    parser.add_argument(
        "--data_mode",
        default="both",
        choices=["clean", "randomized", "both"],
        help="与 RobotWinTaskDataset.data_mode 一致：选用数据划分",
    )
    parser.add_argument(
        "--episodes_per_task",
        type=int,
        default=10,
        help="每个 (split, task) 随机采样的 episode 数量",
    )
    parser.add_argument(
        "--num_video_frames",
        type=int,
        default=DEFAULT_NUM_VIDEO_FRAMES,
        help="每个 episode 采样的视频帧数（序列长度）",
    )
    parser.add_argument(
        "--global_downsample_rate",
        type=int,
        default=DEFAULT_GLOBAL_DOWNSAMPLE_RATE,
        help="与 RobotWinTaskDataset 一致",
    )
    parser.add_argument(
        "--video_action_freq_ratio",
        type=int,
        default=DEFAULT_VIDEO_ACTION_FREQ_RATIO,
        help="与 RobotWinTaskDataset 一致",
    )
    parser.add_argument(
        "--robotwin_sampling_mode",
        default=DEFAULT_SAMPLING_MODE,
        choices=["tail_padding", "strict", "random_start_strict_fill"],
        help="与 RobotWinTaskDataset.robotwin_sampling_mode 一致（默认 tail_padding）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    if args.data_mode == "both":
        splits = ["clean", "randomized"]
    else:
        splits = [args.data_mode]

    out_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "out_root": str(out_root),
        "data_mode": args.data_mode,
        "episodes_per_task": args.episodes_per_task,
        "num_video_frames": args.num_video_frames,
        "global_downsample_rate": args.global_downsample_rate,
        "video_action_freq_ratio": args.video_action_freq_ratio,
        "robotwin_sampling_mode": args.robotwin_sampling_mode,
        "seed": args.seed,
        "samples": [],
        "skipped": [],
    }

    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            manifest["skipped"].append({"reason": "missing_split", "split": split})
            continue

        task_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        for task_dir in task_dirs:
            task = task_dir.name
            episode_names = _list_valid_episodes(task_dir)
            if not episode_names:
                manifest["skipped"].append(
                    {"reason": "no_episodes", "split": split, "task": task}
                )
                continue
            n = min(args.episodes_per_task, len(episode_names))
            picked = random.sample(episode_names, n) if n < len(episode_names) else list(episode_names)
            for ep in picked:
                sample = prepare_one_episode(
                    split,
                    task,
                    ep,
                    task_dir,
                    out_root,
                    args.num_video_frames,
                    args.global_downsample_rate,
                    args.video_action_freq_ratio,
                    args.robotwin_sampling_mode,
                )
                if sample is None:
                    manifest["skipped"].append(
                        {"reason": "prepare_failed", "split": split, "task": task, "episode": ep}
                    )
                    continue
                manifest["samples"].append(sample.__dict__)

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"完成：写入 {len(manifest['samples'])} 条样本 -> {out_root}")
    print(f"Manifest: {out_root / 'manifest.json'}")
    print(f"跳过记录数: {len(manifest['skipped'])}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build a windowed RobotWin2 dataset and augment each sample with Motus forward-dynamics video prediction.

Key requirements (from user):
  - Sampling uses the same indexing logic as RobotWinTaskDataset with `random_start_strict_fill`:
      * choose a random feasible condition frame when possible (prefix constraints)
      * then generate strict-style indices with clamp/repeat to episode end (tail fill by repeating last frame)
  - Sliding windows are NON-overlapping (stride = full physical chunk length).
  - After sampling each window, run Motus inference in "forward dynamics" mode:
      * actions are ground-truth (teacher forcing for action branch)
      * action expert diffusion effectively at T=0
      * video prediction denoises normally to produce predicted frames
  - The predicted frames are saved into the exported sample.

This script is self-contained and does NOT modify existing training scripts.

python "/work/sme-wangr/Motus/data/robotwin2/build_robotwin_fd_augmented_dataset.py" \
  --dataset_dir "/work/sme-wangr/Motus/Dataset/robotwin_dataset" \
  --out_dir "/work/sme-wangr/Motus/Dataset/robotwin_fd_dataset_pos_clean10random50" \
  --n_clean_per_task 10 \
  --n_random_per_task 50 \
  --seed 0 \
  --ckpt "/work/sme-wangr/Motus/checkpoints/robotwin/robotwin_20260329_050331/checkpoint_step_40000" \
  --wan_path "/work/sme-wangr/Motus/checkpoints/pretrained_models/Wan2.2-TI2V-5B" \
  --vlm_path "/work/sme-wangr/Motus/checkpoints/pretrained_models/Qwen3-VL-2B-Instruct" \
  --num_inference_steps 10 \
  --device cuda \
  --log_every 20 \
  --save_predicted_tokens 1 \
  --save_predicted_frames 1 \
  --save_und_tokens 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor

# Ensure `Motus/` is on sys.path so imports like `data.*`, `utils.*`, `models.*` work
MOTUS_ROOT = Path(__file__).resolve().parents[2]  # .../Motus
if str(MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTUS_ROOT))

from data.utils.image_utils import load_video_frames, get_video_frame_count, tensor_to_pil
from utils.vlm_utils import preprocess_vlm_messages

from models.motus import Motus, MotusConfig
from models.motus_fd import (
    inference_video_tokens_given_actions,
    inference_video_tokens_and_und_tokens_given_actions,
    inference_video_given_actions,
)


@dataclass(frozen=True)
class Episode:
    split: str
    task: str
    episode: str
    qpos_path: Path
    video_path: Path
    lang_path: Path
    meta_path: Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("robotwin_fd_export")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if re-imported
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path) for h in logger.handlers):
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)
    return logger


def _scan_task_folder(task_path: Path) -> List[Episode]:
    qpos_dir = task_path / "qpos"
    videos_dir = task_path / "videos"
    umt5_dir = task_path / "umt5_wan"
    metas_dir = task_path / "metas"
    if not (qpos_dir.exists() and videos_dir.exists() and umt5_dir.exists() and metas_dir.exists()):
        return []

    split = task_path.parent.name
    task = task_path.name
    out: List[Episode] = []
    for qpos_file in sorted(qpos_dir.glob("*.pt")):
        ep = qpos_file.stem
        video_file = videos_dir / f"{ep}.mp4"
        lang_file = umt5_dir / f"{ep}.pt"
        meta_file = metas_dir / f"{ep}.txt"
        if video_file.exists() and lang_file.exists() and meta_file.exists():
            out.append(
                Episode(
                    split=split,
                    task=task,
                    episode=ep,
                    qpos_path=qpos_file,
                    video_path=video_file,
                    lang_path=lang_file,
                    meta_path=meta_file,
                )
            )
    return out


def _list_tasks(dataset_dir: Path, split: str) -> List[Path]:
    base = dataset_dir / split
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def _pick_episodes_per_task(dataset_dir: Path, split: str, n_per_task: int, seed: int) -> List[Episode]:
    rng = random.Random(seed + (0 if split == "clean" else 1000003))
    picked: List[Episode] = []
    for task_path in _list_tasks(dataset_dir, split):
        eps = _scan_task_folder(task_path)
        if not eps:
            continue
        chosen = eps if n_per_task >= len(eps) else rng.sample(eps, n_per_task)
        picked.extend(chosen)
    return picked


def _max_condition_index_min_valid_prefix(
    total_frames: int,
    *,
    num_video_frames: int,
    action_chunk_size: int,
    ratio: int,
    ds: int,
) -> Optional[int]:
    need_valid_actions = 2 * ratio
    if action_chunk_size < need_valid_actions:
        return None
    need_video_slots = min(2, num_video_frames)
    max_c_vid = total_frames - 1 - need_video_slots * ratio * ds
    max_c_act = total_frames - 1 - need_valid_actions * ds
    max_c = min(max_c_vid, max_c_act)
    return None if max_c < 0 else int(max_c)


def _strict_style_indices_from_condition(
    condition_frame_idx: int,
    total_frames: int,
    *,
    num_video_frames: int,
    action_chunk_size: int,
    ratio: int,
    ds: int,
) -> Tuple[List[int], List[int]]:
    action_indices: List[int] = []
    for i in range(action_chunk_size):
        action_idx = condition_frame_idx + (i + 1) * ds
        action_indices.append(min(action_idx, total_frames - 1))
    video_indices: List[int] = []
    for i in range(num_video_frames):
        action_step = (i + 1) * ratio - 1
        video_indices.append(action_indices[action_step] if action_step < len(action_indices) else action_indices[-1])
    return video_indices, action_indices


def _random_start_strict_fill_start(
    total_frames: int,
    *,
    num_video_frames: int,
    action_chunk_size: int,
    ratio: int,
    ds: int,
    rng: random.Random,
) -> int:
    max_c = _max_condition_index_min_valid_prefix(
        total_frames,
        num_video_frames=num_video_frames,
        action_chunk_size=action_chunk_size,
        ratio=ratio,
        ds=ds,
    )
    if max_c is None:
        # Legacy strict fallback: random in [0, max_condition] else 0
        physical_chunk = action_chunk_size * ds
        max_condition_idx = total_frames - physical_chunk - 1
        if max_condition_idx < 0:
            return 0
        return rng.randint(0, int(max_condition_idx))
    return rng.randint(0, int(max_c))


def _iter_nonoverlap_windows(
    total_frames: int,
    *,
    start_condition: int,
    num_video_frames: int,
    action_chunk_size: int,
    ratio: int,
    ds: int,
) -> List[Tuple[int, List[int], List[int]]]:
    """
    Non-overlapping sliding windows using strict clamp/repeat tail fill.
    stride = full physical chunk length (action_chunk_size * ds).
    """
    physical_chunk = action_chunk_size * ds
    windows: List[Tuple[int, List[int], List[int]]] = []
    c = int(start_condition)
    while c < total_frames:
        v_idx, a_idx = _strict_style_indices_from_condition(
            c,
            total_frames,
            num_video_frames=num_video_frames,
            action_chunk_size=action_chunk_size,
            ratio=ratio,
            ds=ds,
        )
        windows.append((c, v_idx, a_idx))
        c += physical_chunk
    return windows


def _load_language_and_instruction(ep: Episode, rng: random.Random) -> Tuple[List[torch.Tensor], str, int]:
    loaded = torch.load(str(ep.lang_path), map_location="cpu")
    if isinstance(loaded, torch.Tensor):
        # Older format (unlikely for robotwin2)
        t5_list = [loaded.squeeze(0) if loaded.dim() == 3 else loaded]
        instr_idx = 0
    elif isinstance(loaded, list):
        instr_idx = rng.randint(0, len(loaded) - 1)
        t5_list = [loaded[instr_idx]]
    else:
        raise ValueError(f"Unsupported language embedding format: {type(loaded)}")

    lines = ep.meta_path.read_text(encoding="utf-8").strip().splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        raise ValueError(f"No instructions in meta: {ep.meta_path}")
    if 0 <= instr_idx < len(lines):
        instruction = lines[instr_idx]
    else:
        instruction = lines[0]
    return t5_list, instruction, instr_idx


def _to_frames_tchw(predicted_frames: torch.Tensor) -> torch.Tensor:
    """
    Motus returns predicted_frames as [B, C, T, H, W] for the default inference.
    Convert to [T, C, H, W] (squeezed batch).
    """
    if predicted_frames.dim() != 5:
        raise ValueError(f"Unexpected predicted_frames shape: {tuple(predicted_frames.shape)}")
    b, c, t, h, w = predicted_frames.shape
    if b != 1:
        raise ValueError("This exporter currently runs inference with batch=1")
    return predicted_frames.squeeze(0).permute(1, 0, 2, 3).contiguous()  # [T,C,H,W]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="/work/sme-wangr/Motus/Dataset/robotwin_dataset")
    ap.add_argument("--out_dir", type=str, default="/work/sme-wangr/Motus/Dataset/robotwin_fd_dataset")
    ap.add_argument("--n_clean_per_task", type=int, default=10)
    ap.add_argument("--n_random_per_task", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)

    # Sampling params (match Motus/configs/robotwin.yaml common)
    ap.add_argument("--num_video_frames", type=int, default=16)
    ap.add_argument("--video_action_freq_ratio", type=int, default=4)
    ap.add_argument("--global_downsample_rate", type=int, default=3)
    ap.add_argument("--video_height", type=int, default=384)
    ap.add_argument("--video_width", type=int, default=320)

    # Inference params
    ap.add_argument("--ckpt", type=str, required=True, help="Motus checkpoint path (checkpoint_step_xxxxx)")
    ap.add_argument("--wan_path", type=str, required=True, help="WAN root path (contains models_t5_umt5..., google/umt5-xxl, Wan2.2_VAE.pth)")
    ap.add_argument("--vlm_path", type=str, required=True, help="VLM checkpoint path for AutoProcessor (e.g., Qwen3-VL-2B-Instruct)")
    ap.add_argument(
        "--motus_yaml",
        type=str,
        default=str(Path("/work/sme-wangr/Motus/configs/robotwin.yaml")),
        help="YAML config used to build MotusConfig (e.g., Motus/configs/robotwin.yaml)",
    )
    ap.add_argument("--num_inference_steps", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_every", type=int, default=20, help="Log every N written samples")
    ap.add_argument(
        "--save_predicted_tokens",
        type=int,
        default=1,
        help="1 to save predicted_video_tokens (default), 0 to skip",
    )
    ap.add_argument(
        "--save_predicted_frames",
        type=int,
        default=0,
        help="1 to also save predicted_frames (decoder output images), 0 to skip (default)",
    )
    ap.add_argument(
        "--save_und_tokens",
        type=int,
        default=0,
        help="1 to also save und_tokens_last (Understanding Expert last-layer tokens), 0 to skip (default)",
    )

    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    samples_root = out_dir / "samples"
    _ensure_dir(samples_root)
    logs_dir = out_dir / "logs"
    _ensure_dir(logs_dir)
    logger = _setup_logger(logs_dir / "build_fd_dataset.log")
    logger.info("Starting build_robotwin_fd_augmented_dataset")
    logger.info("Args: %s", vars(args))

    rng = random.Random(args.seed)

    # Pick episodes
    episodes: List[Episode] = []
    episodes.extend(_pick_episodes_per_task(dataset_dir, "clean", args.n_clean_per_task, args.seed))
    episodes.extend(_pick_episodes_per_task(dataset_dir, "randomized", args.n_random_per_task, args.seed))
    episodes = sorted(episodes, key=lambda e: (e.split, e.task, e.episode))
    logger.info("Picked episodes: %d", len(episodes))

    # Build Motus model (load from checkpoint; do NOT load pretrained backbones here)
    t0 = time.time()
    cfg = yaml.safe_load(Path(args.motus_yaml).read_text(encoding="utf-8"))
    common = cfg["common"]
    model_cfg = cfg["model"]
    # YAML may parse scientific notation like "1e-5" as string; cast eps explicitly.
    und_norm_eps = float(model_cfg.get("und_expert", {}).get("norm_eps", 1e-5))
    action_norm_eps = float(model_cfg.get("action_expert", {}).get("norm_eps", 1e-6))
    mc = MotusConfig(
        # Paths for config loading only; weights come from --ckpt
        wan_checkpoint_path=str(args.wan_path),
        vae_path=str(Path(args.wan_path) / "Wan2.2_VAE.pth"),
        wan_config_path=str(args.wan_path),
        video_precision=model_cfg["wan"]["precision"],
        vlm_checkpoint_path=str(args.vlm_path),
        und_expert_hidden_size=model_cfg.get("und_expert", {}).get("hidden_size", 512),
        und_expert_ffn_dim_multiplier=model_cfg.get("und_expert", {}).get("ffn_dim_multiplier", 4),
        und_expert_norm_eps=und_norm_eps,
        vlm_adapter_input_dim=model_cfg.get("und_expert", {}).get("vlm", {}).get("input_dim", 2048),
        vlm_adapter_projector_type=model_cfg.get("und_expert", {}).get("vlm", {}).get("projector_type", "mlp3x_silu"),
        num_layers=30,
        action_state_dim=common["state_dim"],
        action_dim=common["action_dim"],
        action_expert_dim=model_cfg["action_expert"]["hidden_size"],
        action_expert_ffn_dim_multiplier=model_cfg["action_expert"]["ffn_dim_multiplier"],
        action_expert_norm_eps=action_norm_eps,
        global_downsample_rate=common["global_downsample_rate"],
        video_action_freq_ratio=common["video_action_freq_ratio"],
        num_video_frames=common["num_video_frames"],
        video_height=common["video_height"],
        video_width=common["video_width"],
        batch_size=1,
        video_loss_weight=model_cfg["loss_weights"]["video_loss_weight"],
        action_loss_weight=model_cfg["loss_weights"]["action_loss_weight"],
        training_mode="finetune",
        load_pretrained_backbones=False,
    )
    device = torch.device(args.device)
    model = Motus(mc).to(device)
    model.load_checkpoint(str(args.ckpt), strict=False)
    model.eval()
    logger.info("Loaded Motus checkpoint in %.1fs", time.time() - t0)

    # VLM processor for preprocessing messages (tokenization only)
    t1 = time.time()
    vlm_processor = AutoProcessor.from_pretrained(str(args.vlm_path), trust_remote_code=True)
    logger.info("Loaded VLM processor in %.1fs", time.time() - t1)

    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    action_chunk_size = int(args.num_video_frames * args.video_action_freq_ratio)
    n_written = 0
    n_attempted = 0
    build_start = time.time()
    last_log_t = build_start

    with manifest_path.open("a", encoding="utf-8") as mf:
        for ep_i, ep in enumerate(episodes):
            try:
                total_frames = int(get_video_frame_count(str(ep.video_path)))
            except Exception:
                logger.warning("Skip episode (video frame count failed): %s/%s/%s", ep.split, ep.task, ep.episode)
                continue
            if total_frames <= 0:
                logger.warning("Skip episode (no frames): %s/%s/%s", ep.split, ep.task, ep.episode)
                continue

            # Choose random start per random_start_strict_fill
            start_c = _random_start_strict_fill_start(
                total_frames,
                num_video_frames=int(args.num_video_frames),
                action_chunk_size=action_chunk_size,
                ratio=int(args.video_action_freq_ratio),
                ds=int(args.global_downsample_rate),
                rng=rng,
            )

            windows = _iter_nonoverlap_windows(
                total_frames,
                start_condition=start_c,
                num_video_frames=int(args.num_video_frames),
                action_chunk_size=action_chunk_size,
                ratio=int(args.video_action_freq_ratio),
                ds=int(args.global_downsample_rate),
            )
            if not windows:
                logger.warning("Skip episode (no windows): %s/%s/%s total_frames=%d", ep.split, ep.task, ep.episode, total_frames)
                continue

            qpos = torch.load(str(ep.qpos_path), map_location="cpu")
            # Load language embedding + matching instruction text
            try:
                t5_list, instruction_text, instr_idx = _load_language_and_instruction(ep, rng)
            except Exception:
                logger.warning("Skip episode (lang/meta load failed): %s/%s/%s", ep.split, ep.task, ep.episode)
                continue

            logger.info(
                "[%d/%d] %s/%s/%s total_frames=%d start_c=%d windows=%d",
                ep_i + 1,
                len(episodes),
                ep.split,
                ep.task,
                ep.episode,
                total_frames,
                int(start_c),
                len(windows),
            )

            for k, (c_idx, v_idx, a_idx) in enumerate(windows):
                try:
                    n_attempted += 1
                    # Clamp condition index to qpos length (consistent with dataset)
                    c_safe = min(int(c_idx), int(qpos.shape[0]) - 1)
                    # NOTE: load_video_frames expects target_size=(H, W)
                    first_frame = load_video_frames(
                        str(ep.video_path),
                        [c_safe],
                        (int(args.video_height), int(args.video_width)),
                    )
                    first_frame = first_frame.squeeze(0).float()  # [C,H,W]

                    # Sampled (ground-truth) video frames aligned to indices
                    video_frames = load_video_frames(
                        str(ep.video_path),
                        v_idx,
                        (int(args.video_height), int(args.video_width)),
                    ).float()  # [T,C,H,W]

                    # State + actions (strict-fill indices already clamped to total_frames; still clamp to qpos length)
                    initial_state = qpos[c_safe].float()
                    a_idx_safe = [min(int(i), int(qpos.shape[0]) - 1) for i in a_idx]
                    action_sequence = torch.stack([qpos[i].float() for i in a_idx_safe], dim=0)  # [chunk,14]

                    # Build VLM inputs
                    # Match policy's scene prefix style to keep distribution similar
                    scene_prefix = (
                        "The whole scene is in a realistic, industrial art style with three views: "
                        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                        "The aloha robot is currently performing the following task: "
                    )
                    instruction = f"{scene_prefix}{instruction_text}"
                    first_frame_pil: Image.Image = tensor_to_pil(first_frame)
                    vlm_inputs = preprocess_vlm_messages(instruction, first_frame_pil, vlm_processor)

                    # Forward-dynamics video prediction (GT actions, action diffusion T=0)
                    predicted_video_tokens = None
                    predicted_frames = None
                    und_tokens_last = None
                    und_attention_mask = None
                    if int(args.save_predicted_tokens) == 1:
                        if int(args.save_und_tokens) == 1:
                            pred_tokens, und_tok = inference_video_tokens_and_und_tokens_given_actions(
                                model,
                                first_frame=first_frame.unsqueeze(0),
                                state=initial_state.unsqueeze(0),
                                actions=action_sequence.unsqueeze(0),
                                num_inference_steps=int(args.num_inference_steps),
                                language_embeddings=t5_list,
                                vlm_inputs=[vlm_inputs],
                            )
                            predicted_video_tokens = pred_tokens.squeeze(0).detach().to("cpu", dtype=torch.float16)
                            und_tokens_last = und_tok.squeeze(0).detach().to("cpu", dtype=torch.float16)
                        else:
                            pred_tokens = inference_video_tokens_given_actions(
                                model,
                                first_frame=first_frame.unsqueeze(0),
                                state=initial_state.unsqueeze(0),
                                actions=action_sequence.unsqueeze(0),
                                num_inference_steps=int(args.num_inference_steps),
                                language_embeddings=t5_list,
                                vlm_inputs=[vlm_inputs],
                            )
                            predicted_video_tokens = pred_tokens.squeeze(0).detach().to("cpu", dtype=torch.float16)
                    elif int(args.save_und_tokens) == 1:
                        # Need to run the same forward process to obtain und tokens even if video tokens are skipped.
                        pred_tokens, und_tok = inference_video_tokens_and_und_tokens_given_actions(
                            model,
                            first_frame=first_frame.unsqueeze(0),
                            state=initial_state.unsqueeze(0),
                            actions=action_sequence.unsqueeze(0),
                            num_inference_steps=int(args.num_inference_steps),
                            language_embeddings=t5_list,
                            vlm_inputs=[vlm_inputs],
                        )
                        und_tokens_last = und_tok.squeeze(0).detach().to("cpu", dtype=torch.float16)

                    if int(args.save_und_tokens) == 1:
                        try:
                            am = vlm_inputs.get("attention_mask", None)
                            if am is not None:
                                und_attention_mask = am.squeeze(0).detach().to("cpu")
                        except Exception:
                            und_attention_mask = None
                    if int(args.save_predicted_frames) == 1:
                        pred = inference_video_given_actions(
                            model,
                            first_frame=first_frame.unsqueeze(0),
                            state=initial_state.unsqueeze(0),
                            actions=action_sequence.unsqueeze(0),
                            num_inference_steps=int(args.num_inference_steps),
                            language_embeddings=t5_list,
                            vlm_inputs=[vlm_inputs],
                        )
                        predicted_frames = _to_frames_tchw(pred).detach().to("cpu", dtype=torch.float16)

                    sample = {
                        "first_frame": first_frame,  # [C,H,W]
                        "video_frames": video_frames,  # [num_video_frames,C,H,W]
                        "initial_state": initial_state,  # [14]
                        "action_sequence": action_sequence,  # [action_chunk_size,14]
                        "instruction_text": instruction_text,  # raw line from metas/{episode}.txt
                        "instruction": instruction,  # actual instruction used for VLM preprocessing
                        "label": 1,
                        "split": ep.split,
                        "task": ep.task,
                        "episode": ep.episode,
                        "instruction_idx": int(instr_idx),
                        "condition_frame_idx": int(c_safe),
                        "video_indices": [int(x) for x in v_idx],
                        "action_indices": [int(x) for x in a_idx_safe],
                        "sampling_mode": "random_start_strict_fill_nonoverlap",
                    }
                    if predicted_video_tokens is not None:
                        sample["predicted_video_tokens"] = predicted_video_tokens
                    if predicted_frames is not None:
                        sample["predicted_frames"] = predicted_frames
                    if und_tokens_last is not None:
                        sample["und_tokens_last"] = und_tokens_last
                    if und_attention_mask is not None:
                        sample["und_attention_mask"] = und_attention_mask

                    sample_dir = samples_root / ep.split / ep.task / ep.episode
                    _ensure_dir(sample_dir)
                    sample_path = sample_dir / f"sample_{k:06d}.pt"
                    torch.save(sample, str(sample_path))

                    mf.write(
                        json.dumps(
                            {
                                "path": str(sample_path.relative_to(out_dir)),
                                "label": 1,
                                "split": ep.split,
                                "task": ep.task,
                                "episode": ep.episode,
                                "instruction_idx": int(instr_idx),
                                "instruction_text": instruction_text,
                                "condition_frame_idx": int(c_safe),
                                "video_indices": [int(x) for x in v_idx],
                                "action_indices": [int(x) for x in a_idx_safe],
                                "total_frames": int(total_frames),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_written += 1

                    if args.log_every > 0 and (n_written % int(args.log_every) == 0):
                        now = time.time()
                        dt = now - last_log_t
                        total_dt = now - build_start
                        logger.info(
                            "Progress: written=%d attempted=%d (last %d in %.1fs, total %.1fs)",
                            n_written,
                            n_attempted,
                            int(args.log_every),
                            dt,
                            total_dt,
                        )
                        last_log_t = now
                except Exception:
                    # Log a few detailed failures per run to avoid silent "written=0"
                    if n_written == 0 and n_attempted <= 5:
                        logger.warning(
                            "Window failed: %s/%s/%s k=%d c_idx=%s; err=%s",
                            ep.split,
                            ep.task,
                            ep.episode,
                            int(k),
                            str(c_idx),
                            traceback.format_exc(limit=5),
                        )
                    continue

    stats = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "n_clean_per_task": int(args.n_clean_per_task),
        "n_random_per_task": int(args.n_random_per_task),
        "seed": int(args.seed),
        "num_video_frames": int(args.num_video_frames),
        "video_action_freq_ratio": int(args.video_action_freq_ratio),
        "global_downsample_rate": int(args.global_downsample_rate),
        "num_inference_steps": int(args.num_inference_steps),
        "n_samples_written": int(n_written),
        "n_samples_attempted": int(n_attempted),
        "elapsed_sec": float(time.time() - build_start),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Done. written=%d attempted=%d elapsed=%.1fs", n_written, n_attempted, time.time() - build_start)


if __name__ == "__main__":
    main()


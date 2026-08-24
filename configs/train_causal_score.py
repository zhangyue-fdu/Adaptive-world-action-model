#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

# Ensure Motus root on path
MOTUS_ROOT = Path(__file__).resolve().parents[1]
if str(MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTUS_ROOT))

from models.motus import Motus, MotusConfig  # noqa: E402

from causal_attn_score_sample_online.dataset_fd_binary import (  # noqa: E402
    collate_fd_binary,
)
from causal_attn_score_sample_online.fd_online_sampling import OnlineLongFDBatchSampler  # noqa: E402
from causal_attn_score_sample_online.model_causal_score import (  # noqa: E402
    CausalAttnScoreModel,
    CausalScoreConfig,
    MotusVideoVaeTokenEncoder,
)

logger = logging.getLogger("causal_attn_score_sample_online")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _setup_logging(out_dir: Path) -> None:
    _ensure_dir(out_dir)
    log_path = out_dir / "train.log"
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train causal score with online sampling from long FD dataset_root (1:1 pos/neg inside each batch)."
    )
    ap.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Long FD root (manifest.jsonl + samples/). Online pos/neg sampling only.",
    )
    ap.add_argument("--manifest_name", type=str, default="manifest.jsonl")
    ap.add_argument(
        "--cache_sources",
        type=int,
        default=512,
        help="LRU cache size for loaded long .pt sources (online mode).",
    )
    ap.add_argument("--gripper_indices", type=str, default="6,13")
    ap.add_argument("--half_noise_sigma", type=float, default=0.5)
    ap.add_argument(
        "--allow_missing_und",
        action="store_true",
        help="Do not skip sources missing und_tokens_last (may still error in collate if None).",
    )
    ap.add_argument(
        "--results_root",
        type=str,
        default=str(MOTUS_ROOT / "causal_attn_score_sample_online" / "results"),
        help="All outputs will be written under results_root/<timestamp>/",
    )
    ap.add_argument("--seed", type=int, default=0)

    # dataloader (online sampler internally enforces 1:1 pos/neg)
    ap.add_argument("--batch_size", type=int, default=8)

    # model cfg
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--dropout", type=float, default=0.0)

    # training
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=20000)

    # motus/vae encoder (frozen)
    ap.add_argument(
        "--motus_yaml",
        type=str,
        default=str(MOTUS_ROOT / "configs" / "robotwin.yaml"),
        help="Path to Motus YAML config (default: Motus/configs/robotwin.yaml)",
    )
    ap.add_argument("--motus_ckpt", type=str, required=True)
    ap.add_argument("--wan_path", type=str, required=True)
    ap.add_argument("--vlm_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    if args.batch_size % 2 != 0:
        ap.error("--batch_size must be even for 1:1 pos/neg online sampling.")

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_root) / run_ts
    _ensure_dir(out_dir)
    _setup_logging(out_dir)
    logger.info("Starting causal_attn_score_sample_online training (online-only mode)")
    logger.info("out_dir=%s", str(out_dir))
    logger.info("device=%s", str(device))

    # Online sampler over long FD dataset
    online = OnlineLongFDBatchSampler(
        dataset_root=Path(args.dataset_root),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        gripper_indices=str(args.gripper_indices),
        half_noise_sigma=float(args.half_noise_sigma),
        cache_sources=int(args.cache_sources),
        manifest_name=str(args.manifest_name),
        require_und_tokens=not bool(args.allow_missing_und),
    )
    logger.info(
        "Sampling: ONLINE long FD root=%s N_sources=%d batch_size=%d (1:1 pos/neg)",
        args.dataset_root,
        online.N,
        int(args.batch_size),
    )

    def get_batch() -> Dict[str, Any]:
        return online.next_batch(collate_fd_binary)

    # Load Motus (frozen) to reuse VAE encoder + tokenization
    import yaml

    cfg = yaml.safe_load(Path(args.motus_yaml).read_text(encoding="utf-8"))
    common = cfg["common"]
    model_cfg = cfg["model"]
    und_norm_eps = float(model_cfg.get("und_expert", {}).get("norm_eps", 1e-5))
    action_norm_eps = float(model_cfg.get("action_expert", {}).get("norm_eps", 1e-6))
    mc = MotusConfig(
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
    motus = Motus(mc).to(device)
    motus.load_checkpoint(str(args.motus_ckpt), strict=False)
    motus.eval()
    for p in motus.parameters():
        p.requires_grad_(False)
    vae_enc = MotusVideoVaeTokenEncoder(motus)

    # Classifier model
    mcfg = CausalScoreConfig(
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
    )
    model = CausalAttnScoreModel(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # Save config
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "model_cfg": asdict(mcfg),
                "out_dir": str(out_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote config.json")

    t0 = time.time()
    n = 0
    while n < int(args.steps):
        batch = get_batch()

        # Move to device
        und_tokens = batch["und_tokens_last"].to(device, non_blocking=True)
        und_valid = batch["und_padding_mask"].to(device, non_blocking=True)
        actions17 = batch["action_sequence_17"].to(device, non_blocking=True)
        pred_hist = batch["predicted_token_block"].to(device, non_blocking=True)
        pred_fut = batch["predicted_token_block_future"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # Encode GT video frames to tokens using Motus VAE encoder
        gt_video_tokens = vae_enc.encode_image1_to_tokens(batch["video_frame_1"].to(device, non_blocking=True))

        out = model(
            und_tokens_last=und_tokens,
            und_padding_mask=und_valid,
            gt_video_tokens=gt_video_tokens,
            pred_hist_tokens=pred_hist,
            pred_future_tokens=pred_fut,
            actions17=actions17,
            labels=labels,
        )
        loss = out["loss"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        n += 1
        if int(args.log_every) > 0 and (n % int(args.log_every) == 0):
            dt = time.time() - t0
            logger.info("step=%d loss=%.6f dt=%.1fs", int(n), float(loss.item()), float(dt))

        if int(args.save_every) > 0 and (n % int(args.save_every) == 0):
            ckpt = {
                "step": int(n),
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "model_cfg": asdict(mcfg),
                "args": vars(args),
            }
            ckpt_path = out_dir / f"ckpt_step_{n:07d}.pt"
            torch.save(ckpt, str(ckpt_path))
            logger.info("Saved checkpoint: %s", str(ckpt_path))


if __name__ == "__main__":
    main()


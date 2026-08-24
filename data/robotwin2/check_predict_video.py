#!/usr/bin/env python3
"""
Decode `predicted_video_tokens` from an FD augmented `.pt` sample back to RGB frames.

Tokens are the output of `VideoModule.prepare_input(latent)` (Conv3d patch embedding + flatten),
same as saved by `build_robotwin_fd_augmented_dataset.py`. We invert patch embedding with
`conv_transpose3d` (subtract Conv3d bias first), then run `WanVideoModel.decode_video`.

python data/robotwin2/check_predict_video.py \
  --sample_pt /data/250010061/Motus/Dataset/robotwin_fd_dataset_hangingmug/randomized/sample_000000.pt \
  --out_dir /data/250010061/Motus/Dataset/check/predict_video \
  --ckpt /data/share/250010061/Motus/checkpoints/robotwin/robotwin_20260329_050331/checkpoint_step_40000 \
  --wan_path /data/share/250010061/Motus/checkpoints/pretrained_models/Wan2.2-TI2V-5B \
  --vlm_path /data/share/250010061/Motus/checkpoints/pretrained_models/Qwen3-VL-2B-Instruct \
  --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

MOTUS_ROOT = Path(__file__).resolve().parents[2]
if str(MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTUS_ROOT))

from models.motus import Motus, MotusConfig  # noqa: E402


def _tensor01_to_pil(x: torch.Tensor) -> Image.Image:
    """x: [3,H,W] float in [0,1]."""
    x = x.detach().float().cpu().clamp(0, 1)
    if x.dim() != 3:
        raise ValueError(f"expected [3,H,W], got {tuple(x.shape)}")
    arr = (x * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def patch_tokens_to_video_latent(
    model: Motus,
    tokens: torch.Tensor,
    first_frame: torch.Tensor,
    *,
    num_video_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Infer (Hp,Wp) from VAE-encoded condition frame; invert patch embedding to latent."""
    wan = model.video_model.wan_model
    pe = wan.patch_embedding
    lat_t = 1 + int(num_video_frames) // 4
    L, D = int(tokens.shape[0]), int(tokens.shape[1])
    if D != pe.out_channels:
        raise ValueError(f"token dim {D} != patch_embedding.out_channels {pe.out_channels}")

    ff = first_frame.to(device=device, dtype=dtype)
    if ff.dim() == 3:
        ff = ff.unsqueeze(0)
    ffn = (ff * 2.0 - 1.0).unsqueeze(2)
    with torch.no_grad():
        ce = model.video_model.encode_video(ffn)
    _, _c, _t0, hp, wp = ce.shape
    if _t0 != 1:
        raise ValueError(f"expected condition latent T=1, got {_t0}")
    if lat_t * hp * wp != L:
        raise ValueError(
            f"token length L={L} != lat_t*hp*wp={lat_t}*{hp}*{wp}={lat_t * hp * wp}. "
            "Check num_video_frames / sample compatibility."
        )

    x = tokens.to(device=device, dtype=dtype).unsqueeze(0)  # [1, L, D]
    x = x.transpose(1, 2).contiguous().view(1, pe.out_channels, lat_t, hp, wp)
    if pe.bias is not None:
        x = x - pe.bias.to(dtype=dtype, device=device).view(1, -1, 1, 1, 1)

    latent = F.conv_transpose3d(
        x,
        pe.weight,
        bias=None,
        stride=pe.stride,
        padding=pe.padding,
        dilation=pe.dilation,
        groups=1,
    )
    return latent


def decode_latent_to_pixels(model: Motus, video_latent: torch.Tensor) -> torch.Tensor:
    """[B,C,T,H,W] in [0,1], includes condition frame in time dim; caller may drop t=0."""
    with torch.no_grad():
        dec = model.video_model.decode_video(video_latent)
    out = (dec + 1.0) / 2.0
    return torch.clamp(out.float(), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--motus_yaml",
        type=str,
        default=str(MOTUS_ROOT / "configs" / "robotwin.yaml"),
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="/work/sme-wangr/Motus/checkpoints/robotwin/robotwin_20260329_050331/checkpoint_step_40000",
    )
    ap.add_argument(
        "--wan_path",
        type=str,
        default="/work/sme-wangr/Motus/checkpoints/pretrained_models/Wan2.2-TI2V-5B",
    )
    ap.add_argument(
        "--vlm_path",
        type=str,
        default="/work/sme-wangr/Motus/checkpoints/pretrained_models/Qwen3-VL-2B-Instruct",
    )
    ap.add_argument(
        "--sample_pt",
        type=str,
        default="/work/sme-wangr/Motus/Dataset/robotwin_fd_dataset_all_40k/samples/randomized/hanging_mug/0/sample_000000.pt",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/work/sme-wangr/Motus/Dataset/check/predict_video",
    )
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.motus_yaml).read_text(encoding="utf-8"))
    common = cfg["common"]
    model_cfg = cfg["model"]
    und_norm_eps = float(model_cfg.get("und_expert", {}).get("norm_eps", 1e-5))
    action_norm_eps = float(model_cfg.get("action_expert", {}).get("norm_eps", 1e-6))

    mc = MotusConfig(
        wan_checkpoint_path=str(Path(args.wan_path)),
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

    model = Motus(mc).to(device)
    model.load_checkpoint(str(args.ckpt), strict=False)
    model.eval()

    sample = torch.load(str(args.sample_pt), map_location="cpu", weights_only=False)
    if "predicted_video_tokens" not in sample:
        raise KeyError("sample missing key predicted_video_tokens")
    if "first_frame" not in sample:
        raise KeyError("sample missing key first_frame (needed to infer latent spatial size)")

    tokens = sample["predicted_video_tokens"].float()
    first_frame = sample["first_frame"].float()

    # Match Motus / WAN compute dtype
    wan_dtype = model.video_model.precision
    tokens_d = tokens.to(device=device, dtype=wan_dtype)
    ff_d = first_frame.to(device=device, dtype=wan_dtype)

    latent = patch_tokens_to_video_latent(
        model,
        tokens_d,
        ff_d,
        num_video_frames=int(common["num_video_frames"]),
        device=device,
        dtype=wan_dtype,
    )
    pixels = decode_latent_to_pixels(model, latent)

    # Align with training decode: drop condition frame in pixel space (first temporal slice)
    pred = pixels[:, :, 1:, :, :]
    b, c, t, h, w = pred.shape
    assert b == 1
    stem = Path(args.sample_pt).stem
    for ti in range(t):
        pil = _tensor01_to_pil(pred[0, :, ti])
        out_path = out_dir / f"{stem}_pred_{ti:03d}.png"
        pil.save(str(out_path))
    meta = {
        "sample_pt": str(args.sample_pt),
        "ckpt": str(args.ckpt),
        "latent_shape": list(latent.shape),
        "pixels_shape": list(pixels.shape),
        "pred_frames_saved": int(t),
        "token_shape": list(tokens.shape),
    }
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", t, "frames under", str(out_dir))


if __name__ == "__main__":
    main()

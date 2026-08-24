from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CausalScoreConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # token dims
    wan_token_dim: int = 3072
    und_dim: int = 512
    action_dim: int = 14

    # fixed token counts in this exported dataset
    gt_video_block_tokens: int = 120
    pred_video_block_tokens: int = 120
    action_len: int = 17


class PreNormTransformerBlock(nn.Module):
    """
    Lightweight pre-norm block (LN -> MHA -> residual, LN -> MLP -> residual).
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float, attn_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [B,L,D]
        *,
        attn_mask: Optional[torch.Tensor] = None,  # [L,L] bool OR float
        key_padding_mask: Optional[torch.Tensor] = None,  # [B,L] True=pad
    ) -> torch.Tensor:
        h = self.norm1(x)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(y)
        h2 = self.norm2(x)
        x = x + self.drop2(self.mlp(h2))
        return x


def build_group_causal_attn_mask(
    *,
    und_len: int,
    gt_len: int,
    pred_hist_len: int,
    action_len: int,
    pred_future_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build boolean attention mask for MultiheadAttention.

    PyTorch MultiheadAttention expects:
      - attn_mask shape [L,L] where True indicates positions that are NOT allowed (for bool masks).

    Sequence layout:
      [ und_tokens | gt_video_tokens | pred_hist_tokens | action_tokens(0..T-1) | pred_future_tokens | cls ]

    Rules from user:
      1) und + gt + pred_hist can attend to each other freely (within this group only)
      2) action at time t can attend to group1 + actions[0..t] (inclusive)
      3) pred_future tokens can attend to group1 + all actions + within pred_future
      4) cls can attend to everything
    """
    g1 = und_len + gt_len + pred_hist_len
    a0 = g1
    a1 = a0 + action_len
    f0 = a1
    f1 = f0 + pred_future_len
    cls = f1
    L = cls + 1

    allow = torch.zeros((L, L), dtype=torch.bool, device=device)

    # group1: can attend within group1
    allow[:g1, :g1] = True

    # action tokens: time-causal within actions + can see group1
    for t in range(action_len):
        q = a0 + t
        allow[q, :g1] = True
        allow[q, a0 : (a0 + t + 1)] = True

    # future predicted tokens: can see group1 + all actions + within future block
    allow[f0:f1, :g1] = True
    allow[f0:f1, a0:a1] = True
    allow[f0:f1, f0:f1] = True

    # cls: can see all
    allow[cls, : (cls + 1)] = True

    # Disallow all other attention edges
    disallow = ~allow
    return disallow  # True means masked out


class ActionTokenEncoder(nn.Module):
    """
    Encode 17x14 actions into 17 tokens.
    """

    def __init__(self, action_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model, d_model),
        )
        self.pos = nn.Parameter(torch.zeros(1, 17, d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, actions17: torch.Tensor) -> torch.Tensor:
        # actions17: [B,17,14]
        x = self.proj(actions17)
        return x + self.pos[:, : x.shape[1]]


class CausalAttnScoreModel(nn.Module):
    """
    Binary classifier with group-causal attention over:
      und tokens, GT video tokens, predicted history tokens, action tokens, predicted future tokens, CLS.
    """

    def __init__(self, cfg: CausalScoreConfig):
        super().__init__()
        self.cfg = cfg

        # Projections to d_model
        self.proj_und = nn.Linear(cfg.und_dim, cfg.d_model)
        self.proj_wan = nn.Linear(cfg.wan_token_dim, cfg.d_model)
        self.action_enc = ActionTokenEncoder(cfg.action_dim, cfg.d_model)

        # CLS token
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.cls, std=0.02)

        # Transformer
        self.blocks = nn.ModuleList(
            [
                PreNormTransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attn_dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)

        # Head
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(
        self,
        *,
        und_tokens_last: torch.Tensor,  # [B,Lu,512] padded
        und_padding_mask: torch.Tensor,  # [B,Lu] True=valid
        gt_video_tokens: torch.Tensor,  # [B,120,3072]
        pred_hist_tokens: torch.Tensor,  # [B,120,3072]
        pred_future_tokens: torch.Tensor,  # [B,120,3072]
        actions17: torch.Tensor,  # [B,17,14]
        labels: Optional[torch.Tensor] = None,  # [B]
    ) -> Dict[str, torch.Tensor]:
        B = actions17.shape[0]
        device = actions17.device
        dtype = self.cls.dtype

        # Unify dtypes for projections/attention (avoid Half vs Float matmul errors)
        und_tokens_last = und_tokens_last.to(device=device, dtype=dtype)
        gt_video_tokens = gt_video_tokens.to(device=device, dtype=dtype)
        pred_hist_tokens = pred_hist_tokens.to(device=device, dtype=dtype)
        pred_future_tokens = pred_future_tokens.to(device=device, dtype=dtype)
        actions17 = actions17.to(device=device, dtype=dtype)

        # Project tokens
        und = self.proj_und(und_tokens_last)  # [B,Lu,D]
        gt = self.proj_wan(gt_video_tokens)  # [B,120,D]
        ph = self.proj_wan(pred_hist_tokens)  # [B,120,D]
        pf = self.proj_wan(pred_future_tokens)  # [B,120,D]
        a = self.action_enc(actions17)  # [B,17,D]

        cls = self.cls.expand(B, -1, -1)  # [B,1,D]
        x = torch.cat([und, gt, ph, a, pf, cls], dim=1)  # [B,L,D]

        Lu = int(und.shape[1])
        attn_mask = build_group_causal_attn_mask(
            und_len=Lu,
            gt_len=int(gt.shape[1]),
            pred_hist_len=int(ph.shape[1]),
            action_len=int(a.shape[1]),
            pred_future_len=int(pf.shape[1]),
            device=device,
        )

        # Key padding mask: True means "pad and should be ignored".
        # Only und has padding; everything else is valid.
        L = x.shape[1]
        key_padding_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        # und_padding_mask True=valid => pad positions are ~valid
        key_padding_mask[:, :Lu] = ~und_padding_mask.to(torch.bool)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        cls_out = x[:, -1, :]  # [B,D]
        logits = self.head(cls_out).squeeze(-1)  # [B]
        probs = torch.sigmoid(logits)

        out: Dict[str, torch.Tensor] = {"logits": logits, "probs": probs}
        if labels is not None:
            labels_f = labels.float()
            loss = F.binary_cross_entropy_with_logits(logits, labels_f)
            out["loss"] = loss
        return out


class MotusVideoVaeTokenEncoder(nn.Module):
    """
    Utility to reuse Motus' VAE encoder + tokenization (prepare_input) to produce WAN-space tokens.

    Produces a token block [B, 120, 3072] for a single GT frame by encoding it to latent and then tokenizing.
    """

    def __init__(self, motus_model) -> None:
        super().__init__()
        self.motus = motus_model
        # Make explicit: we only use encoder side; keep frozen by default outside.

    @torch.no_grad()
    def encode_image1_to_tokens(self, video_frame_1: torch.Tensor) -> torch.Tensor:
        """
        video_frame_1: [B,3,H,W] in [0,1]
        Returns: [B, 120, 3072]
        """
        if video_frame_1.dim() != 4 or video_frame_1.shape[1] != 3:
            raise ValueError(f"Expected video_frame_1 [B,3,H,W], got {tuple(video_frame_1.shape)}")

        # Motus VAE expects [B,C,T,H,W] in [-1,1]
        x = (video_frame_1 * 2.0 - 1.0).unsqueeze(2).contiguous()  # [B,3,1,H,W]
        latent = self.motus.video_model.encode_video(x.to(self.motus.dtype))  # [B,48,T',H',W']
        tokens = self.motus.video_module.prepare_input(latent.to(self.motus.dtype))  # [B,L,3072]
        return tokens


from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FDSampleRef:
    root: Path
    relpath: str
    label: int  # 1 pos, 0 neg
    meta: Dict[str, Any]

    @property
    def path(self) -> Path:
        return self.root / self.relpath


def _read_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_fd_binary_index(
    *,
    pos_root: Path,
    neg_root: Path,
    max_pos: Optional[int] = None,
    max_neg: Optional[int] = None,
) -> List[FDSampleRef]:
    """
    Build a single index list from:
      - pos_root/manifest.jsonl (label=1)
      - neg_root/manifest.jsonl (label=0)
    """
    pos_manifest = pos_root / "manifest.jsonl"
    neg_manifest = neg_root / "manifest.jsonl"
    if not pos_manifest.exists():
        raise FileNotFoundError(f"pos manifest not found: {pos_manifest}")
    if not neg_manifest.exists():
        raise FileNotFoundError(f"neg manifest not found: {neg_manifest}")

    pos_rows = _read_manifest(pos_manifest)
    neg_rows = _read_manifest(neg_manifest)
    if max_pos is not None:
        pos_rows = pos_rows[: int(max_pos)]
    if max_neg is not None:
        neg_rows = neg_rows[: int(max_neg)]

    out: List[FDSampleRef] = []
    for r in pos_rows:
        out.append(FDSampleRef(root=pos_root, relpath=str(r["path"]), label=1, meta=dict(r)))
    for r in neg_rows:
        out.append(FDSampleRef(root=neg_root, relpath=str(r["path"]), label=0, meta=dict(r)))
    return out


class RobotWinFDBinaryDataset(Dataset):
    """
    Binary dataset for causal-attention scoring.

    Each item yields:
      - video_frame_1: [3,H,W] float tensor
      - action_sequence_17: [17,14]
      - predicted_token_block: [120,3072]  (history)
      - predicted_token_block_future: [120,3072] (future)
      - und_tokens_last: [Lu,512] (or [B,Lu,512] in older exports; we normalize to 2D here)
      - und_attention_mask: [Lu] bool/int (optional, may be None)
      - label: int64 scalar
      - meta: dict (split/task/episode/neg_type/etc.)
    """

    def __init__(
        self,
        *,
        index: Sequence[FDSampleRef],
        require_und_tokens: bool = True,
        require_future_tokens: bool = True,
    ) -> None:
        self.index = list(index)
        self.require_und_tokens = bool(require_und_tokens)
        self.require_future_tokens = bool(require_future_tokens)

    def __len__(self) -> int:
        return len(self.index)

    def _normalize_und(self, und_tokens_last: torch.Tensor) -> torch.Tensor:
        # Allow [1,Lu,512] or [Lu,512]
        if und_tokens_last.dim() == 3 and und_tokens_last.shape[0] == 1:
            und_tokens_last = und_tokens_last.squeeze(0)
        if und_tokens_last.dim() != 2:
            raise ValueError(f"und_tokens_last must be 2D [Lu,Du], got {tuple(und_tokens_last.shape)}")
        return und_tokens_last

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ref = self.index[idx]
        if not ref.path.exists():
            raise FileNotFoundError(f"sample missing: {ref.path}")
        d = torch.load(str(ref.path), map_location="cpu")

        # Core fields
        img1 = d.get("video_frame_1")
        actions17 = d.get("action_sequence_17")
        tok_hist = d.get("predicted_token_block")
        tok_fut = d.get("predicted_token_block_future")

        if img1 is None or actions17 is None or tok_hist is None:
            raise ValueError(f"Sample missing required keys at {ref.path}")
        if self.require_future_tokens and tok_fut is None:
            raise ValueError(f"Sample missing predicted_token_block_future at {ref.path}")

        # Understanding fields
        und_tokens_last = d.get("und_tokens_last")
        und_attention_mask = d.get("und_attention_mask")
        if self.require_und_tokens and und_tokens_last is None:
            raise ValueError(f"Sample missing und_tokens_last at {ref.path}")
        if und_tokens_last is not None:
            if not isinstance(und_tokens_last, torch.Tensor):
                raise ValueError(f"und_tokens_last is not tensor at {ref.path}: {type(und_tokens_last)}")
            und_tokens_last = self._normalize_und(und_tokens_last)

        # Convert mask to 1D if possible
        if und_attention_mask is not None:
            if not isinstance(und_attention_mask, torch.Tensor):
                raise ValueError(f"und_attention_mask is not tensor at {ref.path}: {type(und_attention_mask)}")
            if und_attention_mask.dim() == 2 and und_attention_mask.shape[0] == 1:
                und_attention_mask = und_attention_mask.squeeze(0)
            if und_attention_mask.dim() != 1:
                # keep but warn via exception for strictness
                raise ValueError(
                    f"und_attention_mask must be 1D [Lu], got {tuple(und_attention_mask.shape)} at {ref.path}"
                )

        y = torch.tensor(int(ref.label), dtype=torch.long)
        return {
            "video_frame_1": img1,
            "action_sequence_17": actions17,
            "predicted_token_block": tok_hist,
            "predicted_token_block_future": tok_fut,
            "und_tokens_last": und_tokens_last,
            "und_attention_mask": und_attention_mask,
            "label": y,
            "meta": ref.meta,
            "path": str(ref.path),
        }


def collate_fd_binary(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate with padding for und_tokens_last.
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")

    img1 = torch.stack([b["video_frame_1"] for b in batch], dim=0)  # [B,3,H,W]
    actions17 = torch.stack([b["action_sequence_17"] for b in batch], dim=0)  # [B,17,14]
    tok_hist = torch.stack([b["predicted_token_block"] for b in batch], dim=0)  # [B,120,3072]
    tok_fut = torch.stack([b["predicted_token_block_future"] for b in batch], dim=0)  # [B,120,3072]
    labels = torch.stack([b["label"] for b in batch], dim=0)  # [B]

    und_list = [b.get("und_tokens_last") for b in batch]
    if any(u is None for u in und_list):
        raise ValueError("und_tokens_last missing in some batch items (set require_und_tokens=True in dataset)")

    und_tokens = [u for u in und_list if isinstance(u, torch.Tensor)]
    und_lens = [int(u.shape[0]) for u in und_tokens]
    max_len = max(und_lens) if und_lens else 0
    du = int(und_tokens[0].shape[1]) if und_tokens else 0

    und_padded = torch.zeros((len(batch), max_len, du), dtype=und_tokens[0].dtype)
    und_pad_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)  # True for valid tokens
    for i, u in enumerate(und_tokens):
        L = int(u.shape[0])
        und_padded[i, :L] = u
        und_pad_mask[i, :L] = True

    # If the original und_attention_mask exists, keep it too (padded to max_len) for exact reconstruction.
    und_attn_mask_list = [b.get("und_attention_mask") for b in batch]
    und_attn_mask_padded = None
    if all(isinstance(m, torch.Tensor) for m in und_attn_mask_list):
        und_attn_mask_padded = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, m in enumerate(und_attn_mask_list):
            m = m.to(torch.bool)
            L = int(m.numel())
            und_attn_mask_padded[i, :L] = m

    return {
        "video_frame_1": img1,
        "action_sequence_17": actions17,
        "predicted_token_block": tok_hist,
        "predicted_token_block_future": tok_fut,
        "und_tokens_last": und_padded,
        "und_padding_mask": und_pad_mask,  # True=valid
        "und_attention_mask": und_attn_mask_padded,  # True=valid (if provided), else None
        "label": labels,
        "meta": [b.get("meta", {}) for b in batch],
        "path": [b.get("path") for b in batch],
    }


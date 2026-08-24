"""
Online sampling from long-sequence FD augmented samples (e.g. Motus/Dataset/robotwin_fd_dataset_all).

Builds training examples on-the-fly using the same windowing / token alignment as
build_robotwin_fd_positive_dataset.py and negative augmentations as
build_robotwin_fd_negative_dataset.py (without materializing sliced .pt files).

For use only inside causal_attn_score_sample_online/.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _token_block_for_time(tokens: torch.Tensor, time_idx_overall: int) -> torch.Tensor:
    start = int(time_idx_overall) * 120
    end = start + 120
    return tokens[start:end]


def _has_tail_padding_by_action_indices(action_indices_window: List[int]) -> bool:
    if not action_indices_window or len(action_indices_window) < 2:
        return False
    for i in range(len(action_indices_window) - 1):
        if int(action_indices_window[i]) == int(action_indices_window[i + 1]):
            return True
    return False


def _parse_gripper_indices(s: str) -> List[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [int(p) for p in parts]


def enumerate_valid_action_starts(d: Dict[str, Any]) -> List[int]:
    video_frames: torch.Tensor = d["video_frames"]
    action_seq: torch.Tensor = d["action_sequence"]
    tokens: torch.Tensor = d["predicted_video_tokens"]

    if video_frames.dim() != 4 or video_frames.shape[0] < 16:
        return []
    if action_seq.dim() != 2 or action_seq.shape[0] < 64:
        return []
    if tokens.dim() != 2 or tokens.shape[0] != 600:
        return []

    start0 = 15
    stride = 16
    win_len = 17
    max_start = int(action_seq.shape[0]) - win_len
    src_action_indices = d.get("action_indices") or d.get("source_action_indices") or []

    out: List[int] = []
    s = start0
    while s <= max_start:
        if (s + 1) % 4 != 0:
            s += stride
            continue
        v4 = (s + 1) // 4 - 1
        v_start = v4 - 3
        if v_start < 0 or (v_start + 4) > video_frames.shape[0]:
            s += stride
            continue
        try:
            idx_window = [int(x) for x in src_action_indices[s : s + win_len]]
        except Exception:
            idx_window = []
        if _has_tail_padding_by_action_indices(idx_window):
            s += stride
            continue
        out.append(int(s))
        s += stride
    return out


def slice_fields_at_s(d: Dict[str, Any], s: int) -> Dict[str, Any]:
    video_frames: torch.Tensor = d["video_frames"]
    action_seq: torch.Tensor = d["action_sequence"]
    tokens: torch.Tensor = d["predicted_video_tokens"]
    win_len = 17

    v4 = (s + 1) // 4 - 1
    v_start = v4 - 3
    gt_video_1 = video_frames[v_start + 3].contiguous()
    gt_actions_17 = action_seq[s : s + win_len].contiguous()

    correct_pred_slice = 1 + (v_start // 4)
    correct_pred_slice = max(1, min(4, int(correct_pred_slice)))
    token_hist = _token_block_for_time(tokens, correct_pred_slice).contiguous()
    future_pred_slice = min(correct_pred_slice + 1, 4)
    token_future = _token_block_for_time(tokens, future_pred_slice).contiguous()

    return {
        "video_frame_1": gt_video_1,
        "action_sequence_17": gt_actions_17,
        "predicted_token_block": token_hist,
        "predicted_token_block_future": token_future,
        "und_tokens_last": d.get("und_tokens_last"),
        "und_attention_mask": d.get("und_attention_mask"),
    }


def normalize_und(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.dim() == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.dim() != 2:
        raise ValueError(f"und_tokens_last must be 2D after normalize, got {tuple(t.shape)}")
    return t


def normalize_und_mask(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if m is None:
        return None
    if m.dim() == 2 and m.shape[0] == 1:
        m = m.squeeze(0)
    if m.dim() != 1:
        raise ValueError(f"und_attention_mask must be 1D, got {tuple(m.shape)}")
    return m


def neg_action_time_swap(actions: torch.Tensor, rng: random.Random, num_swaps: int = 2) -> torch.Tensor:
    a = actions.clone()
    T = a.shape[0]
    for _ in range(max(1, int(num_swaps))):
        i = rng.randrange(0, T)
        j = rng.randrange(0, T)
        if i == j:
            continue
        tmp = a[i].clone()
        a[i] = a[j]
        a[j] = tmp
    return a


def neg_action_joint_swap(
    actions: torch.Tensor,
    rng: random.Random,
    num_timesteps: int = 3,
    num_swaps_per_t: int = 1,
    avoid_dims: Optional[List[int]] = None,
) -> torch.Tensor:
    a = actions.clone()
    T, D = a.shape
    avoid = set(avoid_dims or [])
    dims = [d for d in range(D) if d not in avoid]
    if len(dims) < 2:
        return a
    for _ in range(max(1, int(num_timesteps))):
        t = rng.randrange(0, T)
        for _ in range(max(1, int(num_swaps_per_t))):
            d1, d2 = rng.sample(dims, 2)
            v = a[t, d1].clone()
            a[t, d1] = a[t, d2]
            a[t, d2] = v
    return a


def neg_action_flip_gripper(actions: torch.Tensor, gripper_indices: List[int]) -> torch.Tensor:
    a = actions.clone()
    for gi in gripper_indices:
        if 0 <= gi < a.shape[1]:
            a[:, gi] = -a[:, gi]
    return a


def neg_action_half_noise(actions: torch.Tensor, rng: random.Random, sigma: float) -> torch.Tensor:
    a = actions.clone()
    T = a.shape[0]
    half = T // 2
    noise = torch.randn_like(a[half:]) * float(sigma)
    a[half:] = a[half:] + noise
    return a


NEG_TYPES: Tuple[str, ...] = (
    "action_time_swap",
    "action_joint_swap",
    "action_flip_gripper",
    "action_half_noise",
    "action_swap_between_samples",
)


@dataclass
class _CachedSource:
    d: Dict[str, Any]
    windows: List[int]


class OnlineLongFDBatchSampler:
    """
    1:1 pos/neg batches from a single long-sample FD directory (manifest.jsonl + samples/**.pt).

    Coverage: maintains a permutation over all manifest indices; each time it advances, the next
    *source sample* is taken from this permutation (reshuffled after every full pass). So every
    long .pt is used equally often as the anchor for positive windows over training.

    Negatives: uniform random source, uniform random valid window among NEG_TYPES (uniform),
    each with equal probability.
    """

    def __init__(
        self,
        *,
        dataset_root: Path,
        batch_size: int,
        seed: int,
        gripper_indices: str = "6,13",
        half_noise_sigma: float = 0.5,
        cache_sources: int = 512,
        manifest_name: str = "manifest.jsonl",
        require_und_tokens: bool = True,
    ) -> None:
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even (1:1 pos/neg), got {batch_size}")
        self.root = Path(dataset_root)
        self.batch_size = int(batch_size)
        self.half = self.batch_size // 2
        self.rng = random.Random(int(seed))
        self.gripper_idx = _parse_gripper_indices(gripper_indices)
        self.half_noise_sigma = float(half_noise_sigma)
        self.cache_sources = int(cache_sources)
        self.require_und_tokens = bool(require_und_tokens)

        man = self.root / manifest_name
        if not man.exists():
            raise FileNotFoundError(f"manifest not found: {man}")
        self._rows: List[Dict[str, Any]] = []
        with man.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._rows.append(json.loads(line))
        self.N = len(self._rows)
        if self.N == 0:
            raise ValueError(f"Empty manifest: {man}")

        self._cache: "OrderedDict[int, _CachedSource]" = OrderedDict()
        self._coverage_perm: List[int] = list(range(self.N))
        self.rng.shuffle(self._coverage_perm)
        self._coverage_pos = 0
        self._classified_sources: set[int] = set()
        self._source_has_windows: Dict[int, bool] = {}
        self._coverage_finalized = False
        # After every .pt has been loaded once, restrict pos (and neg) anchors to indices with ≥1 valid window.
        self._eligible_sources: Optional[List[int]] = None

    def _row_path(self, idx: int) -> Path:
        return self.root / str(self._rows[idx]["path"])

    def _get_cached(self, idx: int) -> _CachedSource:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        path = self._row_path(idx)
        if not path.exists():
            raise FileNotFoundError(f"sample missing: {path}")
        d = torch.load(str(path), map_location="cpu")
        wins = enumerate_valid_action_starts(d)
        ent = _CachedSource(d=d, windows=wins)
        self._cache[idx] = ent
        self._cache.move_to_end(idx)
        while len(self._cache) > self.cache_sources:
            self._cache.popitem(last=False)
        if idx not in self._classified_sources:
            self._classified_sources.add(idx)
            self._source_has_windows[idx] = len(ent.windows) > 0
            if len(self._classified_sources) == self.N:
                self._finalize_eligible_pools()
        return ent

    def _finalize_eligible_pools(self) -> None:
        if self._coverage_finalized:
            return
        elig = [i for i in range(self.N) if self._source_has_windows.get(i, False)]
        if not elig:
            raise RuntimeError(
                "All manifest entries lack valid sliding windows (or torch.load failed). Check dataset."
            )
        self._eligible_sources = elig
        self._coverage_perm = list(elig)
        self.rng.shuffle(self._coverage_perm)
        self._coverage_pos = 0
        self._coverage_finalized = True

    def _neg_pick_source_idx(self) -> int:
        pool = self._eligible_sources
        if pool is not None:
            return int(self.rng.choice(pool))
        return int(self.rng.randrange(0, self.N))

    def _next_coverage_source_idx(self) -> int:
        lcov = len(self._coverage_perm)
        sid = int(self._coverage_perm[self._coverage_pos % lcov])
        self._coverage_pos += 1
        if self._coverage_pos % lcov == 0:
            self.rng.shuffle(self._coverage_perm)
        return sid

    def _build_positive_item(self) -> Dict[str, Any]:
        max_tries = max(64, self.N * 4)
        for _ in range(max_tries):
            sid = self._next_coverage_source_idx()
            c = self._get_cached(sid)
            if not c.windows:
                continue
            s = int(self.rng.choice(c.windows))
            core = slice_fields_at_s(c.d, s)
            und = normalize_und(core["und_tokens_last"])
            if self.require_und_tokens and und is None:
                continue
            mask = normalize_und_mask(core["und_attention_mask"])
            return {
                "video_frame_1": core["video_frame_1"],
                "action_sequence_17": core["action_sequence_17"],
                "predicted_token_block": core["predicted_token_block"],
                "predicted_token_block_future": core["predicted_token_block_future"],
                "und_tokens_last": und,
                "und_attention_mask": mask,
                "label": torch.tensor(1, dtype=torch.long),
                "meta": {
                    "online": True,
                    "source_idx": sid,
                    "slice_action_start": s,
                    "neg_type": None,
                },
                "path": str(self._row_path(sid)),
            }
        raise RuntimeError(
            "Could not build a positive item (no valid windows or missing und_tokens). Check dataset."
        )

    def _build_negative_item(self) -> Dict[str, Any]:
        max_tries = max(128, self.N * 8)
        aug_choices = list(NEG_TYPES)
        if self.N < 2:
            aug_choices = [a for a in NEG_TYPES if a != "action_swap_between_samples"]
        elif self._eligible_sources is not None and len(self._eligible_sources) < 2:
            aug_choices = [a for a in NEG_TYPES if a != "action_swap_between_samples"]

        for _ in range(max_tries):
            aug = str(self.rng.choice(aug_choices))

            sid = self._neg_pick_source_idx()
            c = self._get_cached(sid)
            if not c.windows:
                continue
            s = int(self.rng.choice(c.windows))
            core = slice_fields_at_s(c.d, s)
            und = normalize_und(core["und_tokens_last"])
            if self.require_und_tokens and und is None:
                continue
            mask = normalize_und_mask(core["und_attention_mask"])
            actions = core["action_sequence_17"].clone()
            tok_h = core["predicted_token_block"]
            tok_f = core["predicted_token_block_future"]

            meta: Dict[str, Any] = {
                "online": True,
                "source_idx": sid,
                "slice_action_start": s,
                "neg_type": aug,
            }

            if aug == "action_swap_between_samples":
                pool = self._eligible_sources
                if pool is not None and len(pool) >= 2:
                    choices = [i for i in pool if i != sid]
                    sid_b = int(self.rng.choice(choices))
                else:
                    sid_b = int(self.rng.randrange(0, self.N - 1))
                    if sid_b >= sid:
                        sid_b += 1
                cb = self._get_cached(sid_b)
                if not cb.windows:
                    continue
                s_b = int(self.rng.choice(cb.windows))
                partner = slice_fields_at_s(cb.d, s_b)
                actions = partner["action_sequence_17"].clone()
                meta["partner_source_idx"] = sid_b
                meta["partner_slice_action_start"] = s_b
            elif aug == "action_time_swap":
                actions = neg_action_time_swap(actions, self.rng)
            elif aug == "action_joint_swap":
                actions = neg_action_joint_swap(actions, self.rng, avoid_dims=self.gripper_idx)
            elif aug == "action_flip_gripper":
                actions = neg_action_flip_gripper(actions, self.gripper_idx)
            elif aug == "action_half_noise":
                actions = neg_action_half_noise(actions, self.rng, sigma=self.half_noise_sigma)
            else:
                raise RuntimeError(f"Unknown aug {aug}")

            return {
                "video_frame_1": core["video_frame_1"],
                "action_sequence_17": actions,
                "predicted_token_block": tok_h,
                "predicted_token_block_future": tok_f,
                "und_tokens_last": und,
                "und_attention_mask": mask,
                "label": torch.tensor(0, dtype=torch.long),
                "meta": meta,
                "path": str(self._row_path(sid)),
            }

        raise RuntimeError("Could not build a negative item after many tries.")

    def next_batch(self, collate_fn) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        for _ in range(self.half):
            items.append(self._build_positive_item())
        for _ in range(self.half):
            items.append(self._build_negative_item())
        self.rng.shuffle(items)
        return collate_fn(items)

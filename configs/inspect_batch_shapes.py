#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

MOTUS_ROOT = Path(__file__).resolve().parents[1]
if str(MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTUS_ROOT))

from causal_attn_score_sample_online.dataset_fd_binary import (  # noqa: E402
    collate_fd_binary,
)
from causal_attn_score_sample_online.fd_online_sampling import OnlineLongFDBatchSampler  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Print one batch tensor shapes (online long FD only).")
    ap.add_argument("--dataset_root", type=str, required=True, help="Long FD root for online sampling.")
    ap.add_argument("--manifest_name", type=str, default="manifest.jsonl")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.batch_size % 2 != 0:
        ap.error("--batch_size must be even for online 1:1 sampling.")

    sampler = OnlineLongFDBatchSampler(
        dataset_root=Path(args.dataset_root),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        manifest_name=str(args.manifest_name),
    )
    batch = sampler.next_batch(collate_fd_binary)

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, tuple(v.shape), v.dtype)
        elif v is None:
            print(k, None)
        else:
            print(k, type(v), f"len={len(v) if hasattr(v,'__len__') else 'NA'}")


if __name__ == "__main__":
    main()


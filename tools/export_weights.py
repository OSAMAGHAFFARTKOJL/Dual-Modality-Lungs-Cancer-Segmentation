"""Slim the training checkpoint down to deployable inference weights.

    python tools/export_weights.py checkpoints/best.pth weights/cipaformer_v2_ema_fp16.pth

Keeps only the EMA weights and stores them as float16: ~450 MB -> ~118 MB, which
fits comfortably as a GitHub release asset.
"""
import sys
from pathlib import Path

import torch


def main(src: str, dst: str) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state = ckpt.get("ema") or ckpt.get("model") or ckpt
    half = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in state.items()}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema": half, "note": "CIPAFormer-V2 EMA weights, fp16, inference only"}, dst)
    print(f"{src} ({Path(src).stat().st_size/1e6:.0f} MB) -> "
          f"{dst} ({Path(dst).stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pth",
         sys.argv[2] if len(sys.argv) > 2 else "weights/cipaformer_v2_ema_fp16.pth")

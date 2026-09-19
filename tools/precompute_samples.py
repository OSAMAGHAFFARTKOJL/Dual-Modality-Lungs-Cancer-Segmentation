"""Bake every view the bundled sample cases can show into samples/precomputed/.

    python tools/precompute_samples.py

The deployed app serves those arrays directly, so browsing the samples never loads the
network — which is what keeps the app inside a small container's memory budget. Re-run
this whenever the checkpoint, the sample list or the attribution options change.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import metrics as M                      # noqa: E402
from core import explain, inference                # noqa: E402
from core.model import load_model                  # noqa: E402
from core.spec import IMG_SIZE, LAYERS, METHODS    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUT = SAMPLES / "precomputed"


def u16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 65535.0, 0, 65535).astype(np.uint16)


def u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = json.loads((SAMPLES / "index.json").read_text())["cases"]
    model = load_model("cpu")
    methods, layers = list(METHODS), list(LAYERS)

    for case in cases:
        cid = case["id"]
        t0 = time.perf_counter()
        d = SAMPLES / cid
        ct = cv2.imread(str(d / "ct.png"), cv2.IMREAD_GRAYSCALE)
        pet = cv2.imread(str(d / "pet.png"), cv2.IMREAD_GRAYSCALE)
        ct_t, pet_t = inference.preprocess(ct, pet, IMG_SIZE)

        prob, _ = inference.predict(model, ct_t, pet_t, tta=False)
        gates = np.stack([explain.pet_gate_map(model, s, IMG_SIZE) for s in range(4)])
        prob_tta, std_tta = inference.predict(model, ct_t, pet_t, tta=True)
        prob_ct = inference.predict_single_modality(model, ct_t, pet_t, "ct")
        prob_pet = inference.predict_single_modality(model, ct_t, pet_t, "pet")

        cams = np.stack([[explain.cam(model, ct_t, pet_t, layer, method=m)
                          for layer in layers] for m in methods])

        np.savez_compressed(
            OUT / f"{cid}.npz",
            prob=u16(prob), prob_tta=u16(prob_tta), std_tta=u8(std_tta),
            prob_ct=u16(prob_ct), prob_pet=u16(prob_pet),
            gates=u8(gates), cams=u8(cams),
            methods=np.array(methods), layers=np.array(layers))

        gt = M.binarize_mask(cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE))
        size = (OUT / f"{cid}.npz").stat().st_size / 1e6
        print(f"{cid}: dice {M.metrics(prob, gt)['dice']*100:5.1f} | "
              f"{cams.shape[0]}x{cams.shape[1]} cams | {size:.1f} MB | "
              f"{time.perf_counter()-t0:.0f}s")

    print("wrote", OUT)


if __name__ == "__main__":
    main()

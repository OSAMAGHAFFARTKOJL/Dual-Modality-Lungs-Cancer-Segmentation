"""Image prep and segmentation metrics. Pure numpy/OpenCV — importing this must not
pull in torch, because the bundled-sample path runs without a model."""
from __future__ import annotations

import cv2
import numpy as np

from .spec import IMG_SIZE


def to_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    return arr.astype(np.uint8)


def resize(img: np.ndarray, size: int = IMG_SIZE, nearest: bool = False) -> np.ndarray:
    if img.shape[:2] == (size, size):
        return img
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def binarize_mask(mask: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    return (resize(to_gray(mask), size, nearest=True) > 127).astype(np.uint8)


def entropy_map(prob: np.ndarray) -> np.ndarray:
    """Per-pixel binary entropy in bits; 1.0 at p = 0.5."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return (-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).astype(np.float32)


def _surface_distances(pred: np.ndarray, gt: np.ndarray):
    """Symmetric boundary distances in pixels; None when either mask is empty."""
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    pe = cv2.Canny(pred * 255, 100, 200) > 0
    ge = cv2.Canny(gt * 255, 100, 200) > 0
    if pe.sum() == 0 or ge.sum() == 0:
        return None
    dt_g = cv2.distanceTransform((~ge).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((~pe).astype(np.uint8), cv2.DIST_L2, 3)
    return np.concatenate([dt_g[pe], dt_p[ge]])


def metrics(prob: np.ndarray, gt: np.ndarray | None, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(np.uint8)
    out = {
        "pred_px": int(pred.sum()),
        "mean_conf": float(prob[pred > 0].mean()) if pred.sum() else 0.0,
        "max_conf": float(prob.max()),
    }
    if gt is None:
        return out

    p, g = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    safe = lambda a, b: float(a / b) if b else 0.0

    out.update({
        "gt_px": int(g.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "dice": safe(2 * tp, 2 * tp + fp + fn),
        "iou": safe(tp, tp + fp + fn),
        "precision": safe(tp, tp + fp),
        "recall": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "accuracy": safe(tp + tn, tp + tn + fp + fn),
    })

    d = _surface_distances(pred, gt.astype(np.uint8))
    out["hd95"] = float(np.percentile(d, 95)) if d is not None else None
    out["assd"] = float(d.mean()) if d is not None else None
    return out


def sweep_threshold(prob: np.ndarray, gt: np.ndarray, steps: int = 41):
    g = gt.astype(bool)
    rows = []
    for t in np.linspace(0.02, 0.98, steps):
        p = prob >= t
        tp = np.logical_and(p, g).sum()
        denom = p.sum() + g.sum()
        union = np.logical_or(p, g).sum()
        rows.append((float(t),
                     float(2 * tp / denom) if denom else 0.0,
                     float(tp / union) if union else 0.0))
    return rows

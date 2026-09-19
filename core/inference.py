"""Preprocessing, prediction and segmentation metrics."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .model import IMG_SIZE

# Training normalisation: uint8 -> [-1.6, 1.6]
_SCALE, _SHIFT = 3.2, 1.6


def to_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    return arr.astype(np.uint8)


def resize(img: np.ndarray, size: int = IMG_SIZE, nearest: bool = False) -> np.ndarray:
    if img.shape[:2] == (size, size):
        return img
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def preprocess(ct: np.ndarray, pet: np.ndarray, device: str = "cpu"):
    """Grayscale CT/PET -> two 1x3xHxW tensors, exactly as PETCTDataset did."""
    ct_r, pet_r = resize(to_gray(ct)), resize(to_gray(pet))
    stacked = np.stack([pet_r, ct_r], axis=0).astype(np.float32) / 255.0 * _SCALE - _SHIFT
    t = torch.from_numpy(stacked).unsqueeze(0).to(device)
    pet_t = t[:, 0:1].repeat(1, 3, 1, 1)
    ct_t = t[:, 1:2].repeat(1, 3, 1, 1)
    return ct_t, pet_t


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    return (resize(to_gray(mask), nearest=True) > 127).astype(np.uint8)


@torch.no_grad()
def predict(model, ct_t, pet_t, tta: bool = False):
    """Return (probability map HxW, per-view std map HxW)."""
    views = []
    if tta:
        # flipped views first so the module-level activations left behind by the final
        # forward pass (the PET gate maps) belong to the un-flipped image
        views.append(torch.sigmoid(model(ct_t.flip(-1), pet_t.flip(-1))).flip(-1))
        views.append(torch.sigmoid(model(ct_t.flip(-2), pet_t.flip(-2))).flip(-2))
        views.append(torch.sigmoid(model(ct_t.flip(-1, -2), pet_t.flip(-1, -2))).flip(-1, -2))
    views.append(torch.sigmoid(model(ct_t, pet_t)))
    stack = torch.cat(views, dim=0)
    prob = stack.mean(dim=0)[0].cpu().numpy()
    std = stack.std(dim=0)[0].cpu().numpy() if len(views) > 1 else np.zeros_like(prob)
    return prob, std


@torch.no_grad()
def predict_single_modality(model, ct_t, pet_t, keep: str):
    """Prediction with one modality blanked out, for modality-importance analysis."""
    blank_ct = torch.full_like(ct_t, -_SHIFT)
    blank_pet = torch.full_like(pet_t, -_SHIFT)
    if keep == "ct":
        return torch.sigmoid(model(ct_t, blank_pet))[0, 0].cpu().numpy()
    return torch.sigmoid(model(blank_ct, pet_t))[0, 0].cpu().numpy()


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
    ths = np.linspace(0.02, 0.98, steps)
    g = gt.astype(bool)
    rows = []
    for t in ths:
        p = prob >= t
        tp = np.logical_and(p, g).sum()
        denom = p.sum() + g.sum()
        union = np.logical_or(p, g).sum()
        rows.append((float(t),
                     float(2 * tp / denom) if denom else 0.0,
                     float(tp / union) if union else 0.0))
    return rows

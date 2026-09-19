"""Overlay rendering. Everything returns an HxWx3 uint8 RGB array."""
from __future__ import annotations

import cv2
import numpy as np

COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "hot": cv2.COLORMAP_HOT,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "bone": cv2.COLORMAP_BONE,
}

TP_COLOR = (34, 197, 94)     # green  - correct tumour
FP_COLOR = (239, 68, 68)     # red    - false positive
FN_COLOR = (59, 130, 246)    # blue   - missed tumour
GT_COLOR = (34, 197, 94)
PRED_COLOR = (244, 114, 22)


def to_rgb(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)


def window(gray: np.ndarray, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    out = (gray.astype(np.float32) - 127.5) * contrast + 127.5 + brightness * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def colorize(scalar: np.ndarray, cmap: str = "turbo") -> np.ndarray:
    x = np.clip(scalar, 0.0, 1.0)
    bgr = cv2.applyColorMap((x * 255).astype(np.uint8), COLORMAPS.get(cmap, cv2.COLORMAP_TURBO))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def heat_overlay(base: np.ndarray, scalar: np.ndarray, alpha: float = 0.5,
                 cmap: str = "turbo", floor: float = 0.0) -> np.ndarray:
    """Blend a heatmap onto a grayscale or RGB base, fading out where the signal is weak."""
    if base.ndim == 2:
        base = to_rgb(base)
    heat = colorize(scalar, cmap)
    a = (alpha * np.clip((scalar - floor) / max(1e-6, 1 - floor), 0, 1))[..., None]
    return (heat * a + base * (1 - a)).astype(np.uint8)


def mask_overlay(gray: np.ndarray, mask: np.ndarray, color=PRED_COLOR, alpha: float = 0.45,
                 outline: bool = True) -> np.ndarray:
    rgb = to_rgb(gray)
    m = mask.astype(bool)
    tint = np.zeros_like(rgb)
    tint[m] = color
    out = np.where(m[..., None], (alpha * tint + (1 - alpha) * rgb).astype(np.uint8), rgb)
    if outline:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
    return out


def contour_compare(gray: np.ndarray, gt: np.ndarray | None, pred: np.ndarray) -> np.ndarray:
    rgb = to_rgb(gray)
    if gt is not None:
        cg, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cg, -1, GT_COLOR, 2)
    cp, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, cp, -1, PRED_COLOR, 2)
    return rgb


def error_map(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    rgb = to_rgb(gray)
    g, p = gt.astype(bool), pred.astype(bool)
    tint = np.zeros_like(rgb)
    tint[g & p] = TP_COLOR
    tint[~g & p] = FP_COLOR
    tint[g & ~p] = FN_COLOR
    m = (g | p)[..., None]
    return np.where(m, (alpha * tint + (1 - alpha) * rgb).astype(np.uint8), rgb)


def side_by_side(panels: list[tuple[str, np.ndarray]], cols: int = 3, pad: int = 8,
                 label_h: int = 28) -> np.ndarray:
    """Composite labelled panels into one downloadable report image."""
    h, w = panels[0][1].shape[:2]
    rows = int(np.ceil(len(panels) / cols))
    ch, cw = h + label_h + pad, w + pad
    canvas = np.full((rows * ch + pad, cols * cw + pad, 3), 17, np.uint8)
    for i, (title, img) in enumerate(panels):
        r, c = divmod(i, cols)
        y, x = pad + r * ch, pad + c * cw
        cv2.putText(canvas, title, (x + 2, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1,
                    cv2.LINE_AA)
        canvas[y + label_h:y + label_h + h, x:x + w] = img
    return canvas


def png_bytes(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()

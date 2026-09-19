"""Explainability for dense (segmentation) predictions.

Grad-CAM for segmentation differs from the classification recipe: there is no class
logit, so the score being back-propagated is the summed logit energy inside the target
region (predicted tumour by default, ground truth if asked).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

TARGET_LAYERS = {
    "Decoder · up1 (stride 4, sharpest)": lambda m: m.decoder.up1,
    "Decoder · up2 (stride 8)": lambda m: m.decoder.up2,
    "Decoder · up3 (stride 16)": lambda m: m.decoder.up3,
    "Fusion · stage 4 (stride 32, most semantic)": lambda m: m.fusions[3],
    "Fusion · stage 3 (stride 16)": lambda m: m.fusions[2],
    "Fusion · stage 2 (stride 8)": lambda m: m.fusions[1],
}

METHODS = ("Grad-CAM", "HiRes-CAM", "Layer-CAM", "XGrad-CAM")


def _normalise(cam: torch.Tensor) -> np.ndarray:
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).cpu().numpy()


class _Hook:
    def __init__(self, layer):
        self.acts = None
        self.grads = None
        self._f = layer.register_forward_hook(self._fwd)
        self._b = layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, gout):
        self.grads = gout[0].detach()

    def close(self):
        self._f.remove()
        self._b.remove()


def cam(model, ct_t, pet_t, layer_fn, method: str = "Grad-CAM",
        target: np.ndarray | None = None, threshold: float = 0.5) -> np.ndarray:
    """Class-activation map upsampled to the input resolution, normalised to [0, 1]."""
    hook = _Hook(layer_fn(model))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(ct_t, pet_t)
        prob = torch.sigmoid(logits)

        if target is not None:
            region = torch.from_numpy(target.astype(np.float32))[None, None].to(logits.device)
        else:
            region = (prob >= threshold).float()
        score = (logits * region).sum() if region.sum() > 0 else (logits * prob).sum()
        score.backward()

        acts, grads = hook.acts, hook.grads
        if method == "HiRes-CAM":
            weighted = grads * acts
        elif method == "Layer-CAM":
            weighted = F.relu(grads) * acts
        elif method == "XGrad-CAM":
            w = (acts * grads).sum(dim=(2, 3), keepdim=True) / \
                (acts.sum(dim=(2, 3), keepdim=True) + 1e-8)
            weighted = w * acts
        else:
            w = grads.mean(dim=(2, 3), keepdim=True)
            weighted = w * acts

        heat = F.relu(weighted.sum(dim=1, keepdim=True))
        heat = F.interpolate(heat, size=ct_t.shape[2:], mode="bilinear", align_corners=False)
        return _normalise(heat[0, 0])
    finally:
        hook.close()
        model.zero_grad(set_to_none=True)


def pet_gate_map(model, scale: int = 0) -> np.ndarray | None:
    """The learned PET->CT spatial gate at one fusion scale, in [0, 1]."""
    gates = model.pet_gates()
    g = gates[scale]
    if g is None:
        return None
    g = F.interpolate(g, size=(512, 512), mode="bilinear", align_corners=False)
    return g[0, 0].cpu().numpy()


def entropy_map(prob: np.ndarray) -> np.ndarray:
    """Per-pixel binary entropy, normalised so 0.5 probability = 1.0."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return h.astype(np.float32)

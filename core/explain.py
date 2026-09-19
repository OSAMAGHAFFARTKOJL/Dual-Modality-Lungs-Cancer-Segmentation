"""Explainability for dense (segmentation) predictions.

Grad-CAM for segmentation differs from the classification recipe: there is no class
logit, so the score being back-propagated is the summed logit energy inside the target
region (predicted tumour by default, ground truth if asked).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .spec import LAYERS, METHODS  # noqa: F401  (re-exported for the UI)


def _resolve(model, dotted: str):
    obj = model
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


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


def _forward_for(model, ct, pet, dotted: str):
    """Forward pass that only records an autograd graph downstream of the target layer.

    No attribution target lives inside the encoder, and d(score)/d(target activation)
    does not depend on how the target's *inputs* were produced — so running everything
    upstream under no_grad is exact, and it keeps the encoder's activations (by far the
    largest part of the graph at 512x512) out of memory.
    """
    out_size = ct.shape[2:]
    with torch.no_grad():
        cf = model.encoder(ct)
        pf = model.encoder(pet)

    if dotted.startswith("fusions"):
        fused = [model.fusions[i](cf[i].detach(), pf[i].detach()) for i in range(4)]
    else:
        with torch.no_grad():
            fused = [model.fusions[i](cf[i], pf[i]) for i in range(4)]
        fused = [f.detach() for f in fused]

    return model.decoder(fused, out_size)


def cam(model, ct_t, pet_t, layer: str, method: str = "Grad-CAM",
        target: np.ndarray | None = None, threshold: float = 0.5) -> np.ndarray:
    """Class-activation map upsampled to the input resolution, normalised to [0, 1]."""
    dotted = LAYERS[layer]
    hook = _Hook(_resolve(model, dotted))
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = _forward_for(model, ct_t, pet_t, dotted)
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
            weighted = grads.mean(dim=(2, 3), keepdim=True) * acts

        heat = F.relu(weighted.sum(dim=1, keepdim=True))
        heat = F.interpolate(heat, size=ct_t.shape[2:], mode="bilinear", align_corners=False)
        heat = heat[0, 0]
        heat = heat - heat.min()
        return (heat / (heat.max() + 1e-8)).cpu().numpy()
    finally:
        hook.close()
        model.zero_grad(set_to_none=True)


def pet_gate_map(model, scale: int, size: int) -> np.ndarray | None:
    """The learned PET->CT spatial gate at one fusion scale, in [0, 1]."""
    g = model.pet_gates()[scale]
    if g is None:
        return None
    g = F.interpolate(g, size=(size, size), mode="bilinear", align_corners=False)
    return g[0, 0].cpu().numpy()

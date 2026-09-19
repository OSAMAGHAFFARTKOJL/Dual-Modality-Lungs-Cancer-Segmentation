"""Model-dependent inference. Importing this pulls in torch and timm."""
from __future__ import annotations

import numpy as np
import torch

from .metrics import resize, to_gray
from .spec import IMG_SIZE

# Training normalisation: uint8 -> [-1.6, 1.6]
_SCALE, _SHIFT = 3.2, 1.6


def preprocess(ct: np.ndarray, pet: np.ndarray, size: int = IMG_SIZE, device: str = "cpu"):
    """Grayscale CT/PET -> two 1x3xHxW tensors, exactly as PETCTDataset did."""
    ct_r, pet_r = resize(to_gray(ct), size), resize(to_gray(pet), size)
    stacked = np.stack([pet_r, ct_r], axis=0).astype(np.float32) / 255.0 * _SCALE - _SHIFT
    t = torch.from_numpy(stacked).unsqueeze(0).to(device)
    return t[:, 1:2].repeat(1, 3, 1, 1), t[:, 0:1].repeat(1, 3, 1, 1)


@torch.no_grad()
def predict(model, ct_t, pet_t, tta: bool = False):
    """Return (probability map HxW, per-view std map HxW)."""
    views = []
    if tta:
        # flipped views first so the activations left behind by the final forward pass
        # (the PET gate maps) belong to the un-flipped image
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
    if keep == "ct":
        return torch.sigmoid(model(ct_t, torch.full_like(pet_t, -_SHIFT)))[0, 0].cpu().numpy()
    return torch.sigmoid(model(torch.full_like(ct_t, -_SHIFT), pet_t))[0, 0].cpu().numpy()

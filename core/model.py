"""CIPAFormer-V2: PET/CT dual-stream tumour segmentation network.

Architecture must stay byte-identical to the training script or the checkpoint
keys will not line up.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "pvt_v2_b3"
IMG_SIZE = 512

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
WEIGHT_CANDIDATES = (
    WEIGHTS_DIR / "cipaformer_v2_ema_fp16.pth",
    ROOT / "checkpoints" / "best.pth",
)

# Public mirror of the fp16 EMA weights; MODEL_URL overrides it.
DEFAULT_MODEL_URL = ("https://huggingface.co/OsamaGhaffar/lung_segmentation/"
                     "resolve/main/cipaformer_v2_ema_fp16.pth")


class CrossModalFusion(nn.Module):
    """SE channel gate on both streams + a PET-derived spatial gate applied to CT."""

    def __init__(self, dim: int, r: int = 8):
        super().__init__()
        hidden = max(dim * 2 // r, 8)
        self.se = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(inplace=True),
                                nn.Linear(hidden, 2 * dim), nn.Sigmoid())
        self.pet_spatial = nn.Sequential(nn.Conv2d(dim, 1, 7, padding=3), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 3, padding=1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.last_gate = None

    def forward(self, ct, pet):
        C = ct.shape[1]
        z = torch.cat([ct.mean(dim=(2, 3)), pet.mean(dim=(2, 3))], dim=1)
        g = self.se(z)
        gct, gpet = g[:, :C, None, None], g[:, C:, None, None]
        ct = ct + ct * gct
        pet = pet + pet * gpet
        gate = self.pet_spatial(pet)
        self.last_gate = gate.detach()
        ct = ct * (1.0 + gate)
        return self.fuse(torch.cat([ct, pet], dim=1))


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class UNetDecoder(nn.Module):
    def __init__(self, chs, dec=(256, 128, 64), num_classes: int = 1):
        super().__init__()
        c0, c1, c2, c3 = chs
        d2, d1, d0 = dec
        self.up3 = ConvBlock(c3 + c2, d2)
        self.up2 = ConvBlock(d2 + c1, d1)
        self.up1 = ConvBlock(d1 + c0, d0)
        self.pred = nn.Conv2d(d0, num_classes, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, feats, out_size):
        f0, f1, f2, f3 = feats
        x2 = self.up3(torch.cat([self._up(f3, f2), f2], dim=1))
        x1 = self.up2(torch.cat([self._up(x2, f1), f1], dim=1))
        x0 = self.up1(torch.cat([self._up(x1, f0), f0], dim=1))
        return F.interpolate(self.pred(x0), size=out_size, mode="bilinear", align_corners=False)


class CIPAFormerV2(nn.Module):
    def __init__(self, backbone: str = BACKBONE, pretrained: bool = False, num_classes: int = 1):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                         out_indices=(0, 1, 2, 3))
        chs = self.encoder.feature_info.channels()
        self.fusions = nn.ModuleList([CrossModalFusion(c) for c in chs])
        self.decoder = UNetDecoder(chs, num_classes=num_classes)

    def forward(self, ct, pet):
        out_size = ct.shape[2:]
        cf = self.encoder(ct)
        pf = self.encoder(pet)
        fused = [self.fusions[i](cf[i], pf[i]) for i in range(4)]
        return self.decoder(fused, out_size)

    def pet_gates(self):
        """Per-scale PET spatial gate from the last forward pass (1 = PET boosts CT most)."""
        return [f.last_gate for f in self.fusions]


def resolve_weights(progress=None) -> Path:
    """Return a local checkpoint path, downloading it once if configured."""
    for p in WEIGHT_CANDIDATES:
        if p.is_file():
            return p

    url = os.environ.get("MODEL_URL", "").strip()
    if not url:
        try:
            import streamlit as st
            url = str(st.secrets.get("MODEL_URL", "")).strip()
        except Exception:
            url = ""
    url = url or DEFAULT_MODEL_URL

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / "cipaformer_v2_ema_fp16.pth"
    tmp = dest.with_suffix(".part")

    # a token lets the download reach a release asset in a private repo
    token = os.environ.get("MODEL_TOKEN", "").strip()
    if not token:
        try:
            import streamlit as st
            token = str(st.secrets.get("MODEL_TOKEN", "")).strip()
        except Exception:
            token = ""
    headers = {"User-Agent": "cipaformer-demo"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/octet-stream"

    try:
        with urlopen(Request(url, headers=headers)) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(done / total)
    except HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code == 404 and "github.com" in url:
            hint = ("GitHub answers 404 for release assets in a private repo when the request "
                    "is unauthenticated. Host the weights publicly (the Hugging Face Hub works "
                    "well) or add a MODEL_TOKEN secret with repo read access.")
        elif e.code == 404:
            hint = "Nothing is served at that URL — check the path and the file name."
        else:
            hint = "Check that MODEL_URL points straight at the file."
        raise RuntimeError(
            f"Could not download the weights (HTTP {e.code}) from\n\n`{url.split('?')[0]}`\n\n"
            f"{hint}") from None
    except URLError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not reach the weights URL: {e.reason}") from None

    if tmp.stat().st_size < 50_000_000:
        head = tmp.read_bytes()[:200]
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "The weights URL returned something too small to be the checkpoint — it is probably "
            f"an HTML page rather than the file. First bytes: {head[:80]!r}")

    tmp.replace(dest)
    return dest


def load_model(device: str = "cpu", progress=None) -> CIPAFormerV2:
    path = resolve_weights(progress=progress)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "ema" in ckpt:
        state = ckpt["ema"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    state = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in state.items()}

    model = CIPAFormerV2()
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} keys, e.g. {missing[:3]}")
    return model.eval().to(device)

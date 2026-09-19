"""Names the UI and the precompute step must agree on. Deliberately torch-free."""
from __future__ import annotations

IMG_SIZE = 512
UPLOAD_SIZE = 384          # smaller activations for the one path that runs the model live

METHODS = ("Grad-CAM", "HiRes-CAM", "Layer-CAM", "XGrad-CAM")

# display name -> dotted path of the module whose activations are probed
LAYERS = {
    "Decoder · up1 (stride 4, sharpest)": "decoder.up1",
    "Decoder · up2 (stride 8)": "decoder.up2",
    "Decoder · up3 (stride 16)": "decoder.up3",
    "Fusion · stage 4 (stride 32, most semantic)": "fusions.3",
    "Fusion · stage 3 (stride 16)": "fusions.2",
    "Fusion · stage 2 (stride 8)": "fusions.1",
}

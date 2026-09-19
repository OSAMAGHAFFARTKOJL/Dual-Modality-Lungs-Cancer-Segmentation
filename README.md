# RAMF-Net — PET/CT Lung Tumour Segmentation

An interactive Streamlit app for **RAMF-Net**, a dual-stream PET/CT segmentation network
trained on PCLT20K. Pick a bundled slice or upload your own PET/CT pair, and the app runs the
real checkpoint and shows the mask, the metrics, and *why* the network fired where it did.

![Segmentation tab](docs/screenshot.png)

## What it does

**Segmentation** — CT, PET, predicted mask, probability heatmap, ground truth, contour
comparison and a colour-coded error map (true positive / false positive / missed). The mask,
probability map and a composite report figure are all downloadable as PNG.

**Explainability** — four gradient attribution methods (Grad-CAM, HiRes-CAM, Layer-CAM,
XGrad-CAM) adapted to dense prediction: the score back-propagated is the logit energy inside
the predicted tumour, not a class score. Any of six target layers can be probed, from the
sharpest decoder block to the most semantic fusion stage. Also on this tab:

- the network's own **PET→CT spatial gate** at each fusion scale — a built-in interpretability
  signal showing where PET is pushing the CT stream;
- **uncertainty** — predictive entropy, the ambiguous 0.3–0.7 probability band, and a
  test-time-augmentation disagreement map when flip TTA is on;
- **modality importance** — each modality blanked in turn, so you can see what CT alone and
  PET alone recover.

**Metrics** — Dice, IoU, precision, recall, specificity, accuracy, HD95, ASSD, confusion
counts, and a threshold sweep that marks the Dice-optimal operating point.

## How it runs inside a small container

Every view of the four bundled cases is precomputed at 512×512 into `samples/precomputed/`
(~0.4 MB per case) and shipped with the app, so browsing them imports no torch and loads no
weights — the sample path peaks around 150 MB. Only a user-supplied slice runs the network,
at 384×384.

Two things make the live path fit as well. The fp16 checkpoint is copied straight into the
fp32 parameters by `load_state_dict` rather than being converted first, which avoids a
redundant 235 MB. And Grad-CAM runs the encoder under `no_grad`: no attribution target lives
inside the encoder, and the gradient at the target layer does not depend on how that layer's
inputs were produced, so skipping the encoder's graph is exact — verified bit-identical
against the full-graph version across all 24 method/layer combinations. Together these cut
peak RSS from ~1.1 GB to ~850 MB at 512×512.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

That is enough to browse the bundled cases — they are precomputed. To run the model yourself
(uploads, or regenerating the precomputed views) you also need the weights:

```bash
python tools/export_weights.py checkpoints/best.pth weights/cipaformer_v2_ema_fp16.pth
python tools/precompute_samples.py      # only after changing the checkpoint or the sample list
```

With `checkpoints/best.pth` present the app loads it directly; the export step just makes a
118 MB fp16 copy of the 450 MB training checkpoint. With neither file present, the weights are
fetched from the URL below.

## Deploy on Streamlit Community Cloud

The weights are too large for a git repository (GitHub caps files at 100 MB), so they live
outside it and are downloaded once at first run.

1. **Push this folder to GitHub.** `weights/`, `checkpoints/` and the full `PCLT20K/` dataset
   are gitignored; the four slices in `samples/` are committed.

2. **Host the weights on the Hugging Face Hub.** Create a **public** model repo at
   [huggingface.co/new](https://huggingface.co/new) and upload
   `weights/cipaformer_v2_ema_fp16.pth` — either by dragging it into *Files → Add file*, or:

   ```bash
   pip install huggingface_hub
   hf auth login
   hf upload <user>/<repo> weights/cipaformer_v2_ema_fp16.pth
   ```

   The direct link is then
   `https://huggingface.co/<user>/<repo>/resolve/main/cipaformer_v2_ema_fp16.pth`.

3. **Deploy.** Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, and set
   the main file to `app.py`.

No secret is required: `DEFAULT_MODEL_URL` in `core/model.py` already points at the published
weights, and the app fetches them on first boot with a progress bar, then caches them for the
life of the container. To serve them from somewhere else, override with a `MODEL_URL` secret
(or environment variable) holding a direct download link — a Streamlit secret looks like:

```toml
MODEL_URL = "https://huggingface.co/<user>/<repo>/resolve/main/cipaformer_v2_ema_fp16.pth"
```

Note that the link must serve the raw file. On Hugging Face that means `/resolve/` in the path,
not the `/blob/` link the file browser shows.

> **On GitHub release assets:** a release asset works too, but only if the repo is **public**.
> GitHub answers `404` for assets in a private repo unless the request is authenticated, and
> the app downloads anonymously. For a private repo, either host the weights publicly as above,
> or add a fine-grained token with repo read access as a second secret:
> `MODEL_TOKEN = "github_pat_…"`.

### Hugging Face Spaces

Spaces works too and is a better fit for the memory profile: create a Space with the Streamlit
SDK, push the same files, and commit the weights directly (Spaces handles large files through
LFS, so no release asset or `MODEL_URL` is needed).

### Why not Vercel

Vercel runs serverless functions with a 250 MB unzipped bundle limit and short execution
windows. PyTorch plus a 118 MB checkpoint plus a ~3 s CPU forward pass does not fit that
model — a Vercel deployment would need the inference moved to a separate always-on host, at
which point Streamlit Cloud or Spaces is doing the same job with none of the plumbing.

## Project layout

```
app.py                  Streamlit UI
core/spec.py               names the UI and the precompute step share (torch-free)
core/metrics.py            image prep and segmentation metrics (torch-free)
core/viz.py                overlays and composite figures (torch-free)
core/model.py              RAMF-Net architecture + checkpoint resolution
core/inference.py          preprocessing, prediction, TTA
core/explain.py            Grad-CAM family, PET gate maps
samples/                   four PCLT20K slices (CT / PET / mask)
samples/precomputed/       every view of those cases, baked
tools/export_weights.py    training checkpoint -> slim fp16 inference weights
tools/precompute_samples.py  regenerate samples/precomputed/
```

## Model

| | |
|---|---|
| Encoder | PVT-v2-b3, weight-shared across PET and CT, 4 stages (stride 4→32) |
| Fusion | per-stage SE channel gate + 7×7 PET-derived spatial gate applied to CT as `ct × (1 + gate)` |
| Decoder | U-Net, three upsampling blocks with skips, 1×1 head |
| Input | 512×512 grayscale PET and CT, normalised to [-1.6, 1.6] |
| Output | one sigmoid channel |
| Weights | EMA of training weights, fp16 for deployment |

Reported test performance (5 seeds, held-out split): IoU 64.2, Dice 78.2, precision 75.0,
recall 81.8, specificity 99.9. Individual slices vary widely around that mean — the bundled
samples run between roughly 66 and 96 Dice.

## Data

`samples/` contains four slices from **PCLT20K** with their expert masks. To use your own
data, supply a CT slice and a PET slice of the same anatomy as PNG or JPEG; a mask is optional
and only needed for the metrics.

---

**Research demo — not a medical device.** Nothing here is validated for clinical use, and no
output should inform diagnosis or treatment.

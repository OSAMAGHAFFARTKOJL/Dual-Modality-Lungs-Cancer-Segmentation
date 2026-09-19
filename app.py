"""RAMF-Net · PET/CT lung tumour segmentation — interactive demo.

The bundled sample cases are served from precomputed arrays, so browsing them never
imports torch. A user-supplied slice is the only path that loads the network.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core import metrics as M
from core import spec, viz

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
PRECOMPUTED = SAMPLES / "precomputed"

SURFACE = "#12120f"
PANEL = "#1a1a19"
INK = "#f5f5f0"
INK_DIM = "#a3a29b"
SERIES = ["#3987e5", "#d95926", "#199e70"]

st.set_page_config(page_title="RAMF-Net · PET/CT Tumour Segmentation",
                   page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .stApp {{ background: {SURFACE}; }}
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
  h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.015em; }}
  .hero {{ display:flex; align-items:baseline; gap:.75rem; flex-wrap:wrap; margin-bottom:.15rem; }}
  .hero h1 {{ font-size:1.9rem; margin:0; font-weight:650; }}
  .hero .tag {{ font-size:.72rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
               color:{SERIES[0]}; border:1px solid {SERIES[0]}44; background:{SERIES[0]}14;
               padding:.18rem .5rem; border-radius:999px; }}
  .sub {{ color:{INK_DIM}; font-size:.92rem; margin-bottom:1.4rem; }}
  .tile {{ background:{PANEL}; border:1px solid #2e2e2b; border-radius:12px; padding:.85rem 1rem; }}
  .tile .label {{ color:{INK_DIM}; font-size:.7rem; text-transform:uppercase; letter-spacing:.07em; }}
  .tile .value {{ color:{INK}; font-size:1.65rem; font-weight:650; line-height:1.25; }}
  .tile .unit {{ font-size:.85rem; color:{INK_DIM}; font-weight:500; margin-left:.15rem; }}
  .tile .foot {{ color:{INK_DIM}; font-size:.72rem; }}
  .legend {{ display:flex; gap:1.1rem; flex-wrap:wrap; color:{INK_DIM}; font-size:.8rem;
             margin:.4rem 0 .1rem; }}
  .legend span.dot {{ display:inline-block; width:10px; height:10px; border-radius:3px;
                      margin-right:.35rem; vertical-align:middle; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:.25rem; border-bottom:1px solid #2e2e2b; }}
  .stTabs [data-baseweb="tab"] {{ height:2.5rem; padding:0 1rem; color:{INK_DIM}; }}
  .stTabs [aria-selected="true"] {{ color:{INK} !important; }}
  section[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid #2e2e2b; }}
  div[data-testid="stImage"] img {{ border-radius:10px; border:1px solid #2e2e2b; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------- precomputed path
@st.cache_data(show_spinner=False)
def load_sample_index() -> list[dict]:
    return json.loads((SAMPLES / "index.json").read_text())["cases"]


@st.cache_data(show_spinner=False)
def load_sample_images(case_id: str):
    d = SAMPLES / case_id
    read = lambda n: cv2.imread(str(d / n), cv2.IMREAD_GRAYSCALE)
    return read("ct.png"), read("pet.png"), read("mask.png")


@st.cache_data(show_spinner=False, max_entries=2)
def load_precomputed(case_id: str) -> dict:
    with np.load(PRECOMPUTED / f"{case_id}.npz") as z:
        return {k: z[k] for k in z.files}


class Precomputed:
    """Serves the baked arrays for a bundled case. Never touches the model."""

    live = False
    size = spec.IMG_SIZE

    def __init__(self, case_id: str, tta: bool):
        z = load_precomputed(case_id)
        self._cams = z["cams"]
        self._methods = list(z["methods"])
        self._layers = list(z["layers"])
        self.prob = z["prob_tta" if tta else "prob"].astype(np.float32) / 65535.0
        self.std = z["std_tta"].astype(np.float32) / 255.0 if tta else None
        self.gates = [g.astype(np.float32) / 255.0 for g in z["gates"]]
        self.prob_ct = z["prob_ct"].astype(np.float32) / 65535.0
        self.prob_pet = z["prob_pet"].astype(np.float32) / 65535.0

    def cam(self, method: str, layer: str) -> np.ndarray:
        i, j = self._methods.index(method), self._layers.index(layer)
        return self._cams[i, j].astype(np.float32) / 255.0


# -------------------------------------------------------------------------- live path
@st.cache_resource(show_spinner=False)
def get_model():
    from core.model import load_model
    bar = st.sidebar.progress(0.0, "Downloading weights…")
    try:
        return load_model("cpu",
                          progress=lambda f: bar.progress(min(f, 1.0), "Downloading weights…"))
    except (RuntimeError, FileNotFoundError) as e:
        st.error(f"**The model weights could not be loaded.**\n\n{e}")
        st.stop()
    finally:
        bar.empty()


@st.cache_data(show_spinner=False, max_entries=2)
def _live_predict(key: str, ct: np.ndarray, pet: np.ndarray, tta: bool):
    from core import explain, inference
    model = get_model()
    ct_t, pet_t = inference.preprocess(ct, pet, spec.UPLOAD_SIZE)
    prob, std = inference.predict(model, ct_t, pet_t, tta=tta)
    gates = [explain.pet_gate_map(model, s, spec.UPLOAD_SIZE) for s in range(4)]
    return prob, std, gates


@st.cache_data(show_spinner=False, max_entries=4)
def _live_cam(key: str, ct: np.ndarray, pet: np.ndarray, method: str, layer: str):
    from core import explain, inference
    model = get_model()
    ct_t, pet_t = inference.preprocess(ct, pet, spec.UPLOAD_SIZE)
    return explain.cam(model, ct_t, pet_t, layer, method=method)


@st.cache_data(show_spinner=False, max_entries=2)
def _live_ablation(key: str, ct: np.ndarray, pet: np.ndarray):
    from core import inference
    model = get_model()
    ct_t, pet_t = inference.preprocess(ct, pet, spec.UPLOAD_SIZE)
    return (inference.predict_single_modality(model, ct_t, pet_t, "ct"),
            inference.predict_single_modality(model, ct_t, pet_t, "pet"))


class Live:
    """Runs the network on a user-supplied slice, at a reduced size to stay inside the
    container's memory budget."""

    live = True
    size = spec.UPLOAD_SIZE

    def __init__(self, key: str, ct: np.ndarray, pet: np.ndarray, tta: bool):
        self._args = (key, ct, pet)
        self.prob, std, self.gates = _live_predict(key, ct, pet, tta)
        self.std = std if tta else None

    def cam(self, method: str, layer: str) -> np.ndarray:
        return _live_cam(*self._args, method, layer)

    @property
    def ablation(self):
        return _live_ablation(*self._args)


def decode_upload(file) -> np.ndarray | None:
    if file is None:
        return None
    img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.sidebar.error(f"Could not read {file.name}")
    return img


def tile(col, label: str, value: str, unit: str = "", foot: str = ""):
    col.markdown(
        f'<div class="tile"><div class="label">{label}</div>'
        f'<div class="value">{value}<span class="unit">{unit}</span></div>'
        f'<div class="foot">{foot}</div></div>', unsafe_allow_html=True)


def themed(chart: alt.Chart) -> alt.Chart:
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(labelColor=INK_DIM, titleColor=INK_DIM, gridColor="#2b2b28",
                            domainColor="#2b2b28", tickColor="#2b2b28", labelFontSize=11,
                            titleFontSize=11, titlePadding=8)
            .configure_legend(labelColor=INK, titleColor=INK_DIM, labelFontSize=11,
                              titleFontSize=11))


# -------------------------------------------------------------------------- sidebar
st.sidebar.markdown("### Input")
source = st.sidebar.radio("Source", ["Sample cases", "Upload PET/CT"], label_visibility="collapsed",
                          horizontal=True)

ct_img = pet_img = gt_raw = None

if source == "Sample cases":
    cases = load_sample_index()
    labels = {f'{c["id"]} · {c["note"]}': c["id"] for c in cases}
    case_id = labels[st.sidebar.selectbox("Case", list(labels))]
    ct_img, pet_img, gt_raw = load_sample_images(case_id)
    case_label = f"PCLT20K · patient {case_id.split('_')[0]} · slice {int(case_id.split('_')[1])}"
    key = case_id
else:
    up_ct = st.sidebar.file_uploader("CT slice", type=["png", "jpg", "jpeg", "bmp", "tif"])
    up_pet = st.sidebar.file_uploader("PET slice", type=["png", "jpg", "jpeg", "bmp", "tif"])
    up_gt = st.sidebar.file_uploader("Ground-truth mask (optional)",
                                     type=["png", "jpg", "jpeg", "bmp", "tif"])
    ct_img, pet_img, gt_raw = decode_upload(up_ct), decode_upload(up_pet), decode_upload(up_gt)
    case_label = "Uploaded slice"
    key = hashlib.sha1(b"".join(f.getvalue() for f in (up_ct, up_pet)
                                if f is not None)).hexdigest()[:16]

st.sidebar.markdown("### Inference")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
tta = st.sidebar.toggle("Flip TTA (4 views)", value=False,
                        help="Averages 4 flipped views — steadier masks, plus a "
                             "disagreement map that reads as uncertainty.")

st.sidebar.markdown("### Display")
cmap = st.sidebar.selectbox("Heatmap colormap", list(viz.COLORMAPS), index=0)
alpha = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.50, 0.05)
with st.sidebar.expander("CT windowing"):
    brightness = st.slider("Brightness", -0.5, 0.5, 0.0, 0.05)
    contrast = st.slider("Contrast", 0.5, 2.5, 1.0, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption("Research demo · not a medical device")


# ----------------------------------------------------------------------------- head
st.markdown(
    '<div class="hero"><h1>PET/CT Lung Tumour Segmentation</h1>'
    '<span class="tag">RAMF-Net</span></div>'
    '<div class="sub">Dual-stream PVT-v2 encoder with reliability-aware cross-modal fusion '
    'and a U-Net decoder. Trained on PCLT20K.</div>', unsafe_allow_html=True)

if ct_img is None or pet_img is None:
    st.info("Upload a **CT** and a **PET** slice in the sidebar, or switch to *Sample cases* "
            "to try one of the bundled PCLT20K slices.")
    st.stop()

if ct_img.shape[:2] != pet_img.shape[:2]:
    st.warning(f"CT is {ct_img.shape[:2]} and PET is {pet_img.shape[:2]} — both are resampled "
               "before inference.")

t0 = time.perf_counter()
if source == "Sample cases":
    result = Precomputed(key, tta)
else:
    with st.spinner("Running RAMF-Net on your slice…"):
        result = Live(key, ct_img, pet_img, tta)
elapsed = time.perf_counter() - t0

size = result.size
prob = result.prob
pred = (prob >= threshold).astype(np.uint8)
gt = M.binarize_mask(gt_raw, size) if gt_raw is not None else None
stats = M.metrics(prob, gt, threshold)

ct_disp = viz.window(M.resize(M.to_gray(ct_img), size), brightness, contrast)
pet_disp = M.resize(M.to_gray(pet_img), size)
pet_rgb = viz.colorize(pet_disp.astype(np.float32) / 255.0, "inferno")

cols = st.columns(5)
if gt is not None:
    tile(cols[0], "Dice", f"{stats['dice']*100:.1f}", "%", "overlap with ground truth")
    tile(cols[1], "IoU", f"{stats['iou']*100:.1f}", "%", "Jaccard index")
    tile(cols[2], "Precision", f"{stats['precision']*100:.1f}", "%",
         f"{stats['fp']} false-positive px")
    tile(cols[3], "Recall", f"{stats['recall']*100:.1f}", "%", f"{stats['fn']} missed px")
    hd = stats["hd95"]
    tile(cols[4], "HD95", f"{hd:.1f}" if hd is not None else "—", " px", "95th-pct boundary error")
else:
    tile(cols[0], "Tumour area", f"{stats['pred_px']:,}", " px", f"at threshold {threshold:.2f}")
    tile(cols[1], "Peak confidence", f"{stats['max_conf']*100:.1f}", "%", "max sigmoid output")
    tile(cols[2], "Mean confidence", f"{stats['mean_conf']*100:.1f}", "%", "inside prediction")
    tile(cols[3], "Resolution", f"{size}", " px", "live inference" if result.live else "precomputed")
    tile(cols[4], "Ground truth", "—", "", "upload a mask for metrics")

st.markdown("")
tab_seg, tab_xai, tab_metrics = st.tabs(["Segmentation", "Explainability", "Metrics"])


# ---------------------------------------------------------------------- segmentation
with tab_seg:
    mode = f"{elapsed:.1f}s live at {size}px" if result.live else "precomputed"
    st.caption(f"{case_label} · {mode}{' · flip TTA' if tta else ''}")
    c = st.columns(4)
    c[0].image(ct_disp, caption="CT", width="stretch")
    c[1].image(pet_rgb, caption="PET", width="stretch")
    c[2].image(viz.mask_overlay(ct_disp, pred, viz.PRED_COLOR, alpha),
               caption=f"Prediction (p ≥ {threshold:.2f})", width="stretch")
    c[3].image(viz.heat_overlay(ct_disp, prob, alpha, cmap, floor=0.05),
               caption="Tumour probability", width="stretch")

    st.markdown("")
    c2 = st.columns(4)
    if gt is not None:
        c2[0].image(viz.mask_overlay(ct_disp, gt, viz.GT_COLOR, alpha),
                    caption="Ground truth", width="stretch")
        c2[1].image(viz.contour_compare(ct_disp, gt, pred),
                    caption="Contours · GT vs prediction", width="stretch")
        c2[2].image(viz.error_map(ct_disp, gt, pred, alpha),
                    caption="Error map", width="stretch")
        c2[3].image(viz.heat_overlay(pet_rgb, prob, alpha, cmap, floor=0.05),
                    caption="Probability on PET", width="stretch")
        st.markdown(
            f'<div class="legend">'
            f'<span><span class="dot" style="background:rgb{viz.TP_COLOR}"></span>True positive'
            f' — {stats["tp"]:,} px</span>'
            f'<span><span class="dot" style="background:rgb{viz.FP_COLOR}"></span>False positive'
            f' — {stats["fp"]:,} px</span>'
            f'<span><span class="dot" style="background:rgb{viz.FN_COLOR}"></span>Missed'
            f' — {stats["fn"]:,} px</span></div>', unsafe_allow_html=True)
    else:
        c2[0].image(viz.contour_compare(ct_disp, None, pred),
                    caption="Predicted contour", width="stretch")
        c2[1].image(viz.heat_overlay(pet_rgb, prob, alpha, cmap, floor=0.05),
                    caption="Probability on PET", width="stretch")
        c2[2].image(viz.colorize(prob, cmap), caption="Raw probability map", width="stretch")
        c2[3].empty()

    panels = [("CT", viz.to_rgb(ct_disp)), ("PET", pet_rgb),
              ("Probability", viz.heat_overlay(ct_disp, prob, alpha, cmap, floor=0.05)),
              ("Prediction", viz.mask_overlay(ct_disp, pred, viz.PRED_COLOR, alpha))]
    if gt is not None:
        panels += [("Ground truth", viz.mask_overlay(ct_disp, gt, viz.GT_COLOR, alpha)),
                   ("Error map", viz.error_map(ct_disp, gt, pred, alpha))]

    d = st.columns(3)
    d[0].download_button("Download mask (PNG)", viz.png_bytes(viz.to_rgb(pred * 255)),
                         f"{key}_mask.png", "image/png", width="stretch")
    d[1].download_button("Download probability map (PNG)", viz.png_bytes(viz.colorize(prob, cmap)),
                         f"{key}_probability.png", "image/png", width="stretch")
    d[2].download_button("Download report figure (PNG)",
                         viz.png_bytes(viz.side_by_side(panels, cols=3)),
                         f"{key}_report.png", "image/png", width="stretch")


# -------------------------------------------------------------------- explainability
with tab_xai:
    left, right = st.columns(2)
    method = left.selectbox("Attribution method", list(spec.METHODS),
                            help="Gradient saliency adapted to segmentation: the score "
                                 "back-propagated is the logit energy inside the predicted "
                                 "tumour, not a class score.")
    layer = right.selectbox("Target layer", list(spec.LAYERS))

    with st.spinner(f"Computing {method}…"):
        heat = result.cam(method, layer)

    c = st.columns(3)
    c[0].image(viz.heat_overlay(ct_disp, heat, alpha, cmap),
               caption=f"{method} on CT", width="stretch")
    c[1].image(viz.heat_overlay(pet_rgb, heat, alpha, cmap),
               caption=f"{method} on PET", width="stretch")
    c[2].image(viz.heat_overlay(viz.contour_compare(ct_disp, gt, pred), heat, alpha, cmap,
                                floor=0.15),
               caption="Attribution + contours", width="stretch")
    inside = float(heat[pred > 0].mean()) if pred.sum() else 0.0
    st.caption(f"Mean attribution inside the predicted region: **{inside:.2f}** vs "
               f"**{float(heat[pred == 0].mean()):.2f}** outside — the wider that gap, the more "
               f"the evidence is concentrated on the lesion.")

    st.markdown("#### Learned PET→CT gate")
    st.caption("The network modulates CT features by a spatial gate derived from PET "
               "(`ct × (1 + gate)`). These are that gate at each fusion scale — bright means "
               "PET is pushing the CT stream hardest there.")
    gc = st.columns(4)
    for i, g in enumerate(result.gates):
        stride = 4 * 2 ** i
        if g is None:
            continue
        lo, hi = float(g.min()), float(g.max())
        if hi < 0.02:
            gc[i].markdown(f'<div class="tile"><div class="label">Stage {i+1} · stride {stride}'
                           f'</div><div class="value">{hi:.3f}</div>'
                           f'<div class="foot">gate is off at this scale — PET adds nothing '
                           f'here</div></div>', unsafe_allow_html=True)
            continue
        gc[i].image(viz.heat_overlay(ct_disp, g, alpha, cmap),
                    caption=f"Stage {i+1} · stride {stride} · gate {lo:.2f}–{hi:.2f}",
                    width="stretch")

    st.markdown("#### Uncertainty")
    uc = st.columns(3)
    ent = M.entropy_map(prob)
    uc[0].image(viz.heat_overlay(ct_disp, ent, alpha, cmap),
                caption=f"Predictive entropy · peak {ent.max():.2f} bits", width="stretch")
    band = ((prob > 0.3) & (prob < 0.7)).astype(np.uint8)
    uc[1].image(viz.mask_overlay(ct_disp, band, (234, 179, 8), alpha),
                caption=f"Ambiguous band 0.3–0.7 · {int(band.sum()):,} px", width="stretch")
    if result.std is not None:
        std = result.std
        uc[2].image(viz.heat_overlay(ct_disp, std / (std.max() + 1e-8), alpha, cmap),
                    caption=f"TTA disagreement · max σ {std.max():.3f}", width="stretch")
    else:
        uc[2].info("Enable **Flip TTA** in the sidebar to add a test-time-augmentation "
                   "disagreement map.")

    st.markdown("#### Modality importance")
    st.caption("Each modality is blanked in turn to measure what the other one alone recovers.")
    prob_ct, prob_pet = (result.ablation if result.live else (result.prob_ct, result.prob_pet))

    mc = st.columns(3)
    for col, p, cap in ((mc[0], prob, "PET + CT (full model)"),
                        (mc[1], prob_ct, "CT only (PET blanked)"),
                        (mc[2], prob_pet, "PET only (CT blanked)")):
        col.image(viz.mask_overlay(ct_disp, (p >= threshold).astype(np.uint8),
                                   viz.PRED_COLOR, alpha), caption=cap, width="stretch")

    if gt is not None:
        df = pd.DataFrame([("PET + CT", M.metrics(prob, gt, threshold)["dice"] * 100),
                           ("CT only", M.metrics(prob_ct, gt, threshold)["dice"] * 100),
                           ("PET only", M.metrics(prob_pet, gt, threshold)["dice"] * 100)],
                          columns=["input", "pct"])
        y_enc = alt.Y("input:N", sort=None, title=None)
        x_enc = alt.X("pct:Q", title="Dice (%)", scale=alt.Scale(domain=[0, 100]),
                      axis=alt.Axis(tickCount=6))
        bars = alt.Chart(df).mark_bar(color=SERIES[0], cornerRadiusEnd=4, size=18).encode(
            y=y_enc, x=x_enc,
            tooltip=[alt.Tooltip("input:N", title="Input"),
                     alt.Tooltip("pct:Q", title="Dice (%)", format=".1f")])
        labels = alt.Chart(df).mark_text(align="left", dx=8, color=INK_DIM, fontSize=11).encode(
            y=y_enc, x=alt.X("pct:Q", scale=alt.Scale(domain=[0, 100])),
            text=alt.Text("pct:Q", format=".1f"))
        st.altair_chart(themed((bars + labels).properties(height=170, padding={"right": 34})),
                        width="stretch")


# --------------------------------------------------------------------------- metrics
with tab_metrics:
    if gt is None:
        st.info("Metrics need a ground-truth mask. Pick a sample case, or upload a mask "
                "alongside your PET/CT pair.")
    else:
        rows = [
            ("Dice / F1", f"{stats['dice']*100:.2f} %"),
            ("IoU (Jaccard)", f"{stats['iou']*100:.2f} %"),
            ("Precision", f"{stats['precision']*100:.2f} %"),
            ("Recall / sensitivity", f"{stats['recall']*100:.2f} %"),
            ("Specificity", f"{stats['specificity']*100:.2f} %"),
            ("Pixel accuracy", f"{stats['accuracy']*100:.2f} %"),
            ("HD95", f"{stats['hd95']:.2f} px" if stats["hd95"] is not None else "—"),
            ("ASSD", f"{stats['assd']:.2f} px" if stats["assd"] is not None else "—"),
            ("Predicted area", f"{stats['pred_px']:,} px"),
            ("Ground-truth area", f"{stats['gt_px']:,} px"),
            ("True positives", f"{stats['tp']:,} px"),
            ("False positives", f"{stats['fp']:,} px"),
            ("False negatives", f"{stats['fn']:,} px"),
        ]
        mleft, mright = st.columns([1, 1.4])
        mleft.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]),
                        hide_index=True, width="stretch", height=490)

        sweep = pd.DataFrame(M.sweep_threshold(prob, gt), columns=["threshold", "Dice", "IoU"])
        long = sweep.melt("threshold", var_name="metric", value_name="score")
        long["score"] *= 100
        best = sweep.loc[sweep["Dice"].idxmax()]

        hover = alt.selection_point(fields=["threshold"], nearest=True, on="pointermove",
                                    empty=False)
        color = alt.Color("metric:N", title=None,
                          scale=alt.Scale(domain=["Dice", "IoU"], range=SERIES[:2]),
                          legend=alt.Legend(orient="top", direction="horizontal",
                                            symbolType="stroke", symbolStrokeWidth=3))
        x = alt.X("threshold:Q", title="Decision threshold",
                  scale=alt.Scale(domain=[0, 1], nice=False),
                  axis=alt.Axis(tickCount=6, format=".1f"))
        y = alt.Y("score:Q", title="Score (%)", scale=alt.Scale(domain=[0, 100]))

        lines = alt.Chart(long).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
        rule_now = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(
            color=INK_DIM, strokeDash=[4, 4]).encode(x="t:Q")
        rule_best = alt.Chart(pd.DataFrame({"t": [best["threshold"]]})).mark_rule(
            color=SERIES[2], strokeWidth=2).encode(x="t:Q")
        points = alt.Chart(long).mark_circle(size=70).encode(
            x=x, y=y, color=color.legend(None),
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("threshold:Q", format=".2f", title="Threshold"),
                     alt.Tooltip("metric:N", title="Metric"),
                     alt.Tooltip("score:Q", format=".1f", title="Score (%)")]).add_params(hover)

        mright.altair_chart(
            themed((lines + rule_now + rule_best + points).properties(height=300)),
            width="stretch")
        mright.caption(
            f"Solid green line = best Dice **{best['Dice']*100:.1f}%** at threshold "
            f"**{best['threshold']:.2f}**; dashed line = your current threshold "
            f"**{threshold:.2f}** (Dice {stats['dice']*100:.1f}%).")


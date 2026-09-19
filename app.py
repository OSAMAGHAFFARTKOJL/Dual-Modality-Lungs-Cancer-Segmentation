"""CIPAFormer-V2 · PET/CT lung tumour segmentation — interactive demo."""
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
import torch

from core import explain, inference, viz
from core.model import load_model

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"

SURFACE = "#12120f"
PANEL = "#1a1a19"
INK = "#f5f5f0"
INK_DIM = "#a3a29b"
SERIES = ["#3987e5", "#d95926", "#199e70"]

st.set_page_config(page_title="CIPAFormer-V2 · PET/CT Tumour Segmentation",
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
  .caption {{ color:{INK_DIM}; font-size:.78rem; margin-top:-.4rem; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:.25rem; border-bottom:1px solid #2e2e2b; }}
  .stTabs [data-baseweb="tab"] {{ height:2.5rem; padding:0 1rem; color:{INK_DIM}; }}
  .stTabs [aria-selected="true"] {{ color:{INK} !important; }}
  section[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid #2e2e2b; }}
  div[data-testid="stImage"] img {{ border-radius:10px; border:1px solid #2e2e2b; }}
  .disclaimer {{ background:#2a1f12; border:1px solid #6b4a1e; color:#e8d5b5; border-radius:10px;
                 padding:.7rem .9rem; font-size:.82rem; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_sample_index() -> list[dict]:
    idx = json.loads((SAMPLES / "index.json").read_text())
    return idx["cases"]


@st.cache_data(show_spinner=False)
def load_sample(case_id: str):
    d = SAMPLES / case_id
    read = lambda n: cv2.imread(str(d / n), cv2.IMREAD_GRAYSCALE)
    return read("ct.png"), read("pet.png"), read("mask.png")


def decode_upload(file) -> np.ndarray | None:
    if file is None:
        return None
    buf = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.sidebar.error(f"Could not read {file.name}")
    return img


# ---------------------------------------------------------------------------- model
@st.cache_resource(show_spinner=False)
def get_model():
    bar = st.sidebar.progress(0.0, "Downloading weights…")
    try:
        return load_model("cpu",
                          progress=lambda f: bar.progress(min(f, 1.0), "Downloading weights…"))
    except (RuntimeError, FileNotFoundError) as e:
        st.error(f"**The model weights could not be loaded.**\n\n{e}")
        st.stop()
    finally:
        bar.empty()


@st.cache_data(show_spinner=False, max_entries=8)
def run_inference(key: str, ct: np.ndarray, pet: np.ndarray, tta: bool):
    model = get_model()
    ct_t, pet_t = inference.preprocess(ct, pet)
    t0 = time.perf_counter()
    prob, std = inference.predict(model, ct_t, pet_t, tta=tta)
    gates = [explain.pet_gate_map(model, s) for s in range(4)]
    return prob, std, gates, time.perf_counter() - t0


@st.cache_data(show_spinner=False, max_entries=8)
def run_cam(key: str, ct: np.ndarray, pet: np.ndarray, method: str, layer: str, threshold: float):
    model = get_model()
    ct_t, pet_t = inference.preprocess(ct, pet)
    return explain.cam(model, ct_t, pet_t, explain.TARGET_LAYERS[layer], method=method,
                       threshold=threshold)


@st.cache_data(show_spinner=False, max_entries=8)
def run_modality_ablation(key: str, ct: np.ndarray, pet: np.ndarray):
    model = get_model()
    ct_t, pet_t = inference.preprocess(ct, pet)
    return (inference.predict_single_modality(model, ct_t, pet_t, "ct"),
            inference.predict_single_modality(model, ct_t, pet_t, "pet"))


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
case_label = ""

if source == "Sample cases":
    cases = load_sample_index()
    labels = {f'{c["id"]} · {c["note"]}': c["id"] for c in cases}
    case_id = labels[st.sidebar.selectbox("Case", list(labels))]
    ct_img, pet_img, gt_raw = load_sample(case_id)
    case_label = f"PCLT20K · patient {case_id.split('_')[0]} · slice {int(case_id.split('_')[1])}"
    key = f"sample:{case_id}"
else:
    up_ct = st.sidebar.file_uploader("CT slice (PNG/JPG)", type=["png", "jpg", "jpeg", "bmp", "tif"])
    up_pet = st.sidebar.file_uploader("PET slice (PNG/JPG)", type=["png", "jpg", "jpeg", "bmp", "tif"])
    up_gt = st.sidebar.file_uploader("Ground-truth mask (optional)",
                                     type=["png", "jpg", "jpeg", "bmp", "tif"])
    ct_img, pet_img, gt_raw = decode_upload(up_ct), decode_upload(up_pet), decode_upload(up_gt)
    case_label = "Uploaded slice"
    key = "upload:" + hashlib.sha1(
        b"".join(f.getvalue() for f in (up_ct, up_pet) if f is not None)).hexdigest()[:16]

st.sidebar.markdown("### Inference")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
tta = st.sidebar.toggle("Flip TTA (4 views, slower)", value=False,
                        help="Averages 4 flipped views — steadier masks and a variance-based "
                             "uncertainty map, at ~4× the compute.")

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
    '<span class="tag">CIPAFormer-V2</span></div>'
    '<div class="sub">Dual-stream PVT-v2 encoder with cross-modal SE + PET-spatial gating, '
    'U-Net decoder. Trained on PCLT20K.</div>', unsafe_allow_html=True)

if ct_img is None or pet_img is None:
    st.info("Upload a **CT** and a **PET** slice in the sidebar, or switch to *Sample cases* "
            "to try one of the bundled PCLT20K slices.")
    st.stop()

if ct_img.shape[:2] != pet_img.shape[:2]:
    st.warning(f"CT is {ct_img.shape[:2]} and PET is {pet_img.shape[:2]} — both are resampled "
               "to 512×512 before inference.")

with st.spinner("Running CIPAFormer-V2…"):
    prob, std, gates, elapsed = run_inference(key, ct_img, pet_img, tta)

pred = (prob >= threshold).astype(np.uint8)
gt = inference.binarize_mask(gt_raw) if gt_raw is not None else None
stats = inference.metrics(prob, gt, threshold)

ct_disp = viz.window(inference.resize(inference.to_gray(ct_img)), brightness, contrast)
pet_disp = inference.resize(inference.to_gray(pet_img))
pet_rgb = viz.colorize(pet_disp.astype(np.float32) / 255.0, "inferno")

cols = st.columns(5)
if gt is not None:
    tile(cols[0], "Dice", f"{stats['dice']*100:.1f}", "%", "overlap with ground truth")
    tile(cols[1], "IoU", f"{stats['iou']*100:.1f}", "%", "Jaccard index")
    tile(cols[2], "Precision", f"{stats['precision']*100:.1f}", "%", f"{stats['fp']} false-positive px")
    tile(cols[3], "Recall", f"{stats['recall']*100:.1f}", "%", f"{stats['fn']} missed px")
    hd = stats["hd95"]
    tile(cols[4], "HD95", f"{hd:.1f}" if hd is not None else "—", " px", "95th-pct boundary error")
else:
    tile(cols[0], "Tumour area", f"{stats['pred_px']:,}", " px", f"at threshold {threshold:.2f}")
    tile(cols[1], "Peak confidence", f"{stats['max_conf']*100:.1f}", "%", "max sigmoid output")
    tile(cols[2], "Mean confidence", f"{stats['mean_conf']*100:.1f}", "%", "inside predicted region")
    tile(cols[3], "Runtime", f"{elapsed:.1f}", " s", "CPU forward pass" + (" ×4 TTA" if tta else ""))
    tile(cols[4], "Ground truth", "—", "", "upload a mask for metrics")

st.markdown("")
tab_seg, tab_xai, tab_metrics, tab_about = st.tabs(
    ["Segmentation", "Explainability", "Metrics", "Model & data"])


# ---------------------------------------------------------------------- segmentation
with tab_seg:
    st.caption(f"{case_label} · {elapsed:.1f}s on CPU{' · flip TTA' if tta else ''}")
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
        c2[2].image(viz.colorize(prob, cmap), caption="Raw probability map",
                    width="stretch")
        c2[3].empty()

    panels = [("CT", viz.to_rgb(ct_disp)),
              ("PET", pet_rgb),
              ("Probability", viz.heat_overlay(ct_disp, prob, alpha, cmap, floor=0.05)),
              ("Prediction", viz.mask_overlay(ct_disp, pred, viz.PRED_COLOR, alpha))]
    if gt is not None:
        panels += [("Ground truth", viz.mask_overlay(ct_disp, gt, viz.GT_COLOR, alpha)),
                   ("Error map", viz.error_map(ct_disp, gt, pred, alpha))]

    d = st.columns(3)
    d[0].download_button("Download mask (PNG)", viz.png_bytes(viz.to_rgb(pred * 255)),
                         f"{key.split(':')[-1]}_mask.png", "image/png", width="stretch")
    d[1].download_button("Download probability map (PNG)",
                         viz.png_bytes(viz.colorize(prob, cmap)),
                         f"{key.split(':')[-1]}_probability.png", "image/png",
                         width="stretch")
    d[2].download_button("Download report figure (PNG)",
                         viz.png_bytes(viz.side_by_side(panels, cols=3)),
                         f"{key.split(':')[-1]}_report.png", "image/png", width="stretch")


# -------------------------------------------------------------------- explainability
with tab_xai:
    left, right = st.columns([1, 1])
    with left:
        method = st.selectbox("Attribution method", explain.METHODS,
                              help="Gradient-based saliency adapted to segmentation: the score "
                                   "back-propagated is the logit energy inside the predicted "
                                   "tumour region.")
    with right:
        layer = st.selectbox("Target layer", list(explain.TARGET_LAYERS))

    with st.spinner(f"Computing {method}…"):
        heat = run_cam(key, ct_img, pet_img, method, layer, threshold)

    c = st.columns(3)
    c[0].image(viz.heat_overlay(ct_disp, heat, alpha, cmap),
               caption=f"{method} on CT", width="stretch")
    c[1].image(viz.heat_overlay(pet_rgb, heat, alpha, cmap),
               caption=f"{method} on PET", width="stretch")
    contour = viz.contour_compare(ct_disp, gt, pred)
    c[2].image(viz.heat_overlay(contour, heat, alpha, cmap, floor=0.15),
               caption="Attribution + contours", width="stretch")
    inside = float(heat[pred > 0].mean()) if pred.sum() else 0.0
    st.caption(f"Mean attribution inside the predicted region: **{inside:.2f}** "
               f"vs **{float(heat[pred == 0].mean()):.2f}** outside — higher contrast means the "
               f"evidence is better localised on the lesion.")

    st.markdown("#### Learned PET→CT gate")
    st.caption("This network modulates CT features by a spatial gate derived from PET "
               "(`ct × (1 + gate)`). The maps below are that gate at each fusion scale — "
               "bright means PET is pushing the CT stream hardest there.")
    gc = st.columns(4)
    for i, g in enumerate(gates):
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
    ent = explain.entropy_map(prob)
    uc[0].image(viz.heat_overlay(ct_disp, ent, alpha, cmap),
                caption=f"Predictive entropy · peak {ent.max():.2f} bits",
                width="stretch")
    band = ((prob > 0.3) & (prob < 0.7)).astype(np.uint8)
    uc[1].image(viz.mask_overlay(ct_disp, band, (234, 179, 8), alpha),
                caption=f"Ambiguous band 0.3–0.7 · {int(band.sum()):,} px",
                width="stretch")
    if tta:
        uc[2].image(viz.heat_overlay(ct_disp, std / (std.max() + 1e-8), alpha, cmap),
                    caption=f"TTA disagreement · max σ {std.max():.3f}", width="stretch")
    else:
        uc[2].info("Enable **Flip TTA** in the sidebar to add a test-time-augmentation "
                   "disagreement map.")

    st.markdown("#### Modality importance")
    st.caption("Each modality is blanked in turn to measure what the other one alone can recover.")
    with st.spinner("Running single-modality passes…"):
        prob_ct, prob_pet = run_modality_ablation(key, ct_img, pet_img)

    mc = st.columns(3)
    mc[0].image(viz.mask_overlay(ct_disp, (prob >= threshold).astype(np.uint8),
                                 viz.PRED_COLOR, alpha),
                caption="PET + CT (full model)", width="stretch")
    mc[1].image(viz.mask_overlay(ct_disp, (prob_ct >= threshold).astype(np.uint8),
                                 viz.PRED_COLOR, alpha),
                caption="CT only (PET blanked)", width="stretch")
    mc[2].image(viz.mask_overlay(ct_disp, (prob_pet >= threshold).astype(np.uint8),
                                 viz.PRED_COLOR, alpha),
                caption="PET only (CT blanked)", width="stretch")

    if gt is not None:
        rows = [("PET + CT", inference.metrics(prob, gt, threshold)["dice"]),
                ("CT only", inference.metrics(prob_ct, gt, threshold)["dice"]),
                ("PET only", inference.metrics(prob_pet, gt, threshold)["dice"])]
        df = pd.DataFrame(rows, columns=["input", "dice"])
        df["pct"] = df["dice"] * 100
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
        st.info("Metrics need a ground-truth mask. Pick a sample case, or upload a mask alongside "
                "your PET/CT pair.")
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

        sweep = pd.DataFrame(inference.sweep_threshold(prob, gt), columns=["threshold", "Dice", "IoU"])
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
                  scale=alt.Scale(domain=[0, 1], nice=False), axis=alt.Axis(tickCount=6, format=".1f"))
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


# ---------------------------------------------------------------------------- about
with tab_about:
    a, b = st.columns(2)
    a.markdown("""
#### Architecture
**CIPAFormer-V2** — a dual-stream encoder–decoder for PET/CT tumour segmentation.

- **Encoder** · PVT-v2-b3, weight-shared across PET and CT (4 stages, strides 4→32)
- **Fusion** · per-stage `CrossModalFusion`: an SE channel gate over the concatenated
  streams, plus a 7×7 PET-derived spatial gate that modulates CT as `ct × (1 + gate)`
- **Decoder** · U-Net style, three upsampling blocks with skip connections, 1×1 prediction head
- **Inference** · 512×512, single sigmoid channel, optional 4-view flip TTA
- **Weights** · exponential moving average of the training weights
""")
    b.markdown("""
#### Dataset
**PCLT20K** — paired PET/CT lung slices with expert tumour masks, one folder per patient
(`<patient>_<slice>_CT.png`, `_PET.png`, `_mask.png`). Images are 512×512 grayscale and are
normalised to [-1.6, 1.6] before the forward pass, matching training exactly.

Four slices are bundled with this app under `samples/`, chosen to span an easy large lesion
through to a small low-contrast one.

#### Reported test performance
5-seed mean on the held-out split: **IoU 64.2**, **Dice 78.2**, precision 75.0, recall 81.8,
specificity 99.9. Per-slice scores in this demo vary widely around that mean — the bundled
samples are individually easier than the dataset average.
""")
    st.markdown(
        '<div class="disclaimer"><b>Research demo — not a medical device.</b> '
        'Nothing here is validated for clinical use, and no output should inform diagnosis '
        'or treatment.</div>', unsafe_allow_html=True)

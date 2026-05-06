"""
Streamlit deployment for the Flange-Invariant Acoustic Bolt-Looseness Detector.

Designed for Streamlit Community Cloud (https://share.streamlit.io). Reuses the
exact backend modules (feature_extraction.py, segmentation.py, inference.py)
that the FastAPI version uses, so predictions are identical.

Layout (three tabs):
    1. Predict   — record / upload / classify
    2. Features  — per-hit 150-D feature breakdown, spectrograms, heatmap
    3. About     — method explanation, photos, LOFO confusion matrices

Run locally:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from inference import CLASSES_REF, FlangeInvariantLR

MODEL_DIR = ROOT / "backend" / "model"
ASSETS_DIR = ROOT / "assets"

# ---------------------------------------------------------------- #
# Page config + CSS
# ---------------------------------------------------------------- #
st.set_page_config(
    page_title="Flange-Invariant Bolt-Looseness Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
  .main .block-container { padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1280px; }
  header[data-testid="stHeader"] { background: transparent; }
  #MainMenu, footer { visibility: hidden; }

  .stApp {
    background:
      radial-gradient(1200px 700px at 10% -10%, rgba(34, 211, 238, 0.10), transparent 60%),
      radial-gradient(1100px 600px at 90% 110%, rgba(16, 185, 129, 0.08), transparent 60%),
      linear-gradient(180deg, #05070d 0%, #0a0d15 100%);
  }

  .ribbon {
    display: inline-flex; align-items: center; gap: 0.55rem;
    padding: 0.45rem 0.85rem; border-radius: 9999px;
    background: rgba(34, 211, 238, 0.08);
    border: 1px solid rgba(34, 211, 238, 0.25);
    color: #67e8f9;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
  }

  .hero-title {
    font-size: clamp(1.9rem, 4vw, 3.6rem);
    font-weight: 800; line-height: 1.05; letter-spacing: -0.02em;
    color: #f5f7fa; margin-top: 0.85rem;
  }
  .hero-grad {
    background: linear-gradient(90deg, #10b981 0%, #22d3ee 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .hero-sub { color: #9aa6b9; max-width: 720px; margin-top: 0.8rem; font-size: 0.98rem; line-height: 1.55; }

  .pipe-strip {
    display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center;
    margin-top: 1.2rem; font-family: ui-monospace, SFMono-Regular, JetBrains Mono, monospace;
    font-size: 0.82rem; color: #cbd3df;
  }
  .pipe-pill {
    padding: 0.35rem 0.7rem; border-radius: 0.55rem;
    background: linear-gradient(180deg, rgba(75, 86, 111, 0.22) 0%, rgba(38, 45, 63, 0.55) 100%);
    border: 1px solid rgba(155, 166, 185, 0.16);
  }
  .pipe-pill.final {
    background: rgba(16, 185, 129, 0.12); color: #6ee7b7; border-color: rgba(16, 185, 129, 0.35);
  }
  .pipe-arrow { color: #22d3ee; }

  .glass {
    background: rgba(24, 29, 44, 0.65);
    backdrop-filter: blur(14px) saturate(120%);
    border: 1px solid rgba(155, 166, 185, 0.12);
    border-radius: 1rem;
    padding: 1.05rem 1.15rem;
  }
  .glass + .glass { margin-top: 0.85rem; }

  .stage-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.55rem; margin-top: 1.4rem;
  }
  @media (min-width: 900px) { .stage-grid { grid-template-columns: repeat(4, 1fr); } }
  .stage {
    display: flex; gap: 0.6rem; align-items: flex-start;
    background: rgba(24, 29, 44, 0.6);
    border: 1px solid rgba(155, 166, 185, 0.12);
    border-radius: 0.85rem; padding: 0.75rem 0.95rem;
  }
  .stage-num {
    width: 2rem; height: 2rem; border-radius: 0.6rem; flex: 0 0 auto;
    display: flex; align-items: center; justify-content: center;
    background: rgba(34, 211, 238, 0.12); color: #22d3ee; font-weight: 700;
  }
  .stage-num.final { background: rgba(16, 185, 129, 0.15); color: #10b981; }
  .stage-title { font-weight: 600; color: #f5f7fa; font-size: 0.92rem; }
  .stage-sub { color: #9aa6b9; font-size: 0.75rem; margin-top: 0.1rem; }

  .label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #cbd3df;
  }
  .label-sub { font-size: 0.7rem; color: #9aa6b9; margin-bottom: 0.55rem; }

  .final-card { padding: 1.4rem 1.5rem; border-radius: 1.1rem; }
  .final-card.high   { box-shadow: 0 0 36px -8px rgba(16, 185, 129, 0.55); border-color: rgba(16, 185, 129, 0.25); }
  .final-card.medium { box-shadow: 0 0 36px -8px rgba(245, 158, 11, 0.55); border-color: rgba(245, 158, 11, 0.25); }
  .final-card.low    { box-shadow: 0 0 36px -8px rgba(239, 68, 68, 0.55);  border-color: rgba(239, 68, 68, 0.25); }
  .final-badge {
    font-size: clamp(1.55rem, 3.2vw, 2.5rem); font-weight: 800;
    letter-spacing: -0.01em; line-height: 1.1;
  }
  .final-badge.high   { color: #10b981; }
  .final-badge.medium { color: #f59e0b; }
  .final-badge.low    { color: #ef4444; }
  .final-meta { color: #9aa6b9; font-size: 0.8rem; margin-top: 0.3rem; }
  .conf-num { font-family: ui-monospace, monospace; font-size: 1.55rem; color: white; }

  .pill {
    display: inline-flex; padding: 0.18rem 0.55rem; border-radius: 9999px;
    font-size: 0.66rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
  }
  .pill.high   { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.35); }
  .pill.medium { background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.35); }
  .pill.low    { background: rgba(239, 68, 68, 0.15);  color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.35); }
  .pill.cyan   { background: rgba(34, 211, 238, 0.12); color: #67e8f9; border: 1px solid rgba(34, 211, 238, 0.25); }
  .pill.green  { background: rgba(16, 185, 129, 0.12); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.30); }

  .probrow { margin-top: 0.65rem; }
  .probrow .head {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.76rem; font-family: ui-monospace, monospace; margin-bottom: 0.22rem;
  }
  .probrow .head .lbl { color: #9aa6b9; }
  .probrow .head .lbl.win { color: white; font-weight: 700; }
  .probrow .head .val { color: #cbd3df; }
  .probrow .head .val.win { color: white; }
  .probtrack {
    height: 0.5rem; border-radius: 9999px; background: rgba(75, 86, 111, 0.35); overflow: hidden;
  }
  .probfill { height: 100%; background: #4b566f; }
  .probfill.win { background: linear-gradient(90deg, #10b981 0%, #22d3ee 100%); }

  .explain {
    display: grid; grid-template-columns: repeat(1, 1fr); gap: 0.55rem; margin-top: 0.95rem;
  }
  @media (min-width: 700px) { .explain { grid-template-columns: repeat(3, 1fr); } }
  .explain .tile {
    background: rgba(24, 29, 44, 0.55);
    border: 1px solid rgba(155, 166, 185, 0.14);
    border-radius: 0.65rem; padding: 0.5rem 0.7rem;
  }
  .explain .tile .k {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6b7892;
  }
  .explain .tile .v { color: #cbd3df; font-size: 0.8rem; margin-top: 0.12rem; }

  .hit-grid {
    display: grid; grid-template-columns: repeat(1, 1fr); gap: 0.55rem;
  }
  @media (min-width: 700px)  { .hit-grid { grid-template-columns: repeat(2, 1fr); } }
  @media (min-width: 1100px) { .hit-grid { grid-template-columns: repeat(3, 1fr); } }
  .hit-card {
    background: linear-gradient(180deg, rgba(75, 86, 111, 0.22) 0%, rgba(38, 45, 63, 0.55) 100%);
    border: 1px solid rgba(155, 166, 185, 0.16);
    border-radius: 0.8rem; padding: 0.7rem 0.8rem;
  }
  .hit-head {
    display: flex; justify-content: space-between; align-items: center;
    font-family: ui-monospace, monospace; font-size: 0.7rem; color: #9aa6b9;
    margin-bottom: 0.4rem;
  }
  .mini-bar {
    height: 0.32rem; background: rgba(38, 45, 63, 0.7); border-radius: 9999px; overflow: hidden;
  }
  .mini-bar .fill { height: 100%; background: #4b566f; }
  .mini-bar .fill.win { background: linear-gradient(90deg, #10b981, #22d3ee); }
  .mini-row {
    display: flex; justify-content: space-between; align-items: center;
    font-family: ui-monospace, monospace; font-size: 0.62rem; color: #9aa6b9;
    margin-top: 0.32rem;
  }
  .mini-row .v { color: #e4e9f0; }

  .why {
    border-left: 3px solid #22d3ee; padding-left: 0.85rem;
    margin-top: 0.5rem; color: #cbd3df; font-size: 0.85rem; line-height: 1.55;
  }

  div[data-testid="stRadio"] > div { gap: 0.4rem; }
  .stButton > button {
    width: 100%; height: 3.05rem;
    background: linear-gradient(135deg, #10b981 0%, #22d3ee 100%);
    color: #05070d; font-weight: 700; border: none;
    box-shadow: 0 0 32px -10px rgba(16, 185, 129, 0.55);
  }
  .stButton > button:hover { transform: translateY(-1px); }
  .stButton > button:disabled {
    background: rgba(75, 86, 111, 0.35); color: #6b7892; box-shadow: none;
  }
  .stRadio label[data-testid="stWidgetLabel"] { display: none; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem; background: transparent; border-bottom: 1px solid rgba(155,166,185,0.15);
  }
  .stTabs [data-baseweb="tab"] {
    background: rgba(24, 29, 44, 0.55);
    border: 1px solid rgba(155, 166, 185, 0.12);
    border-bottom: none;
    border-radius: 0.55rem 0.55rem 0 0;
    color: #9aa6b9; font-weight: 600;
    padding: 0.55rem 1rem;
  }
  .stTabs [data-baseweb="tab"]:hover { color: #f5f7fa; }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(34, 211, 238, 0.18) 0%, rgba(24, 29, 44, 0.85) 100%) !important;
    color: #67e8f9 !important;
    border-color: rgba(34, 211, 238, 0.4) !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem; }

  /* Warnings */
  .warning {
    background: rgba(245, 158, 11, 0.10);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #fcd34d;
    padding: 0.65rem 0.85rem; border-radius: 0.65rem; font-size: 0.83rem;
    display: flex; align-items: flex-start; gap: 0.55rem;
  }
  .warning + .warning { margin-top: 0.4rem; }

  /* About-tab specific */
  .photo-card {
    background: rgba(24, 29, 44, 0.65); border: 1px solid rgba(155, 166, 185, 0.14);
    border-radius: 0.85rem; overflow: hidden;
  }
  .photo-cap {
    padding: 0.55rem 0.85rem; font-size: 0.78rem; color: #9aa6b9;
    border-top: 1px solid rgba(155, 166, 185, 0.10);
  }
  .stat-card {
    text-align: center; padding: 1rem 0.85rem;
    background: rgba(24, 29, 44, 0.65); border: 1px solid rgba(155, 166, 185, 0.14);
    border-radius: 0.85rem;
  }
  .stat-card .num {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #10b981, #22d3ee);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    line-height: 1;
  }
  .stat-card .lbl {
    color: #9aa6b9; font-size: 0.72rem; letter-spacing: 0.05em; text-transform: uppercase;
    margin-top: 0.45rem; font-weight: 600;
  }
  .feat-tile {
    background: rgba(24, 29, 44, 0.55); border: 1px solid rgba(155, 166, 185, 0.14);
    border-radius: 0.7rem; padding: 0.7rem 0.85rem;
  }
  .feat-tile .k { font-size: 0.72rem; font-weight: 700; color: #f5f7fa; }
  .feat-tile .n { font-size: 0.68rem; color: #67e8f9; font-family: ui-monospace, monospace; margin-left: 0.3rem; }
  .feat-tile .v { font-size: 0.72rem; color: #9aa6b9; margin-top: 0.25rem; line-height: 1.4; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_model() -> FlangeInvariantLR | None:
    try:
        return FlangeInvariantLR(MODEL_DIR)
    except Exception:
        return None


def _flatten_html(s: str) -> str:
    s = textwrap.dedent(s)
    return "\n".join(line.lstrip() for line in s.splitlines() if line.strip())


def _render_html(html: str) -> None:
    cleaned = _flatten_html(html)
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


def confidence_class(level: str) -> str:
    return level if level in ("high", "medium", "low") else "medium"


def _stretch_image(path: str, caption: str = "") -> None:
    """Render an image stretched to its container, working across Streamlit versions."""
    try:
        st.image(path, width="stretch", caption=caption)
    except TypeError:
        try:
            st.image(path, use_container_width=True, caption=caption)
        except TypeError:
            st.image(path, use_column_width=True, caption=caption)


def _stretch_df(df, height: int = 300) -> None:
    try:
        st.dataframe(df, height=height, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(df, height=height, hide_index=True, use_container_width=True)


def _stretch_plot(fig, key: str | None = None) -> None:
    cfg = {"displayModeBar": False}
    try:
        if key is not None:
            st.plotly_chart(fig, width="stretch", config=cfg, key=key)
        else:
            st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:
        if key is not None:
            st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)
        else:
            st.plotly_chart(fig, use_container_width=True, config=cfg)


def render_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    html = "".join(
        f'<div class="warning"><span style="font-weight:700;">!</span><span>{w}</span></div>'
        for w in warnings
    )
    _render_html(html)


def build_main_waveform_figure(j: dict) -> go.Figure:
    sr = j.get("sample_rate") or 1
    duration = j.get("duration_sec") or 1.0
    wave = j.get("waveform", {}) or {}
    env = j.get("envelope", {}) or {}
    peaks = j.get("hit_times_sec") or []

    fig = go.Figure()

    wv = wave.get("values") or []
    if wv:
        x = np.linspace(0, duration, num=len(wv))
        fig.add_trace(go.Scatter(
            x=x, y=wv, mode="lines",
            line=dict(color="rgba(34, 211, 238, 0.85)", width=1),
            name="raw waveform", hoverinfo="skip",
        ))

    ev = env.get("values") or []
    if ev:
        env_off = env.get("offset_sec", 0.0)
        env_len_samples = env.get("length", len(ev))
        env_dur_sec = env_len_samples / sr if sr else 0.0
        x = np.linspace(env_off, env_off + env_dur_sec, num=len(ev))
        fig.add_trace(go.Scatter(
            x=x, y=ev, mode="lines",
            line=dict(color="rgba(16, 185, 129, 0.95)", width=1.6),
            name="envelope", hoverinfo="skip",
        ))

    for i, t in enumerate(peaks):
        fig.add_vline(
            x=t, line_color="rgba(245, 158, 11, 0.9)", line_width=1.5,
            annotation_text=f"#{i+1}", annotation_position="top",
            annotation_font_color="#fcd34d", annotation_font_size=10,
        )

    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5, 7, 13, 0.55)",
        xaxis=dict(title="time (s)", showgrid=True, gridcolor="rgba(155, 166, 185, 0.08)",
                   zeroline=False, color="#9aa6b9"),
        yaxis=dict(showgrid=True, gridcolor="rgba(155, 166, 185, 0.08)",
                   zeroline=False, color="#9aa6b9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd3df", size=10)),
    )
    return fig


def render_prob_bars(avg_proba: dict, winning_class: int) -> str:
    rows = []
    for c in CLASSES_REF:
        c_int = int(c)
        p = float(avg_proba.get(str(c_int), 0.0))
        win = c_int == winning_class
        win_cls = "win" if win else ""
        rows.append(f"""
            <div class="probrow">
                <div class="head">
                    <span class="lbl {win_cls}">{c_int} ft-lbs</span>
                    <span class="val {win_cls}">{p * 100:.1f}%</span>
                </div>
                <div class="probtrack">
                    <div class="probfill {win_cls}" style="width: {p * 100:.2f}%"></div>
                </div>
            </div>
        """)
    return "".join(rows)


def render_hit_card(hit: dict) -> str:
    cls = int(hit["predicted_torque"])
    conf = float(hit["confidence"])
    lvl = "high" if conf >= 0.7 else "medium" if conf >= 0.5 else "low"
    t = hit.get("time_sec")
    t_txt = f"{t:.2f}s" if t is not None else ""

    bars = []
    for c in CLASSES_REF:
        c_int = int(c)
        p = float(hit["probabilities"].get(str(c_int), 0.0))
        win_cls = "win" if c_int == cls else ""
        bars.append(f"""
            <div class="mini-row">
                <span>{c_int} ft-lbs</span>
                <span class="v">{p * 100:.1f}%</span>
            </div>
            <div class="mini-bar">
                <div class="fill {win_cls}" style="width: {p * 100:.2f}%"></div>
            </div>
        """)

    return f"""
        <div class="hit-card">
            <div class="hit-head">
                <span>Hit #{hit['hit_id']} · {t_txt}</span>
                <span class="pill {lvl}">{cls} ft-lbs</span>
            </div>
            {''.join(bars)}
        </div>
    """


# ---------------------------------------------------------------- #
# Header / hero
# ---------------------------------------------------------------- #
_render_html("""
<div class="ribbon">
    <span style="display:inline-block; width:8px; height:8px; border-radius:9999px; background:#22d3ee;"></span>
    UH Machine Learning Competition 2026 · Bolted-Flange Looseness Detection
</div>
<div class="hero-title">
    Flange-Invariant<br/>
    <span class="hero-grad">Acoustic Bolt-Looseness Detector</span>
</div>
<p class="hero-sub">
    Live percussion-based torque classification for the
    <strong style="color:#f5f7fa;">UH ML Competition</strong>. Strike a bolted flange and
    the model reports the most likely bolt preload &mdash;
    <span style="color:#10b981;font-family:ui-monospace,monospace;">0</span>,
    <span style="color:#22d3ee;font-family:ui-monospace,monospace;">25</span>, or
    <span style="color:#f59e0b;font-family:ui-monospace,monospace;">50 ft-lbs</span>
    &mdash; from its acoustic ring-down.
</p>
<div class="pipe-strip">
    <span class="pipe-pill">Raw Audio</span>
    <span class="pipe-arrow">&rarr;</span>
    <span class="pipe-pill">Hit Detection</span>
    <span class="pipe-arrow">&rarr;</span>
    <span class="pipe-pill">150 Features</span>
    <span class="pipe-arrow">&rarr;</span>
    <span class="pipe-pill">Flange-Invariant LR</span>
    <span class="pipe-arrow">&rarr;</span>
    <span class="pipe-pill final">Torque Prediction</span>
</div>
""")

st.write("")

# Model load + status pill
model = load_model()
if model is None:
    _render_html("""
    <div class="warning">
        <span style="font-weight:700;">!</span>
        <span>Model artifacts could not be loaded. Run
        <code>python backend/train_and_export.py</code> to regenerate them.</span>
    </div>
    """)
else:
    acc = model.metadata.get("lofo_calibrated_hit_accuracy")
    file_acc = model.metadata.get("lofo_file_accuracy_softvote")
    bits = ['<span class="pill green">model ready</span>']
    if acc is not None:
        bits.append(f'<span class="pill cyan">LOFO hit {acc * 100:.1f}%</span>')
    if file_acc is not None:
        bits.append(f'<span class="pill cyan">LOFO file {file_acc * 100:.1f}%</span>')
    _render_html(" ".join(bits))


# ---------------------------------------------------------------- #
# TABS
# ---------------------------------------------------------------- #
tab_predict, tab_features, tab_about = st.tabs(["Predict", "Features", "About"])


# ================================================================ #
# TAB 1: Predict
# ================================================================ #
with tab_predict:
    left, right = st.columns([1, 2], gap="large")

    with left:
        _render_html('<div class="label">Flange ID</div>'
                     '<div class="label-sub">Required for per-flange centering</div>')
        flange_id = st.radio(
            "flange",
            options=[1, 2, 3, 4],
            format_func=lambda x: f"F{x}",
            horizontal=True,
            label_visibility="collapsed",
        )

        st.write("")
        _render_html('<div class="label">Live Recording</div>'
                     '<div class="label-sub">Strike the flange 3–6 times, then stop.</div>')
        if hasattr(st, "audio_input"):
            recorded = st.audio_input("Record hammer hits", label_visibility="collapsed")
        else:
            recorded = None
            st.info("Live recording requires Streamlit ≥ 1.39. Use the file uploader below.")

        st.write("")
        _render_html('<div class="label">Or Upload Audio</div>'
                     '<div class="label-sub">.m4a · .wav · .mp3 · .webm · .ogg</div>')
        uploaded = st.file_uploader(
            "audio",
            type=["m4a", "wav", "mp3", "webm", "ogg", "flac"],
            label_visibility="collapsed",
        )

        audio_bytes: bytes | None = None
        audio_name: str | None = None
        if recorded is not None:
            audio_bytes = recorded.getvalue()
            ext = ".wav"
            if hasattr(recorded, "type") and recorded.type:
                t = recorded.type.lower()
                if "webm" in t: ext = ".webm"
                elif "mp4" in t: ext = ".m4a"
                elif "ogg" in t: ext = ".ogg"
                elif "wav" in t: ext = ".wav"
            audio_name = f"recording{ext}"
        elif uploaded is not None:
            audio_bytes = uploaded.getvalue()
            audio_name = uploaded.name

        st.write("")
        predict = st.button(
            "Predict Torque",
            disabled=(audio_bytes is None or model is None),
            type="primary",
        )

        _render_html("""
        <div class="glass" style="margin-top:1.2rem;">
            <div class="label">Why flange-invariant?</div>
            <div class="why">
                Each physical flange has its own resonant fingerprint &mdash; geometry, mounting,
                microphone position. The model subtracts that flange's typical acoustic signature
                before classifying, so it focuses on <strong>torque-related sound changes</strong>
                rather than memorising the flange itself.
            </div>
        </div>
        """)

    with right:
        if predict and audio_bytes is not None and model is not None:
            with st.spinner("Analyzing audio…"):
                ext = (Path(audio_name or "audio.wav").suffix or ".wav").lower()
                try:
                    result = model.predict_from_audio(
                        audio_bytes=audio_bytes,
                        flange_id=int(flange_id),
                        file_extension=ext,
                    )
                except Exception as e:
                    result = {
                        "ok": False, "warnings": [f"Prediction failed: {e}"],
                        "n_hits": 0, "per_hit": [],
                        "averaged_probabilities": None, "final_prediction": None,
                        "flange_id": int(flange_id), "sample_rate": None,
                        "duration_sec": None,
                        "waveform": {"values": []}, "envelope": {"values": []},
                        "hit_times_sec": [],
                    }
            st.session_state["last_result"] = result

        result = st.session_state.get("last_result")

        if result is None:
            _render_html("""
            <div class="glass" style="text-align:center; padding: 2.5rem 1.5rem;">
                <div style="font-size:1.05rem; font-weight:600; color:#f5f7fa;">No prediction yet</div>
                <div style="color:#9aa6b9; font-size:0.85rem; max-width:480px; margin: 0.5rem auto 0;">
                    Pick a flange, then either record a few hammer strikes or upload an audio file.
                    The detected hits, calibrated probabilities, and final torque will appear here.
                </div>
            </div>
            """)
        else:
            render_warnings(result.get("warnings", []) or [])

            fp = result.get("final_prediction")
            if fp:
                torque = int(fp["torque_ftlbs"])
                conf = float(fp["confidence"])
                lvl = confidence_class(fp.get("confidence_level", "medium"))
                avg = result["averaged_probabilities"]
                sr = result.get("sample_rate")
                dur = result.get("duration_sec")
                n_hits = result.get("n_hits", 0)

                _render_html(f"""
                <div class="glass final-card {lvl}">
                    <div style="display:flex; flex-wrap:wrap; gap:1rem; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <div class="label">Final Prediction</div>
                            <div class="final-badge {lvl}">Predicted Torque: {torque} ft-lbs</div>
                            <div class="final-meta">
                                Flange F{result.get('flange_id')} · {n_hits} hit{'s' if n_hits != 1 else ''}
                                · {dur:.2f} s · {sr} Hz
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div class="label">Confidence</div>
                            <div class="conf-num">{conf * 100:.1f}%</div>
                            <div style="margin-top:0.3rem;"><span class="pill {lvl}">{lvl} confidence</span></div>
                        </div>
                    </div>
                    {render_prob_bars(avg, torque)}
                    <div class="explain">
                        <div class="tile"><div class="k">Hits detected</div><div class="v">{n_hits}</div></div>
                        <div class="tile"><div class="k">Aggregation</div><div class="v">Soft-vote average across hits</div></div>
                        <div class="tile"><div class="k">Decision rule</div><div class="v">argmax of averaged probabilities</div></div>
                    </div>
                </div>
                """)

            if result.get("waveform", {}).get("values"):
                _render_html(f"""
                <div class="glass" style="margin-top:0.85rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div class="label">Waveform &amp; Detected Hits</div>
                            <div class="label-sub">Smoothed envelope and peak markers over the raw signal.</div>
                        </div>
                        <span class="pill cyan">{result.get('n_hits', 0)} hit{'s' if result.get('n_hits', 0) != 1 else ''}</span>
                    </div>
                </div>
                """)
                fig = build_main_waveform_figure(result)
                _stretch_plot(fig)

            per_hit = result.get("per_hit") or []
            if per_hit:
                _render_html("""
                <div class="glass" style="margin-top:0.85rem;">
                    <div class="label">Per-Hit Predictions</div>
                    <div class="label-sub">Each card shows one hammer strike and its calibrated probabilities.</div>
                </div>
                """)
                cards_html = '<div class="hit-grid">' + ''.join(render_hit_card(h) for h in per_hit) + '</div>'
                _render_html(cards_html)


# ================================================================ #
# TAB 2: Features
# ================================================================ #
with tab_features:
    result = st.session_state.get("last_result")

    if result is None or not (result.get("per_hit") or []):
        _render_html("""
        <div class="glass" style="text-align:center; padding: 2.5rem 1.5rem;">
            <div style="font-size:1.05rem; font-weight:600; color:#f5f7fa;">Run a prediction first</div>
            <div style="color:#9aa6b9; font-size:0.85rem; max-width:480px; margin: 0.5rem auto 0;">
                Once you analyse audio in the <strong>Predict</strong> tab, this view will show the
                full 150-D feature vector for every detected hit, the mel-spectrograms,
                and where each hit's signal lives across feature groups.
            </div>
        </div>
        """)
    else:
        groups: dict = result.get("feature_groups", {})
        per_hit = result["per_hit"]
        n_hits = len(per_hit)
        feat_dim = len(per_hit[0]["features_full"])
        sel_idx = set(result.get("selected_feature_indices", []))

        # ---------- Feature group breakdown ----------
        _render_html(f"""
        <div class="glass">
            <div class="label">Feature Anatomy</div>
            <div class="label-sub">
                Each detected hit is collapsed into a {feat_dim}-D vector
                ({len(sel_idx)} of these survive ANOVA selection for the LR).
                Below: what each block measures, and how strongly it speaks for this recording.
            </div>
        </div>
        """)

        cols = st.columns(3)
        for i, (gname, g) in enumerate(groups.items()):
            n_sel_in_group = sum(1 for idx in sel_idx
                                 if g["start"] <= idx < g["start"] + g["length"])
            with cols[i % 3]:
                _render_html(f"""
                <div class="feat-tile">
                    <div><span class="k">{gname}</span><span class="n">{g['length']} dims</span></div>
                    <div class="v">{g['blurb']}</div>
                    <div class="v" style="margin-top:0.4rem; color:#22d3ee;">
                        {n_sel_in_group} / {g['length']} kept by feature selection
                    </div>
                </div>
                """)

        # ---------- Group energy bar chart (averaged across hits) ----------
        st.write("")
        _render_html("""
        <div class="glass">
            <div class="label">Where Each Hit's Signal Lives</div>
            <div class="label-sub">
                Mean |z-score| of each feature group, per hit. Higher bars mean
                that group is responding strongly relative to the rest of the vector.
            </div>
        </div>
        """)

        group_names = list(groups.keys())
        ge_matrix = np.array([
            [hit["feature_group_energy"].get(g, 0.0) for g in group_names]
            for hit in per_hit
        ])  # shape (n_hits, n_groups)

        fig_ge = go.Figure()
        for gi, gname in enumerate(group_names):
            fig_ge.add_trace(go.Bar(
                name=gname,
                x=[f"#{h['hit_id']}" for h in per_hit],
                y=ge_matrix[:, gi],
                marker_color=[
                    "#22d3ee", "#10b981", "#f59e0b", "#a78bfa", "#f472b6", "#fb7185",
                ][gi % 6],
            ))
        fig_ge.update_layout(
            barmode="stack", height=320, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,7,13,0.55)",
            xaxis=dict(title="hit", color="#9aa6b9", showgrid=False),
            yaxis=dict(title="mean |z-score| per group", color="#9aa6b9",
                       gridcolor="rgba(155,166,185,0.08)"),
            legend=dict(font=dict(color="#cbd3df", size=10),
                        bgcolor="rgba(0,0,0,0)"),
        )
        _stretch_plot(fig_ge)

        # ---------- 150-D feature heatmap (z-scored, per-hit rows) ----------
        st.write("")
        _render_html("""
        <div class="glass">
            <div class="label">150-D Feature Matrix</div>
            <div class="label-sub">
                Each row is one detected hit, each column is one of the 150 raw features
                (in extraction order, before flange-invariant centering). Colour is the
                z-score across this recording — bright spots are the features that
                make these hits distinctive.
            </div>
        </div>
        """)

        F = np.array([h["features_full"] for h in per_hit])  # (n_hits, 150)
        Fz = (F - F.mean(axis=0, keepdims=True)) / (F.std(axis=0, keepdims=True) + 1e-9)

        # Build group boundary annotations
        shapes = []
        annotations = []
        for gname, g in groups.items():
            shapes.append(dict(
                type="line", xref="x", yref="paper",
                x0=g["start"] - 0.5, x1=g["start"] - 0.5, y0=0, y1=1,
                line=dict(color="rgba(155,166,185,0.4)", width=1, dash="dot"),
            ))
            annotations.append(dict(
                xref="x", yref="paper",
                x=g["start"] + g["length"] / 2 - 0.5, y=1.04,
                showarrow=False, text=gname,
                font=dict(color="#9aa6b9", size=10),
            ))

        fig_heat = go.Figure(go.Heatmap(
            z=Fz, colorscale="RdBu_r", zmid=0, zmin=-3, zmax=3,
            colorbar=dict(title="z", thickness=10, tickfont=dict(color="#9aa6b9")),
            hovertemplate="hit #%{y}<br>feature %{x}<br>z=%{z:.2f}<extra></extra>",
        ))
        fig_heat.update_layout(
            height=max(220, 30 + 22 * n_hits),
            margin=dict(l=40, r=10, t=44, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,7,13,0.55)",
            xaxis=dict(title="feature index (0…149)", color="#9aa6b9",
                       showgrid=False, zeroline=False),
            yaxis=dict(title="hit", color="#9aa6b9",
                       tickmode="array",
                       tickvals=list(range(n_hits)),
                       ticktext=[f"#{h['hit_id']}" for h in per_hit],
                       autorange="reversed"),
            shapes=shapes, annotations=annotations,
        )
        _stretch_plot(fig_heat)

        # ---------- Per-hit spectrograms ----------
        st.write("")
        _render_html("""
        <div class="glass">
            <div class="label">Per-Hit Mel Spectrograms</div>
            <div class="label-sub">
                Time-frequency energy of each hammer impact (dB scale).
                Tighter rings of energy near the top = brisker decay = tighter bolt.
            </div>
        </div>
        """)

        cols_per_row = 3
        for row_start in range(0, n_hits, cols_per_row):
            row_hits = per_hit[row_start: row_start + cols_per_row]
            row_cols = st.columns(cols_per_row)
            for ci, h in enumerate(row_hits):
                ms = h.get("mel_spectrogram") or {}
                values = ms.get("values") or []
                with row_cols[ci]:
                    if not values:
                        _render_html(f"""
                        <div class="glass" style="padding:0.7rem;">
                            <div class="label">Hit #{h['hit_id']}</div>
                            <div class="label-sub">spectrogram unavailable</div>
                        </div>
                        """)
                        continue
                    z = np.array(values)
                    fig_s = go.Figure(go.Heatmap(
                        z=z, colorscale="Viridis", zmin=-80, zmax=0,
                        showscale=False,
                        hovertemplate="mel %{y}<br>frame %{x}<br>%{z:.1f} dB<extra></extra>",
                    ))
                    cls = int(h["predicted_torque"])
                    fig_s.update_layout(
                        height=180, margin=dict(l=8, r=8, t=30, b=8),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,7,13,0.55)",
                        title=dict(
                            text=f"Hit #{h['hit_id']} → {cls} ft-lbs ({h['confidence']*100:.0f}%)",
                            font=dict(color="#cbd3df", size=12), x=0.02, xanchor="left",
                        ),
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                    )
                    _stretch_plot(fig_s, key=f"spec_{h['hit_id']}")

        # ---------- Raw feature table ----------
        st.write("")
        _render_html("""
        <div class="glass">
            <div class="label">Raw Feature Values</div>
            <div class="label-sub">
                Numerical values for every hit and every feature, with the
                feature group and whether it survived ANOVA selection.
            </div>
        </div>
        """)

        # Build a long-form dataframe
        idx_to_group = []
        for fi in range(feat_dim):
            for gname, g in groups.items():
                if g["start"] <= fi < g["start"] + g["length"]:
                    idx_to_group.append(gname)
                    break
            else:
                idx_to_group.append("?")

        import pandas as pd  # local import keeps the cold start tighter
        rows = []
        for h in per_hit:
            for fi, val in enumerate(h["features_full"]):
                rows.append({
                    "hit": h["hit_id"],
                    "feature_idx": fi,
                    "group": idx_to_group[fi],
                    "selected": "✓" if fi in sel_idx else "",
                    "value": float(val),
                })
        df = pd.DataFrame(rows)
        _stretch_df(df, height=300)


# ================================================================ #
# TAB 3: About
# ================================================================ #
with tab_about:
    md = (model.metadata if model is not None else {}) or {}

    # Method statement
    _render_html("""
    <div class="glass">
        <div class="label">The Problem</div>
        <div class="why" style="border-left-color:#10b981;">
            Bolted flanges hold pressurised pipelines together. As bolts loosen, they release
            their preload, and the flange's acoustic ring-down after a hammer strike changes —
            ringing longer and brighter when bolts are loose, shorter and damped when fully torqued.
            Our task is to read that change and recover the bolt preload from sound alone.
        </div>
        <div class="label" style="margin-top:1rem;">Our Method</div>
        <div class="why">
            We strike each flange with a steel hammer at four locations (Areas 1–4) and record
            at 48 kHz. The signal is segmented into individual impact events, and each event is
            converted into a 150-D feature vector that captures spectral content
            (Welch log-PSD, MFCCs, spectral statistics) <em>and</em> ring-down dynamics
            (per-band T60-style decay).
            <br/><br/>
            The classifier is a <strong>Flange-Invariant Logistic Regression</strong>:
            we ANOVA-select the 100 features whose F-ratio for torque is largest relative
            to flange identity, subtract each flange's mean feature vector, and fit a
            logistic regression with per-class isotonic calibration. At inference, we
            soft-vote the calibrated per-hit probabilities to produce one prediction per recording.
        </div>
    </div>
    """)

    # ---- Stats row
    if md:
        st.write("")
        cols = st.columns(4)
        stats = [
            ("Training files",      f"{md.get('n_training_files', '—')}"),
            ("Single-hit samples",  f"{md.get('n_training_hits', '—')}"),
            ("LOFO hit accuracy",   f"{(md.get('lofo_calibrated_hit_accuracy') or 0) * 100:.1f}%"),
            ("LOFO file accuracy",  f"{(md.get('lofo_file_accuracy_softvote') or 0) * 100:.1f}%"),
        ]
        for col, (lbl, num) in zip(cols, stats):
            with col:
                _render_html(f"""
                <div class="stat-card">
                    <div class="num">{num}</div>
                    <div class="lbl">{lbl}</div>
                </div>
                """)

    # ---- Photo gallery
    st.write("")
    _render_html("""
    <div class="glass">
        <div class="label">Experimental Setup</div>
        <div class="label-sub">
            Four bolted flanges along a steel pipeline. Each flange is struck at four
            locations (Areas 1–4) for redundancy and the audio is captured by a phone microphone.
        </div>
    </div>
    """)

    # Pedagogical assembly diagram (kept from poster) + the user's lab photos
    p_assembly = ASSETS_DIR / "pipeline_assembly.png"
    p_flanges  = ROOT / "flanges.png"
    p_close    = ROOT / "IMG_1976.jpg"
    p_lab      = ROOT / "IMG_1979.jpg"
    p_torque   = ROOT / "IMG_1984.jpg"

    # Top row: the three "what" photos
    col_a, col_b, col_c = st.columns([1, 1, 1.2])
    with col_a:
        if p_assembly.exists():
            _stretch_image(str(p_assembly),
                           "Four flanges along the pipeline (F1 nearest, F4 farthest).")
    with col_b:
        if p_flanges.exists():
            _stretch_image(str(p_flanges),
                           "One flange labelled with the 4 hammer-strike areas.")
    with col_c:
        if p_close.exists():
            _stretch_image(str(p_close),
                           "Close-up of flange F4 — three bolts secure each flange.")

    # Bottom row: the two "how" photos
    col_d, col_e = st.columns([1.4, 1])
    with col_d:
        if p_lab.exists():
            _stretch_image(str(p_lab),
                           "Lab setup: the steel pipeline mounted on stands, "
                           "instrumented for percussion measurements.")
    with col_e:
        if p_torque.exists():
            _stretch_image(str(p_torque),
                           "Setting bolt preload with a torque wrench before each session "
                           "(0, 25, or 50 ft-lbs).")

    # ---- Confusion matrices
    if md.get("confusion_matrices"):
        st.write("")
        _render_html("""
        <div class="glass">
            <div class="label">Leave-One-Flange-Out Performance</div>
            <div class="label-sub">
                Honest cross-validation: train on three flanges, test on the held-out one.
                Repeat for each flange, concatenate predictions. This is the regime the
                competition rewards.
            </div>
        </div>
        """)
        cm_blocks = md["confusion_matrices"]
        labels = cm_blocks.get("labels", [0, 25, 50])

        def cm_figure(cm: list, title: str) -> go.Figure:
            cm_arr = np.array(cm, dtype=float)
            row_sum = cm_arr.sum(axis=1, keepdims=True)
            cm_pct = np.divide(cm_arr, np.where(row_sum == 0, 1, row_sum)) * 100.0
            text = [[f"<b>{int(cm_arr[i, j])}</b><br>{cm_pct[i, j]:.0f}%"
                     for j in range(cm_arr.shape[1])]
                    for i in range(cm_arr.shape[0])]
            fig = go.Figure(go.Heatmap(
                z=cm_pct,
                x=[f"{c} ft-lbs" for c in labels],
                y=[f"{c} ft-lbs" for c in labels],
                text=text, texttemplate="%{text}",
                textfont=dict(color="white", size=14),
                colorscale=[[0, "rgba(34,211,238,0.0)"], [0.5, "rgba(34,211,238,0.5)"],
                            [1, "rgba(16,185,129,1.0)"]],
                zmin=0, zmax=100, showscale=False,
                hovertemplate="true %{y}<br>predicted %{x}<br>%{z:.1f}%<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=title, font=dict(color="#cbd3df", size=13), x=0.02),
                height=300, margin=dict(l=10, r=10, t=44, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,7,13,0.55)",
                xaxis=dict(title="predicted", color="#9aa6b9", side="bottom"),
                yaxis=dict(title="true", color="#9aa6b9", autorange="reversed"),
            )
            return fig

        c1, c2 = st.columns(2)
        with c1:
            _stretch_plot(cm_figure(cm_blocks["hit_level_calibrated"], "Hit-level (calibrated)"))
        with c2:
            _stretch_plot(cm_figure(cm_blocks["file_level"], "File-level (soft-vote)"))

    # ---- Per-class recall bar
    if md.get("per_class_recall"):
        rec = md["per_class_recall"]
        labels = md["confusion_matrices"]["labels"]
        fig = go.Figure()
        for series_name, key, color in [
            ("hit-level (raw)",        "hit_level_raw",        "rgba(34,211,238,0.6)"),
            ("hit-level (calibrated)", "hit_level_calibrated", "rgba(16,185,129,0.9)"),
            ("file-level (soft-vote)", "file_level",           "rgba(245,158,11,0.9)"),
        ]:
            data = rec.get(key, {})
            fig.add_trace(go.Bar(
                name=series_name,
                x=[f"{c} ft-lbs" for c in labels],
                y=[float(data.get(str(int(c)), 0.0)) * 100.0 for c in labels],
                marker_color=color,
            ))
        fig.update_layout(
            barmode="group", height=300, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,7,13,0.55)",
            xaxis=dict(color="#9aa6b9"),
            yaxis=dict(title="recall (%)", color="#9aa6b9",
                       gridcolor="rgba(155,166,185,0.08)", range=[0, 100]),
            legend=dict(font=dict(color="#cbd3df", size=10),
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.write("")
        _render_html("""
        <div class="glass"><div class="label">Per-Class Recall</div>
        <div class="label-sub">Calibration mainly helps the middle class (25 ft-lbs); soft-vote
        on top of the calibrated probabilities is what tips file-level accuracy past 85%.</div>
        </div>""")
        _stretch_plot(fig)

    # ---- Training data composition
    if md.get("training_data"):
        td = md["training_data"]
        c1, c2 = st.columns(2)

        def comp_fig(d: dict, title: str, x_label: str) -> go.Figure:
            xs = sorted(d.keys(), key=lambda k: int(k))
            ys = [d[k] for k in xs]
            fig = go.Figure(go.Bar(
                x=[str(k) for k in xs], y=ys,
                marker_color=["#22d3ee", "#10b981", "#f59e0b", "#a78bfa"][:len(xs)],
                text=ys, textposition="outside",
                textfont=dict(color="#cbd3df", size=11),
            ))
            fig.update_layout(
                title=dict(text=title, font=dict(color="#cbd3df", size=13), x=0.02),
                height=240, margin=dict(l=10, r=10, t=44, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,7,13,0.55)",
                xaxis=dict(title=x_label, color="#9aa6b9"),
                yaxis=dict(title="hits", color="#9aa6b9",
                           gridcolor="rgba(155,166,185,0.08)"),
                showlegend=False,
            )
            return fig

        with c1:
            _stretch_plot(comp_fig(td.get("hits_per_class", {}),
                                   "Training hits per class", "torque (ft-lbs)"))
        with c2:
            _stretch_plot(comp_fig(td.get("hits_per_flange", {}),
                                   "Training hits per flange", "flange ID"))

    # ---- Why flange-invariant (extended)
    st.write("")
    _render_html("""
    <div class="glass">
        <div class="label">Why "Flange-Invariant"?</div>
        <div class="why">
            A naïve classifier on raw audio features tends to memorise <em>which flange</em>
            it is hearing, not <em>how loose</em> it is. Each physical flange has its own
            resonant fingerprint — geometry, mounting, microphone placement, room acoustics.
            <br/><br/>
            Our pipeline does two things to break that crutch:
            <br/>
            <strong>1. Feature selection by F(torque) / (F(flange) + 1)</strong> — keep only the
            100 features whose variance is best explained by torque relative to flange identity.
            <br/>
            <strong>2. Per-flange centering</strong> — subtract the training mean of each flange
            from its features. Whatever survives is, by construction, the deviation from that
            flange's typical signature.
            <br/><br/>
            That is why the same model generalises across recording sessions and devices.
        </div>
    </div>
    """)


# ---------------------------------------------------------------- #
# Footer
# ---------------------------------------------------------------- #
_render_html("""
<div style="margin-top:2rem; padding-top:1rem; border-top:1px solid rgba(155,166,185,0.12);
            text-align:center; color:#6b7892; font-size:0.75rem;">
    Built for the UH Machine Learning Competition 2026 · Model: Flange-Invariant LR with per-class isotonic calibration · Soft-vote across hits.
</div>
""")

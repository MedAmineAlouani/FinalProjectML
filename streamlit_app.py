"""
Streamlit deployment for the Flange-Invariant Acoustic Bolt-Looseness Detector.

Designed for Streamlit Community Cloud (https://share.streamlit.io). Reuses the
exact backend modules (feature_extraction.py, segmentation.py, inference.py)
that the FastAPI version uses, so predictions are identical.

Run locally:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from inference import CLASSES_REF, FlangeInvariantLR

MODEL_DIR = ROOT / "backend" / "model"

# ---------------------------------------------------------------- #
# Page + style
# ---------------------------------------------------------------- #
st.set_page_config(
    page_title="Flange-Invariant Bolt-Looseness Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
  /* layout */
  .main .block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1280px; }
  header[data-testid="stHeader"] { background: transparent; }
  #MainMenu, footer { visibility: hidden; }

  /* page background gradient */
  .stApp {
    background:
      radial-gradient(1200px 700px at 10% -10%, rgba(34, 211, 238, 0.10), transparent 60%),
      radial-gradient(1100px 600px at 90% 110%, rgba(16, 185, 129, 0.08), transparent 60%),
      linear-gradient(180deg, #05070d 0%, #0a0d15 100%);
  }

  /* ribbon banner */
  .ribbon {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.55rem 0.9rem;
    border-radius: 9999px;
    background: rgba(34, 211, 238, 0.08);
    border: 1px solid rgba(34, 211, 238, 0.25);
    color: #67e8f9;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
    width: max-content;
  }

  /* hero title */
  .hero-title {
    font-size: clamp(2rem, 4vw, 3.8rem);
    font-weight: 800; line-height: 1.05; letter-spacing: -0.02em;
    color: #f5f7fa; margin-top: 1rem;
  }
  .hero-grad {
    background: linear-gradient(90deg, #10b981 0%, #22d3ee 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .hero-sub { color: #9aa6b9; max-width: 720px; margin-top: 0.9rem; font-size: 1rem; line-height: 1.55; }

  /* pipeline strip */
  .pipe-strip {
    display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
    margin-top: 1.5rem; font-family: ui-monospace, SFMono-Regular, JetBrains Mono, monospace;
    font-size: 0.85rem; color: #cbd3df;
  }
  .pipe-pill {
    padding: 0.4rem 0.75rem; border-radius: 0.6rem;
    background: linear-gradient(180deg, rgba(75, 86, 111, 0.22) 0%, rgba(38, 45, 63, 0.55) 100%);
    border: 1px solid rgba(155, 166, 185, 0.16);
  }
  .pipe-pill.final {
    background: rgba(16, 185, 129, 0.12); color: #6ee7b7; border-color: rgba(16, 185, 129, 0.35);
  }
  .pipe-arrow { color: #22d3ee; }

  /* generic glass card */
  .glass {
    background: rgba(24, 29, 44, 0.65);
    backdrop-filter: blur(14px) saturate(120%);
    border: 1px solid rgba(155, 166, 185, 0.12);
    border-radius: 1rem;
    padding: 1.1rem 1.2rem;
  }
  .glass + .glass { margin-top: 0.85rem; }

  /* stage cards */
  .stage-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.7rem; margin-top: 1.6rem;
  }
  @media (min-width: 900px) { .stage-grid { grid-template-columns: repeat(4, 1fr); } }
  .stage {
    display: flex; gap: 0.65rem; align-items: flex-start;
    background: rgba(24, 29, 44, 0.6);
    border: 1px solid rgba(155, 166, 185, 0.12);
    border-radius: 0.9rem; padding: 0.85rem 1rem;
  }
  .stage-num {
    width: 2.1rem; height: 2.1rem; border-radius: 0.65rem; flex: 0 0 auto;
    display: flex; align-items: center; justify-content: center;
    background: rgba(34, 211, 238, 0.12); color: #22d3ee; font-weight: 700;
  }
  .stage-num.final { background: rgba(16, 185, 129, 0.15); color: #10b981; }
  .stage-title { font-weight: 600; color: #f5f7fa; font-size: 0.95rem; }
  .stage-sub { color: #9aa6b9; font-size: 0.78rem; margin-top: 0.15rem; }

  /* section labels */
  .label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #cbd3df;
  }
  .label-sub { font-size: 0.72rem; color: #9aa6b9; margin-bottom: 0.6rem; }

  /* final prediction badge */
  .final-card { padding: 1.4rem 1.5rem; border-radius: 1.1rem; }
  .final-card.high   { box-shadow: 0 0 36px -8px rgba(16, 185, 129, 0.55); border-color: rgba(16, 185, 129, 0.25); }
  .final-card.medium { box-shadow: 0 0 36px -8px rgba(245, 158, 11, 0.55); border-color: rgba(245, 158, 11, 0.25); }
  .final-card.low    { box-shadow: 0 0 36px -8px rgba(239, 68, 68, 0.55);  border-color: rgba(239, 68, 68, 0.25); }
  .final-badge {
    font-size: clamp(1.6rem, 3.2vw, 2.6rem); font-weight: 800;
    letter-spacing: -0.01em; line-height: 1.1;
  }
  .final-badge.high   { color: #10b981; }
  .final-badge.medium { color: #f59e0b; }
  .final-badge.low    { color: #ef4444; }
  .final-meta { color: #9aa6b9; font-size: 0.82rem; margin-top: 0.35rem; }
  .conf-num { font-family: ui-monospace, monospace; font-size: 1.6rem; color: white; }
  .pill {
    display: inline-flex; padding: 0.2rem 0.6rem; border-radius: 9999px;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
  }
  .pill.high   { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.35); }
  .pill.medium { background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.35); }
  .pill.low    { background: rgba(239, 68, 68, 0.15);  color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.35); }
  .pill.cyan   { background: rgba(34, 211, 238, 0.12); color: #67e8f9; border: 1px solid rgba(34, 211, 238, 0.25); }
  .pill.green  { background: rgba(16, 185, 129, 0.12); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.30); }

  /* probability bars */
  .probrow { margin-top: 0.7rem; }
  .probrow .head {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.78rem; font-family: ui-monospace, monospace; margin-bottom: 0.25rem;
  }
  .probrow .head .lbl  { color: #9aa6b9; }
  .probrow .head .lbl.win { color: white; font-weight: 700; }
  .probrow .head .val  { color: #cbd3df; }
  .probrow .head .val.win { color: white; }
  .probtrack {
    height: 0.55rem; border-radius: 9999px; background: rgba(75, 86, 111, 0.35);
    overflow: hidden;
  }
  .probfill { height: 100%; background: #4b566f; }
  .probfill.win {
    background: linear-gradient(90deg, #10b981 0%, #22d3ee 100%);
  }

  /* explain strip */
  .explain {
    display: grid; grid-template-columns: repeat(1, 1fr); gap: 0.55rem; margin-top: 1rem;
  }
  @media (min-width: 700px) { .explain { grid-template-columns: repeat(3, 1fr); } }
  .explain .tile {
    background: rgba(24, 29, 44, 0.55);
    border: 1px solid rgba(155, 166, 185, 0.14);
    border-radius: 0.65rem; padding: 0.55rem 0.75rem;
  }
  .explain .tile .k {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6b7892;
  }
  .explain .tile .v { color: #cbd3df; font-size: 0.82rem; margin-top: 0.15rem; }

  /* per-hit cards */
  .hit-grid {
    display: grid; grid-template-columns: repeat(1, 1fr); gap: 0.55rem;
  }
  @media (min-width: 700px)  { .hit-grid { grid-template-columns: repeat(2, 1fr); } }
  @media (min-width: 1100px) { .hit-grid { grid-template-columns: repeat(3, 1fr); } }
  .hit-card {
    background: linear-gradient(180deg, rgba(75, 86, 111, 0.22) 0%, rgba(38, 45, 63, 0.55) 100%);
    border: 1px solid rgba(155, 166, 185, 0.16);
    border-radius: 0.85rem; padding: 0.7rem 0.8rem;
  }
  .hit-head {
    display: flex; justify-content: space-between; align-items: center;
    font-family: ui-monospace, monospace; font-size: 0.72rem; color: #9aa6b9;
    margin-bottom: 0.45rem;
  }
  .mini-bar {
    height: 0.35rem; background: rgba(38, 45, 63, 0.7); border-radius: 9999px; overflow: hidden;
  }
  .mini-bar .fill { height: 100%; background: #4b566f; }
  .mini-bar .fill.win { background: linear-gradient(90deg, #10b981, #22d3ee); }
  .mini-row {
    display: flex; justify-content: space-between; align-items: center;
    font-family: ui-monospace, monospace; font-size: 0.65rem; color: #9aa6b9;
    margin-top: 0.35rem;
  }
  .mini-row .v { color: #e4e9f0; }

  /* why-flange box */
  .why {
    border-left: 3px solid #22d3ee; padding-left: 0.85rem;
    margin-top: 0.5rem; color: #cbd3df; font-size: 0.85rem; line-height: 1.55;
  }

  /* tweak streamlit widgets */
  div[data-testid="stRadio"] > div { gap: 0.4rem; }
  .stButton > button {
    width: 100%; height: 3.1rem;
    background: linear-gradient(135deg, #10b981 0%, #22d3ee 100%);
    color: #05070d; font-weight: 700; border: none;
    box-shadow: 0 0 32px -10px rgba(16, 185, 129, 0.55);
  }
  .stButton > button:hover { transform: translateY(-1px); }
  .stButton > button:disabled {
    background: rgba(75, 86, 111, 0.35); color: #6b7892; box-shadow: none;
  }

  /* hide labels we override */
  .stRadio label[data-testid="stWidgetLabel"] { display: none; }

  /* warnings */
  .warning {
    background: rgba(245, 158, 11, 0.10);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #fcd34d;
    padding: 0.7rem 0.9rem; border-radius: 0.7rem; font-size: 0.85rem;
    display: flex; align-items: flex-start; gap: 0.6rem;
  }
  .warning + .warning { margin-top: 0.4rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_model() -> FlangeInvariantLR | None:
    """Load and cache the Flange-Invariant LR pipeline. Returns None on error."""
    try:
        return FlangeInvariantLR(MODEL_DIR)
    except Exception:
        return None


def confidence_class(level: str) -> str:
    return level if level in ("high", "medium", "low") else "medium"


def render_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    html = "".join(
        f'<div class="warning"><span style="font-weight:700;">!</span><span>{w}</span></div>'
        for w in warnings
    )
    st.markdown(html, unsafe_allow_html=True)


def build_main_waveform_figure(j: dict) -> go.Figure:
    sr = j.get("sample_rate") or 1
    duration = j.get("duration_sec") or 1.0

    wave = j.get("waveform", {}) or {}
    env  = j.get("envelope", {}) or {}
    peaks = j.get("hit_times_sec") or []

    fig = go.Figure()

    # Raw waveform
    wv = wave.get("values") or []
    if wv:
        x = np.linspace(0, duration, num=len(wv))
        fig.add_trace(go.Scatter(
            x=x, y=wv, mode="lines",
            line=dict(color="rgba(34, 211, 238, 0.85)", width=1),
            name="raw waveform", hoverinfo="skip",
        ))

    # Envelope
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

    # Peak markers
    for i, t in enumerate(peaks):
        fig.add_vline(
            x=t, line_color="rgba(245, 158, 11, 0.9)", line_width=1.5,
            annotation_text=f"#{i+1}", annotation_position="top",
            annotation_font_color="#fcd34d", annotation_font_size=10,
        )

    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5, 7, 13, 0.55)",
        xaxis=dict(
            title="time (s)", showgrid=True, gridcolor="rgba(155, 166, 185, 0.08)",
            zeroline=False, color="#9aa6b9",
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(155, 166, 185, 0.08)",
            zeroline=False, color="#9aa6b9",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
            bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd3df", size=10),
        ),
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
st.markdown("""
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

<div class="stage-grid">
    <div class="stage"><div class="stage-num">1</div>
        <div><div class="stage-title">Record</div>
        <div class="stage-sub">Capture hammer hits live or upload an audio file.</div></div></div>
    <div class="stage"><div class="stage-num">2</div>
        <div><div class="stage-title">Segment</div>
        <div class="stage-sub">Envelope + peak detection isolates each impact.</div></div></div>
    <div class="stage"><div class="stage-num">3</div>
        <div><div class="stage-title">Extract Features</div>
        <div class="stage-sub">PSD &middot; MFCC &middot; spectral stats &middot; per-band T60 decay.</div></div></div>
    <div class="stage"><div class="stage-num final">4</div>
        <div><div class="stage-title">Predict</div>
        <div class="stage-sub">Flange-Invariant LR + soft-vote across hits.</div></div></div>
</div>
""", unsafe_allow_html=True)

st.write("")  # small spacer

# ---------------------------------------------------------------- #
# Model status pill
# ---------------------------------------------------------------- #
model = load_model()
if model is None:
    st.markdown("""
    <div class="warning">
        <span style="font-weight:700;">!</span>
        <span>Model artifacts could not be loaded. Run
        <code>python backend/train_and_export.py</code> to regenerate them.</span>
    </div>
    """, unsafe_allow_html=True)
else:
    acc = model.metadata.get("lofo_calibrated_hit_accuracy")
    pill = (
        f'<span class="pill green">model ready'
        f'{f" · LOFO {acc * 100:.1f}%" if acc is not None else ""}</span>'
    )
    st.markdown(pill, unsafe_allow_html=True)

# ---------------------------------------------------------------- #
# Main two-column layout
# ---------------------------------------------------------------- #
left, right = st.columns([1, 2], gap="large")

with left:
    # Flange selector
    st.markdown('<div class="label">Flange ID</div>'
                '<div class="label-sub">Required for per-flange centering</div>',
                unsafe_allow_html=True)
    flange_id = st.radio(
        "flange",
        options=[1, 2, 3, 4],
        format_func=lambda x: f"F{x}",
        horizontal=True,
        label_visibility="collapsed",
    )

    st.write("")
    # Live recording (Streamlit's built-in mic widget)
    st.markdown('<div class="label">Live Recording</div>'
                '<div class="label-sub">Strike the flange 3–6 times, then stop.</div>',
                unsafe_allow_html=True)
    if hasattr(st, "audio_input"):
        recorded = st.audio_input("Record hammer hits", label_visibility="collapsed")
    else:
        recorded = None
        st.info("Live recording requires Streamlit ≥ 1.39. Use the file uploader below.")

    st.write("")
    # File upload
    st.markdown('<div class="label">Or Upload Audio</div>'
                '<div class="label-sub">.m4a · .wav · .mp3 · .webm · .ogg</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "audio",
        type=["m4a", "wav", "mp3", "webm", "ogg", "flac"],
        label_visibility="collapsed",
    )

    # Choose source: prefer the live recording when both are present.
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

    # Why flange-invariant
    st.markdown("""
    <div class="glass" style="margin-top:1.2rem;">
        <div class="label">Why flange-invariant?</div>
        <div class="why">
            Each physical flange has its own resonant fingerprint &mdash; geometry, mounting,
            microphone position. The model subtracts that flange's typical acoustic signature
            before classifying, so it focuses on <strong>torque-related sound changes</strong>
            rather than memorising the flange itself. That is why it generalises across recording
            sessions and devices.
        </div>
    </div>
    """, unsafe_allow_html=True)


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
                    "ok": False,
                    "warnings": [f"Prediction failed: {e}"],
                    "n_hits": 0,
                    "per_hit": [],
                    "averaged_probabilities": None,
                    "final_prediction": None,
                    "flange_id": int(flange_id),
                    "sample_rate": None,
                    "duration_sec": None,
                    "waveform": {"values": []},
                    "envelope": {"values": []},
                    "hit_times_sec": [],
                }
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")

    if result is None:
        st.markdown("""
        <div class="glass" style="text-align:center; padding: 2.5rem 1.5rem;">
            <div style="font-size:1.05rem; font-weight:600; color:#f5f7fa;">No prediction yet</div>
            <div style="color:#9aa6b9; font-size:0.85rem; max-width:480px; margin: 0.5rem auto 0;">
                Pick a flange, then either record a few hammer strikes or upload an audio file.
                The detected hits, calibrated probabilities, and final torque will appear here.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # -------- Warnings -------- #
        render_warnings(result.get("warnings", []) or [])

        # -------- Final prediction card -------- #
        fp = result.get("final_prediction")
        if fp:
            torque = int(fp["torque_ftlbs"])
            conf = float(fp["confidence"])
            lvl = confidence_class(fp.get("confidence_level", "medium"))
            avg = result["averaged_probabilities"]
            sr = result.get("sample_rate")
            dur = result.get("duration_sec")
            n_hits = result.get("n_hits", 0)

            html = f"""
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
            """
            st.markdown(html, unsafe_allow_html=True)

        # -------- Waveform -------- #
        if result.get("waveform", {}).get("values"):
            st.markdown(f"""
            <div class="glass" style="margin-top:0.85rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="label">Waveform &amp; Detected Hits</div>
                        <div class="label-sub">Smoothed envelope and peak markers over the raw signal.</div>
                    </div>
                    <span class="pill cyan">{result.get('n_hits', 0)} hit{'s' if result.get('n_hits', 0) != 1 else ''}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            fig = build_main_waveform_figure(result)
            try:
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            except TypeError:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # -------- Per-hit gallery -------- #
        per_hit = result.get("per_hit") or []
        if per_hit:
            st.markdown("""
            <div class="glass" style="margin-top:0.85rem;">
                <div class="label">Per-Hit Predictions</div>
                <div class="label-sub">Each card shows one hammer strike and its calibrated probabilities.</div>
            </div>
            """, unsafe_allow_html=True)
            cards_html = '<div class="hit-grid">' + ''.join(render_hit_card(h) for h in per_hit) + '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)


# ---------------------------------------------------------------- #
# Footer
# ---------------------------------------------------------------- #
st.markdown("""
<div style="margin-top:2rem; padding-top:1rem; border-top:1px solid rgba(155,166,185,0.12);
            text-align:center; color:#6b7892; font-size:0.75rem;">
    Built for the UH Machine Learning Competition 2026 · Model: Flange-Invariant LR with per-class isotonic calibration
    · Soft-vote across hits.
</div>
""", unsafe_allow_html=True)

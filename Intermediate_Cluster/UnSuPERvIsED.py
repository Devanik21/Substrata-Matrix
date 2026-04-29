"""
UnSuPERvIsED.py — Substrata-Matrix Interactive Clustering Workbench

Streamlit-based interface orchestrating the complete clustering pipeline:
data ingestion, preprocessing, algorithm selection, parallel execution, 
evaluation, visualization, stability analysis, consensus clustering,
and interpretability.    
"""      

# ─────────────────────────────────────────────────────────────────
# IMPORTS & PATH SETUP  
# ─────────────────────────────────────────────────────────────────

import sys, os, io, time, json, warnings, hashlib, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

warnings.filterwarnings("ignore")

# Add backend directory to path
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Streamlit page config (MUST be first st call) ────────────────
st.set_page_config(
    page_title="UnSuPERvIsED · Substrata-Matrix",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Substrata-Matrix — Clustering Workbench"},
)

# ── Backend lazy imports ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_backends():
    try:
        from preprocessing import (
            DataLoader, DataProfiler, PreprocessingPipeline,
            PreprocessingConfig, ScalerType, ImputeStrategy,
            OutlierMethod, OutlierAction, FeatureSelectionMethod,
            infer_best_config, get_scaler_options,
            get_imputer_options, get_outlier_options,
        )
        from clustering_registry import (
            get_registry, summarize_registry,
            get_family_options, AlgorithmFamily, AlgorithmTag,
        )
        from clustering_runner import (
            ClusteringRunner, RunnerConfig, ExecutionMode,
            make_runner, run_all_algorithms, compute_cluster_statistics,
        )
        from evaluation import (
            ClusteringEvaluator, AlgorithmRanker,
            ResultsTableBuilder, KSweepAnalyser,
            evaluate_all, build_results_dataframe,
            get_best_algorithm, METRIC_REGISTRY,
            silhouette_grade, format_metric_value,
        )
        from stability import (
            MultiAlgorithmStabilityComparator, StabilityConfig,
            StabilityVisDataBuilder, LabelStabilityMatrix,
            NoiseResponseProfiler, quick_stability_test,
            make_stability_config, build_stability_summary,
            get_stability_color,
        )
        from consensus import (
            ConsensusEngine, ConsensusConfig, ConsensusMethod,
            ConsensusVisBuilder, ConsensusKEstimator,
            run_consensus, get_consensus_method_options,
            analyse_ensemble_diversity,
        )
        from visualization import (
            VisualisationEngine, Theme, get_vis_engine,
            available_embedding_methods, empty_figure, label_colormap,
        )
        from dynamic_patterns import (
            get_pattern_names, 
            get_pattern_info,
            generate_pattern_dataframe,
            PATTERN_CATALOG
        )
        return {
            "ok": True,
            "PreprocessingConfig": PreprocessingConfig,
            "ScalerType": ScalerType,
            "ImputeStrategy": ImputeStrategy,
            "OutlierMethod": OutlierMethod,
            "OutlierAction": OutlierAction,
            "FeatureSelectionMethod": FeatureSelectionMethod,
            "DataLoader": DataLoader,
            "DataProfiler": DataProfiler,
            "PreprocessingPipeline": PreprocessingPipeline,
            "infer_best_config": infer_best_config,
            "get_scaler_options": get_scaler_options,
            "get_imputer_options": get_imputer_options,
            "get_outlier_options": get_outlier_options,
            "get_registry": get_registry,
            "summarize_registry": summarize_registry,
            "ClusteringRunner": ClusteringRunner,
            "RunnerConfig": RunnerConfig,
            "ExecutionMode": ExecutionMode,
            "make_runner": make_runner,
            "compute_cluster_statistics": compute_cluster_statistics,
            "ClusteringEvaluator": ClusteringEvaluator,
            "AlgorithmRanker": AlgorithmRanker,
            "ResultsTableBuilder": ResultsTableBuilder,
            "KSweepAnalyser": KSweepAnalyser,
            "evaluate_all": evaluate_all,
            "build_results_dataframe": build_results_dataframe,
            "get_best_algorithm": get_best_algorithm,
            "METRIC_REGISTRY": METRIC_REGISTRY,
            "silhouette_grade": silhouette_grade,
            "MultiAlgorithmStabilityComparator": MultiAlgorithmStabilityComparator,
            "StabilityConfig": StabilityConfig,
            "StabilityVisDataBuilder": StabilityVisDataBuilder,
            "LabelStabilityMatrix": LabelStabilityMatrix,
            "NoiseResponseProfiler": NoiseResponseProfiler,
            "quick_stability_test": quick_stability_test,
            "make_stability_config": make_stability_config,
            "build_stability_summary": build_stability_summary,
            "get_stability_color": get_stability_color,
            "ConsensusEngine": ConsensusEngine,
            "ConsensusConfig": ConsensusConfig,
            "ConsensusMethod": ConsensusMethod,
            "ConsensusVisBuilder": ConsensusVisBuilder,
            "ConsensusKEstimator": ConsensusKEstimator,
            "run_consensus": run_consensus,
            "get_consensus_method_options": get_consensus_method_options,
            "analyse_ensemble_diversity": analyse_ensemble_diversity,
            "VisualisationEngine": VisualisationEngine,
            "Theme": Theme,
            "get_vis_engine": get_vis_engine,
            "available_embedding_methods": available_embedding_methods,
            "empty_figure": empty_figure,
            "get_pattern_names": get_pattern_names,
            "get_pattern_info": get_pattern_info,
            "generate_pattern_dataframe": generate_pattern_dataframe,
            "PATTERN_CATALOG": PATTERN_CATALOG,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "tb": traceback.format_exc()}

B = _load_backends()

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS — DARK NEON DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #07070f !important;
        color: #e0e0f0 !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── App container ── */
    .main .block-container {
        padding: 1.2rem 2rem 3rem 2rem;
        max-width: 1600px;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0b18 0%, #0d0d1f 100%) !important;
        border-right: 1px solid #1a1a2e !important;
    }
    section[data-testid="stSidebar"] * { color: #c8c8e8 !important; }
    section[data-testid="stSidebar"] .stSelectbox > div,
    section[data-testid="stSidebar"] .stMultiSelect > div {
        background: #10101e !important;
        border: 1px solid #1e1e3e !important;
    }

    /* ── Hero title ── */
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem; font-weight: 700;
        background: linear-gradient(135deg, #00e5ff 0%, #9b59ff 50%, #ff4daa 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; line-height: 1.15;
        letter-spacing: -0.02em; margin: 0;
    }
    .hero-sub {
        font-size: 1.05rem; color: #8888bb; margin-top: .5rem;
        font-weight: 400; letter-spacing: 0.02em;
    }

    /* ── Section headers ── */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem; font-weight: 600; color: #e0e0ff;
        border-left: 3px solid #00e5ff; padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
    .subsection-header {
        font-size: 1.05rem; font-weight: 600; color: #aaaacc;
        letter-spacing: 0.05em; text-transform: uppercase;
        margin: 1.2rem 0 .6rem 0;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #10101e 0%, #141428 100%);
        border: 1px solid #1e1e3a; border-radius: 12px;
        padding: 1.1rem 1.4rem; text-align: center;
        transition: border-color .25s, box-shadow .25s;
        position: relative; overflow: hidden;
    }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #00e5ff, #9b59ff, #ff4daa);
    }
    .metric-card:hover {
        border-color: #3333aa; box-shadow: 0 4px 24px rgba(0,229,255,.12);
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem; font-weight: 700; color: #00e5ff;
        line-height: 1;
    }
    .metric-label {
        font-size: .78rem; color: #6666aa; letter-spacing: .08em;
        text-transform: uppercase; margin-top: .35rem;
    }
    .metric-delta { font-size: .82rem; margin-top: .2rem; }
    .metric-delta.good { color: #00ff88; }
    .metric-delta.bad  { color: #ff4444; }
    .metric-delta.neutral { color: #aaaacc; }

    /* ── Algorithm cards ── */
    .algo-card {
        background: #10101e; border: 1px solid #1e1e32;
        border-radius: 10px; padding: .9rem 1rem;
        margin-bottom: .5rem; cursor: pointer;
        transition: all .2s;
    }
    .algo-card:hover { border-color: #4433aa; background: #13132a; }
    .algo-card.selected { border-color: #00e5ff; background: #0a1a2a; }
    .algo-tag {
        display: inline-block; padding: .18rem .55rem;
        border-radius: 20px; font-size: .7rem; font-weight: 500;
        margin: .1rem .15rem; letter-spacing: .04em;
    }
    .tag-fast     { background: #0a2a10; color: #00ff88; border: 1px solid #005520; }
    .tag-nok      { background: #1a0a2a; color: #9b59ff; border: 1px solid #440088; }
    .tag-noise    { background: #2a1a00; color: #ff8c00; border: 1px solid #884400; }
    .tag-prob     { background: #0a1a2a; color: #00e5ff; border: 1px solid #004466; }
    .tag-scale    { background: #1a1a00; color: #ffd700; border: 1px solid #555500; }
    .tag-default  { background: #1a1a2a; color: #aaaacc; border: 1px solid #333355; }

    /* ── Status badges ── */
    .badge {
        display: inline-block; padding: .2rem .6rem;
        border-radius: 20px; font-size: .72rem; font-weight: 600;
        letter-spacing: .05em;
    }
    .badge-success { background: #002a14; color: #00ff88; border: 1px solid #005520; }
    .badge-error   { background: #2a0010; color: #ff4488; border: 1px solid #660030; }
    .badge-timeout { background: #2a1500; color: #ff8c00; border: 1px solid #664400; }
    .badge-skip    { background: #1a1a1a; color: #666688; border: 1px solid #333344; }
    .badge-cached  { background: #001a2a; color: #00ccff; border: 1px solid #003355; }

    /* ── Info / warning panels ── */
    .info-panel {
        background: #050d14; border: 1px solid #003366;
        border-left: 3px solid #00e5ff; border-radius: 8px;
        padding: .85rem 1.1rem; margin: .75rem 0;
        font-size: .88rem; color: #aaccee;
    }
    .warn-panel {
        background: #140a00; border: 1px solid #553300;
        border-left: 3px solid #ff8c00; border-radius: 8px;
        padding: .85rem 1.1rem; margin: .75rem 0;
        font-size: .88rem; color: #ddaa88;
    }
    .success-panel {
        background: #001a08; border: 1px solid #005520;
        border-left: 3px solid #00ff88; border-radius: 8px;
        padding: .85rem 1.1rem; margin: .75rem 0;
        font-size: .88rem; color: #88ddaa;
    }

    /* ── Data table styling ── */
    .stDataFrame, [data-testid="stDataFrame"] {
        border: 1px solid #1e1e3a !important; border-radius: 8px !important;
    }
    .stDataFrame td, .stDataFrame th {
        background: #0d0d1e !important; color: #d0d0ee !important;
        border: 1px solid #1e1e3a !important;
    }
    .stDataFrame th { color: #00e5ff !important; font-weight: 600 !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1a0040 0%, #220055 100%) !important;
        color: #cc88ff !important; border: 1px solid #440099 !important;
        border-radius: 8px !important; font-weight: 600 !important;
        transition: all .2s !important; letter-spacing: .03em !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #220055 0%, #330077 100%) !important;
        border-color: #6600cc !important; box-shadow: 0 0 16px rgba(155,89,255,.35) !important;
        color: #dd99ff !important;
    }
    .stButton > button:active { transform: scale(.97) !important; }
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #003344 0%, #005566 100%) !important;
        color: #00e5ff !important; border-color: #007799 !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        box-shadow: 0 0 20px rgba(0,229,255,.4) !important;
    }

    /* ── Sliders & inputs ── */
    .stSlider > div > div { background: #1e1e3a !important; }
    .stSlider > div > div > div { background: #00e5ff !important; }
    input[type="text"], input[type="number"], textarea, .stTextInput input, .stNumberInput input {
        background: #0d0d1e !important; border: 1px solid #1e1e3a !important;
        color: #e0e0f0 !important; border-radius: 6px !important;
    }
    input:focus, textarea:focus {
        border-color: #4433aa !important;
        box-shadow: 0 0 0 2px rgba(68,51,170,.3) !important;
    }

    /* ── Select boxes ── */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: #0d0d1e !important; border: 1px solid #1e1e3a !important;
        color: #e0e0f0 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #0b0b18 !important; border-bottom: 1px solid #1e1e3a !important;
        gap: .3rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important; color: #6666aa !important;
        border: none !important; border-radius: 8px 8px 0 0 !important;
        font-weight: 500 !important; padding: .6rem 1.2rem !important;
        transition: all .2s !important;
    }
    .stTabs [aria-selected="true"] {
        background: #10101e !important; color: #00e5ff !important;
        border-top: 2px solid #00e5ff !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #aaaaee !important; }

    /* ── Progress bar ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00e5ff, #9b59ff) !important;
        border-radius: 4px !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: #0d0d1e !important; color: #aaaacc !important;
        border: 1px solid #1e1e3a !important; border-radius: 8px !important;
    }
    .streamlit-expanderHeader:hover { border-color: #3333aa !important; }

    /* ── Checkboxes ── */
    .stCheckbox label { color: #aaaacc !important; }

    /* ── Dividers ── */
    hr { border-color: #1e1e3a !important; margin: 1.5rem 0 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #07070f; }
    ::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a3a6a; }

    /* ── Code blocks ── */
    code, .stCode { font-family: 'JetBrains Mono', monospace !important; }
    .stCode { background: #0d0d1e !important; border: 1px solid #1e1e3a !important; }

    /* ── Tooltips ── */
    .tooltip-text {
        font-size: .78rem; color: #8888bb; margin-top: .25rem; font-style: italic;
    }

    /* ── Rank badges ── */
    .rank-1 { background: linear-gradient(135deg,#664400,#aa7700); color:#ffd700; padding:.2rem .6rem; border-radius:20px; font-weight:700; font-size:.8rem; }
    .rank-2 { background: linear-gradient(135deg,#333,#555); color:#ccc; padding:.2rem .6rem; border-radius:20px; font-weight:700; font-size:.8rem; }
    .rank-3 { background: linear-gradient(135deg,#3a1500,#6a2800); color:#cd7f32; padding:.2rem .6rem; border-radius:20px; font-weight:700; font-size:.8rem; }

    /* ── Gemini AI panel ── */
    .ai-response {
        background: linear-gradient(135deg,#080818,#0a0a1e);
        border: 1px solid #2a1a4a; border-left: 3px solid #9b59ff;
        border-radius: 10px; padding: 1.1rem 1.4rem;
        font-size: .92rem; line-height: 1.7; color: #ccccee;
    }
    .ai-label {
        font-size: .7rem; color: #9b59ff; font-weight: 700;
        letter-spacing: .1em; text-transform: uppercase; margin-bottom: .6rem;
    }

    /* ── Pipeline stepper ── */
    .step-indicator {
        display:flex; align-items:center; gap:.5rem;
        padding:.5rem 1rem; border-radius:8px; margin:.3rem 0;
        font-size:.85rem; font-weight:500;
    }
    .step-done    { background:#001a08; border:1px solid #005520; color:#00ff88; }
    .step-active  { background:#001528; border:1px solid #004466; color:#00e5ff; }
    .step-pending { background:#0d0d1e; border:1px solid #1e1e3a; color:#555577; }

    /* ── Pulse animation for running ── */
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
    .pulsing { animation: pulse 1.2s ease-in-out infinite; }

    /* ── Gradient separator ── */
    .gradient-sep {
        height:1px; background:linear-gradient(90deg,transparent,#2a2a5a,transparent);
        margin:1.5rem 0; border:none;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ─────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "page": "🏠 Home",
    "df_raw": None,
    "df_filename": "",
    "data_profile": None,
    "preproc_result": None,
    "X_processed": None,
    "feature_names": [],
    "preproc_config": None,
    "selected_algorithms": [],
    "n_clusters": 8,
    "batch_result": None,
    "eval_results": [],
    "eval_dict": {},
    "stability_reports": {},
    "consensus_result": None,
    "consensus_all": {},
    "embedding_cache": {},
    "gemini_history": [],
    "run_log": [],
    "execution_stats": {},
    "k_sweep_data": {},
    "noise_profiles": {},
    "ari_matrix": None,
    "last_run_timestamp": None,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _ok(key: str = "ok") -> bool:
    return B.get("ok", False)

def _b(name: str):
    return B.get(name)

def _has_data() -> bool:
    return st.session_state.X_processed is not None

def _has_results() -> bool:
    return (st.session_state.batch_result is not None and
            len(st.session_state.eval_results) > 0)

def _metric_card(value, label, delta=None, delta_good=True, color="#00e5ff"):
    delta_html = ""
    if delta is not None:
        cls = "good" if delta_good else "bad"
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

def _badge(text: str, kind: str = "success") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'

def _info(msg: str):
    st.markdown(f'<div class="info-panel">ℹ️ {msg}</div>', unsafe_allow_html=True)

def _warn(msg: str):
    st.markdown(f'<div class="warn-panel">⚠️ {msg}</div>', unsafe_allow_html=True)

def _success(msg: str):
    st.markdown(f'<div class="success-panel">✅ {msg}</div>', unsafe_allow_html=True)

def _section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def _subsection(title: str):
    st.markdown(f'<div class="subsection-header">{title}</div>', unsafe_allow_html=True)

def _sep():
    st.markdown('<div class="gradient-sep"></div>', unsafe_allow_html=True)

def _get_vis() -> Any:
    if not _ok():
        return None
    vis_cls = _b("get_vis_engine")
    return vis_cls(42) if vis_cls else None

def _dark_plotly(fig, height: int = 500) -> Any:
    T = _b("Theme")
    if T and fig:
        fig.update_layout(
            paper_bgcolor=T.BG_DARK,
            plot_bgcolor=T.BG_CARD,
            font=dict(color=T.TEXT_PRIMARY, family="Inter, sans-serif"),
            height=height,
        )
    return fig

def _safe_plotly(fig):
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


# ─────────────────────────────────────────────────────────────────
# GEMINI AI ORACLE
# ─────────────────────────────────────────────────────────────────

def _gemini_query(prompt: str, context: str = "",
                  show_spinner: bool = True) -> str:
    """Send a query to Gemini and return text response."""
    try:
        import google.generativeai as genai
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            return "⚠️ Gemini API key not found in `st.secrets`. Add `GEMINI_API_KEY` to `.streamlit/secrets.toml`."
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        full_prompt = (
            "You are an expert data scientist and ML researcher specialising in "
            "clustering and unsupervised learning. Be precise, insightful, and concise.\n\n"
            + (f"Context:\n{context}\n\n" if context else "")
            + prompt
        )
        response = model.generate_content(full_prompt)
        return response.text
    except ImportError:
        return "⚠️ `google-generativeai` package not installed. Run: `pip install google-generativeai`"
    except Exception as e:
        return f"⚠️ Gemini error: {str(e)}"


def _render_ai_response(text: str):
    st.markdown(f"""
    <div class="ai-response">
        <div class="ai-label"> Gemini AI Insight</div>
        {text}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────

PAGES = [
    "🏠 Home",
    "📁 Data Ingestion",
    "⚙️ Preprocessing",
    "🧬 Algorithm Arena",
    "⚡ Execution Engine",
    "📊 Results Dashboard",
    "🔬 Visualization Lab",
    "🧪 Stability Lab",
    "🤝 Consensus Forge",
    " AI Oracle",
    "🛠️ Advanced Tools",
]

PIPELINE_STATE = [
    ("📁 Data", _ok() and st.session_state.df_raw is not None),
    ("⚙️ Preprocess", _has_data()),
    ("🧬 Algorithms", len(st.session_state.selected_algorithms) > 0),
    ("⚡ Run", st.session_state.batch_result is not None),
    ("📊 Evaluate", len(st.session_state.eval_results) > 0),
]

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:.8rem 0 1.2rem 0;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700;
             background:linear-gradient(135deg,#00e5ff,#9b59ff); -webkit-background-clip:text;
             -webkit-text-fill-color:transparent; background-clip:text;">
            🧬 UnSuPERvIsED
        </div>
        <div style="font-size:.7rem; color:#555577; letter-spacing:.12em; text-transform:uppercase;">
            ClusterX Intelligence Lab
        </div>
    </div>""", unsafe_allow_html=True)

    st.session_state.page = st.selectbox(
        "Navigation", PAGES,
        index=PAGES.index(st.session_state.page),
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.72rem; color:#555577; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.5rem;">Pipeline Status</div>', unsafe_allow_html=True)
    for step_name, done in PIPELINE_STATE:
        cls = "step-done" if done else "step-pending"
        icon = "✓" if done else "○"
        st.markdown(f'<div class="step-indicator {cls}">{icon} {step_name}</div>',
                    unsafe_allow_html=True)

    if _ok() and _b("summarize_registry"):
        st.markdown("<hr>", unsafe_allow_html=True)
        try:
            reg_summary = _b("summarize_registry")()
            st.markdown(f"""
            <div style="font-size:.78rem; color:#666688; line-height:1.9;">
                <span style="color:#00e5ff; font-weight:600;">{reg_summary['total_algorithms']}</span> algorithms<br>
                <span style="color:#9b59ff; font-weight:600;">{reg_summary['no_k_required']}</span> auto-k<br>
                <span style="color:#ff4daa; font-weight:600;">{len(reg_summary['families'])}</span> families<br>
                <span style="color:#ffd700; font-weight:600;">{reg_summary['probabilistic']}</span> probabilistic
            </div>""", unsafe_allow_html=True)
        except Exception:
            pass

    if st.session_state.last_run_timestamp:
        st.markdown(f'<div style="font-size:.7rem; color:#444466; margin-top:.5rem;">Last run: {st.session_state.last_run_timestamp}</div>', unsafe_allow_html=True)

page = st.session_state.page


# ─────────────────────────────────────────────────────────────────
# ▓▓  PAGE 0 · HOME
# ─────────────────────────────────────────────────────────────────

if page == "🏠 Home":
    # Hero
    col_hero, col_anim = st.columns([2, 1])
    with col_hero:
        st.markdown("""
        <div style="padding: 2rem 0 1rem 0;">
            <div class="hero-title">UnSuPERvIsED</div>
            <div class="hero-title" style="font-size:1.6rem; margin-top:-.4rem;">
                Universal Clustering Intelligence Lab
            </div>
            <div class="hero-sub">
                The most comprehensive open-source clustering workbench.<br>
                60+ algorithms · AI-powered insights · Production-grade stability analysis.
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(" Quick Start →", type="primary", use_container_width=True):
                st.session_state.page = "📁 Data Ingestion"
                st.rerun()
        with c2:
            if st.button("📖 Algorithm Browser", use_container_width=True):
                st.session_state.page = "🧬 Algorithm Arena"
                st.rerun()
        with c3:
            if st.button(" AI Oracle", use_container_width=True):
                st.session_state.page = " AI Oracle"
                st.rerun()

    with col_anim:
        # Dynamic 21-Pattern Morphing Animation
        # Dynamic 21-Pattern Morphing Animation
        @st.cache_data
        def _build_animated_patterns():
            pattern_keys = list(_b("get_pattern_names")().keys())
            frames = []
            n_points = 5000  # Increased density for a serious, particle-system aesthetic
            for pk in pattern_keys:
                df, meta = _b("generate_pattern_dataframe")(pk, n_samples=n_points, n_clusters=7, random_state=42)
                
                # Calculate angle for a cyclical, deep-analytical color gradient
                x_col, y_col = df.columns[0], df.columns[1]
                angle = np.arctan2(df[y_col], df[x_col])
                
                df_temp = pd.DataFrame({
                    "x": df[x_col], "y": df[y_col],
                    "color_val": angle,
                    "Pattern": meta["pattern_name"]
                })
                frames.append(df_temp)
            return pd.concat(frames, ignore_index=True)

        df_anim = _build_animated_patterns()

        # Build the animated scatter plot using a sleek, deep intelligence palette
        fig_demo = px.scatter(
            df_anim, x="x", y="y", animation_frame="Pattern", color="color_val",
            color_continuous_scale=["#00e5ff", "#00ff88", "#ffd700", "#ff4daa", "#9b59ff", "#330088", "#00e5ff"]
                
                
            )

        # Apply dark neon layout and lock axes for smooth, deliberate morphing transitions
        # Build the animated scatter plot using a high-visibility futuristic palette
        fig_demo = px.scatter(
            df_anim, x="x", y="y", animation_frame="Pattern", color="color_val",
            color_continuous_scale=[
                "#330088",  # Deep radiant indigo (visible against #07070f)
                "#7700cc",  # Electric violet
                "#9b59ff",  # Core purple
                "#ff4daa",  # Plasma pink
                "#00e5ff",  # Piercing cyan
                "#00ffcc",  # Radioactive mint
                "#00e5ff",  # Mirroring back down for a seamless cyclical glow
                "#9b59ff", 
                "#330088"
            ]
        )

        # Apply dark neon layout and lock axes for smooth, deliberate morphing transitions
        fig_demo.update_layout(
            paper_bgcolor="#07070f", plot_bgcolor="#07070f",
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=25, b=0), height=350,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", range=[-7.5, 8]), 
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", range=[-8, 7.5]),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=-0.05, x=0.5, xanchor="center", yanchor="top",
                buttons=[dict(
                    label="▶ Initiate Evolution",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=2500, redraw=True), 
                        fromcurrent=True, 
                        transition=dict(duration=2500, easing="cubic-in-out")
                    )]
                )]
            )]
        )
        
        # Micro-markers: density remains high, but opacity allows overlapping glows
        fig_demo.update_traces(marker=dict(size=1.5, opacity=0.85, line=dict(width=0)))

        # Hide the default slider to keep the UI strictly professional
        if "sliders" in fig_demo.layout:
            fig_demo.layout.sliders[0].visible = False

        st.plotly_chart(fig_demo, use_container_width=True, config={"displayModeBar": False})

    _sep()

    # Stats strip
    if _ok():
        try:
            reg = _b("get_registry")()
            rs  = _b("summarize_registry")()
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            with c1: _metric_card(rs["total_algorithms"], "Algorithms", color="#00e5ff")
            with c2: _metric_card(len(rs["families"]), "Algorithm Families", color="#9b59ff")
            with c3: _metric_card(rs["no_k_required"], "Auto-K Methods", color="#ff4daa")
            with c4: _metric_card(rs["probabilistic"], "Probabilistic", color="#ffd700")
            with c5: _metric_card(rs["scalable"], "Scalable (>100K)", color="#00ff88")
            with c6: _metric_card(rs["noise_producing"], "Noise-Aware", color="#ff8c00")
        except Exception:
            pass
    else:
        _warn(f"Backend load failed: {B.get('error','Unknown')}")
        st.code(B.get("tb",""), language="python")

    _sep()

    # Feature grid
    _section("🌌 Feature Constellation")
    features = [
        ("🔬","Deep Data Profiling","Column statistics, normality tests, outlier detection, correlation analysis — before you cluster a single point."),
        ("⚙️","Smart Preprocessing","9 scalers · 7 imputers · 5 outlier methods · PCA/variance/correlation feature selection. Auto-recommended."),
        ("🧬","60+ Algorithms","Every major clustering family: centroid, hierarchical, density, distribution, graph, neural, fuzzy, manifold, ensemble."),
        ("⚡","Parallel Execution","Thread-pool parallel execution with per-algorithm timeout isolation and adaptive dataset-size gating."),
        ("📐","12 Evaluation Metrics","Silhouette · Davies-Bouldin · Calinski-Harabasz · Dunn · Xie-Beni · cluster balance · inertia · geometry."),
        ("🧪","Stability Analysis","Bootstrap · Gaussian noise · Laplacian noise · feature dropout · subset sampling — full robustness profiling."),
        ("🤝","8 Consensus Methods","EAC (3 linkages) · CSPA · Voting · Weighted Voting · Meta-Clustering · Bayesian · Hybrid. Co-association matrix."),
        ("🔭","Multi-Embedding Vis","PCA · UMAP · t-SNE · ISOMAP · LLE — 2D and 3D projections, pair scatter, silhouette bars, centroid heatmaps."),
        ("","Gemini AI Insights","Context-aware AI analysis at every stage: data profiling, algorithm selection, results interpretation, recommendations."),
        ("📊","Pairwise ARI Matrix","Algorithm agreement heatmap — see which algorithms agree and which explore different structure."),
        ("🎯","k-Sweep Analysis","Automated elbow analysis across k=2..N for any algorithm. Silhouette, DB, CH curves. Optimal-k detection."),
        ("💾","Full Export Suite","Labels · metrics · co-association matrix · stability reports — CSV, JSON, and raw numpy download."),
    ]
    rows = [features[i:i+3] for i in range(0, len(features), 3)]
    for row in rows:
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, row):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="text-align:left; margin-bottom:.8rem;">
                    <div style="font-size:1.6rem; margin-bottom:.4rem;">{icon}</div>
                    <div style="font-weight:600; color:#ccccee; font-size:.95rem;">{title}</div>
                    <div style="font-size:.8rem; color:#666688; margin-top:.35rem; line-height:1.55;">{desc}</div>
                </div>""", unsafe_allow_html=True)

    if st.session_state.df_raw is not None:
        _sep()
        _section("📌 Current Session")
        df = st.session_state.df_raw
        c1,c2,c3,c4 = st.columns(4)
        with c1: _metric_card(f"{len(df):,}", "Rows Loaded", color="#00e5ff")
        with c2: _metric_card(str(len(df.columns)), "Columns", color="#9b59ff")
        if _has_data():
            X = st.session_state.X_processed
            with c3: _metric_card(str(X.shape[1]), "Features (processed)", color="#ff4daa")
        if st.session_state.batch_result:
            br = st.session_state.batch_result
            with c4: _metric_card(str(br.n_success), "Successful Runs", color="#00ff88")


# ─────────────────────────────────────────────────────────────────
# ▓▓  PAGE 1 · DATA INGESTION
# ─────────────────────────────────────────────────────────────────

elif page == "📁 Data Ingestion":
    _section("📁 Data Ingestion & Profiling")

    tab_upload, tab_sample, tab_profile = st.tabs(["📤 Upload", "🎲 Sample Datasets", "🔍 Deep Profile"])

    # ── Tab: Upload ───────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Drop your CSV / Excel / JSON here",
            type=["csv", "xlsx", "xls", "json", "tsv"],
            help="Supports CSV, Excel, JSON, TSV. Max ~500K rows auto-sampled.",
        )

        col_sep, col_decimal, col_head = st.columns(3)
        with col_sep:
            sep = st.text_input("CSV Separator", value=",", max_chars=3)
        with col_decimal:
            decimal = st.text_input("Decimal Character", value=".", max_chars=2)
        with col_head:
            header = st.number_input("Header Row (0=first)", value=0, min_value=0)

        if uploaded:
            with st.spinner("📥 Loading data..."):
                try:
                    DL = _b("DataLoader")()
                    df, load_log = DL.load(
                        uploaded, filename=uploaded.name,
                        sep=sep, decimal=decimal, header=int(header),
                    )
                    st.session_state.df_raw = df
                    st.session_state.df_filename = uploaded.name
                    _success(f"Loaded **{len(df):,}** rows × **{len(df.columns)}** columns from `{uploaded.name}`")
                    for msg in load_log:
                        if "WARNING" in msg:
                            _warn(msg)
                except Exception as e:
                    st.error(f"Load failed: {e}")
            # ─────────────────────────────────────────────────────────────────
    # OPTION 3: SYNTHETIC PATTERN GENERATOR (NEW)
    # ─────────────────────────────────────────────────────────────────
    
    _sep()
    _section("🎯 Generate Synthetic Pattern")
    
    col_pat1, col_pat2, col_pat3 = st.columns([2, 1, 1])
    
    with col_pat1:
        pattern_names = _b("get_pattern_names")()
        pattern_id = st.selectbox(
            "Clustering Pattern",
            options=list(pattern_names.keys()),
            format_func=lambda x: pattern_names[x],
            key="pattern_select",
            help="Choose from 21 dynamic clustering patterns"
        )
        
        # Show pattern details
        if pattern_id:
            pattern_info = _b("get_pattern_info")(pattern_id)
            st.caption(f"✨ {pattern_info['description']}")
            st.caption(f"🎯 Best for: {pattern_info['best_for']}")
    
    with col_pat2:
        n_samples = st.number_input(
            "Sample Count", 
            min_value=50, max_value=5000, 
            value=300, step=50,
            key="pat_samples"
        )
    
    with col_pat3:
        n_clusters = st.number_input(
            "Cluster Count", 
            min_value=2, max_value=10, 
            value=5, step=1,
            key="pat_clusters"
        )
    
    if st.button("🎲 Generate Pattern", use_container_width=True, 
                 type="primary", key="gen_pattern_btn"):
        with st.spinner("🔄 Generating pattern..."):
            try:
                df, metadata = _b("generate_pattern_dataframe")(
                    pattern_id, 
                    n_samples=int(n_samples),
                    n_clusters=int(n_clusters),
                    random_state=42
                )
                st.session_state.df_raw = df
                st.session_state.source_file = f"Generated: {metadata['pattern_name']}"
                st.session_state.df_profile = None
                
                _success(f"✅ Generated {metadata['n_samples']} samples "
                        f"from **{metadata['pattern_name']}**")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Show pattern preview
    _sep()
    if st.session_state.df_raw is not None:
        if len(st.session_state.df_raw.columns) >= 2:
            _subsection("📊 Pattern Preview")
            
            col_vis1, col_vis2 = st.columns([3, 1])
            
            with col_vis1:
                import plotly.express as px
                
                fig = px.scatter(
                    st.session_state.df_raw,
                    x=st.session_state.df_raw.columns[0],
                    y=st.session_state.df_raw.columns[1],
                    title="Live Pattern Visualization",
                    opacity=0.65
                )
                T = _b("Theme")
                fig.update_layout(
                    height=350,
                    paper_bgcolor=T.BG_DARK if T else "#07070f",
                    plot_bgcolor=T.BG_CARD if T else "#0d0d1e",
                    font=dict(color=T.TEXT_PRIMARY if T else "#e0e0f0"),
                    margin=dict(l=50, r=20, t=40, b=40),
                    showlegend=False,
                    hovermode="closest"
                )
                fig.update_traces(
                    marker=dict(
                        color="#00e5ff", 
                        size=6, 
                        line=dict(width=0),
                        opacity=0.6
                    )
                )
                st.plotly_chart(fig, use_container_width=True, 
                               config={"displayModeBar": False})
            
            with col_vis2:
                _metric_card(
                    str(len(st.session_state.df_raw)), 
                    "Total Samples", 
                    color="#00e5ff"
                )
                _metric_card(
                    str(len(st.session_state.df_raw.columns)), 
                    "Features", 
                    color="#9b59ff"
                )
                st.divider()
                if st.button("📋 Show Data", use_container_width=True, key="show_synth"):
                    st.dataframe(
                        st.session_state.df_raw.head(10), 
                        use_container_width=True
                    )
    


        if st.session_state.df_raw is not None:
            df = st.session_state.df_raw
            _subsection("Preview")
            preview_rows = st.slider("Rows to preview", 5, 100, 20)
            st.dataframe(df.head(preview_rows), use_container_width=True, height=340)

            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: _metric_card(f"{len(df):,}", "Rows")
            with c2: _metric_card(str(len(df.columns)), "Columns")
            with c3: _metric_card(str(df.select_dtypes(include=np.number).shape[1]), "Numeric Cols", color="#00ff88")
            with c4: _metric_card(str(df.select_dtypes(exclude=np.number).shape[1]), "Categorical Cols", color="#9b59ff")
            with c5: _metric_card(f"{df.isnull().mean().mean()*100:.1f}%", "Missing %", color="#ff8c00")

            st.markdown("")
            if st.button("🔍 Run Deep Profile", type="primary", use_container_width=True):
                with st.spinner("Profiling dataset..."):
                    try:
                        profiler = _b("DataProfiler")()
                        profile = profiler.profile(df)
                        st.session_state.data_profile = profile
                        _success("Profile complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Profiling failed: {e}")

    # ── Tab: Sample Datasets ──────────────────────────────────────
    with tab_sample:
        _subsection("Built-in Sample Datasets")
        datasets = {
            "Blobs (n=1000, k=5)": ("sklearn", dict(n_samples=1000, centers=5, random_state=42)),
            "Moons (n=800)":        ("moons",  dict(n_samples=800, noise=0.08, random_state=42)),
            "Circles (n=800)":      ("circles",dict(n_samples=800, noise=0.05, factor=0.5, random_state=42)),
            "Anisotropic Blobs":    ("aniso",  dict(n_samples=1500, random_state=42)),
            "Varied Variance Blobs":("varied_blobs", dict(n_samples=1500, random_state=42)),
            "Iris (UCI)":           ("iris",   {}),
            "Digits (8×8, n=1797)": ("digits", {}),
            "Wine (UCI)":           ("wine",   {}),
            "Breast Cancer":        ("cancer", {}),
            "S-Curve Manifold":     ("scurve", dict(n_samples=1000, noise=0.05, random_state=42)),
        }

        sel_ds = st.selectbox("Select dataset", list(datasets.keys()))
        if st.button("📥 Load Sample Dataset", use_container_width=True):
            kind, kwargs = datasets[sel_ds]
            try:
                if kind == "sklearn":
                    from sklearn.datasets import make_blobs
                    X, y = make_blobs(**kwargs)
                elif kind == "moons":
                    from sklearn.datasets import make_moons
                    X, y = make_moons(**kwargs)
                elif kind == "circles":
                    from sklearn.datasets import make_circles
                    X, y = make_circles(**kwargs)
                elif kind == "aniso":
                    from sklearn.datasets import make_blobs
                    from sklearn.preprocessing import StandardScaler
                    X, y = make_blobs(**kwargs)
                    transform = [[0.6,-0.6],[-0.4,0.8]]
                    X = X @ np.array(transform)
                elif kind == "varied_blobs":
                    from sklearn.datasets import make_blobs
                    X, y = make_blobs(cluster_std=[1.0,2.5,0.5], **kwargs)
                elif kind == "iris":
                    from sklearn.datasets import load_iris
                    d = load_iris(); X, y = d.data, d.target
                elif kind == "digits":
                    from sklearn.datasets import load_digits
                    d = load_digits(); X, y = d.data, d.target
                elif kind == "wine":
                    from sklearn.datasets import load_wine
                    d = load_wine(); X, y = d.data, d.target
                elif kind == "cancer":
                    from sklearn.datasets import load_breast_cancer
                    d = load_breast_cancer(); X, y = d.data, d.target
                elif kind == "scurve":
                    from sklearn.datasets import make_s_curve
                    X3, t = make_s_curve(**kwargs)
                    X = X3[:, [0,2]]; y = (t * 4).astype(int)
                else:
                    raise ValueError(f"Unknown: {kind}")

                cols = [f"feature_{i}" for i in range(X.shape[1])]
                df = pd.DataFrame(X, columns=cols)
                df["__true_label__"] = y
                st.session_state.df_raw = df
                st.session_state.df_filename = sel_ds
                _success(f"Loaded **{sel_ds}**: {len(df)} rows × {len(df.columns)} cols")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Failed: {e}")

    # ── Tab: Deep Profile ─────────────────────────────────────────
    with tab_profile:
        profile = st.session_state.data_profile
        if profile is None:
            _info("Upload data and click **Run Deep Profile** to see statistics.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            with c1: _metric_card(f"{profile.n_rows:,}", "Rows")
            with c2: _metric_card(str(profile.n_numeric), "Numeric", color="#00ff88")
            with c3: _metric_card(f"{profile.total_missing_pct:.1f}%", "Missing", color="#ff8c00")
            with c4: _metric_card(f"{profile.duplicate_pct:.1f}%", "Duplicates", color="#ff4daa")

            _sep()
            vis_engine = _get_vis()

            col_left, col_right = st.columns([1.2, 1])
            with col_left:
                _subsection("Column Statistics")
                rows = []
                for col, cp in profile.column_profiles.items():
                    rows.append({
                        "Column": col[:30], "Type": cp.dtype,
                        "Unique": cp.n_unique, "Missing%": cp.missing_pct,
                        "Mean": round(cp.mean,3) if cp.mean else None,
                        "Std": round(cp.std,3) if cp.std else None,
                        "Skew": round(cp.skewness,3) if cp.skewness else None,
                        "Outliers%": cp.outlier_pct,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=380)

            with col_right:
                if profile.total_missing > 0:
                    _subsection("Missing Values")
                    fig_miss = vis_engine.missing_values_bar(profile) if vis_engine else None
                    _safe_plotly(fig_miss)

            if profile.correlation_matrix is not None and vis_engine:
                _sep()
                _subsection("Correlation Heatmap")
                max_corr_cols = st.slider("Max columns in correlation matrix", 5, 50, 25)
                fig_corr = vis_engine.correlation_heatmap(
                    profile.correlation_matrix, max_cols=max_corr_cols)
                _safe_plotly(fig_corr)

            if profile.high_corr_pairs:
                _subsection(f"High-Correlation Pairs (top {min(10, len(profile.high_corr_pairs))})")
                hc_rows = [{"Feature A": a[:25], "Feature B": b[:25], "Pearson |r|": round(r,4)}
                           for a,b,r in profile.high_corr_pairs[:10]]
                st.dataframe(pd.DataFrame(hc_rows), use_container_width=True)

            _sep()
            _subsection("Feature Distribution Explorer")
            num_cols = [c for c, cp in profile.column_profiles.items() if cp.is_numeric]
            if num_cols and vis_engine:
                sel_col = st.selectbox("Select feature", num_cols)
                df = st.session_state.df_raw
                if sel_col in df.columns:
                    fig_hist = vis_engine.distribution_histogram(df[sel_col], sel_col)
                    _safe_plotly(fig_hist)

            if profile.warnings:
                _subsection("Profiler Warnings")
                for w in profile.warnings[:10]:
                    _warn(w)


# ─────────────────────────────────────────────────────────────────
# ▓▓  PAGE 2 · PREPROCESSING
# ─────────────────────────────────────────────────────────────────

elif page == "⚙️ Preprocessing":
    _section("⚙️ Preprocessing Studio")

    if st.session_state.df_raw is None:
        _warn("No data loaded. Go to **📁 Data Ingestion** first.")
        st.stop()

    df = st.session_state.df_raw
    profile = st.session_state.data_profile

    # Auto-recommend
    col_rec, col_run = st.columns([3,1])
    with col_rec:
        if profile is not None:
            _info("💡 Smart Config recommended based on your data profile. Review and adjust below.")
        else:
            _info("Run **Deep Profile** first for auto-recommendations (Data Ingestion → Deep Profile).")

    with col_run:
        auto_rec = st.button("✨ Auto-Recommend Config", use_container_width=True)

    if auto_rec and profile:
        rec_cfg = _b("infer_best_config")(profile)
        st.session_state._rec_cfg = rec_cfg

    # Identify columns
    num_cols_all = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_all = df.select_dtypes(exclude=np.number).columns.tolist()

    tab_basic, tab_advanced, tab_preview = st.tabs(["🔧 Basic Config", "🔬 Advanced Config", "👁️ Preview"])

    with tab_basic:
        col_a, col_b = st.columns(2)

        with col_a:
            _subsection("Column Management")
            drop_cols = st.multiselect(
                "Columns to drop",
                df.columns.tolist(),
                default=[c for c in ["__true_label__"] if c in df.columns],
                help="Select columns to exclude from clustering"
            )

            _subsection("Scaler")
            scaler_options = {o["value"]: o for o in _b("get_scaler_options")()}
            rec_scaler = getattr(st.session_state.get("_rec_cfg"), "scaler_type",
                                  _b("ScalerType").STANDARD).value if hasattr(st.session_state.get("_rec_cfg",""), "scaler_type") else "standard"
            scaler_val = st.selectbox(
                "Scaling method",
                list(scaler_options.keys()),
                index=list(scaler_options.keys()).index(rec_scaler)
                      if rec_scaler in scaler_options else 0,
                format_func=lambda x: scaler_options[x]["label"],
            )
            st.caption(scaler_options[scaler_val].get("description",""))

        with col_b:
            _subsection("Missing Values")
            imputer_options = {o["value"]: o for o in _b("get_imputer_options")()}
            imputer_val = st.selectbox(
                "Imputation strategy",
                list(imputer_options.keys()),
                format_func=lambda x: imputer_options[x]["label"],
            )
            if imputer_val == "knn":
                knn_k = st.slider("KNN neighbors", 3, 20, 5)
            elif imputer_val == "constant":
                fill_val = st.number_input("Fill value", value=0.0)

            _subsection("Outlier Handling")
            outlier_options = {o["value"]: o for o in _b("get_outlier_options")()}
            outlier_val = st.selectbox(
                "Outlier detection method",
                list(outlier_options.keys()),
                format_func=lambda x: outlier_options[x]["label"],
            )
            if outlier_val != "none":
                outlier_action = st.selectbox("Outlier action",
                    ["flag","remove","clip","none"],
                    format_func=lambda x: {"flag":"Flag (add __outlier__ col)",
                                           "remove":"Remove rows",
                                           "clip":"Clip to [1%,99%]",
                                           "none":"No action"}[x])
                if outlier_val == "zscore":
                    outlier_thresh = st.slider("Z-score threshold", 1.5, 5.0, 3.0, 0.1)
                else:
                    outlier_contam = st.slider("Contamination fraction", 0.01, 0.3, 0.05, 0.01)

    with tab_advanced:
        col_c, col_d = st.columns(2)

        with col_c:
            _subsection("Feature Selection")
            feat_sel_map = {
                "none":"None — use all features",
                "variance_threshold":"Variance Threshold (remove near-zero variance)",
                "correlation":"Correlation Filter (remove highly correlated)",
                "pca_reduce":"PCA Reduction",
                "manual":"Manual Column Selection",
            }
            feat_sel = st.selectbox("Method", list(feat_sel_map.keys()),
                                     format_func=lambda x: feat_sel_map[x])
            if feat_sel == "variance_threshold":
                var_thresh = st.slider("Variance threshold", 0.0, 0.5, 0.01, 0.001)
            elif feat_sel == "correlation":
                corr_thresh = st.slider("Correlation threshold", 0.7, 0.99, 0.95, 0.01)
            elif feat_sel == "pca_reduce":
                pca_var = st.slider("Variance to explain", 0.80, 0.999, 0.95, 0.005)
                pca_n   = st.number_input("Or fixed n_components (0=use variance)", 0, 200, 0)
            elif feat_sel == "manual":
                manual_cols = st.multiselect("Select features to keep", num_cols_all,
                                              default=num_cols_all[:min(10, len(num_cols_all))])

        with col_d:
            _subsection("Categorical Encoding")
            encode_cats = st.checkbox("Encode categorical columns", value=True)
            max_card    = st.slider("Max cardinality for encoding", 5, 200, 50)

            _subsection("Other Settings")
            random_state = st.number_input("Random seed", value=42, min_value=0)
            encode_cats_flag = encode_cats

    with tab_preview:
        _info("Configure settings in other tabs, then click **Run Preprocessing** below to see output.")

    _sep()
    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
    with c_btn1:
        run_preproc = st.button("⚙️ Run Preprocessing Pipeline", type="primary", use_container_width=True)
    with c_btn2:
        reset_preproc = st.button("🔄 Reset", use_container_width=True)

    if reset_preproc:
        st.session_state.preproc_result = None
        st.session_state.X_processed = None
        st.rerun()

    if run_preproc:
        with st.spinner("Running preprocessing pipeline..."):
            try:
                ScalerType = _b("ScalerType")
                ImputeStrategy = _b("ImputeStrategy")
                OutlierMethod = _b("OutlierMethod")
                OutlierAction = _b("OutlierAction")
                FeatureSelectionMethod = _b("FeatureSelectionMethod")
                PreprocessingConfig = _b("PreprocessingConfig")

                config = PreprocessingConfig(
                    drop_columns=drop_cols,
                    scaler_type=ScalerType(scaler_val),
                    impute_strategy=ImputeStrategy(imputer_val),
                    impute_constant=locals().get("fill_val", 0.0),
                    knn_neighbors=locals().get("knn_k", 5),
                    outlier_method=OutlierMethod(outlier_val),
                    outlier_action=OutlierAction(locals().get("outlier_action","flag"))
                                   if outlier_val != "none"
                                   else OutlierAction("none"),
                    outlier_threshold=locals().get("outlier_thresh", 3.0),
                    outlier_contamination=locals().get("outlier_contam", 0.05),
                    feature_selection=FeatureSelectionMethod(feat_sel),
                    variance_threshold=locals().get("var_thresh", 0.01),
                    correlation_threshold=locals().get("corr_thresh", 0.95),
                    pca_variance_explained=locals().get("pca_var", 0.95),
                    pca_n_components=int(locals().get("pca_n",0)) or None,
                    encode_categoricals=encode_cats_flag,
                    max_categorical_cardinality=max_card,
                    random_state=random_state,
                )

                pipeline = _b("PreprocessingPipeline")(config)
                result = pipeline.run(df)
                st.session_state.preproc_result = result
                st.session_state.X_processed = result.X_processed.values
                st.session_state.feature_names = result.feature_names
                st.session_state.preproc_config = config
                _success(f"Preprocessing done. Output shape: **{result.X_processed.shape}**")
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                st.code(traceback.format_exc(), language="python")

    # Show results
    if st.session_state.preproc_result is not None:
        result = st.session_state.preproc_result
        _sep()
        _subsection("Preprocessing Output")

        c1,c2,c3,c4 = st.columns(4)
        with c1: _metric_card(f"{result.X_processed.shape[0]:,}", "Rows")
        with c2: _metric_card(str(result.X_processed.shape[1]), "Features", color="#00ff88")
        with c3: _metric_card(str(result.n_outliers), "Outliers Detected", color="#ff8c00")
        with c4: _metric_card(str(len(result.dropped_columns)), "Cols Dropped", color="#9b59ff")

        with st.expander("📋 Processing Log"):
            for msg in result.log:
                if "===" in msg:
                    st.markdown(f"**{msg}**")
                elif "WARNING" in msg or "⚠" in msg:
                    st.markdown(f"⚠️ {msg}")
                else:
                    st.text(msg)

        _subsection("Processed Feature Statistics")
        df_proc = result.X_processed
        stats = df_proc.describe().T.round(4)
        st.dataframe(stats, use_container_width=True, height=280)

        if result.outlier_mask is not None and result.outlier_mask.any():
            _subsection("Outlier Scatter Preview")
            vis_engine = _get_vis()
            if vis_engine and result.X_processed.shape[1] >= 2:
                fig_out = vis_engine.outlier_scatter(
                    result.X_processed.values, result.outlier_mask)
                _safe_plotly(fig_out)


# ─────────────────────────────────────────────────────────────────
# ▓▓  PAGE 3 · ALGORITHM ARENA
# ─────────────────────────────────────────────────────────────────

elif page == "🧬 Algorithm Arena":
    _section("🧬 Algorithm Arena")

    if not _ok():
        _warn("Backends not loaded.")
        st.stop()

    registry = _b("get_registry")()

    tab_browse, tab_select, tab_detail = st.tabs(
        ["🗂️ Algorithm Browser", "✅ Selection Panel", "🔬 Algorithm Inspector"])

    # ── Browser ───────────────────────────────────────────────────
    with tab_browse:
        col_filter, col_info = st.columns([1,2])
        with col_filter:
            _subsection("Filters")
            families = registry.family_map()
            sel_families = st.multiselect(
                "Algorithm Families",
                list(families.keys()),
                default=list(families.keys()),
            )
            show_no_k  = st.checkbox("Only auto-K (no k required)", False)
            show_noise = st.checkbox("Only noise-aware algorithms", False)
            search_q   = st.text_input("Search by name/ID", "")

        with col_info:
            filtered_ids = []
            for fam_name, specs in families.items():
                if fam_name not in sel_families:
                    continue
                st.markdown(f"""
                <div style="margin:.6rem 0 .3rem 0; font-size:.85rem; font-weight:600;
                     color:#9b59ff; letter-spacing:.08em; text-transform:uppercase;">{fam_name}</div>""",
                unsafe_allow_html=True)
                for spec in specs:
                    if show_no_k and spec.requires_n_clusters:
                        continue
                    if show_noise and not spec.produces_noise_label:
                        continue
                    if search_q and search_q.lower() not in (spec.name+spec.id).lower():
                        continue
                    filtered_ids.append(spec.id)
                    tags_html = ""
                    tag_map = {
                        "fast":"fast","scalable":"scale","no_k_needed":"nok",
                        "noise_robust":"noise","probabilistic":"prob","online":"scale"
                    }
                    for tag in spec.tags[:4]:
                        cls = tag_map.get(tag.value,"default")
                        tags_html += f'<span class="algo-tag tag-{cls}">{tag.value}</span>'
                    selected = spec.id in st.session_state.selected_algorithms
                    card_cls = "algo-card selected" if selected else "algo-card"
                    st.markdown(f"""
                    <div class="{card_cls}">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <div>
                                <span style="font-weight:600; color:#e0e0ff; font-size:.92rem;">{spec.name}</span>
                                <span style="color:#444466; font-size:.75rem; margin-left:.5rem; font-family:JetBrains Mono;">{spec.id}</span>
                            </div>
                            <span style="font-size:.7rem; color:#444466;">{spec.time_complexity.value}</span>
                        </div>
                        <div style="font-size:.78rem; color:#666688; margin:.3rem 0;">{spec.description[:110]}...</div>
                        <div>{tags_html}</div>
                    </div>""", unsafe_allow_html=True)

    # ── Selection ────────────────────────────────────────────────
    with tab_select:
        _subsection("Quick-Select Presets")
        preset_cols = st.columns(4)
        presets = {
            "⚡ Fast Essentials": ["kmeans","minibatch_kmeans","dbscan","agglomerative_ward","gmm_full"],
            "🔬 Full Density Suite": ["dbscan","optics","hdbscan","mean_shift","denclue"],
            "🎯 Top 15 (Balanced)": ["kmeans","kmeans_pp","minibatch_kmeans",
                                      "agglomerative_ward","agglomerative_complete",
                                      "birch","dbscan","hdbscan","optics",
                                      "gmm_full","gmm_diag","spectral_kmeans",
                                      "affinity_propagation","fuzzy_cmeans","autoencoder_kmeans"],
            " All Algorithms": registry.ids(),
        }
        for (pname, pids), col in zip(presets.items(), preset_cols):
            with col:
                if st.button(pname, use_container_width=True):
                    valid = [aid for aid in pids if aid in registry.ids()]
                    st.session_state.selected_algorithms = valid
                    _success(f"Selected {len(valid)} algorithms")
                    st.rerun()

        st.markdown("---")
        _subsection(f"Manual Selection ({len(st.session_state.selected_algorithms)} selected)")

        all_ids = registry.ids()
        sel = st.multiselect(
            "Select algorithms to run",
            all_ids,
            default=st.session_state.selected_algorithms,
            format_func=lambda x: f"{registry.get(x).name} [{x}]",
        )
        st.session_state.selected_algorithms = sel

        # k config
        _sep()
        _subsection("Clustering Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.n_clusters = st.slider("Target k (n_clusters)", 2, 30, st.session_state.n_clusters)
        with c2:
            timeout = st.slider("Per-algorithm timeout (s)", 10, 300, 120)
        with c3:
            max_workers = st.slider("Parallel workers", 1, 8, 4)

        st.session_state._run_timeout = timeout
        st.session_state._max_workers = max_workers

        if sel:
            _subsection("Selected Algorithms Summary")
            fam_counts: Dict[str,int] = {}
            for aid in sel:
                fam = registry.get(aid).family.value
                fam_counts[fam] = fam_counts.get(fam, 0) + 1
            c1, c2, c3 = st.columns(3)
            with c1: _metric_card(str(len(sel)), "Algorithms Selected", color="#00e5ff")
            with c2: _metric_card(str(len(fam_counts)), "Families Covered", color="#9b59ff")
            with c3: _metric_card(
                str(sum(1 for a in sel if not registry.get(a).requires_n_clusters)),
                "Auto-K Algorithms", color="#ff4daa")

            fam_df = pd.DataFrame(list(fam_counts.items()), columns=["Family","Count"])
            fig_fam = px.bar(fam_df, x="Count", y="Family", orientation="h",
                             color="Count", color_continuous_scale=["#1a003a","#9b59ff","#00e5ff"])
            fig_fam.update_layout(paper_bgcolor="#07070f", plot_bgcolor="#0d0d1e",
                                   font=dict(color="#e0e0f0"), height=300,
                                   showlegend=False, coloraxis_showscale=False,
                                   margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_fam, use_container_width=True, config={"displayModeBar":False})

    # ── Inspector ─────────────────────────────────────────────────
    with tab_detail:
        _subsection("Inspect Algorithm Details")
        inspect_id = st.selectbox("Select algorithm to inspect",
                                   registry.ids(),
                                   format_func=lambda x: f"{registry.get(x).name}")
        if inspect_id:
            spec = registry.get(inspect_id)
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.markdown(f"""
                <div class="metric-card" style="text-align:left;">
                    <div style="font-family:'Space Grotesk'; font-size:1.3rem; font-weight:700; color:#00e5ff;">{spec.name}</div>
                    <div style="color:#555577; font-size:.8rem; font-family:JetBrains Mono; margin:.3rem 0;">{spec.id}</div>
                    <div style="color:#8888bb; font-size:.88rem; line-height:1.65; margin-top:.6rem;">{spec.description}</div>
                    {'<div style="color:#555577; font-size:.78rem; margin-top:.6rem;">Paper: ' + spec.paper_ref + '</div>' if spec.paper_ref else ''}
                </div>""", unsafe_allow_html=True)
            with c2:
                _metric_card(spec.family.value, "Family", color="#9b59ff")
                st.markdown("")
                _metric_card(spec.time_complexity.value, "Time Complexity", color="#ffd700")
                st.markdown("")
                props = []
                if not spec.requires_n_clusters: props.append("✅ Auto-K")
                if spec.produces_noise_label: props.append("🔇 Noise label")
                for p in props:
                    st.markdown(f'<span class="badge badge-success" style="margin:.2rem;">{p}</span>',
                                unsafe_allow_html=True)

            if spec.hyper_params:
                _subsection("Hyper-Parameters")
                hp_rows = [{"Parameter": hp.name, "Type": hp.dtype,
                             "Default": hp.default, "Range": f"[{hp.min_val}, {hp.max_val}]"
                             if hp.min_val is not None else "—",
                             "Description": hp.description}
                            for hp in spec.hyper_params]
                st.dataframe(pd.DataFrame(hp_rows), use_container_width=True)

            _tag_cls_map = {
                "fast":"fast","scalable":"scale","no_k_needed":"nok",
                "noise_robust":"noise","probabilistic":"prob",
            }
            tags_html = "".join(
                f'<span class="algo-tag tag-{_tag_cls_map.get(t.value, "default")}">{t.value}</span>'
                for t in spec.tags
            )
            st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# ▓▓  PAGE 4 · EXECUTION ENGINE
# ─────────────────────────────────────────────────────────────────

elif page == "⚡ Execution Engine":
    _section("⚡ Execution Engine")

    if not _has_data():
        _warn("No preprocessed data. Complete **⚙️ Preprocessing** first.")
        st.stop()
    if not st.session_state.selected_algorithms:
        _warn("No algorithms selected. Go to **🧬 Algorithm Arena** first.")
        st.stop()

    X = st.session_state.X_processed
    n_clusters = st.session_state.n_clusters
    sel_algos  = st.session_state.selected_algorithms
    timeout    = st.session_state.get("_run_timeout", 120)
    max_wk     = st.session_state.get("_max_workers", 4)

    c1,c2,c3,c4 = st.columns(4)
    with c1: _metric_card(f"{X.shape[0]:,}", "Samples")
    with c2: _metric_card(str(X.shape[1]), "Features", color="#9b59ff")
    with c3: _metric_card(str(len(sel_algos)), "Algorithms Queued", color="#ff4daa")
    with c4: _metric_card(str(n_clusters), "Target k", color="#ffd700")

    _sep()

    exec_mode = st.selectbox(
        "Execution mode",
        ["🔀 Adaptive (recommended)", "⚡ Parallel", "🔁 Sequential"],
    )
    mode_map = {
        "🔀 Adaptive (recommended)":"adaptive",
        "⚡ Parallel":"parallel",
        "🔁 Sequential":"sequential",
    }

    _sep()
    col_run, col_eval, _ = st.columns([1,1,2])
    with col_run:
        do_run = st.button(" Launch Clustering", type="primary", use_container_width=True)
    with col_eval:
        do_eval_only = st.button("📊 Re-Evaluate (existing results)", use_container_width=True)

    if do_run:
        with st.spinner(""):
            run_progress = st.progress(0)
            run_status   = st.empty()
            run_log_box  = st.empty()
            log_lines    = []

            def _progress_cb(alg_id, completed, total):
                pct = int(completed / max(total,1) * 100)
                run_progress.progress(pct)
                run_status.markdown(
                    f'<div class="info-panel pulsing">⚡ Running: <b>{alg_id}</b> ({completed}/{total})</div>',
                    unsafe_allow_html=True)
                log_lines.append(f"[{completed:3d}/{total}] {alg_id}")
                if len(log_lines) > 8:
                    log_lines.pop(0)
                run_log_box.code("\n".join(log_lines), language="text")

            from clustering_runner import RunnerConfig, ExecutionMode, ClusteringRunner
            emode = {
                "adaptive":  ExecutionMode.ADAPTIVE,
                "parallel":  ExecutionMode.PARALLEL,
                "sequential":ExecutionMode.SEQUENTIAL,
            }[mode_map[exec_mode]]

            config = RunnerConfig(
                n_clusters=n_clusters,
                execution_mode=emode,
                max_workers=max_wk,
                timeout_seconds=timeout,
                use_cache=True,
                skip_slow_on_large=True,
                progress_callback=_progress_cb,
                verbose=False,
            )
            runner = ClusteringRunner(config)
            t0 = time.perf_counter()
            batch = runner.run_algorithms(sel_algos, X)
            elapsed = time.perf_counter() - t0

            st.session_state.batch_result = batch
            st.session_state.last_run_timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.run_log = runner.run_log

            run_progress.progress(100)
            run_status.empty()
            run_log_box.empty()

            # Evaluate
            eval_results = _b("evaluate_all")(X, batch, fast_mode=False)
            st.session_state.eval_results = eval_results
            st.session_state.eval_dict = {r.algorithm_id: r for r in eval_results}

            _success(f"Run complete in **{elapsed:.1f}s** — {batch.n_success} succeeded, {batch.n_failed} failed, {batch.n_timeout} timeout.")

    if do_eval_only and st.session_state.batch_result:
        with st.spinner("Re-evaluating..."):
            eval_results = _b("evaluate_all")(X, st.session_state.batch_result, fast_mode=False)
            st.session_state.eval_results = eval_results
            st.session_state.eval_dict = {r.algorithm_id: r for r in eval_results}
            _success("Re-evaluation complete.")

    # Show run summary
    br = st.session_state.batch_result
    if br:
        _sep()
        _subsection("Run Summary")
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: _metric_card(str(br.n_success), "✓ Success", color="#00ff88")
        with c2: _metric_card(str(br.n_failed), "✗ Failed", color="#ff4444")
        with c3: _metric_card(str(br.n_timeout), "⏱ Timeout", color="#ff8c00")
        with c4: _metric_card(str(br.n_skipped), "⊘ Skipped", color="#888888")
        with c5: _metric_card(f"{br.total_runtime:.1f}s", "Total Time", color="#ffd700")

        _subsection("Per-Algorithm Status")
        status_rows = []
        for aid, cr in br.results.items():
            status_rows.append({
                "Algorithm": getattr(cr, "algorithm_name", aid)[:35],
                "Family": getattr(cr, "algorithm_family", "—")[:20],
                "Status": cr.status.value,
                "k Found": cr.n_clusters_found,
                "Noise%": f"{cr.noise_ratio*100:.1f}%",
                "Time(s)": round(cr.runtime_seconds, 3),
                "Error": (cr.error_message or "")[:60],
            })
        status_df = pd.DataFrame(status_rows)
        st.dataframe(status_df, use_container_width=True, height=380)

        if st.session_state.run_log:
            with st.expander("📋 Run Log"):
                st.code("\n".join(st.session_state.run_log[-50:]), language="text")

        if st.session_state.eval_results:
            best = _b("get_best_algorithm")(st.session_state.eval_results)
            if best:
                _sep()
                st.markdown(f"""
                <div class="success-panel">
                    🏆 <b>Best Algorithm:</b> {best.algorithm_name}
                    — Composite Score: <b>{best.composite_score:.1f}/100</b>
                    — Silhouette: <b>{(best.metric_value('silhouette') or 0):.4f}</b>
                    — k: <b>{best.n_clusters}</b>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# ▓▓  PAGE 5 · RESULTS DASHBOARD
# ─────────────────────────────────────────────────────────────────

elif page == "📊 Results Dashboard":
    _section("📊 Results Dashboard")

    if not _has_results():
        _warn("No results yet. Run clustering in **⚡ Execution Engine** first.")
        st.stop()

    eval_results = st.session_state.eval_results
    X = st.session_state.X_processed
    vis_engine = _get_vis()

    best = _b("get_best_algorithm")(eval_results)

    # Top KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: _metric_card(str(len(eval_results)), "Evaluated", color="#00e5ff")
    with c2: _metric_card(best.algorithm_name[:18] if best else "—", "🏆 Best", color="#ffd700")
    with c3: _metric_card(f"{best.composite_score:.1f}" if best else "—", "Best Score", color="#00ff88")
    with c4: _metric_card(f"{(best.metric_value('silhouette') or 0):.4f}" if best else "—", "Best Silhouette", color="#9b59ff")
    with c5: _metric_card(str(best.n_clusters) if best else "—", "Best k", color="#ff4daa")

    tab_table, tab_radar, tab_heatmap, tab_violin, tab_rank = st.tabs([
        "📋 Ranking Table", "📡 Radar Chart", "🌡️ Metric Heatmap", "🎻 Violin Plot", "🏅 Score Board"
    ])

    with tab_table:
        _subsection("Algorithm Rankings")
        top_n = st.slider("Show top N algorithms", 5, len(eval_results), min(20, len(eval_results)))
        df_results = _b("build_results_dataframe")(eval_results[:top_n])

        if not df_results.empty:
            st.dataframe(df_results.style.background_gradient(
                subset=["Score"] if "Score" in df_results.columns else None,
                cmap="Blues"), use_container_width=True, height=420)

        col_dl1, _ = st.columns([1,4])
        with col_dl1:
            csv = df_results.to_csv(index=False).encode()
            st.download_button("💾 Download Rankings CSV", csv,
                                "clusterx_rankings.csv", "text/csv")

    with tab_radar:
        top_k = st.slider("Algorithms in radar", 3, min(10, len(eval_results)), 6)
        if vis_engine:
            fig_radar = vis_engine.radar_chart(eval_results, top_n=top_k)
            _safe_plotly(fig_radar)

    with tab_heatmap:
        if vis_engine:
            from evaluation import ResultsTableBuilder
            builder = ResultsTableBuilder()
            hmap_df = builder.build_score_heatmap_data(eval_results)
            if not hmap_df.empty:
                fig_hmap = vis_engine.metric_heatmap(hmap_df, "Normalised Metric Scores — All Algorithms")
                _safe_plotly(fig_hmap)

    with tab_violin:
        if vis_engine:
            sel_metrics = st.multiselect(
                "Select metrics",
                ["silhouette","davies_bouldin","calinski_harabasz","dunn_index"],
                default=["silhouette","davies_bouldin"],
            )
            if sel_metrics:
                fig_vio = vis_engine.violin_metric(eval_results, metric_ids=sel_metrics)
                _safe_plotly(fig_vio)

    with tab_rank:
        _subsection("Full Composite Score Bar")
        if vis_engine:
            fig_bar = vis_engine.composite_score_bar(eval_results)
            _safe_plotly(fig_bar)

        _sep()
        _subsection("Individual Metric Comparison")
        sel_metric_bar = st.selectbox("Metric", list(_b("METRIC_REGISTRY").keys()))
        spec = _b("METRIC_REGISTRY")[sel_metric_bar]
        if vis_engine:
            fig_met = vis_engine.metric_bar(eval_results, sel_metric_bar, spec.name)
            _safe_plotly(fig_met)

    _sep()
    _subsection("Algorithm Interpretation Cards")
    top_show = st.slider("Show interpretations for top N", 1, min(10, len(eval_results)), 5)
    for er in eval_results[:top_show]:
        with st.expander(f"{'🥇' if er.rank==1 else '🥈' if er.rank==2 else '🥉' if er.rank==3 else f'#{er.rank}'} {er.algorithm_name} — Score {er.composite_score:.1f}"):
            col_a, col_b = st.columns([1.5,1])
            with col_a:
                st.markdown(f'<div class="info-panel">{er.interpretation}</div>', unsafe_allow_html=True)
                for w in er.warnings[:3]:
                    _warn(w)
            with col_b:
                m_rows = []
                for mid, mv in er.metrics.items():
                    if mv.is_valid:
                        spec = _b("METRIC_REGISTRY").get(mid)
                        m_rows.append({
                            "Metric": spec.name if spec else mid,
                            "Value": _b("format_metric_value")(mid, mv.value),
                            "Grade": mv.grade,
                        })
                if m_rows:
                    st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────
# ▓▓  PAGE 6 · VISUALIZATION LAB
# ─────────────────────────────────────────────────────────────────

elif page == "🔬 Visualization Lab":
    _section("🔬 Visualization Lab")

    if not _has_results():
        _warn("No results. Run clustering first.")
        st.stop()

    X = st.session_state.X_processed
    eval_results = st.session_state.eval_results
    batch_result = st.session_state.batch_result
    feature_names = st.session_state.feature_names
    vis_engine = _get_vis()

    # Algorithm selector
    alg_options = {
        er.algorithm_id: f"#{er.rank} {er.algorithm_name} (Score: {er.composite_score:.1f})"
        for er in eval_results
    }
    sel_alg_id = st.selectbox(
        "Select algorithm to visualise",
        list(alg_options.keys()),
        format_func=lambda x: alg_options[x],
    )
    sel_result = batch_result.results.get(sel_alg_id)
    if sel_result is None or not sel_result.succeeded:
        _warn("Selected algorithm has no valid labels.")
        st.stop()

    labels = sel_result.labels

    tab_2d, tab_3d, tab_multi, tab_profile, tab_pair = st.tabs([
        "🗺️ 2D Projection","🌐 3D Projection","🔭 Multi-Embedding","📊 Cluster Profile","🔲 Pair Matrix"
    ])

    with tab_2d:
        col_ctrl, _ = st.columns([1,2])
        with col_ctrl:
            method_2d = st.selectbox("Embedding method", _b("available_embedding_methods")(), key="vis2d")
            show_noise = st.checkbox("Show noise points", True)
            marker_size = st.slider("Marker size", 2, 12, 5)
            opacity = st.slider("Opacity", 0.3, 1.0, 0.8)

        if vis_engine:
            with st.spinner(f"Computing {method_2d} embedding..."):
                fig_2d = vis_engine.scatter_2d(
                    X, labels, method=method_2d,
                    algorithm_name=sel_result.algorithm_name,
                    show_noise=show_noise, marker_size=marker_size, opacity=opacity,
                )
            _safe_plotly(fig_2d)

    with tab_3d:
        method_3d = st.selectbox("Embedding method", _b("available_embedding_methods")(), key="vis3d")
        if vis_engine:
            with st.spinner(f"Computing {method_3d} 3D..."):
                fig_3d = vis_engine.scatter_3d(X, labels, method=method_3d,
                                                algorithm_name=sel_result.algorithm_name)
            _safe_plotly(fig_3d)

    with tab_multi:
        methods_multi = st.multiselect(
            "Embedding methods to compare",
            _b("available_embedding_methods")(),
            default=["PCA","t-SNE"],
        )
        if methods_multi and vis_engine:
            with st.spinner("Computing multiple embeddings..."):
                fig_multi = vis_engine.multi_embedding(
                    X, labels, methods=methods_multi,
                    algorithm_name=sel_result.algorithm_name)
            _safe_plotly(fig_multi)

    with tab_profile:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if vis_engine:
                fig_size = vis_engine.cluster_size_distribution(
                    labels, sel_result.algorithm_name)
                _safe_plotly(fig_size)

        with col_p2:
            if vis_engine:
                from evaluation import SilhouetteComputer
                sil_comp = SilhouetteComputer()
                sil_vals = sil_comp.per_sample(X, labels, max_samples=5000)
                if sil_vals is not None:
                    fig_sil = vis_engine.silhouette_bar(sil_vals, labels,
                                                         sel_result.algorithm_name)
                    _safe_plotly(fig_sil)

        if vis_engine and len(feature_names) > 0:
            _sep()
            _subsection("Cluster Centroid Heatmap")
            fig_cent = vis_engine.centroid_heatmap(X, labels, feature_names,
                                                     sel_result.algorithm_name)
            _safe_plotly(fig_cent)

            _sep()
            _subsection("Feature Importance for Clustering")
            top_feat_n = st.slider("Top N features", 5, min(50, len(feature_names)), 20)
            fig_feat = vis_engine.feature_importance(X, labels, feature_names, top_n=top_feat_n)
            _safe_plotly(fig_feat)

    with tab_pair:
        if vis_engine and len(feature_names) >= 2:
            max_feat = st.slider("Max features in pair matrix", 2, min(8, len(feature_names)), 5)
            with st.spinner("Building pair scatter matrix..."):
                fig_pair = vis_engine.pair_scatter_matrix(X, labels, feature_names, max_features=max_feat)
            _safe_plotly(fig_pair)

    _sep()
    _subsection("PCA Scree Plot")
    if vis_engine:
        var_data = vis_engine.variance_explained(X)
        fig_scree = vis_engine.scree_plot(var_data)
        _safe_plotly(fig_scree)


# ─────────────────────────────────────────────────────────────────
# ▓▓  PAGE 7 · STABILITY LAB
# ─────────────────────────────────────────────────────────────────

elif page == "🧪 Stability Lab":
    _section("🧪 Stability Lab")

    if not _has_results():
        _warn("No results available. Run clustering first.")
        st.stop()

    X = st.session_state.X_processed
    eval_results = st.session_state.eval_results
    batch_result = st.session_state.batch_result
    vis_engine = _get_vis()

    tab_config, tab_run, tab_ari_matrix, tab_noise = st.tabs([
        "⚙️ Config", "🧪 Run Tests", "🔲 ARI Matrix", "📉 Noise Response"
    ])

    with tab_config:
        _subsection("Stability Test Configuration")
        c1, c2, c3 = st.columns(3)
        with c1:
            n_boot = st.slider("Bootstrap runs", 3, 30, 8)
            boot_frac = st.slider("Bootstrap fraction", 0.5, 0.95, 0.80, 0.05)
        with c2:
            noise_levels_str = st.text_input("Noise levels (comma-separated)", "0.02,0.05,0.10,0.20")
            n_noise_runs = st.slider("Runs per noise level", 2, 10, 3)
        with c3:
            enable_dropout = st.checkbox("Feature dropout", True)
            enable_subset  = st.checkbox("Subset sampling", True)
            fast_mode = st.checkbox("Fast mode (fewer runs)", False)

        sel_test_algos = st.multiselect(
            "Algorithms to test (top-10 by default)",
            [er.algorithm_id for er in eval_results],
            default=[er.algorithm_id for er in eval_results[:min(8, len(eval_results))]],
            format_func=lambda x: next((er.algorithm_name for er in eval_results if er.algorithm_id==x), x),
        )

    with tab_run:
        col_btn, _ = st.columns([1,3])
        with col_btn:
            run_stab = st.button("🧪 Run Stability Analysis", type="primary", use_container_width=True)

        if run_stab:
            if not sel_test_algos:
                _warn("Select algorithms to test.")
            else:
                noise_levels = [float(x.strip()) for x in noise_levels_str.split(",") if x.strip()]
                stab_config = _b("StabilityConfig")(
                    n_bootstrap_runs=n_boot,
                    bootstrap_fraction=boot_frac,
                    noise_levels=noise_levels,
                    n_noise_runs_per_level=n_noise_runs,
                    enable_feature_dropout=enable_dropout,
                    enable_subset_sampling=enable_subset,
                    max_workers=3,
                    sample_cap=8000,
                )

                prog_bar = st.progress(0)
                status_ph = st.empty()
                reports = {}

                for i, aid in enumerate(sel_test_algos):
                    pct = int(i / len(sel_test_algos) * 100)
                    prog_bar.progress(pct)
                    status_ph.markdown(
                        f'<div class="info-panel pulsing">🧪 Testing: <b>{aid}</b> ({i+1}/{len(sel_test_algos)})</div>',
                        unsafe_allow_html=True)
                    cr = batch_result.results.get(aid)
                    if cr and cr.succeeded and len(cr.labels) > 0:
                        try:
                            from stability import AlgorithmStabilityTester
                            tester = AlgorithmStabilityTester(stab_config)
                            report = tester.test(aid, X, cr.labels, st.session_state.n_clusters)
                            report.algorithm_name = cr.algorithm_name
                            reports[aid] = report
                        except Exception as e:
                            st.warning(f"Stability test failed for {aid}: {e}")

                prog_bar.progress(100)
                status_ph.empty()
                st.session_state.stability_reports = reports
                _success(f"Stability analysis complete for {len(reports)} algorithms.")

        # Display results
        reports = st.session_state.stability_reports
        if reports:
            _subsection("Stability Scorecard")
            from stability import MultiAlgorithmStabilityComparator
            comparator = MultiAlgorithmStabilityComparator()
            scorecard_df = comparator.build_scorecard(reports)
            if vis_engine and not scorecard_df.empty:
                fig_sc = vis_engine.stability_scorecard_table(scorecard_df)
                _safe_plotly(fig_sc)

            _sep()
            _subsection("ARI Distribution — Box Plots")
            from stability import StabilityVisDataBuilder
            vis_data = StabilityVisDataBuilder()
            ari_data = vis_data.ari_box_data(reports)
            if vis_engine:
                fig_box = vis_engine.ari_boxplot(ari_data)
                _safe_plotly(fig_box)

            _sep()
            _subsection("Per-Algorithm Stability Details")
            sel_stab_alg = st.selectbox(
                "Select algorithm for detailed stability",
                list(reports.keys()),
                format_func=lambda x: reports[x].algorithm_name,
            )
            if sel_stab_alg:
                rpt = reports[sel_stab_alg]
                c1,c2,c3,c4 = st.columns(4)
                with c1: _metric_card(f"{rpt.stability_score:.1f}", "Stability Score",
                                       color=_b("get_stability_color")(rpt.stability_grade.value))
                with c2: _metric_card(rpt.stability_grade.value, "Grade",
                                       color=_b("get_stability_color")(rpt.stability_grade.value))
                with c3: _metric_card(f"{rpt.mean_ari:.4f}", "Mean ARI", color="#00e5ff")
                with c4: _metric_card(f"{rpt.std_ari:.4f}", "ARI Std", color="#ff8c00")

                if rpt.cluster_persistence and vis_engine:
                    _subsection("Cluster Persistence")
                    fig_persist = vis_engine.persistence_bar(
                        rpt.cluster_persistence, rpt.algorithm_name)
                    _safe_plotly(fig_persist)

                _subsection("ARI by Perturbation Type")
                type_df = pd.DataFrame([
                    {"Type": k, "ARI": round(v,4)}
                    for k,v in rpt.ari_by_type.items()
                ])
                if not type_df.empty:
                    st.dataframe(type_df, use_container_width=True)

    with tab_ari_matrix:
        col_btn2, _ = st.columns([1,3])
        with col_btn2:
            build_matrix = st.button("🔲 Build ARI Agreement Matrix", use_container_width=True)

        if build_matrix:
            with st.spinner("Computing pairwise ARI matrix..."):
                from stability import LabelStabilityMatrix
                mat_builder = LabelStabilityMatrix()
                algo_names = {aid: batch_result.results[aid].algorithm_name
                               for aid in batch_result.results
                               if batch_result.results[aid].succeeded}
                ari_mat = mat_builder.build_ari_matrix(batch_result.results, algo_names)
                st.session_state.ari_matrix = ari_mat

        if st.session_state.ari_matrix is not None and vis_engine:
            fig_mat = vis_engine.ari_matrix_heatmap(st.session_state.ari_matrix)
            _safe_plotly(fig_mat)
            csv_mat = st.session_state.ari_matrix.to_csv().encode()
            st.download_button("💾 Download ARI Matrix CSV", csv_mat, "ari_matrix.csv", "text/csv")

    with tab_noise:
        _subsection("Noise Robustness Curves")
        noise_sel_algos = st.multiselect(
            "Algorithms for noise profiling",
            [er.algorithm_id for er in eval_results[:10]],
            default=[er.algorithm_id for er in eval_results[:min(5, len(eval_results))]],
            format_func=lambda x: next((er.algorithm_name for er in eval_results if er.algorithm_id==x), x),
        )
        noise_levels_inp = st.text_input("Noise levels", "0.0,0.02,0.05,0.1,0.15,0.2,0.3")
        col_noise_btn, _ = st.columns([1,3])
        with col_noise_btn:
            run_noise = st.button("📉 Profile Noise Response", use_container_width=True)

        if run_noise and noise_sel_algos:
            from stability import NoiseResponseProfiler
            noise_levels_list = [float(x) for x in noise_levels_inp.split(",") if x.strip()]
            profiles = {}
            prog = st.progress(0)
            for i, aid in enumerate(noise_sel_algos):
                cr = batch_result.results.get(aid)
                if cr and cr.succeeded:
                    profiler = NoiseResponseProfiler(
                        noise_levels=noise_levels_list, n_runs=3)
                    try:
                        prof = profiler.profile(aid, X, cr.labels, st.session_state.n_clusters)
                        cr_name = cr.algorithm_name
                        profiles[cr_name] = prof
                    except Exception:
                        pass
                prog.progress(int((i+1)/len(noise_sel_algos)*100))
            st.session_state.noise_profiles = profiles
            _success("Noise profiling complete.")

        if st.session_state.noise_profiles and vis_engine:
            fig_noise = vis_engine.noise_degradation(st.session_state.noise_profiles)
            _safe_plotly(fig_noise)


# ─────────────────────────────────────────────────────────────────
# ▓▓  PAGE 8 · CONSENSUS FORGE
# ─────────────────────────────────────────────────────────────────

elif page == "🤝 Consensus Forge":
    _section("🤝 Consensus Forge")

    if not _has_results():
        _warn("No results. Run clustering first.")
        st.stop()

    X = st.session_state.X_processed
    batch_result = st.session_state.batch_result
    eval_results = st.session_state.eval_results
    eval_dict    = st.session_state.eval_dict
    vis_engine   = _get_vis()

    tab_config, tab_run, tab_compare, tab_diversity = st.tabs([
        "⚙️ Configuration","🤝 Build Consensus","📊 Method Comparison","🌈 Diversity"
    ])

    with tab_config:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            method_options = _b("get_consensus_method_options")()
            method_val = st.selectbox(
                "Consensus Method",
                [m["value"] for m in method_options],
                format_func=lambda x: next(m["label"] for m in method_options if m["value"]==x),
            )
            weight_by_metric = st.checkbox("Weight algorithms by metric score", True)
            diversity_aware  = st.checkbox("Diversity-aware weighting", True)
        with col_c2:
            top_k_algos = st.number_input(
                "Use top-K algorithms (0 = all successful)", 0, len(eval_results), 0)
            consensus_k = st.slider("Consensus k (n_clusters)", 2, 30,
                                     st.session_state.n_clusters)

    with tab_run:
        col_btn1, col_btn2, _ = st.columns([1,1,2])
        with col_btn1:
            run_single = st.button("🤝 Build Consensus (selected method)", type="primary", use_container_width=True)
        with col_btn2:
            run_all_methods = st.button("⚡ Run All 8 Methods", use_container_width=True)

        if run_single or run_all_methods:
            with st.spinner("Building consensus..."):
                try:
                    ConsensusConfig = _b("ConsensusConfig")
                    ConsensusMethod = _b("ConsensusMethod")
                    ConsensusEngine = _b("ConsensusEngine")

                    if run_all_methods:
                        cfg = ConsensusConfig(
                            n_clusters=consensus_k,
                            weight_by_metric=weight_by_metric,
                            diversity_aware=diversity_aware,
                            top_k_algorithms=int(top_k_algos) if top_k_algos > 0 else None,
                        )
                        engine = ConsensusEngine(cfg)
                        all_results = engine.run_all_methods(
                            X, batch_result.results, eval_dict)
                        st.session_state.consensus_all = all_results
                        # pick best by quality
                        best_m = max(all_results.values(), key=lambda r: r.quality_score, default=None)
                        st.session_state.consensus_result = best_m
                    else:
                        cfg = ConsensusConfig(
                            method=ConsensusMethod(method_val),
                            n_clusters=consensus_k,
                            weight_by_metric=weight_by_metric,
                            diversity_aware=diversity_aware,
                            top_k_algorithms=int(top_k_algos) if top_k_algos > 0 else None,
                        )
                        engine = ConsensusEngine(cfg)
                        result = engine.run(X, batch_result.results, eval_dict)
                        st.session_state.consensus_result = result
                except Exception as e:
                    st.error(f"Consensus failed: {e}")
                    st.code(traceback.format_exc())

        cr = st.session_state.consensus_result
        if cr:
            _sep()
            c1,c2,c3,c4 = st.columns(4)
            with c1: _metric_card(str(cr.n_clusters), "Clusters Found", color="#00e5ff")
            with c2: _metric_card(f"{cr.quality_score:.1f}", "Quality Score", color="#00ff88")
            with c3: _metric_card(str(cr.n_algorithms), "Algorithms Used", color="#9b59ff")
            with c4: _metric_card(f"{cr.diversity_score:.3f}", "Ensemble Diversity", color="#ffd700")

            st.markdown(f'<div class="info-panel">{cr.interpretation}</div>', unsafe_allow_html=True)

            # Consensus scatter
            if vis_engine:
                _subsection("Consensus Cluster Projection")
                method_vis = st.selectbox("Embedding", _b("available_embedding_methods")(), key="cons_vis")
                with st.spinner("Projecting consensus labels..."):
                    fig_cons = vis_engine.scatter_2d(X, cr.labels, method=method_vis,
                                                      algorithm_name=f"Consensus ({cr.method.value})")
                _safe_plotly(fig_cons)

            # Co-association heatmap
            if cr.coassoc_matrix is not None and vis_engine:
                _sep()
                _subsection("Co-Association Matrix")
                from consensus import ConsensusVisBuilder
                cvb = ConsensusVisBuilder()
                co_sorted, labels_sorted = cvb.coassoc_heatmap(cr.coassoc_matrix, cr.labels)
                if co_sorted is not None:
                    fig_coassoc = vis_engine.coassoc_heatmap(co_sorted, labels_sorted)
                    _safe_plotly(fig_coassoc)

            # Algorithm weights
            if cr.algorithm_weights and vis_engine:
                _sep()
                _subsection("Algorithm Weights")
                from consensus import ConsensusVisBuilder
                cvb = ConsensusVisBuilder()
                wdf = cvb.weight_barchart_data(cr)
                fig_wts = vis_engine.weight_barchart(wdf)
                _safe_plotly(fig_wts)

            # Download consensus labels
            _sep()
            labels_df = pd.DataFrame({"consensus_label": cr.labels})
            st.download_button("💾 Download Consensus Labels CSV",
                                labels_df.to_csv(index=False).encode(),
                                "consensus_labels.csv", "text/csv")

    with tab_compare:
        if st.session_state.consensus_all:
            from consensus import ConsensusVisBuilder
            cvb = ConsensusVisBuilder()
            comp_df = cvb.method_comparison_table(st.session_state.consensus_all)
            if vis_engine and not comp_df.empty:
                fig_comp = vis_engine.consensus_method_comparison(comp_df)
                _safe_plotly(fig_comp)
        else:
            _info("Run **All 8 Methods** in the Run tab to compare.")

    with tab_diversity:
        if batch_result:
            with st.spinner("Analysing ensemble diversity..."):
                diversity = _b("analyse_ensemble_diversity")(batch_result.results)
            c1,c2,c3 = st.columns(3)
            with c1: _metric_card(f"{diversity.diversity_score:.3f}", "Diversity Score", color="#9b59ff")
            with c2: _metric_card(f"{diversity.mean_pairwise_disagreement:.3f}", "Mean Disagreement", color="#ff4daa")
            with c3: _metric_card(f"{diversity.cluster_count_variance:.1f}", "k Variance", color="#ffd700")
            _info("High diversity → consensus adds real value. Low diversity → algorithms agree; consensus confirms.")

            if st.session_state.stability_reports and vis_engine:
                from stability import StabilityVisDataBuilder
                vis_data = StabilityVisDataBuilder()
                radar_data = vis_data.stability_radar_data(st.session_state.stability_reports)
                if radar_data:
                    fig_rad = vis_engine.diversity_radar(radar_data)
                    _safe_plotly(fig_rad)


# ─────────────────────────────────────────────────────────────────
# ▓▓  PAGE 9 · AI ORACLE
# ─────────────────────────────────────────────────────────────────

elif page == " AI Oracle":
    _section(" AI Oracle — Gemini Intelligence")

    tab_insights, tab_chat, tab_recommend, tab_explain = st.tabs([
        "💡 Auto Insights", "💬 Free Chat", "🎯 Recommendations", "📐 Deep Explain"
    ])

    with tab_insights:
        _subsection("Context-Aware Automated Insights")
        insight_topics = []
        if st.session_state.data_profile:
            insight_topics.append("📊 Data Profile Analysis")
        if _has_results():
            insight_topics.append("🏆 Results Summary & Best Algorithm")
            insight_topics.append("⚠️ Algorithm Failure Analysis")
        if st.session_state.stability_reports:
            insight_topics.append("🧪 Stability Insights")
        if st.session_state.consensus_result:
            insight_topics.append("🤝 Consensus Quality Analysis")

        if not insight_topics:
            _info("Load data and run clustering to generate AI insights.")
        else:
            sel_topic = st.selectbox("Generate insight for:", insight_topics)
            if st.button("✨ Generate AI Insight", type="primary", use_container_width=True):
                context = ""
                prompt  = ""

                if "Data Profile" in sel_topic and st.session_state.data_profile:
                    p = st.session_state.data_profile
                    context = (f"Dataset: {p.n_rows} rows, {p.n_cols} columns, "
                                f"{p.n_numeric} numeric, {p.total_missing_pct:.1f}% missing, "
                                f"{p.duplicate_pct:.1f}% duplicates, "
                                f"{len(p.high_corr_pairs)} high-correlation pairs.")
                    prompt = ("Analyse this dataset profile for clustering suitability. "
                              "Comment on missing values, correlations, dimensionality, "
                              "and suggest the most appropriate preprocessing steps and "
                              "3 specific clustering algorithm families with rationale.")

                elif "Results" in sel_topic and _has_results():
                    ev = st.session_state.eval_results[:5]
                    rows_str = "\n".join([f"  {r.rank}. {r.algorithm_name}: score={r.composite_score:.1f}, "
                                          f"sil={r.metric_value('silhouette') or 'N/A':.4f}, k={r.n_clusters}"
                                          for r in ev])
                    context = f"Top 5 clustering results:\n{rows_str}"
                    prompt  = ("Interpret these clustering results. Which algorithm is best and why? "
                                "What do the silhouette scores tell us? "
                                "Are there signs of overfitting to k? Suggest next steps.")

                elif "Failure" in sel_topic and st.session_state.batch_result:
                    failed = st.session_state.batch_result.failed()
                    fail_str = "\n".join([f"  {aid}: {cr.error_message[:80]}"
                                           for aid, cr in list(failed.items())[:8]])
                    context = f"Failed algorithms:\n{fail_str}"
                    prompt  = "Diagnose these clustering failures. What are the likely causes and how to fix them?"

                elif "Stability" in sel_topic and st.session_state.stability_reports:
                    rpts = st.session_state.stability_reports
                    stab_str = "\n".join([f"  {r.algorithm_name}: score={r.stability_score:.1f}, "
                                           f"grade={r.stability_grade.value}, ARI={r.mean_ari:.4f}"
                                           for r in list(rpts.values())[:5]])
                    context = f"Stability results:\n{stab_str}"
                    prompt  = ("Interpret these stability results. Which algorithms are robust? "
                                "What explains the instability of the lower-scoring ones? "
                                "Recommend the most reliable algorithm for production use.")

                elif "Consensus" in sel_topic and st.session_state.consensus_result:
                    cr = st.session_state.consensus_result
                    context = (f"Consensus: method={cr.method.value}, k={cr.n_clusters}, "
                                f"quality={cr.quality_score:.1f}, diversity={cr.diversity_score:.3f}, "
                                f"n_algorithms={cr.n_algorithms}")
                    prompt  = ("Evaluate this consensus clustering result. "
                                "Is the ensemble diversity sufficient? "
                                "What does the quality score indicate? "
                                "How should the analyst interpret and validate the final clusters?")

                if prompt:
                    with st.spinner(" Querying Gemini..."):
                        response = _gemini_query(prompt, context)
                    _render_ai_response(response)
                    st.session_state.gemini_history.append(
                        {"topic": sel_topic, "prompt": prompt, "response": response,
                         "ts": datetime.now().strftime("%H:%M:%S")})

    with tab_chat:
        _subsection("Free-Form Clustering Q&A")
        if st.session_state.gemini_history:
            st.markdown('<div style="max-height:320px; overflow-y:auto;">', unsafe_allow_html=True)
            for entry in st.session_state.gemini_history[-5:]:
                st.markdown(f'<div style="background:#0d0d1e; border-radius:8px; padding:.7rem 1rem; margin-bottom:.4rem;">'
                             f'<div style="color:#9b59ff; font-size:.75rem; margin-bottom:.3rem;">You ({entry["ts"]})</div>'
                             f'<div style="color:#ccccee; font-size:.88rem;">{entry["prompt"][:200]}</div></div>',
                             unsafe_allow_html=True)
                _render_ai_response(entry["response"][:600] + ("..." if len(entry["response"])>600 else ""))
            st.markdown('</div>', unsafe_allow_html=True)
            _sep()

        user_q = st.text_area(
            "Ask anything about clustering, your data, or results:",
            placeholder="e.g. What clustering algorithm is best for high-dimensional genomics data with noise?",
            height=100,
        )

        build_ctx = st.checkbox("Include current session context", True)
        col_send, _ = st.columns([1,3])
        with col_send:
            send_q = st.button("📨 Send", type="primary", use_container_width=True)

        if send_q and user_q.strip():
            ctx = ""
            if build_ctx:
                ctx_parts = []
                if st.session_state.df_raw is not None:
                    df = st.session_state.df_raw
                    ctx_parts.append(f"Dataset: {len(df)} rows, {len(df.columns)} cols")
                if _has_data():
                    ctx_parts.append(f"Processed features: {st.session_state.X_processed.shape[1]}")
                if _has_results():
                    best = _b("get_best_algorithm")(st.session_state.eval_results)
                    if best:
                        ctx_parts.append(f"Best algorithm: {best.algorithm_name} (score={best.composite_score:.1f})")
                ctx = ". ".join(ctx_parts)
            with st.spinner(" Thinking..."):
                response = _gemini_query(user_q, ctx)
            _render_ai_response(response)
            st.session_state.gemini_history.append(
                {"topic":"chat","prompt":user_q,"response":response,
                 "ts": datetime.now().strftime("%H:%M:%S")})

        if st.button("🗑️ Clear History", use_container_width=False):
            st.session_state.gemini_history = []
            st.rerun()

    with tab_recommend:
        _subsection("Algorithm Recommendation Engine")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            rec_n_samples = st.number_input("Dataset size (rows)", 100, 5_000_000,
                                              len(st.session_state.df_raw) if st.session_state.df_raw is not None else 1000)
            rec_n_features = st.number_input("Number of features", 1, 10000,
                                               st.session_state.X_processed.shape[1] if _has_data() else 10)
            rec_has_noise  = st.checkbox("Data likely contains noise/outliers", True)
        with col_r2:
            rec_k_known = st.checkbox("Number of clusters (k) is known", False)
            rec_need_soft = st.checkbox("Need soft/probabilistic assignments", False)
            rec_interpretable = st.checkbox("Prioritise interpretability", True)
            rec_speed = st.select_slider("Speed priority", ["Accuracy","Balanced","Speed"], "Balanced")

        if st.button("🎯 Get AI Recommendations", type="primary", use_container_width=True):
            prompt = (
                f"Recommend the best clustering algorithms for this scenario:\n"
                f"- Dataset: {rec_n_samples:,} rows, {rec_n_features} features\n"
                f"- Contains noise: {rec_has_noise}\n"
                f"- k known: {rec_k_known}\n"
                f"- Need soft assignments: {rec_need_soft}\n"
                f"- Interpretability priority: {rec_interpretable}\n"
                f"- Speed priority: {rec_speed}\n\n"
                f"Give top 5 specific algorithms with: (1) algorithm name, "
                f"(2) why it's suitable, (3) key parameters to tune, (4) expected limitations."
            )
            with st.spinner(" Generating recommendations..."):
                response = _gemini_query(prompt)
            _render_ai_response(response)

    with tab_explain:
        _subsection("Deep Explain Any Concept")
        explain_topics = [
            "What is the Silhouette Score and how to interpret it?",
            "When does DBSCAN outperform K-Means?",
            "How does the co-association matrix work in consensus clustering?",
            "What is the elbow method and its limitations?",
            "Explain Gaussian Mixture Models vs K-Means",
            "How to choose between Adjusted Rand Index and NMI?",
            "What is UMAP and why is it better than t-SNE for clustering?",
            "How does Affinity Propagation work and when to use it?",
            "Custom question (type below)",
        ]
        sel_explain = st.selectbox("Select topic", explain_topics)
        custom_q = ""
        if sel_explain == "Custom question (type below)":
            custom_q = st.text_input("Enter your question")

        if st.button("📐 Deep Explain", type="primary", use_container_width=True):
            q = custom_q if custom_q else sel_explain
            prompt = (f"Give a deep, rigorous explanation of: '{q}'. "
                       f"Include mathematical intuition where appropriate, "
                       f"practical implications, and concrete examples.")
            with st.spinner(" Generating explanation..."):
                resp = _gemini_query(prompt)
            _render_ai_response(resp)


# ─────────────────────────────────────────────────────────────────
# ▓▓  PAGE 10 · ADVANCED TOOLS
# ─────────────────────────────────────────────────────────────────

elif page == "🛠️ Advanced Tools":
    _section("🛠️ Advanced Tools")

    tab_ksweep, tab_kest, tab_feat, tab_export, tab_diagnostics = st.tabs([
        "📈 k-Sweep", "🎯 k-Estimator", "🔍 Feature Analysis", "💾 Export Studio", "🔧 Diagnostics"
    ])

    # ── k-Sweep ───────────────────────────────────────────────────
    with tab_ksweep:
        _subsection("k-Sweep Elbow Analysis")
        if not _has_data():
            _info("Preprocess data first.")
        else:
            X = st.session_state.X_processed
            vis_engine = _get_vis()
            registry = _b("get_registry")()

            c1, c2 = st.columns(2)
            with c1:
                sweep_alg = st.selectbox(
                    "Algorithm for k-sweep",
                    [aid for aid in registry.ids() if registry.get(aid).requires_n_clusters],
                    format_func=lambda x: registry.get(x).name,
                )
            with c2:
                k_min = st.number_input("k min", 2, 20, 2)
                k_max = st.number_input("k max", 3, 50, 15)

            if st.button("📈 Run k-Sweep", type="primary", use_container_width=True):
                with st.spinner(f"Running k-sweep for {sweep_alg}..."):
                    try:
                        from evaluation import KSweepAnalyser
                        analyser = KSweepAnalyser()
                        result = analyser.run_sweep(sweep_alg, X, range(k_min, k_max+1))
                        st.session_state.k_sweep_data[sweep_alg] = result
                        if result["optimal_k"]:
                            _success(f"Optimal k = **{result['optimal_k']}** (by silhouette)")
                    except Exception as e:
                        st.error(f"k-sweep failed: {e}")

            if sweep_alg in st.session_state.k_sweep_data:
                rd = st.session_state.k_sweep_data[sweep_alg]
                mk = rd["metrics_by_k"]
                if mk["k"] and vis_engine:
                    fig_elbow = vis_engine.k_sweep_elbow(
                        mk["k"], mk["silhouette"],
                        optimal_k=rd.get("optimal_k"),
                        algorithm_name=registry.get(sweep_alg).name,
                    )
                    _safe_plotly(fig_elbow)

                    # Multi-metric sweep
                    fig_multi_k = make_subplots(rows=1, cols=3,
                        subplot_titles=["Silhouette","Davies-Bouldin","Calinski-Harabász"])
                    for i, (metric, col_idx) in enumerate(
                        [("silhouette",1),("davies_bouldin",2),("calinski_harabasz",3)]):
                        vals = mk.get(metric,[])
                        if vals:
                            color = ["#00e5ff","#ff4daa","#ffd700"][i]
                            fig_multi_k.add_trace(
                                go.Scatter(x=mk["k"], y=vals, mode="lines+markers",
                                           marker=dict(color=color,size=6),
                                           line=dict(color=color,width=2),
                                           showlegend=False),
                                row=1, col=col_idx)
                    fig_multi_k.update_layout(
                        paper_bgcolor="#07070f", plot_bgcolor="#0d0d1e",
                        font=dict(color="#e0e0f0"), height=350,
                        margin=dict(l=40,r=40,t=50,b=40))
                    st.plotly_chart(fig_multi_k, use_container_width=True)

    # ── k-Estimator ───────────────────────────────────────────────
    with tab_kest:
        _subsection("Optimal k Estimator via Consensus")
        if not st.session_state.consensus_result or \
           st.session_state.consensus_result.coassoc_matrix is None:
            _info("Build consensus with a valid co-association matrix first (Consensus Forge).")
        else:
            coassoc = st.session_state.consensus_result.coassoc_matrix
            k_min_e = st.number_input("k min", 2, 10, 2, key="ke_min")
            k_max_e = st.number_input("k max", 3, 30, 12, key="ke_max")
            if st.button("🎯 Estimate Optimal k", type="primary", use_container_width=True):
                from consensus import ConsensusKEstimator
                est = ConsensusKEstimator()
                result = est.estimate(coassoc, k_range=range(k_min_e, k_max_e+1))
                _success(f"Consensus-estimated optimal k = **{result['optimal_k']}**")
                if result.get("k_values"):
                    vis_engine = _get_vis()
                    if vis_engine:
                        fig_ke = vis_engine.k_sweep_elbow(
                            result["k_values"], result["sil_values"],
                            optimal_k=result["optimal_k"],
                            algorithm_name="Consensus k-Estimator",
                        )
                        _safe_plotly(fig_ke)

    # ── Feature Analysis ─────────────────────────────────────────
    with tab_feat:
        _subsection("Feature Clustering Analysis")
        if not _has_results():
            _info("Run clustering first.")
        else:
            X = st.session_state.X_processed
            eval_results = st.session_state.eval_results
            feature_names = st.session_state.feature_names
            vis_engine = _get_vis()
            batch = st.session_state.batch_result

            sel_feat_alg = st.selectbox(
                "Algorithm",
                [er.algorithm_id for er in eval_results],
                format_func=lambda x: next((er.algorithm_name for er in eval_results if er.algorithm_id==x), x),
            )
            labels = batch.results[sel_feat_alg].labels if sel_feat_alg in batch.results else None

            if labels is not None and vis_engine:
                c1, c2 = st.columns(2)
                with c1:
                    _subsection("Feature Importance")
                    top_n = st.slider("Top N features", 5, min(40, len(feature_names)), 15, key="fn_top")
                    fig_fi = vis_engine.feature_importance(X, labels, feature_names, top_n=top_n)
                    _safe_plotly(fig_fi)

                with c2:
                    _subsection("Centroid Heatmap")
                    fig_ch = vis_engine.centroid_heatmap(X, labels, feature_names, sel_feat_alg)
                    _safe_plotly(fig_ch)

                _sep()
                _subsection("Compute Cluster Statistics")
                from clustering_runner import compute_cluster_statistics
                stats = compute_cluster_statistics(labels)
                c1,c2,c3,c4 = st.columns(4)
                with c1: _metric_card(str(stats["n_clusters"]), "k Found")
                with c2: _metric_card(str(stats["min_size"]), "Min Cluster Size", color="#ff8c00")
                with c3: _metric_card(str(stats["max_size"]), "Max Cluster Size", color="#00ff88")
                with c4: _metric_card(f"{stats['balance_ratio']:.3f}", "Balance Ratio", color="#9b59ff")

    # ── Export Studio ─────────────────────────────────────────────
    with tab_export:
        _section("💾 Export Studio")
        if not _has_results():
            _info("Run clustering to generate exportable results.")
        else:
            batch  = st.session_state.batch_result
            eval_r = st.session_state.eval_results
            X      = st.session_state.X_processed

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**📊 Rankings Table**")
                df_exp = _b("build_results_dataframe")(eval_r)
                st.download_button("💾 CSV",
                                    df_exp.to_csv(index=False).encode(),
                                    "clusterx_rankings.csv", "text/csv",
                                    use_container_width=True)

                st.markdown("**🏷️ All Labels (wide)**")
                lbl_dict = {}
                for aid, cr in batch.results.items():
                    if cr.succeeded and len(cr.labels) == len(X):
                        lbl_dict[aid[:30]] = cr.labels
                if lbl_dict:
                    lbl_df = pd.DataFrame(lbl_dict)
                    st.download_button("💾 CSV",
                                        lbl_df.to_csv(index=False).encode(),
                                        "all_labels.csv", "text/csv",
                                        use_container_width=True)

            with c2:
                st.markdown("**🤝 Consensus Labels**")
                if st.session_state.consensus_result:
                    cr = st.session_state.consensus_result
                    cdf = pd.DataFrame({"consensus_label": cr.labels})
                    st.download_button("💾 CSV",
                                        cdf.to_csv(index=False).encode(),
                                        "consensus_labels.csv", "text/csv",
                                        use_container_width=True)

                st.markdown("**📐 Full Metrics JSON**")
                metrics_export = [r.to_dict() for r in eval_r]
                st.download_button("💾 JSON",
                                    json.dumps(metrics_export, indent=2).encode(),
                                    "metrics.json", "application/json",
                                    use_container_width=True)

            with c3:
                st.markdown("**🧪 Stability Reports JSON**")
                if st.session_state.stability_reports:
                    stab_export = {
                        k: v.to_dict() for k,v in st.session_state.stability_reports.items()
                    }
                    st.download_button("💾 JSON",
                                        json.dumps(stab_export, indent=2).encode(),
                                        "stability.json", "application/json",
                                        use_container_width=True)

                st.markdown("**🔲 ARI Matrix**")
                if st.session_state.ari_matrix is not None:
                    st.download_button("💾 CSV",
                                        st.session_state.ari_matrix.to_csv().encode(),
                                        "ari_matrix.csv", "text/csv",
                                        use_container_width=True)

            _sep()
            _subsection("📦 Processed Feature Matrix")
            st.markdown(f"Shape: **{X.shape[0]:,} × {X.shape[1]}**")
            X_df = pd.DataFrame(X, columns=st.session_state.feature_names or
                                 [f"f{i}" for i in range(X.shape[1])])
            c_dl, _ = st.columns([1,3])
            with c_dl:
                st.download_button("💾 Download Processed Data (CSV)",
                                    X_df.to_csv(index=False).encode(),
                                    "processed_features.csv", "text/csv",
                                    use_container_width=True)

    # ── Diagnostics ───────────────────────────────────────────────
    with tab_diagnostics:
        _section("🔧 System Diagnostics")

        c1, c2 = st.columns(2)
        with c1:
            _subsection("Backend Status")
            if _ok():
                _success("All backends loaded successfully.")
                reg = _b("get_registry")()
                rs  = _b("summarize_registry")()
                diag_rows = [
                    {"Module": "preprocessing", "Status": "✅ OK"},
                    {"Module": "clustering_registry", "Status": f"✅ OK ({rs['total_algorithms']} algos)"},
                    {"Module": "clustering_runner", "Status": "✅ OK"},
                    {"Module": "evaluation", "Status": "✅ OK"},
                    {"Module": "stability", "Status": "✅ OK"},
                    {"Module": "consensus", "Status": "✅ OK"},
                    {"Module": "visualization", "Status": "✅ OK"},
                ]
                st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)
            else:
                _warn("Backend load failed.")
                st.code(B.get("tb",""), language="python")

        with c2:
            _subsection("Session State")
            state_rows = [
                {"Key": "Data loaded", "Value": "Yes" if st.session_state.df_raw is not None else "No"},
                {"Key": "Preprocessed", "Value": "Yes" if _has_data() else "No"},
                {"Key": "Algorithms selected", "Value": str(len(st.session_state.selected_algorithms))},
                {"Key": "Run completed", "Value": "Yes" if st.session_state.batch_result else "No"},
                {"Key": "Evaluated", "Value": str(len(st.session_state.eval_results))},
                {"Key": "Stability tested", "Value": str(len(st.session_state.stability_reports))},
                {"Key": "Consensus built", "Value": "Yes" if st.session_state.consensus_result else "No"},
                {"Key": "AI queries", "Value": str(len(st.session_state.gemini_history))},
            ]
            st.dataframe(pd.DataFrame(state_rows), use_container_width=True, hide_index=True)

        _sep()
        _subsection("Optional Dependencies")
        dep_rows = []
        for pkg, purpose in [
            ("umap", "UMAP embeddings"),
            ("hdbscan","HDBSCAN algorithm"),
            ("sklearn_extra","K-Medoids"),
            ("tensorflow","Autoencoder KMeans"),
            ("google.generativeai","Gemini AI Oracle"),
        ]:
            try:
                __import__(pkg)
                dep_rows.append({"Package": pkg, "Purpose": purpose, "Status": "✅ Available"})
            except ImportError:
                dep_rows.append({"Package": pkg, "Purpose": purpose, "Status": "⚠️ Not installed"})
        st.dataframe(pd.DataFrame(dep_rows), use_container_width=True, hide_index=True)

        _sep()
        if st.button("🗑️ Reset Full Session", use_container_width=False):
            for k in _DEFAULTS:
                st.session_state[k] = _DEFAULTS[k]
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding:2.5rem 0 1rem 0; border-top:1px solid #1a1a2e; margin-top:3rem;">
    <div style="font-family:'Space Grotesk'; font-size:.95rem; font-weight:600;
         background:linear-gradient(90deg,#00e5ff,#9b59ff,#ff4daa);
         -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
        UnSuPERvIsED · ClusterX Universal Clustering Intelligence Lab
    </div>
    <div style="font-size:.72rem; color:#333355; margin-top:.5rem; letter-spacing:.06em;">
        Built with ❤️ by ClusterX Intelligence Lab · Powered by Anthropic Claude &amp; Google Gemini
    </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# FINAL POLISH — ADVANCED FEATURE PANELS
# Injected at module level; called from Advanced Tools page
# ══════════════════════════════════════════════════════════════════

def _render_clusterability_panel():
    """Full clusterability analysis panel — Hopkins, Gap, DBCV, Confidence."""
    if not _has_data():
        _info("Preprocess data first.")
        return

    X = st.session_state.X_processed
    vis_engine = _get_vis()

    _section("🧭 Clusterability Analysis")
    _info(
        "Run these tests **before** choosing algorithms. They tell you whether "
        "clustering is meaningful on your data and what the optimal k is."
    )

    col_h, col_g = st.columns(2)

    # ── Hopkins Statistic ─────────────────────────────────────────
    with col_h:
        st.markdown("#### 🔬 Hopkins Statistic")
        st.caption("Measures spatial randomness. H > 0.6 → data is clusterable.")
        n_hop = st.slider("Sample size for Hopkins", 50, 300, 150, key="hop_n")
        if st.button("▶ Compute Hopkins", use_container_width=True, key="run_hop"):
            with st.spinner("Computing Hopkins statistic..."):
                try:
                    from evaluation import HopkinsStatistic
                    result = HopkinsStatistic(n_samples=n_hop).compute(X)
                    st.session_state["_hop_result"] = result
                except Exception as e:
                    st.error(str(e))

        if "_hop_result" in st.session_state and st.session_state["_hop_result"]:
            res = st.session_state["_hop_result"]
            H   = res.get("hopkins")
            if H is not None and vis_engine and hasattr(vis_engine, "clusterability_gauge"):
                fig_gauge = vis_engine.clusterability_gauge(H, res.get("interpretation",""))
                st.plotly_chart(fig_gauge, use_container_width=True,
                                config={"displayModeBar": False})
            st.markdown(f"""
            <div class="{'success-panel' if res.get('is_clusterable') else 'warn-panel'}">
                <b>H = {H:.4f if H else 'N/A'}</b><br>
                {res.get('interpretation','')}<br>
                <span style="font-size:.8rem; color:#aaaacc;">{res.get('recommendation','')}</span>
            </div>""", unsafe_allow_html=True)

    # ── Gap Statistic ─────────────────────────────────────────────
    with col_g:
        st.markdown("#### 📈 Gap Statistic")
        st.caption("Tibshirani et al. (2001) — gold standard for optimal k.")
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1: k_min_gap = st.number_input("k min", 1, 10, 1, key="gap_kmin")
        with col_g2: k_max_gap = st.number_input("k max", 2, 20, 10, key="gap_kmax")
        with col_g3: n_refs    = st.number_input("Refs", 3, 20, 8, key="gap_refs")

        if st.button("▶ Compute Gap Stat", use_container_width=True, key="run_gap"):
            with st.spinner("Running Gap Statistic (this may take 30–60s)..."):
                try:
                    from evaluation import run_gap_statistic
                    gap_res = run_gap_statistic(
                        X, k_range=range(int(k_min_gap), int(k_max_gap)+1),
                        n_refs=int(n_refs))
                    st.session_state["_gap_result"] = gap_res
                    _success(f"Optimal k = **{gap_res['optimal_k']}**")
                except Exception as e:
                    st.error(str(e))

        if "_gap_result" in st.session_state and vis_engine:
            if hasattr(vis_engine, "gap_statistic_plot"):
                fig_gap = vis_engine.gap_statistic_plot(st.session_state["_gap_result"])
                st.plotly_chart(fig_gap, use_container_width=True,
                                config={"displayModeBar": True})

    _sep()

    # ── DBCV ─────────────────────────────────────────────────────
    if _has_results():
        st.markdown("#### 🌊 Density-Based Cluster Validity (DBCV)")
        st.caption("Moulavi et al. (2014) — proper validity for DBSCAN/HDBSCAN/density clusters.")

        eval_results = st.session_state.eval_results
        batch = st.session_state.batch_result
        dbcv_alg = st.selectbox(
            "Algorithm for DBCV",
            [er.algorithm_id for er in eval_results if batch.results.get(er.algorithm_id, None) and
             batch.results[er.algorithm_id].succeeded],
            format_func=lambda x: next((er.algorithm_name for er in eval_results if er.algorithm_id==x), x),
            key="dbcv_alg",
        )
        if st.button("▶ Compute DBCV", use_container_width=True, key="run_dbcv"):
            with st.spinner("Computing DBCV..."):
                try:
                    from evaluation import compute_dbcv
                    labs = batch.results[dbcv_alg].labels
                    dbcv_res = compute_dbcv(X, labs)
                    st.session_state["_dbcv_result"] = (dbcv_alg, dbcv_res)
                except Exception as e:
                    st.error(str(e))

        if "_dbcv_result" in st.session_state:
            _aid, dbcv_res = st.session_state["_dbcv_result"]
            if dbcv_res.get("dbcv") is not None:
                c1, c2 = st.columns(2)
                with c1: _metric_card(f"{dbcv_res['dbcv']:.4f}", "DBCV Score", color="#9b59ff")
                with c2: st.markdown(f"""
                    <div class="info-panel" style="margin-top:.8rem;">
                        {dbcv_res.get('interpretation','')}
                    </div>""", unsafe_allow_html=True)
                if dbcv_res.get("per_cluster"):
                    pc_df = pd.DataFrame([
                        {"Cluster": f"C{k}", "DBCV": v}
                        for k, v in dbcv_res["per_cluster"].items()
                    ])
                    st.dataframe(pc_df, use_container_width=True, hide_index=True)
            else:
                _warn(f"DBCV error: {dbcv_res.get('error','unknown')}")

    _sep()

    # ── Confidence Map ────────────────────────────────────────────
    if _has_results():
        st.markdown("#### 🎯 Assignment Confidence Map")
        st.caption(
            "Per-point confidence: how strongly does each point belong to its cluster? "
            "Based on silhouette + k-NN label consistency."
        )
        eval_results = st.session_state.eval_results
        batch = st.session_state.batch_result
        conf_alg = st.selectbox(
            "Algorithm",
            [er.algorithm_id for er in eval_results],
            format_func=lambda x: next((er.algorithm_name for er in eval_results if er.algorithm_id==x), x),
            key="conf_alg",
        )
        conf_method = st.selectbox("Embedding", _b("available_embedding_methods")(), key="conf_method")

        if st.button("▶ Compute Confidence Map", use_container_width=True, key="run_conf"):
            with st.spinner("Scoring point confidence..."):
                try:
                    from stability import ClusterConfidenceScorer
                    labs = batch.results[conf_alg].labels
                    scorer = ClusterConfidenceScorer(n_neighbors=15)
                    conf_scores = scorer.score(X, labs)
                    st.session_state["_conf_scores"] = (conf_alg, labs, conf_scores)

                    n_unc = int((conf_scores < 0.4).sum())
                    n_tot = len(conf_scores)
                    _success(f"Confidence computed. {n_unc}/{n_tot} ({n_unc/n_tot*100:.1f}%) "
                              f"points are uncertain (confidence < 0.4).")
                except Exception as e:
                    st.error(str(e))

        if "_conf_scores" in st.session_state:
            _caid, _clabs, _cscores = st.session_state["_conf_scores"]
            if vis_engine and hasattr(vis_engine, "confidence_heatmap"):
                with st.spinner(f"Projecting via {conf_method}..."):
                    fig_conf = vis_engine.confidence_heatmap(
                        X, _clabs, _cscores, method=conf_method,
                        algorithm_name=_caid)
                _safe_plotly(fig_conf)

            c1,c2,c3 = st.columns(3)
            valid = _cscores[_clabs != -1]
            with c1: _metric_card(f"{float(valid.mean()):.3f}", "Mean Confidence", color="#00e5ff")
            with c2: _metric_card(f"{int((_cscores<0.4).sum())}", "Uncertain Points", color="#ff8c00")
            with c3: _metric_card(f"{float((_cscores>=0.7).mean()*100):.1f}%", "High-Confidence %", color="#00ff88")


def _render_density_network_panel():
    """Density contour + cluster network graph."""
    if not _has_results():
        _info("Run clustering first.")
        return
    X = st.session_state.X_processed
    batch = st.session_state.batch_result
    eval_results = st.session_state.eval_results
    vis_engine = _get_vis()
    if not vis_engine:
        return

    _section("🔭 Advanced Geometry Views")

    alg_choices = [er.algorithm_id for er in eval_results
                   if batch.results.get(er.algorithm_id) and batch.results[er.algorithm_id].succeeded]
    if not alg_choices:
        _info("No successful results to visualise.")
        return

    sel_alg = st.selectbox("Algorithm", alg_choices,
                            format_func=lambda x: next((er.algorithm_name for er in eval_results if er.algorithm_id==x), x),
                            key="geom_alg")
    labels = batch.results[sel_alg].labels

    col_a, col_b = st.columns(2)

    with col_a:
        _subsection("Density Contour Overlay")
        dc_method = st.selectbox("Embedding", _b("available_embedding_methods")(), key="dc_method")
        if st.button("▶ Draw Density Contour", use_container_width=True, key="run_dc"):
            with st.spinner("Computing KDE contours..."):
                if hasattr(vis_engine, "density_contour"):
                    fig_dc = vis_engine.density_contour(X, labels, method=dc_method,
                                                         algorithm_name=sel_alg)
                    st.session_state["_dc_fig"] = fig_dc
        if "_dc_fig" in st.session_state:
            _safe_plotly(st.session_state["_dc_fig"])

    with col_b:
        _subsection("Cluster Network Graph")
        if st.button("▶ Build Network", use_container_width=True, key="run_net"):
            with st.spinner("Building cluster network..."):
                if hasattr(vis_engine, "cluster_network"):
                    fig_net = vis_engine.cluster_network(X, labels, algorithm_name=sel_alg)
                    st.session_state["_net_fig"] = fig_net
        if "_net_fig" in st.session_state:
            _safe_plotly(st.session_state["_net_fig"])

    _sep()
    _subsection("Cluster Separability Matrix")
    if st.button("▶ Compute Separability Matrix", use_container_width=True, key="run_sep"):
        with st.spinner("Computing Mahalanobis separability..."):
            try:
                from evaluation import ClusterSeparabilityMatrix
                sep = ClusterSeparabilityMatrix()
                result = sep.compute(X, labels)
                st.session_state["_sep_result"] = result
                if result["worst_pair"]:
                    _warn(f"Closest cluster pair: C{result['worst_pair'][0]} & "
                           f"C{result['worst_pair'][1]} "
                           f"(Mahalanobis dist = {result['min_separation']:.3f})")
            except Exception as e:
                st.error(str(e))

    if "_sep_result" in st.session_state:
        res = st.session_state["_sep_result"]
        if res.get("matrix") is not None and vis_engine and hasattr(vis_engine, "separability_matrix"):
            fig_sep = vis_engine.separability_matrix(res["matrix"], sel_alg)
            _safe_plotly(fig_sep)


# ──────────────────────────────────────────────────────────────────
# PATCH ADVANCED TOOLS PAGE TO INCLUDE NEW PANELS
# ──────────────────────────────────────────────────────────────────

# NOTE: The Advanced Tools page above uses tab_ksweep, tab_kest, tab_feat,
# tab_export, tab_diagnostics. We extend it at runtime by checking session page.

if page == "🛠️ Advanced Tools":
    # The main page block already rendered; add the extra sub-sections.
    _sep()

    extra_tab_a, extra_tab_b = st.tabs(["🧭 Clusterability Suite", "🔭 Advanced Geometry"])
    with extra_tab_a:
        _render_clusterability_panel()
    with extra_tab_b:
        _render_density_network_panel()


# ──────────────────────────────────────────────────────────────────
# LIVE METRICS TICKER (sidebar injection when results exist)
# ──────────────────────────────────────────────────────────────────

if _has_results() and page not in ["🏠 Home", "📁 Data Ingestion"]:
    with st.sidebar:
        _sep_html = '<div style="height:1px;background:linear-gradient(90deg,transparent,#2a2a5a,transparent);margin:.8rem 0;"></div>'
        st.markdown(_sep_html, unsafe_allow_html=True)
        best = _b("get_best_algorithm")(st.session_state.eval_results)
        if best:
            sil = best.metric_value("silhouette")
            db  = best.metric_value("davies_bouldin")
            st.markdown(f"""
            <div style="font-size:.72rem; color:#555577; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.5rem;">Best Result</div>
            <div style="font-size:.82rem; color:#00e5ff; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                🏆 {best.algorithm_name[:22]}
            </div>
            <div style="font-size:.75rem; color:#666688; margin-top:.3rem; line-height:1.8;">
                Score: <span style="color:#00ff88; font-weight:600;">{best.composite_score:.1f}</span>/100<br>
                Sil: <span style="color:#9b59ff;">{f"{sil:.4f}" if sil is not None else 'N/A'}</span><br>
                DB: <span style="color:#ff8c00;">{f"{db:.4f}" if db is not None else 'N/A'}</span><br>
                k: <span style="color:#ffd700;">{best.n_clusters}</span>
            </div>""", unsafe_allow_html=True)

        if st.session_state.stability_reports:
            st.markdown(_sep_html, unsafe_allow_html=True)
            best_stab = max(st.session_state.stability_reports.values(),
                            key=lambda r: r.stability_score, default=None)
            if best_stab:
                gc = _b("get_stability_color")
                st.markdown(f"""
                <div style="font-size:.72rem; color:#555577; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.5rem;">Most Stable</div>
                <div style="font-size:.82rem; color:#ffd700; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                    🧪 {best_stab.algorithm_name[:22]}
                </div>
                <div style="font-size:.75rem; color:#666688; margin-top:.3rem; line-height:1.8;">
                    Score: <span style="color:{gc(best_stab.stability_grade.value) if gc else '#00ff88'}; font-weight:600;">{best_stab.stability_score:.1f}</span>/100<br>
                    Grade: {best_stab.stability_grade.value}<br>
                    ARI: <span style="color:#00e5ff;">{best_stab.mean_ari:.4f}</span>
                </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# KEYBOARD SHORTCUT HINT (footer micro-bar)
# ──────────────────────────────────────────────────────────────────

if page == "🏠 Home":
    _sep()
    st.markdown("""
    <div style="background:#0a0a18; border:1px solid #1a1a2e; border-radius:8px;
         padding:.7rem 1.4rem; margin-top:1rem;">
        <div style="font-size:.72rem; color:#444466; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.5rem;">Quick Reference</div>
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:.5rem; font-size:.78rem;">
            <div><span style="color:#00e5ff; font-weight:600;">📁→⚙️→🧬→⚡</span> <span style="color:#555577;">Standard pipeline</span></div>
            <div><span style="color:#9b59ff; font-weight:600;">Hopkins H > 0.6</span> <span style="color:#555577;">Data is clusterable</span></div>
            <div><span style="color:#ff4daa; font-weight:600;">Silhouette > 0.5</span> <span style="color:#555577;">Reasonable structure</span></div>
            <div><span style="color:#ffd700; font-weight:600;">DB Index < 1.0</span> <span style="color:#555577;">Good separation</span></div>
            <div><span style="color:#00ff88; font-weight:600;">ARI > 0.7</span> <span style="color:#555577;">Highly stable</span></div>
            <div><span style="color:#ff8c00; font-weight:600;">Diversity > 0.3</span> <span style="color:#555577;">Consensus adds value</span></div>
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# FINAL POLISH PANELS — WIRED TO ALL BACKENDS
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# SMART DATA FINGERPRINT PANEL  (shown on Data Ingestion page)
# ──────────────────────────────────────────────────────────────────

def _render_smart_detection_panel():
    """Auto-detect data types, skew, bimodality, ID columns."""
    if st.session_state.df_raw is None:
        return
    df = st.session_state.df_raw
    _section("🧠 Smart Data Detection")
    _info("Automatic detection of data quality issues, special column types, and preprocessing recommendations.")

    if st.button("🔍 Run Smart Detection", use_container_width=True, key="smart_det"):
        with st.spinner("Analysing data characteristics..."):
            try:
                from preprocessing import SmartDataTypeDetector
                detector = SmartDataTypeDetector()
                detection = detector.detect(df)
                st.session_state["_smart_detection"] = detection
            except Exception as e:
                st.error(str(e))

    det = st.session_state.get("_smart_detection")
    if det:
        c1, c2, c3, c4 = st.columns(4)
        with c1: _metric_card(str(det.get("n_skewed",0)), "Skewed Features", color="#ff8c00")
        with c2: _metric_card(str(det.get("n_bimodal",0)), "Bimodal Features", color="#9b59ff")
        with c3: _metric_card(str(det.get("n_timeseries",0)), "Time-Series Cols", color="#ffd700")
        with c4: _metric_card(str(det.get("n_id_like",0)), "ID-Like Cols (drop!)", color="#ff4444")

        _subsection("Recommendations")
        for rec in det.get("preprocessing_recommendations", []):
            kind = "success" if "✅" in rec else "warn" if "⚠️" in rec else "info"
            st.markdown(f'<div class="{kind}-panel">{rec}</div>', unsafe_allow_html=True)

        if det.get("flag_summary"):
            _subsection("Flag Summary")
            flag_df = pd.DataFrame(
                list(det["flag_summary"].items()), columns=["Flag","Count"]
            ).sort_values("Count", ascending=False)
            st.dataframe(flag_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────
# DIMENSIONALITY REDUCTION BENCHMARKER PANEL
# ──────────────────────────────────────────────────────────────────

def _render_dim_reduc_benchmark():
    """Compare PCA vs ICA vs UMAP vs t-SNE trustworthiness."""
    if not _has_data():
        _info("Preprocess data first.")
        return

    X = st.session_state.X_processed
    _section("📐 Dimensionality Reduction Benchmark")
    _info("Find the best embedding for your data before visualising. "
          "Trustworthiness measures how well local neighbourhood structure is preserved.")

    avail_methods = ["PCA", "ICA", "TruncatedSVD"]
    try:
        import umap; avail_methods.append("UMAP")
    except ImportError: pass

    sel_methods = st.multiselect("Methods to benchmark", avail_methods,
                                  default=["PCA","ICA","TruncatedSVD"])
    n_comp = st.slider("Target components", 2, min(20, X.shape[1]-1), 5)
    max_s  = st.slider("Max sample size", 500, 10000, 3000)

    if st.button("📐 Run Benchmark", type="primary", use_container_width=True, key="dr_bench"):
        with st.spinner("Benchmarking embeddings..."):
            try:
                from preprocessing import DimReducBenchmarker
                bench = DimReducBenchmarker(n_components=n_comp, max_samples=max_s)
                results = bench.benchmark(X, methods=sel_methods)
                st.session_state["_dr_bench"] = results
            except Exception as e:
                st.error(str(e))

    dr = st.session_state.get("_dr_bench")
    if dr:
        bench_df = pd.DataFrame([{
            "Method": r["method"],
            "Trustworthiness": r.get("trustworthiness"),
            "Reconstruction Error": r.get("reconstruction_error"),
            "Variance Explained": r.get("variance_explained"),
            "Runtime(s)": r.get("runtime_seconds"),
            "Status": r.get("status",""),
        } for r in dr])
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

        if len(dr) > 0:
            best_m = max(dr, key=lambda r: r.get("trustworthiness") or 0)
            _success(f"Best embedding: **{best_m['method']}** "
                      f"(trustworthiness = {best_m.get('trustworthiness','N/A')})")

        # Bar chart
        valid_bench = [r for r in dr if r.get("trustworthiness") is not None]
        if valid_bench:
            vis_engine = _get_vis()
            if vis_engine:
                fig_bench = go.Figure(go.Bar(
                    x=[r["method"] for r in valid_bench],
                    y=[r["trustworthiness"] for r in valid_bench],
                    marker_color=[_b("Theme").cluster_color(i) for i in range(len(valid_bench))],
                    text=[f"{r['trustworthiness']:.3f}" for r in valid_bench],
                    textposition="outside",
                ))
                fig_bench.update_layout(
                    paper_bgcolor=_b("Theme").BG_DARK,
                    plot_bgcolor=_b("Theme").BG_CARD,
                    font=dict(color=_b("Theme").TEXT_PRIMARY),
                    height=350,
                    yaxis=dict(range=[0,1.05], title="Trustworthiness"),
                    title="Embedding Trustworthiness Comparison",
                    showlegend=False,
                )
                st.plotly_chart(fig_bench, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# PERFORMANCE LEADERBOARD PANEL
# ──────────────────────────────────────────────────────────────────

def _render_leaderboard_panel():
    """Cross-dataset algorithm performance tracker."""
    _section("🏆 Algorithm Performance Leaderboard")
    _info("Track algorithm performance across multiple datasets and runs. "
          "Helps identify consistently top-performing algorithms for your domain.")

    if "leaderboard" not in st.session_state:
        from clustering_runner import PerformanceLeaderboard
        st.session_state["leaderboard"] = PerformanceLeaderboard()

    lb = st.session_state["leaderboard"]

    col_save, col_clear, _ = st.columns([1,1,3])
    with col_save:
        if _has_results() and st.button("💾 Save Current Run", use_container_width=True):
            ds_name = st.session_state.df_filename or f"dataset_{len(lb._history)}"
            X = st.session_state.X_processed
            lb.record(ds_name, st.session_state.eval_results, X.shape[0], X.shape[1])
            _success(f"Saved {len(st.session_state.eval_results)} results for '{ds_name}'")
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            lb.clear(); st.rerun()

    if lb._history:
        top_df = lb.top_algorithms(top_n=15)
        if not top_df.empty:
            _subsection("Top Algorithms (Mean Composite Score)")
            st.dataframe(top_df, use_container_width=True, hide_index=True)

        win_df = lb.win_rates()
        if not win_df.empty:
            _subsection("Win Rates (% of datasets where ranked #1)")
            st.dataframe(win_df, use_container_width=True, hide_index=True)

        _subsection("Full History")
        st.dataframe(lb.to_dataframe(), use_container_width=True, height=350)
        st.download_button("💾 Export Leaderboard CSV",
                            lb.to_dataframe().to_csv(index=False).encode(),
                            "leaderboard.csv", "text/csv")
    else:
        _info("No runs recorded yet. Run clustering and click **Save Current Run**.")


# ──────────────────────────────────────────────────────────────────
# WHITENING TRANSFORM PANEL
# ──────────────────────────────────────────────────────────────────

def _render_whitening_panel():
    """Apply ZCA/PCA whitening to processed data."""
    if not _has_data():
        _info("Preprocess data first.")
        return

    X = st.session_state.X_processed
    _section("⬜ Whitening Transform")
    _info(
        "Whitening removes correlations and normalises variance — "
        "critical for GMM, K-Means, and distance-sensitive algorithms on correlated data."
    )

    method = st.selectbox("Whitening method",
                           ["ZCA (preserves feature space)", "PCA (maximal decorrelation)"])
    eps = st.number_input("Epsilon (regularisation)", value=1e-5,
                           format="%.1e", min_value=1e-8, max_value=0.1)

    if st.button("⬜ Apply Whitening", type="primary", use_container_width=True):
        with st.spinner("Whitening data..."):
            try:
                from preprocessing import WhiteningTransform
                m = "zca" if "ZCA" in method else "pca"
                wt = WhiteningTransform(method=m, epsilon=float(eps))
                X_white, _ = wt.fit_transform(X)
                st.session_state.X_processed = X_white
                _success(f"Applied {m.upper()} whitening. Shape: {X_white.shape}")

                # Show correlation before/after
                vis_engine = _get_vis()
                if vis_engine:
                    corr_before = pd.DataFrame(X[:, :min(15, X.shape[1])]).corr()
                    corr_after  = pd.DataFrame(X_white[:, :min(15, X_white.shape[1])]).corr()
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Before Whitening**")
                        _safe_plotly(vis_engine.correlation_heatmap(corr_before, 15))
                    with col_b:
                        st.markdown("**After Whitening**")
                        _safe_plotly(vis_engine.correlation_heatmap(corr_after, 15))
            except Exception as e:
                st.error(str(e))


# ──────────────────────────────────────────────────────────────────
# RESULT DEDUPLICATION PANEL
# ──────────────────────────────────────────────────────────────────

def _render_deduplication_panel():
    """Find and remove identical clustering solutions."""
    if not st.session_state.batch_result:
        _info("Run clustering first.")
        return

    _section("🧹 Result Deduplication")
    _info("Remove algorithms that produced identical partitions — "
          "reduces evaluation noise and speeds up stability analysis.")

    if st.button("🔍 Find Duplicates", use_container_width=True):
        with st.spinner("Fingerprinting all results..."):
            from clustering_runner import ResultFingerprinter
            fp = ResultFingerprinter()
            results = st.session_state.batch_result.results
            unique, dup_map = fp.deduplicate(results)
            st.session_state["_dup_map"] = dup_map
            st.session_state["_unique_results"] = unique

    dup_map = st.session_state.get("_dup_map")
    if dup_map is not None:
        c1, c2 = st.columns(2)
        with c1: _metric_card(str(len(dup_map)), "Duplicate Solutions Found", color="#ff8c00")
        with c2: _metric_card(
            str(len(st.session_state.batch_result.results) - len(dup_map)),
            "Unique Solutions", color="#00ff88")

        if dup_map:
            dup_df = pd.DataFrame([
                {"Duplicate Algorithm": k, "Same As": v}
                for k, v in dup_map.items()
            ])
            st.dataframe(dup_df, use_container_width=True, hide_index=True)
            _info("These algorithms found identical partitions. "
                  "Consider running only one representative per group in future.")


# ──────────────────────────────────────────────────────────────────
# COMPLEXITY ESTIMATOR PANEL
# ──────────────────────────────────────────────────────────────────

def _render_complexity_panel():
    """Empirically estimate algorithm time complexity on this data."""
    if not _has_data():
        _info("Preprocess data first.")
        return

    X = st.session_state.X_processed
    _section("⏱️ Runtime Complexity Estimator")
    _info("Empirically fits a power-law T = a·nᵇ to predict runtime at full scale. "
          "Helps you decide whether an algorithm is feasible before running it.")

    registry = _b("get_registry")()
    alg_sel = st.selectbox("Algorithm to profile",
                            registry.ids(),
                            format_func=lambda x: registry.get(x).name,
                            key="cplx_alg")
    n_clusters = st.slider("k for test", 2, 20, st.session_state.n_clusters, key="cplx_k")

    if st.button("⏱️ Estimate Complexity", type="primary", use_container_width=True):
        with st.spinner("Running complexity estimation (takes ~30s)..."):
            try:
                from clustering_runner import ComplexityEstimator
                est = ComplexityEstimator(sample_sizes=[200, 500, 1000, 2000, 4000])
                result = est.estimate(alg_sel, X, n_clusters=n_clusters)
                st.session_state["_cplx_result"] = result
            except Exception as e:
                st.error(str(e))

    cres = st.session_state.get("_cplx_result")
    if cres and cres.get("status") == "ok":
        c1, c2, c3 = st.columns(3)
        b = cres.get("exponent_b")
        pred = cres.get("predicted_runtime_s")
        cc = cres.get("complexity_class","")
        with c1: _metric_card(f"{b:.2f}" if b else "N/A", "Complexity Exponent b", color="#9b59ff")
        with c2: _metric_card(f"{pred:.1f}s" if pred else "N/A", f"Predicted Runtime (n={X.shape[0]})", color="#ff8c00")
        with c3: st.markdown(f'<div class="info-panel" style="margin-top:.5rem;">{cc}</div>', unsafe_allow_html=True)

        # Plot measured vs fit
        sizes = cres.get("sizes_tested",[])
        times = cres.get("runtimes_s",[])
        if sizes and times and b is not None:
            import numpy as _np
            a_val = float(_np.exp(_np.polyfit(_np.log(sizes), _np.log(times), 1)[1]))
            fit_x = list(range(int(min(sizes)), int(X.shape[0])+1, max(1, X.shape[0]//50)))
            fit_y = [a_val * (n ** b) for n in fit_x]
            fig_cplx = go.Figure()
            fig_cplx.add_trace(go.Scatter(
                x=sizes, y=times, mode="markers", name="Measured",
                marker=dict(color=_b("Theme").ACCENT_CYAN, size=10)))
            fig_cplx.add_trace(go.Scatter(
                x=fit_x, y=fit_y, mode="lines", name=f"Fit: T∝n^{b:.2f}",
                line=dict(color=_b("Theme").ACCENT_VIOLET, width=2, dash="dash")))
            fig_cplx.update_layout(
                paper_bgcolor=_b("Theme").BG_DARK, plot_bgcolor=_b("Theme").BG_CARD,
                font=dict(color=_b("Theme").TEXT_PRIMARY), height=340,
                xaxis_title="Sample size (n)", yaxis_title="Runtime (s)",
                title=f"Complexity Fit: {registry.get(alg_sel).name}")
            st.plotly_chart(fig_cplx, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# MULTI-RESOLUTION CONSENSUS PANEL
# ──────────────────────────────────────────────────────────────────

def _render_multi_resolution_panel():
    """Multi-k consensus analysis to find the most stable k."""
    if not _has_results():
        _info("Run clustering first.")
        return

    _section("🔬 Multi-Resolution Consensus")
    _info("Tests consensus clustering at multiple k values. "
          "The k with the best consensus silhouette is the most structurally supported.")

    X = st.session_state.X_processed
    batch = st.session_state.batch_result
    eval_results = st.session_state.eval_results

    k_vals_str = st.text_input("k values to test", "2,3,4,5,6,7,8,10,12")
    top_n_algos = st.slider("Use top N algorithms for ensemble", 3, 20, 10)

    if st.button("🔬 Run Multi-Resolution", type="primary", use_container_width=True):
        with st.spinner("Running multi-resolution consensus..."):
            try:
                from consensus import MultiResolutionConsensus, CoAssociationMatrixBuilder
                k_vals = [int(k.strip()) for k in k_vals_str.split(",") if k.strip()]
                top_ids = [er.algorithm_id for er in eval_results[:top_n_algos]
                           if batch.results.get(er.algorithm_id) and
                           batch.results[er.algorithm_id].succeeded and
                           len(batch.results[er.algorithm_id].labels) == len(X)]
                label_arrays = [batch.results[aid].labels for aid in top_ids]
                if len(label_arrays) < 2:
                    _warn("Need at least 2 successful results.")
                else:
                    mrc = MultiResolutionConsensus(k_range=k_vals)
                    res = mrc.run(X, label_arrays)
                    st.session_state["_mrc_result"] = res
            except Exception as e:
                st.error(str(e))

    mrc = st.session_state.get("_mrc_result")
    if mrc and mrc.get("best_k"):
        _success(f"**Optimal consensus k = {mrc['best_k']}** "
                  f"(silhouette = {mrc['quality_by_k'].get(mrc['best_k'],'N/A')})")
        st.markdown(f'<div class="info-panel">{mrc.get("interpretation","")}</div>',
                    unsafe_allow_html=True)

        vis_engine = _get_vis()
        if vis_engine and mrc.get("k_values"):
            fig_mrc = go.Figure()
            fig_mrc.add_trace(go.Scatter(
                x=mrc["k_values"], y=mrc["silhouettes"],
                mode="lines+markers",
                line=dict(color=_b("Theme").ACCENT_CYAN, width=2),
                marker=dict(color=_b("Theme").ACCENT_CYAN, size=8),
                name="Consensus Silhouette",
            ))
            fig_mrc.add_vline(x=mrc["best_k"], line_dash="dash",
                               line_color=_b("Theme").ACCENT_GREEN,
                               annotation_text=f"Best k={mrc['best_k']}",
                               annotation_font_color=_b("Theme").ACCENT_GREEN)
            fig_mrc.update_layout(
                paper_bgcolor=_b("Theme").BG_DARK,
                plot_bgcolor=_b("Theme").BG_CARD,
                font=dict(color=_b("Theme").TEXT_PRIMARY), height=380,
                xaxis_title="k", yaxis_title="Consensus Silhouette",
                title="Multi-Resolution Consensus Quality")
            st.plotly_chart(fig_mrc, use_container_width=True)

        if mrc.get("labels") is not None and vis_engine:
            _subsection("Best-k Consensus Projection")
            mrc_method = st.selectbox("Embedding", _b("available_embedding_methods")(), key="mrc_vis")
            fig_mrc_scatter = vis_engine.scatter_2d(
                X, mrc["labels"], method=mrc_method,
                algorithm_name=f"Multi-Res Consensus k={mrc['best_k']}")
            _safe_plotly(fig_mrc_scatter)


# ──────────────────────────────────────────────────────────────────
# CLUSTER PROFILES (per-cluster feature signatures)
# ──────────────────────────────────────────────────────────────────

def _render_cluster_profiles_panel():
    """Detailed per-cluster characterisation with AI naming."""
    if not _has_results():
        _info("Run clustering first.")
        return

    X = st.session_state.X_processed
    batch = st.session_state.batch_result
    eval_results = st.session_state.eval_results
    feature_names = st.session_state.feature_names
    vis_engine = _get_vis()

    _section("🔬 Cluster Profile Explorer")
    _info("Detailed characterisation of each cluster: feature signatures, "
          "outlier rates, cohesion scores, and auto-generated descriptive labels.")

    alg_options = {er.algorithm_id: er.algorithm_name for er in eval_results}
    sel_alg = st.selectbox("Algorithm", list(alg_options.keys()),
                            format_func=lambda x: alg_options[x], key="cp_alg")
    labels = batch.results[sel_alg].labels if sel_alg in batch.results else None
    if labels is None or not batch.results[sel_alg].succeeded:
        _warn("No valid labels for selected algorithm.")
        return

    if st.button("🔬 Generate Cluster Profiles", use_container_width=True):
        with st.spinner("Profiling clusters..."):
            try:
                from consensus import ConsensusClusterProfiler
                profiler = ConsensusClusterProfiler()
                profiles = profiler.profile(X, labels, feature_names)
                st.session_state["_cluster_profiles"] = profiles
            except Exception as e:
                st.error(str(e))

    profiles = st.session_state.get("_cluster_profiles")
    if profiles:
        for cid, prof in sorted(profiles.items()):
            color = _b("Theme").cluster_color(cid) if _b("Theme") else "#00e5ff"
            with st.expander(f"Cluster {cid} — '{prof['label']}' "
                              f"(n={prof['size']}, {prof['fraction']*100:.1f}%)"):
                c1, c2, c3 = st.columns(3)
                with c1: _metric_card(str(prof["size"]), "Size", color=color)
                with c2: _metric_card(f"{prof['outlier_rate']*100:.1f}%",
                                       "Outlier Rate",
                                       color="#ff8c00" if prof["outlier_rate"]>0.15 else "#00ff88")
                with c3: _metric_card(f"{max(0,prof['cohesion_score']):.3f}",
                                       "Cohesion", color=color)

                _subsection("Feature Signature (most distinctive features)")
                sig_df = pd.DataFrame(prof["feature_signature"])
                if not sig_df.empty:
                    st.dataframe(sig_df, use_container_width=True, hide_index=True)

        # AI cluster naming
        _sep()
        if st.button(" AI: Name All Clusters", use_container_width=True):
            context = "\n".join([
                f"Cluster {cid}: size={p['size']}, "
                f"top features: {[s['feature'] + '(' + s['direction'] + ')' for s in p['feature_signature'][:3]]}"
                for cid, p in profiles.items()
            ])
            prompt = ("Given these cluster descriptions, suggest a memorable, "
                       "domain-agnostic 2-3 word label for each cluster that a "
                       "data scientist could use in a report. Format: 'Cluster N: Label'")
            with st.spinner(" Naming clusters..."):
                resp = _gemini_query(prompt, context)
            _render_ai_response(resp)


# ──────────────────────────────────────────────────────────────────
# INJECT ALL NEW PANELS INTO CORRECT PAGES
# ──────────────────────────────────────────────────────────────────

# These run AFTER their respective page blocks, only when that page is active

if page == "📁 Data Ingestion" and st.session_state.df_raw is not None:
    _sep()
    _render_smart_detection_panel()

elif page == "⚙️ Preprocessing" and _has_data():
    _sep()
    with st.expander("⬜ Apply Whitening Transform (advanced)", expanded=False):
        _render_whitening_panel()
    with st.expander("📐 Dimensionality Reduction Benchmark", expanded=False):
        _render_dim_reduc_benchmark()

elif page == "⚡ Execution Engine":
    if st.session_state.batch_result:
        _sep()
        with st.expander("🧹 Duplicate Result Detector", expanded=False):
            _render_deduplication_panel()

elif page == "🛠️ Advanced Tools":
    _sep()
    _tabs_extra = st.tabs([
        "🏆 Leaderboard", "⏱️ Complexity Estimator",
        "🔬 Multi-Resolution Consensus", "📊 Cluster Profiles"
    ])
    with _tabs_extra[0]:
        _render_leaderboard_panel()
    with _tabs_extra[1]:
        _render_complexity_panel()
    with _tabs_extra[2]:
        _render_multi_resolution_panel()
    with _tabs_extra[3]:
        _render_cluster_profiles_panel()

elif page == "🤝 Consensus Forge" and _has_results():
    _sep()
    with st.expander("🔬 Multi-Resolution Consensus", expanded=False):
        _render_multi_resolution_panel()


# ══════════════════════════════════════════════════════════════════
# EXECUTION TIMELINE CHART (shown after any run)
# ══════════════════════════════════════════════════════════════════

def _render_execution_timeline():
    """Gantt-style chart showing algorithm execution order and duration."""
    br = st.session_state.batch_result
    if not br:
        return
    _section("⏱️ Execution Timeline")
    rows = []
    cumulative = 0.0
    for aid, cr in sorted(br.results.items(),
                           key=lambda x: x[1].runtime_seconds):
        if cr.status.value in ("skipped",""):
            continue
        rows.append({
            "Algorithm": getattr(cr,"algorithm_name",aid)[:30],
            "Start": round(cumulative, 3),
            "End": round(cumulative + cr.runtime_seconds, 3),
            "Runtime": round(cr.runtime_seconds, 3),
            "Status": cr.status.value,
        })
        cumulative += cr.runtime_seconds

    if not rows:
        return

    T = _b("Theme")
    status_color = {
        "success":"#00ff88","cached":"#00ccff",
        "failed":"#ff4444","timeout":"#ff8c00","skipped":"#555577"
    }
    fig_tl = go.Figure()
    for r in rows:
        col = status_color.get(r["Status"],"#9b59ff")
        fig_tl.add_trace(go.Bar(
            x=[r["Runtime"]], y=[r["Algorithm"]],
            base=[r["Start"]], orientation="h",
            marker_color=col, marker_line_width=0,
            name=r["Status"], showlegend=False,
            hovertemplate=f"<b>{r['Algorithm']}</b><br>{r['Runtime']}s<extra></extra>",
        ))
    fig_tl.update_layout(
        paper_bgcolor=T.BG_DARK if T else "#07070f",
        plot_bgcolor=T.BG_CARD if T else "#0d0d1e",
        font=dict(color=T.TEXT_PRIMARY if T else "#e0e0f0"),
        height=max(400, 22*len(rows)),
        xaxis_title="Cumulative Time (s)",
        title="Algorithm Execution Timeline",
        barmode="stack",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar":False})


if page == "⚡ Execution Engine" and st.session_state.batch_result:
    _sep()
    with st.expander("⏱️ Execution Timeline", expanded=False):
        _render_execution_timeline()


# ══════════════════════════════════════════════════════════════════
# FINAL KEYBOARD SHORTCUTS REFERENCE (collapsible)
# ══════════════════════════════════════════════════════════════════

if page == "🛠️ Advanced Tools":
    with st.expander("📖 Complete API & Method Reference"):
        st.markdown("""
        #### Evaluation Metrics
        | Metric | Range | Ideal | Direction |
        |--------|-------|-------|-----------|
        | Silhouette | [-1, 1] | 1 | Higher better |
        | Davies-Bouldin | [0, ∞) | 0 | Lower better |
        | Calinski-Harabász | [0, ∞) | ∞ | Higher better |
        | Dunn Index | [0, ∞) | ∞ | Higher better |
        | Xie-Beni | [0, ∞) | 0 | Lower better |
        | DBCV | [-1, 1] | 1 | Higher better |
        | Hopkins H | [0, 1] | 1 | Higher = more clusterable |
        | Noise Ratio | [0, 1] | 0 | Lower better |

        #### Stability Metrics
        | Metric | Meaning |
        |--------|---------|
        | ARI | Agreement with reference (0=random, 1=identical) |
        | AMI | Mutual info adjusted for chance |
        | NMI | Normalised mutual information |
        | Jaccard | Set overlap between matched clusters |

        #### Consensus Methods
        | Method | Best for |
        |--------|----------|
        | EAC Average | General purpose, most reliable |
        | CSPA | Non-convex shapes, large k |
        | Bayesian | When quality varies widely across algorithms |
        | Hybrid | High diversity ensembles |
        | Meta-Clustering | When label arrays are high-dimensional |
        """)

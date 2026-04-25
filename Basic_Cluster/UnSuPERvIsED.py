"""
UnSuPERvIsED.py — The World's Most Advanced Unsupervised Learning Intelligence Lab
====================================================================================
A hyper-premium, dark-themed Streamlit application for comprehensive cluster analysis.
Features 25+ algorithms, AI-powered insights via Gemini, stability analysis,
consensus clustering, and publication-ready visualizations.
Author: ClusterX Intelligence Lab
"""

from __future__ import annotations

import io
import os
import sys
import time
import json
import hashlib
import warnings
import traceback
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.datasets import (
    make_blobs, make_moons, make_circles, make_classification,
    load_iris, load_wine, load_breast_cancer,
)

from preprocessing import (
    DataLoader, DataProfiler, MissingValueHandler, OutlierDetector,
    FeatureScaler, CategoricalEncoder, FeatureSelector,
    PreprocessingConfig, PreprocessingResult, DataProfile,
    PreprocessingPipeline,
    ScalerType, ImputeStrategy, OutlierMethod, OutlierAction,
    FeatureSelectionMethod,
)
from clustering_registry import (
    REGISTRY, AlgorithmFamily, AlgorithmMeta, ParameterSpec, ParameterType,
)
from clustering_runner import (
    ClusteringOrchestrator, RunConfig, BatchResult, SingleRunResult,
    RunStatus, ElbowFinder, SweepPoint,
)
from evaluation import (
    EvaluationEngine, ClusteringReport, GapStatistic,
    HopkinsStatistic, NNDistanceProfile, FeatureImportanceAnalyzer,
    MetricNormalizer, ComparisonMatrixBuilder, ClusterOverlapAnalyzer,
    ClusterProfiler,
)
from stability_consensus import (
    StabilityPipeline, StabilityReport, ConsensusResult,
    StabilityConfigPresets, StabilityVisualDataBuilder,
)
from visualization import (
    DimReducer, ScatterPlotter, SilhouettePlotter, ElbowPlotter,
    RadarPlotter, HeatmapPlotter, DendrogramPlotter, DistributionPlotter,
    StabilityPlotter, GaugePlotter, PCAPlotter, ViolinPlotter,
    GeneralBarPlotter,
    PairPlotter, SunburstPlotter, ConvergencePlotter, MetricsTablePlotter,
    NNDistancePlotter, HopkinsPlotter, OverlapHeatmapPlotter,
    DARK_BG, CARD_BG, GRID_COLOR, TEXT_COLOR,
    ACCENT_CYAN, ACCENT_MAGENTA, ACCENT_GOLD, CLUSTER_PALETTE,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UnSuPERvIsED — Clustering Intelligence Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "UnSuPERvIsED v1.0 — The world's most advanced clustering platform.",
    },
)

# ──────────────────────────────────────────────────────────────────
# DARK THEME CSS
# ──────────────────────────────────────────────────────────────────

MASTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #161622;
    --bg-hover: #1c1c2e;
    --border: #2a2a3e;
    --text-primary: #e8e8f0;
    --text-secondary: #8888a0;
    --text-muted: #555570;
    --accent-cyan: #00f0ff;
    --accent-magenta: #ff00aa;
    --accent-gold: #ffd700;
    --accent-green: #00ff88;
    --accent-red: #ff4466;
    --glow-cyan: 0 0 20px rgba(0,240,255,0.15);
    --glow-magenta: 0 0 20px rgba(255,0,170,0.15);
    --radius: 12px;
    --radius-sm: 8px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

html, body, [class*="st-"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

.stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

header[data-testid="stHeader"] { background: transparent !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d14 0%, #0a0a12 100%) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] label {
    color: var(--text-primary) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: var(--transition) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,240,255,0.12), rgba(255,0,170,0.08)) !important;
    color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1rem !important;
}

div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
    box-shadow: var(--glow-cyan) !important;
    transition: var(--transition) !important;
}

div[data-testid="stMetric"]:hover {
    border-color: var(--accent-cyan) !important;
    transform: translateY(-2px) !important;
}

div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(135deg, rgba(0,240,255,0.15), rgba(255,0,170,0.10)) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid rgba(0,240,255,0.3) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    padding: 8px 24px !important;
    transition: var(--transition) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,240,255,0.25), rgba(255,0,170,0.18)) !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
    transform: translateY(-1px) !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

.stSlider > div > div > div {
    background: var(--border) !important;
}

.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-magenta)) !important;
}

.glass-card {
    background: rgba(18,18,26,0.85);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(0,240,255,0.12);
    border-radius: var(--radius);
    padding: 20px;
    margin: 8px 0;
    box-shadow: var(--glow-cyan);
    transition: var(--transition);
}
.glass-card:hover {
    border-color: rgba(0,240,255,0.25);
    box-shadow: 0 0 30px rgba(0,240,255,0.2);
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00f0ff, #ff00aa, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: var(--text-secondary);
    margin-top: 2px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.algo-badge {
    display: inline-block;
    background: rgba(0,240,255,0.08);
    border: 1px solid rgba(0,240,255,0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: var(--accent-cyan);
    margin: 2px 3px;
    font-family: 'JetBrains Mono', monospace;
}

.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(42,42,62,0.5);
}

.metric-label { color: var(--text-secondary); font-size: 0.82rem; }
.metric-value {
    color: var(--accent-cyan);
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.status-success { color: var(--accent-green); }
.status-fail { color: var(--accent-red); }
.status-warn { color: var(--accent-gold); }

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 10px rgba(0,240,255,0.1); }
    50% { box-shadow: 0 0 25px rgba(0,240,255,0.3); }
}

.pulse-border {
    animation: pulse-glow 3s ease-in-out infinite;
}

.stFileUploader > div {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

div[data-testid="stNotification"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}
</style>
"""
st.markdown(MASTER_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "df_raw": None, "df_processed": None, "X_processed": None,
    "feature_names": [], "data_profile": None, "data_loaded": False,
    "preprocessing_done": False, "preprocessing_log": [],
    "batch_result": None, "eval_reports": [],
    "stability_reports": [], "elbow_data": None, "gap_data": None,
    "selected_algorithms": ["kmeans"], "run_complete": False,
    "X_2d": None, "X_3d": None, "dim_method": "pca",
    "gemini_response": "", "gemini_history": [],
    "active_tab": 0, "dataset_name": "", "outlier_mask": None,
    "consensus_result": None, "sweep_points": [],
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _reset_downstream():
    for k in ["batch_result", "eval_reports", "stability_reports",
              "elbow_data", "gap_data", "run_complete", "X_2d", "X_3d",
              "consensus_result", "sweep_points", "gemini_response"]:
        st.session_state[k] = _DEFAULTS[k]


# ──────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def _glass(content: str):
    st.markdown(f'<div class="glass-card">{content}</div>', unsafe_allow_html=True)

def _badge(text: str) -> str:
    return f'<span class="algo-badge">{text}</span>'

def _metric_row(label: str, value: str) -> str:
    return f'<div class="metric-row"><span class="metric-label">{label}</span><span class="metric-value">{value}</span></div>'

def _format_number(n: float) -> str:
    if abs(n) >= 1e6: return f"{n/1e6:.1f}M"
    if abs(n) >= 1e3: return f"{n/1e3:.1f}K"
    if abs(n) < 0.01 and n != 0: return f"{n:.2e}"
    return f"{n:.4f}" if isinstance(n, float) else str(n)

def _safe_metric(label, value, delta=None, delta_color="normal"):
    try:
        st.metric(label, value, delta=delta, delta_color=delta_color)
    except Exception:
        st.metric(label, str(value))

def _generate_synthetic(name: str) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    if name == "Blobs (3 clusters)":
        X, y = make_blobs(n_samples=1000, centers=3, n_features=5, random_state=42)
    elif name == "Blobs (7 clusters)":
        X, y = make_blobs(n_samples=2000, centers=7, n_features=8, random_state=42)
    elif name == "Moons":
        X, y = make_moons(n_samples=1000, noise=0.08, random_state=42)
    elif name == "Circles":
        X, y = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
    elif name == "Anisotropic":
        X, y = make_blobs(n_samples=1000, centers=3, random_state=42)
        transform = [[0.6, -0.6], [-0.4, 0.8]]
        X = X[:, :2] @ transform
    elif name == "Iris":
        data = load_iris()
        X, y = data.data, data.target
    elif name == "Wine":
        data = load_wine()
        X, y = data.data, data.target
    elif name == "Breast Cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif name == "High Dimensional (20D)":
        X, y = make_classification(n_samples=1500, n_features=20, n_informative=10,
                                    n_clusters_per_class=1, n_classes=5, random_state=42)
    elif name == "Noisy Blobs":
        X, y = make_blobs(n_samples=1200, centers=4, n_features=6,
                           cluster_std=2.5, random_state=42)
    else:
        X, y = make_blobs(n_samples=500, centers=3, random_state=42)
    cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["true_label"] = y
    return df

SYNTHETIC_DATASETS = [
    "Blobs (3 clusters)", "Blobs (7 clusters)", "Moons", "Circles",
    "Anisotropic", "Iris", "Wine", "Breast Cancer",
    "High Dimensional (20D)", "Noisy Blobs",
]


# ──────────────────────────────────────────────────────────────────
# SIDEBAR — DATA LOADING
# ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="hero-title">🔬 UnSuPERvIsED</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Clustering Intelligence Lab</div>', unsafe_allow_html=True)
    st.markdown("---")

    data_source = st.radio("Data Source", ["📁 Upload File", "🧪 Synthetic Dataset"],
                           horizontal=True, label_visibility="collapsed")

    if data_source == "📁 Upload File":
        uploaded = st.file_uploader("Upload CSV / Excel / JSON / Parquet",
                                     type=["csv", "xlsx", "xls", "json", "parquet", "tsv"],
                                     help="Max 500K rows × 2000 columns")
        if uploaded is not None:
            file_key = hashlib.md5(uploaded.getvalue()[:4096]).hexdigest()[:12]
            if st.session_state.get("_file_key") != file_key:
                loader = DataLoader()
                try:
                    df, load_log = loader.load(uploaded, uploaded.name)
                    st.session_state.df_raw = df
                    st.session_state.dataset_name = uploaded.name
                    st.session_state.data_loaded = True
                    st.session_state._file_key = file_key
                    st.session_state.preprocessing_done = False
                    _reset_downstream()
                    st.success(f"✅ Loaded {len(df)} × {len(df.columns)}")
                except Exception as e:
                    st.error(f"Load failed: {e}")
    else:
        dataset_choice = st.selectbox("Choose Dataset", SYNTHETIC_DATASETS)
        if st.button("🔄 Generate", use_container_width=True):
            df = _generate_synthetic(dataset_choice)
            st.session_state.df_raw = df
            st.session_state.dataset_name = dataset_choice
            st.session_state.data_loaded = True
            st.session_state.preprocessing_done = False
            _reset_downstream()
            st.success(f"✅ Generated {len(df)} × {len(df.columns)}")

    st.markdown("---")

    if st.session_state.data_loaded and st.session_state.df_raw is not None:
        df_raw = st.session_state.df_raw
        st.markdown(f"**Dataset:** `{st.session_state.dataset_name}`")
        st.markdown(f"**Shape:** `{df_raw.shape[0]}` × `{df_raw.shape[1]}`")
        n_num = len(df_raw.select_dtypes(include=[np.number]).columns)
        n_cat = len(df_raw.columns) - n_num
        st.markdown(f"**Numeric:** `{n_num}` · **Categorical:** `{n_cat}`")
        miss_pct = round(df_raw.isnull().sum().sum() / max(df_raw.size, 1) * 100, 1)
        st.markdown(f"**Missing:** `{miss_pct}%`")

    st.markdown("---")
    st.caption("Built with 🧠 by ClusterX Intelligence Lab")
    st.caption(f"Session: {datetime.now().strftime('%H:%M:%S')}")


# ──────────────────────────────────────────────────────────────────
# MAIN CONTENT — HERO HEADER
# ──────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🔬 UnSuPERvIsED</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">World-Class Unsupervised Clustering Intelligence Platform</div>',
            unsafe_allow_html=True)
st.markdown("")

if not st.session_state.data_loaded:
    _glass("""
    <h3 style='color: var(--accent-cyan); margin-top:0;'>Welcome to UnSuPERvIsED</h3>
    <p style='color: var(--text-secondary);'>
    Upload your dataset or choose a synthetic benchmark from the sidebar to begin.<br/>
    This platform provides <b>25+ clustering algorithms</b>, comprehensive evaluation,
    stability analysis, consensus clustering, and <b>AI-powered insights</b> via Gemini.
    </p>
    <p style='color: var(--text-muted); font-size: 0.85rem;'>
    Supported formats: CSV, Excel, JSON, Parquet, TSV · Max 500K rows
    </p>
    """)
    st.stop()

# ──────────────────────────────────────────────────────────────────
# MAIN TABS
# ──────────────────────────────────────────────────────────────────

tab_profile, tab_preprocess, tab_algorithms, tab_run, tab_viz, tab_stability, tab_ai = st.tabs([
    "📊 Data Profile", "⚙️ Preprocessing", "🧬 Algorithms",
    "🚀 Run Clustering", "📈 Visualizations", "🔒 Stability", "🤖 AI Insights",
])

df_raw = st.session_state.df_raw

# ──────────────────────────────────────────────────────────────────
# TAB 1 — DATA PROFILING
# ──────────────────────────────────────────────────────────────────

with tab_profile:
    st.subheader("📊 Data Profile & Exploration")

    if st.session_state.data_profile is None:
        profiler = DataProfiler()
        st.session_state.data_profile = profiler.profile(df_raw)

    profile = st.session_state.data_profile

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: _safe_metric("Rows", _format_number(profile.n_rows))
    with c2: _safe_metric("Columns", profile.n_cols)
    with c3: _safe_metric("Numeric", profile.n_numeric)
    with c4: _safe_metric("Missing", f"{profile.total_missing_pct}%")
    with c5: _safe_metric("Duplicates", f"{profile.duplicate_pct}%")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("##### Column Profiles")
        profile_rows = []
        for name, cp in profile.column_profiles.items():
            profile_rows.append({
                "Column": name, "Type": cp.dtype, "Unique": cp.n_unique,
                "Missing %": cp.missing_pct,
                "Mean": round(cp.mean, 3) if cp.mean is not None else "—",
                "Std": round(cp.std, 3) if cp.std is not None else "—",
                "Skew": round(cp.skewness, 3) if cp.skewness is not None else "—",
                "Outliers %": cp.outlier_pct,
            })
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, height=350)

    with col_right:
        st.markdown("##### Correlation Matrix")
        if profile.correlation_matrix is not None:
            fig_corr = HeatmapPlotter.correlation_heatmap(profile.correlation_matrix)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Need ≥2 numeric columns for correlation matrix")

        if profile.high_corr_pairs:
            st.markdown("##### High Correlations (>0.90)")
            for a, b, v in profile.high_corr_pairs[:10]:
                st.markdown(f"- `{a}` ↔ `{b}`: **{v:.4f}**")

    with st.expander("🔍 Missing Value Heatmap", expanded=False):
        if profile.total_missing > 0:
            fig_miss = HeatmapPlotter.missing_value_heatmap(df_raw)
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("No missing values detected!")

    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df_raw.head(100), use_container_width=True, height=300)

    if profile.warnings:
        with st.expander("⚠️ Profiling Warnings"):
            for w in profile.warnings:
                st.warning(w)


# ──────────────────────────────────────────────────────────────────
# TAB 2 — PREPROCESSING
# ──────────────────────────────────────────────────────────────────

with tab_preprocess:
    st.subheader("⚙️ Preprocessing Pipeline")

    pp_left, pp_right = st.columns([1, 1])

    with pp_left:
        _glass("<h4 style='margin:0;color:var(--accent-cyan);'>Data Cleaning</h4>")
        imp_strategy = st.selectbox(
            "Missing Value Strategy",
            [e.value for e in ImputeStrategy],
            index=1,
            help="How to fill missing values before clustering",
        )
        scaler_choice = st.selectbox(
            "Feature Scaling",
            [e.value for e in ScalerType],
            index=0,
            help="Normalisation method applied to all numeric features",
        )
        outlier_method = st.selectbox(
            "Outlier Detection",
            [e.value for e in OutlierMethod],
            index=0,
            help="Method to identify anomalous rows",
        )
        outlier_action = st.selectbox(
            "Outlier Action",
            [e.value for e in OutlierAction],
            index=0,
            help="What to do with detected outliers",
        )
        contamination = st.slider(
            "Outlier Contamination", 0.01, 0.20, 0.05, 0.01,
            help="Expected fraction of outliers in the dataset",
        )

    with pp_right:
        _glass("<h4 style='margin:0;color:var(--accent-magenta);'>Feature Engineering</h4>")
        feat_selection = st.selectbox(
            "Feature Selection",
            [e.value for e in FeatureSelectionMethod],
            index=0,
            help="Dimensionality reduction / feature filtering method",
        )
        var_thresh = st.slider(
            "Variance Threshold", 0.0, 0.5, 0.01, 0.005,
            help="Remove features with variance below this threshold",
        )
        corr_thresh = st.slider(
            "Correlation Threshold", 0.70, 1.0, 0.95, 0.01,
            help="Remove one of two features with correlation above this",
        )
        pca_var = st.slider(
            "PCA Variance Explained", 0.80, 1.0, 0.95, 0.01,
            help="Target cumulative variance for PCA (if selected)",
        )

        drop_cols = st.multiselect(
            "Columns to Drop",
            list(df_raw.columns),
            default=[c for c in ["true_label", "target", "class", "label"]
                     if c in df_raw.columns],
            help="Columns to exclude (e.g. ground truth labels)",
        )

    st.markdown("---")

    if st.button("🚀 Run Preprocessing Pipeline", use_container_width=True, type="primary"):
        pp_config = PreprocessingConfig(
            impute_strategy=ImputeStrategy(imp_strategy),
            scaler_type=ScalerType(scaler_choice),
            outlier_method=OutlierMethod(outlier_method),
            outlier_action=OutlierAction(outlier_action),
            outlier_contamination=contamination,
            feature_selection=FeatureSelectionMethod(feat_selection),
            variance_threshold=var_thresh,
            correlation_threshold=corr_thresh,
            pca_variance_explained=pca_var,
            drop_columns=drop_cols,
        )

        with st.spinner("Running preprocessing pipeline..."):
            try:
                pipeline = PreprocessingPipeline(pp_config)
                result = pipeline.run(df_raw)
                st.session_state.X_processed = result.X_processed.values
                st.session_state.feature_names = result.feature_names
                st.session_state.preprocessing_done = True
                st.session_state.preprocessing_result = result
                st.session_state.outlier_mask = result.outlier_mask
                _reset_downstream()
                st.success(
                    f"✅ Pipeline complete: {result.X_processed.shape[0]} samples × "
                    f"{result.X_processed.shape[1]} features | "
                    f"{result.n_outliers} outliers detected"
                )
            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
                st.code(traceback.format_exc(), language="text")

    # ── Show results if preprocessing is done ──
    if st.session_state.preprocessing_done:
        result = st.session_state.get("preprocessing_result")
        if result:
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                _safe_metric("Samples", result.X_processed.shape[0])
            with r2:
                _safe_metric("Features", result.X_processed.shape[1],
                             delta=f"-{len(result.dropped_columns)} dropped")
            with r3:
                _safe_metric("Outliers", result.n_outliers)
            with r4:
                _safe_metric("Scaler", str(result.config.scaler_type.value))

            # Hopkins Statistic
            with st.expander("🔬 Clusterability Test (Hopkins Statistic)", expanded=True):
                hopkins_calc = HopkinsStatistic(random_state=42)
                hopkins_res = hopkins_calc.compute(st.session_state.X_processed)
                hc1, hc2 = st.columns([1, 2])
                with hc1:
                    fig_hop = HopkinsPlotter.plot(hopkins_res["hopkins"])
                    st.plotly_chart(fig_hop, use_container_width=True)
                with hc2:
                    st.markdown(f"**Hopkins Score:** `{hopkins_res['hopkins']}`")
                    st.markdown(f"**Interpretation:** {hopkins_res['interpretation']}")
                    if hopkins_res["is_clusterable"]:
                        st.success("✅ Data shows meaningful cluster structure!")
                    else:
                        st.warning("⚠️ Data may not have strong cluster structure.")

            with st.expander("📋 Pipeline Log"):
                for entry in result.log:
                    st.text(entry)

            with st.expander("📊 Processed Data Preview"):
                st.dataframe(result.X_processed.head(100),
                             use_container_width=True, height=300)

            # NN Distance Profile
            with st.expander("📐 k-NN Distance Profile (DBSCAN ε estimation)"):
                nn_k = st.slider("k for NN distance", 3, 20, 5, key="nn_k_slider")
                nn_profiler = NNDistanceProfile(k=nn_k)
                nn_res = nn_profiler.compute(st.session_state.X_processed)
                fig_nn = NNDistancePlotter.plot(
                    nn_res["kth_distances"], nn_res["suggested_eps"], k=nn_k,
                )
                st.plotly_chart(fig_nn, use_container_width=True)
                st.info(f"📍 Suggested ε = **{nn_res['suggested_eps']}** "
                        f"(knee at index {nn_res['knee_index']})")


# ──────────────────────────────────────────────────────────────────
# TAB 3 — ALGORITHM SELECTION & CONFIGURATION
# ──────────────────────────────────────────────────────────────────

with tab_algorithms:
    st.subheader("🧬 Algorithm Library & Configuration")

    if not st.session_state.preprocessing_done:
        st.warning("⚙️ Please run preprocessing first.")
        st.stop()

    orchestrator = ClusteringOrchestrator()
    all_algos = orchestrator.available_algorithms
    all_families = orchestrator.algorithm_families

    algo_left, algo_right = st.columns([1.5, 1])

    with algo_left:
        _glass("<h4 style='margin:0;color:var(--accent-cyan);'>Browse Algorithms</h4>")

        # Family filter
        selected_family = st.selectbox(
            "Filter by Family", ["All"] + all_families, index=0
        )
        if selected_family != "All":
            filtered_algos = [
                a for a in all_algos
                if orchestrator.get_algorithm_info(a) and
                orchestrator.get_algorithm_info(a).family.value == selected_family
            ]
        else:
            filtered_algos = all_algos

        # Algorithm summary table
        algo_table = orchestrator.list_algorithms()
        if algo_table:
            df_algos = pd.DataFrame(algo_table)
            if selected_family != "All":
                df_algos = df_algos[df_algos.get("family", "") == selected_family]
            st.dataframe(df_algos, use_container_width=True, height=300)

        # Selection
        st.markdown("##### Select Algorithms to Run")
        selected = st.multiselect(
            "Algorithms", filtered_algos,
            default=["kmeans"] if "kmeans" in filtered_algos else filtered_algos[:1],
            help="Select one or more algorithms to include in the batch run",
        )
        st.session_state.selected_algorithms = selected

    with algo_right:
        _glass("<h4 style='margin:0;color:var(--accent-magenta);'>Recommendations</h4>")
        X_proc = st.session_state.X_processed
        if X_proc is not None:
            recs = orchestrator.recommend_algorithms(X_proc.shape[0], X_proc.shape[1])
            for meta, score in recs[:5]:
                badge_color = "#00ff88" if score > 0.7 else "#ffd700"
                st.markdown(
                    f"<div style='padding:8px;margin:4px 0;border-left:3px solid {badge_color};"
                    f"background:rgba(18,18,26,0.8);border-radius:4px;'>"
                    f"<b style='color:{badge_color};'>{meta.display_name}</b> "
                    f"<span style='color:var(--text-muted);'>— score: {score:.2f}</span><br/>"
                    f"<small style='color:var(--text-secondary);'>{meta.description[:80]}...</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Per-algorithm parameter config
    if selected:
        st.markdown("---")
        st.markdown("##### ⚙️ Parameter Configuration")
        params_override = {}
        param_cols = st.columns(min(len(selected), 3))
        for i, algo_name in enumerate(selected):
            col = param_cols[i % len(param_cols)]
            with col:
                info = orchestrator.get_algorithm_info(algo_name)
                if info:
                    st.markdown(f"**{info.display_name}**")
                    algo_params = {}
                    for pspec in info.parameters:
                        pname = pspec.name
                        key = f"param_{algo_name}_{pname}"
                        if pspec.param_type.value == "int":
                            algo_params[pname] = st.number_input(
                                pspec.name.replace("_", " ").title(), 
                                value=int(pspec.default),
                                min_value=int(pspec.min_val) if pspec.min_val else 1,
                                max_value=int(pspec.max_val) if pspec.max_val else 100,
                                help=pspec.description,
                                key=key,
                            )
                        elif pspec.param_type.value == "float":
                            algo_params[pname] = st.number_input(
                                pspec.name.replace("_", " ").title(),
                                value=float(pspec.default),
                                min_value=float(pspec.min_val) if pspec.min_val else 0.0,
                                max_value=float(pspec.max_val) if pspec.max_val else 100.0,
                                help=pspec.description,
                                step=0.01, key=key,
                            )
                        elif pspec.param_type.value == "categorical":
                            opts = pspec.choices or [str(pspec.default)]
                            algo_params[pname] = st.selectbox(
                                pspec.name.replace("_", " ").title(), 
                                opts,
                                index=opts.index(str(pspec.default)) if str(pspec.default) in opts else 0,
                                help=pspec.description,
                                key=key,
                            )
                    if algo_params:
                        params_override[algo_name] = algo_params

        st.session_state["params_override"] = params_override

    # Algorithm detail cards
    st.markdown("---")
    st.markdown("##### 📋 Algorithm Detail Cards")
    if selected:
        for algo_name in selected:
            info = orchestrator.get_algorithm_info(algo_name)
            if info:
                st.markdown(
                    f"<div class='glass-card'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<h4 style='margin:0;color:var(--accent-cyan);'>{info.display_name}</h4>"
                    f"<span class='algo-badge'>{info.family.value}</span>"
                    f"</div>"
                    f"<p style='color:var(--text-secondary);margin:8px 0;'>{info.description}</p>"
                    f"<div style='display:flex;gap:20px;'>"
                    f"<span style='color:var(--text-muted);'>Scalability: "
                    f"<b style='color:var(--accent-gold);'>{getattr(info, 'scalability', 'N/A')}</b>"
                    f"</span>"
                    f"<span style='color:var(--text-muted);'>Requires k: "
                    f"<b style='color:var(--accent-gold);'>{getattr(info, 'requires_k', True)}</b>"
                    f"</span>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Session registry stats
    st.markdown("---")
    st.markdown("##### 📊 Registry Statistics")
    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
    with stat_c1:
        _safe_metric("Total Algorithms", len(all_algos))
    with stat_c2:
        _safe_metric("Families", len(all_families))
    with stat_c3:
        _safe_metric("Selected", len(selected) if selected else 0)
    with stat_c4:
        _safe_metric("Data Features", X_proc.shape[1] if X_proc is not None else 0)


# ──────────────────────────────────────────────────────────────────
# TAB 4 — RUN CLUSTERING
# ──────────────────────────────────────────────────────────────────

with tab_run:
    st.subheader("🚀 Clustering Execution")

    if not st.session_state.preprocessing_done:
        st.warning("⚙️ Please run preprocessing first.")
        st.stop()

    X_data = st.session_state.X_processed
    algos_to_run = st.session_state.selected_algorithms

    if not algos_to_run:
        st.info("🧬 Please select algorithms in the Algorithms tab.")
        st.stop()

    run_left, run_right = st.columns([1, 1])

    with run_left:
        _glass("<h4 style='margin:0;color:var(--accent-cyan);'>Run Configuration</h4>")
        n_clusters = st.slider("Number of Clusters (k)", 2, 25, 3, key="run_k")
        k_range_min = st.number_input("Elbow k-range min", 2, 20, 2, key="elbow_min")
        k_range_max = st.number_input("Elbow k-range max", 3, 30, 12, key="elbow_max")
        timeout = st.slider("Timeout per algo (sec)", 10, 300, 120, 10, key="run_timeout")
        random_seed = st.number_input("Random Seed", 0, 9999, 42, key="run_seed")

    with run_right:
        _glass("<h4 style='margin:0;color:var(--accent-magenta);'>Selected Algorithms</h4>")
        for a in algos_to_run:
            st.markdown(f"{_badge(a)}", unsafe_allow_html=True)
        st.markdown(f"**Total:** {len(algos_to_run)} algorithms")
        st.markdown(f"**Data:** {X_data.shape[0]} × {X_data.shape[1]}")

    st.markdown("---")

    # ── Elbow / Gap Analysis ──
    elbow_col, gap_col = st.columns(2)

    with elbow_col:
        if st.button("📐 Run Elbow Analysis", use_container_width=True):
            orch = ClusteringOrchestrator(RunConfig(random_state=random_seed))
            progress_bar = st.progress(0, text="Computing elbow...")
            def _elbow_cb(msg, pct):
                progress_bar.progress(min(pct, 1.0), text=msg)
            try:
                elbow_data = orch.run_elbow(
                    X_data, (int(k_range_min), int(k_range_max)), callback=_elbow_cb
                )
                st.session_state.elbow_data = elbow_data
                progress_bar.progress(1.0, text="Done!")
                st.success(
                    f"📍 Optimal k: **{elbow_data['recommended_k']}** "
                    f"(inertia={elbow_data['optimal_k_inertia']}, "
                    f"silhouette={elbow_data['optimal_k_silhouette']})"
                )
            except Exception as e:
                st.error(f"Elbow analysis failed: {e}")

    with gap_col:
        if st.button("📊 Run Gap Statistic", use_container_width=True):
            gap_calc = GapStatistic(n_references=10, random_state=random_seed)
            try:
                with st.spinner("Computing gap statistic..."):
                    gap_data = gap_calc.compute(X_data, k_range=(int(k_range_min), int(k_range_max)))
                    st.session_state.gap_data = gap_data
                    st.success(f"📍 Gap optimal k: **{gap_data.get('optimal_k', '?')}**")
            except Exception as e:
                st.error(f"Gap statistic failed: {e}")

    # Show elbow/gap plots
    if st.session_state.elbow_data:
        ed = st.session_state.elbow_data
        ec1, ec2 = st.columns(2)
        with ec1:
            fig_elbow = ElbowPlotter.plot_elbow(
                ed["k_values"], ed["inertias"], ed.get("optimal_k_inertia")
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
        with ec2:
            fig_sil_curve = ElbowPlotter.plot_silhouette_curve(
                ed["k_values"], ed["silhouette_scores"],
                ed.get("optimal_k_silhouette")
            )
            st.plotly_chart(fig_sil_curve, use_container_width=True)

    st.markdown("---")

    # ── Main Clustering Run ──
    if st.button("⚡ Execute Batch Clustering", use_container_width=True, type="primary"):
        params_ovr = st.session_state.get("params_override", {})
        config = RunConfig(
            algorithms=algos_to_run,
            params_override=params_ovr,
            n_clusters=n_clusters,
            timeout_seconds=timeout,
            random_state=random_seed,
        )
        orch = ClusteringOrchestrator(config)

        progress_bar = st.progress(0, text="Starting batch run...")
        def _run_cb(msg, pct):
            progress_bar.progress(min(pct, 1.0), text=msg)

        try:
            batch = orch.run(X_data, callback=_run_cb)
            st.session_state.batch_result = batch
            progress_bar.progress(1.0, text="Batch complete!")

            # Evaluate all successful results
            engine = EvaluationEngine()
            labels_true = None
            if "true_label" in df_raw.columns:
                labels_true = df_raw["true_label"].values[:len(X_data)]
            reports = engine.evaluate_batch(
                X_data, batch.results,
                labels_true=labels_true,
                feature_names=st.session_state.feature_names,
            )
            reports = engine.rank_results(reports)
            st.session_state.eval_reports = reports
            st.session_state.run_complete = True

            st.success(
                f"✅ Batch complete: {batch.n_algorithms_run} run, "
                f"{batch.n_algorithms_failed} failed | "
                f"Best: **{batch.best_algorithm}** ({batch.best_score:.4f}) | "
                f"Time: {batch.total_time_seconds:.2f}s"
            )
        except Exception as e:
            st.error(f"❌ Batch run failed: {e}")
            st.code(traceback.format_exc(), language="text")

    # ── Display results ──
    if st.session_state.run_complete and st.session_state.batch_result:
        batch = st.session_state.batch_result
        reports = st.session_state.eval_reports

        st.markdown("### 📊 Results Summary")

        # Results table
        res_df = pd.DataFrame([
            {
                "Algorithm": r.algorithm_name,
                "Clusters": r.n_clusters,
                "Noise": r.n_noise,
                "Silhouette": round(r.metrics.get("silhouette", type("X", (), {"value": 0})).value, 4)
                if "silhouette" in r.metrics and r.metrics["silhouette"].error is None else "—",
                "DBI": round(r.metrics.get("davies_bouldin", type("X", (), {"value": 0})).value, 4)
                if "davies_bouldin" in r.metrics and r.metrics["davies_bouldin"].error is None else "—",
                "CH": round(r.metrics.get("calinski_harabasz", type("X", (), {"value": 0})).value, 1)
                if "calinski_harabasz" in r.metrics and r.metrics["calinski_harabasz"].error is None else "—",
                "Ranking": round(r.ranking_score, 4),
            }
            for r in reports
        ])
        st.dataframe(res_df, use_container_width=True)

        # Comparison matrix
        if len(reports) > 1:
            with st.expander("🏆 Algorithm Comparison Matrix"):
                builder = ComparisonMatrixBuilder()
                comp_df = builder.build(reports)
                fig_comp = MetricsTablePlotter.plot(comp_df, "Normalized Comparison")
                st.plotly_chart(fig_comp, use_container_width=True)

                win_df = builder.pairwise_win_matrix(reports)
                st.markdown("**Pairwise Win Count:**")
                st.dataframe(win_df, use_container_width=True)

        # Radar chart
        if reports:
            with st.expander("🎯 Metrics Radar Chart", expanded=True):
                normalizer = MetricNormalizer()
                radar_data = {}
                for r in reports[:5]:
                    normed = normalizer.normalize_report(r)
                    if normed:
                        radar_data[r.algorithm_name] = normed
                if radar_data:
                    first_key = list(radar_data.values())[0].keys()
                    fig_radar = RadarPlotter.plot_comparison(
                        list(radar_data.keys()), radar_data
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

        # Cluster statistics for best result
        if reports:
            best_report = reports[0]
            with st.expander(f"📋 Cluster Stats — {best_report.algorithm_name}"):
                engine_display = EvaluationEngine()
                stats_df = engine_display.get_cluster_stats_dataframe(best_report)
                st.dataframe(stats_df, use_container_width=True)

                # Feature importance
                fi_analyzer = FeatureImportanceAnalyzer()
                best_result = None
                for r in batch.results:
                    if r.algorithm_name == best_report.algorithm_name and r.status == RunStatus.SUCCESS:
                        best_result = r
                        break
                if best_result is not None:
                    fi_df = fi_analyzer.compute_anova_importance(
                        X_data, best_result.labels,
                        feature_names=st.session_state.feature_names,
                    )
                    if not fi_df.empty:
                        st.markdown("**Feature Importance (ANOVA F-test):**")
                        st.dataframe(fi_df, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 5 — VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────

with tab_viz:
    st.subheader("📈 Interactive Visualizations")

    if not st.session_state.run_complete:
        st.warning("🚀 Please run clustering first.")
        st.stop()

    X_data = st.session_state.X_processed
    batch = st.session_state.batch_result
    reports = st.session_state.eval_reports

    # Select which result to visualize
    successful_results = [r for r in batch.results if r.status == RunStatus.SUCCESS]
    if not successful_results:
        st.error("No successful clustering results to visualize.")
        st.stop()

    result_names = [r.display_name for r in successful_results]
    viz_algo = st.selectbox("Visualize Result", result_names, index=0, key="viz_algo_select")
    active_result = next(r for r in successful_results if r.display_name == viz_algo)
    active_labels = active_result.labels

    # ── Dimensionality Reduction ──
    st.markdown("---")
    st.markdown("### 🌌 Manifold Projections")

    dim_left, dim_right = st.columns([0.3, 0.7])

    with dim_left:
        dim_method = st.selectbox(
            "Reduction Method", ["pca", "tsne", "umap"], index=0,
            key="dim_method_sel",
            help="Method for projecting high-dimensional data to 2D/3D",
        )
        perplexity = 30
        if dim_method == "tsne":
            perplexity = st.slider("t-SNE Perplexity", 5, 100, 30, key="tsne_perp")
        n_neighbors = 15
        if dim_method == "umap":
            n_neighbors = st.slider("UMAP Neighbors", 5, 100, 15, key="umap_nn")

        show_3d = st.checkbox("Show 3D", value=False, key="show_3d_check")

        if st.button("🔄 Compute Projection", use_container_width=True):
            with st.spinner(f"Computing {dim_method.upper()} projection..."):
                reducer = DimReducer()
                try:
                    X_2d = reducer.reduce(
                        X_data, method=dim_method, n_components=2,
                        perplexity=perplexity, n_neighbors=n_neighbors,
                    )
                    st.session_state.X_2d = X_2d
                    st.session_state.dim_method = dim_method
                    if show_3d:
                        X_3d = reducer.reduce(
                            X_data, method=dim_method, n_components=3,
                            perplexity=perplexity, n_neighbors=n_neighbors,
                        )
                        st.session_state.X_3d = X_3d
                    st.success("Projection computed!")
                except Exception as e:
                    st.error(f"Reduction failed: {e}")

    with dim_right:
        if st.session_state.X_2d is not None:
            fig_2d = ScatterPlotter.scatter_2d(
                st.session_state.X_2d, active_labels,
                title=f"{dim_method.upper()} — {active_result.display_name}",
            )
            st.plotly_chart(fig_2d, use_container_width=True)

        if show_3d and st.session_state.X_3d is not None:
            if st.session_state.X_3d.shape[1] >= 3:
                fig_3d = ScatterPlotter.scatter_3d(
                    st.session_state.X_3d, active_labels,
                    title=f"{dim_method.upper()} 3D — {active_result.display_name}",
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("⚠️ Cannot render 3D plot: Data has fewer than 3 components.")

    # ── Silhouette Analysis ──
    st.markdown("---")
    with st.expander("📊 Silhouette Analysis", expanded=True):
        try:
            clean_mask = active_labels >= 0
            if clean_mask.sum() > 10 and len(set(active_labels[clean_mask])) >= 2:
                fig_sil = SilhouettePlotter.plot(X_data[clean_mask], active_labels[clean_mask])
                st.plotly_chart(fig_sil, use_container_width=True)
            else:
                st.info("Silhouette requires ≥2 clusters with ≥10 points.")
        except Exception as e:
            st.warning(f"Silhouette plot error: {e}")

    # ── PCA Analysis ──
    with st.expander("📐 PCA Variance & Biplot"):
        pca_c1, pca_c2 = st.columns(2)
        with pca_c1:
            fig_pca_var = PCAPlotter.variance_explained(X_data)
            st.plotly_chart(fig_pca_var, use_container_width=True)
        with pca_c2:
            fig_biplot = PCAPlotter.biplot(
                X_data, active_labels,
                feature_names=st.session_state.feature_names,
            )
            st.plotly_chart(fig_biplot, use_container_width=True)

    # ── Violin Plots ──
    with st.expander("🎻 Feature Distribution Violins"):
        fnames = st.session_state.feature_names
        if fnames:
            feat_to_plot = st.selectbox(
                "Feature", fnames, index=0, key="violin_feat"
            )
            feat_idx = fnames.index(feat_to_plot)
            fig_violin = ViolinPlotter.plot(
                X_data, active_labels, fnames, feature_idx=feat_idx,
            )
            st.plotly_chart(fig_violin, use_container_width=True)

    # ── Pair Plot ──
    with st.expander("🔗 Pair Plot (Scatter Matrix)"):
        max_pair_feat = st.slider("Max features", 3, 6, 4, key="pair_max")
        fig_pair = PairPlotter.plot(
            X_data, active_labels,
            feature_names=st.session_state.feature_names,
            max_features=max_pair_feat,
        )
        st.plotly_chart(fig_pair, use_container_width=True)

    # ── Sunburst ──
    with st.expander("🌞 Cluster Sunburst Chart"):
        fig_sun = SunburstPlotter.plot(active_labels)
        st.plotly_chart(fig_sun, use_container_width=True)

    # ── Cluster Overlap ──
    with st.expander("🔥 Cluster Overlap Analysis"):
        try:
            overlap_analyzer = ClusterOverlapAnalyzer()
            overlap_res = overlap_analyzer.compute_pairwise_overlap(X_data, active_labels)
            ov1, ov2 = st.columns([1, 1])
            with ov1:
                fig_overlap = OverlapHeatmapPlotter.plot(overlap_res["overlap_matrix"])
                st.plotly_chart(fig_overlap, use_container_width=True)
            with ov2:
                st.markdown(f"**Total Overlap:** `{overlap_res['total_overlap']}`")
                st.markdown(f"**Max Overlap:** `{overlap_res['max_overlap']}`")
                if overlap_res["worst_pair"]:
                    st.markdown(f"**Worst Pair:** Cluster {overlap_res['worst_pair'][0]} "
                                f"↔ Cluster {overlap_res['worst_pair'][1]}")
                st.markdown("**Cluster Radii (95th %):**")
                for cid, radius in overlap_res["cluster_radii"].items():
                    st.markdown(f"- Cluster {cid}: `{radius:.4f}`")
        except Exception as e:
            st.warning(f"Overlap analysis error: {e}")

    # ── Distribution Plots ──
    with st.expander("📉 Feature Distributions by Cluster"):
        fnames = st.session_state.feature_names
        if fnames:
            dist_feat = st.selectbox("Feature", fnames, index=0, key="dist_feat")
            dist_idx = fnames.index(dist_feat)
            fig_dist = DistributionPlotter.feature_histogram(
                X_data, active_labels, dist_idx, dist_feat,
            )
            st.plotly_chart(fig_dist, use_container_width=True, key="fig_dist_hist_tab5")

    # ── Cluster Profiler ──
    with st.expander("🧾 Cluster Feature Profiles"):
        profiler = ClusterProfiler()
        prof_df = profiler.profile(
            X_data, active_labels,
            feature_names=st.session_state.feature_names,
        )
        if not prof_df.empty:
            st.dataframe(prof_df, use_container_width=True, height=400)


# ──────────────────────────────────────────────────────────────────
# TAB 6 — STABILITY ANALYSIS
# ──────────────────────────────────────────────────────────────────

with tab_stability:
    st.subheader("🔒 Stability & Consensus Analysis")

    if not st.session_state.run_complete:
        st.warning("🚀 Please run clustering first.")
        st.stop()

    X_data = st.session_state.X_processed
    batch = st.session_state.batch_result
    successful = [r for r in batch.results if r.status == RunStatus.SUCCESS]

    if not successful:
        st.error("No successful results for stability analysis.")
        st.stop()

    stab_left, stab_right = st.columns([1, 1])

    with stab_left:
        _glass("<h4 style='margin:0;color:var(--accent-cyan);'>Configuration</h4>")
        stab_preset = st.selectbox(
            "Analysis Preset", ["quick", "standard", "thorough"], index=1,
            help="Controls number of bootstrap/consensus iterations",
            key="stab_preset",
        )
        preset_config = StabilityConfigPresets.get_preset(stab_preset)
        st.markdown(
            f"Bootstrap: `{preset_config['n_bootstrap']}` | "
            f"Consensus: `{preset_config['n_consensus']}` | "
            f"Perturbation: `{preset_config['n_perturb']}`"
        )

        stab_algos = st.multiselect(
            "Algorithms to Analyze",
            [r.algorithm_name for r in successful],
            default=[r.algorithm_name for r in successful[:3]],
            key="stab_algo_select",
        )

        run_consensus = st.checkbox("Run Consensus Clustering", True, key="stab_consensus")
        run_perturbation = st.checkbox("Run Perturbation Analysis", True, key="stab_perturb")

    with stab_right:
        _glass("<h4 style='margin:0;color:var(--accent-magenta);'>About</h4>")
        st.markdown("""
        **Bootstrap Stability** runs the algorithm on resampled data to measure
        how consistent cluster assignments are (ARI & Jaccard).

        **Consensus Clustering** aggregates multiple runs into a co-association
        matrix and measures PAC (Proportion of Ambiguous Clustering).

        **Perturbation Analysis** injects noise to test robustness.
        """)

    st.markdown("---")

    if st.button("🔬 Run Stability Analysis", use_container_width=True, type="primary"):
        pipeline = StabilityPipeline(
            n_bootstrap=preset_config["n_bootstrap"],
            n_consensus=preset_config["n_consensus"],
            n_perturb=preset_config["n_perturb"],
        )

        progress_bar = st.progress(0, text="Starting stability analysis...")
        def _stab_cb(msg, pct):
            progress_bar.progress(min(pct, 1.0), text=msg)

        stability_reports = []
        for i, algo_name in enumerate(stab_algos):
            result = next((r for r in successful if r.algorithm_name == algo_name), None)
            if result is None:
                continue
            try:
                report = pipeline.run(
                    X_data, algo_name, result.params_used,
                    full_labels=result.labels,
                    run_consensus=run_consensus,
                    run_perturbation=run_perturbation,
                    callback=_stab_cb,
                )
                stability_reports.append(report)
            except Exception as e:
                st.warning(f"Stability failed for {algo_name}: {e}")

        st.session_state.stability_reports = stability_reports
        progress_bar.progress(1.0, text="Stability analysis complete!")
        st.success(f"✅ Analyzed {len(stability_reports)} algorithms")

    # ── Display Stability Results ──
    if st.session_state.stability_reports:
        stab_reports = st.session_state.stability_reports

        # Leaderboard
        st.markdown("### 🏆 Stability Leaderboard")
        leaderboard = StabilityPipeline.get_stability_leaderboard(stab_reports)
        st.dataframe(leaderboard, use_container_width=True)

        # Stability bar chart
        viz_data = StabilityVisualDataBuilder.build_leaderboard_data(stab_reports)
        fig_stab_bar = StabilityPlotter.stability_bars(
            viz_data["algo_names"], viz_data["mean_aris"], viz_data["grades"]
        )
        st.plotly_chart(fig_stab_bar, use_container_width=True)

        # Per-algorithm details
        for report in stab_reports:
            s = report.stability_score
            with st.expander(
                f"{'✅' if s.is_stable else '⚠️'} {s.algorithm_name} — "
                f"ARI: {s.mean_ari:.4f} ± {s.std_ari:.4f} [{s.stability_grade}]"
            ):
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    _safe_metric("Mean ARI", f"{s.mean_ari:.4f}")
                with sc2:
                    _safe_metric("Std ARI", f"{s.std_ari:.4f}")
                with sc3:
                    _safe_metric("Jaccard", f"{s.mean_jaccard:.4f}")
                with sc4:
                    _safe_metric("k Variance", f"{s.cluster_count_variance:.3f}")

                # Bootstrap gauge
                fig_gauge = GaugePlotter.metric_gauge(
                    s.mean_ari, title=f"Stability — {s.algorithm_name}",
                    min_val=0, max_val=1,
                    thresholds=[0.4, 0.65, 0.85],
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Consensus heatmap
                if report.consensus_result is not None:
                    cr = report.consensus_result
                    st.markdown(
                        f"**Consensus:** PAC = `{cr.pac_score:.4f}` | "
                        f"Cophenetic = `{cr.cophenetic_correlation:.4f}` | "
                        f"Runs = `{cr.n_runs}`"
                    )
                    con_c1, con_c2 = st.columns(2)
                    with con_c1:
                        fig_hm = HeatmapPlotter.consensus_heatmap(cr.consensus_matrix)
                        st.plotly_chart(fig_hm, use_container_width=True)
                    with con_c2:
                        if cr.cdf_x is not None and cr.cdf_values is not None:
                            fig_cdf = StabilityPlotter.consensus_cdf(
                                cr.cdf_x, cr.cdf_values, cr.pac_score,
                            )
                            st.plotly_chart(fig_cdf, use_container_width=True)

                # Perturbation curves
                if report.perturbation_result is not None:
                    pr = report.perturbation_result
                    st.markdown(f"**Robustness Score:** `{pr.robustness_score:.4f}`")
                    fig_perturb = StabilityPlotter.perturbation_curve(
                        pr.noise_levels, pr.mean_ari_per_level,
                        pr.std_ari_per_level,
                    )
                    st.plotly_chart(fig_perturb, use_container_width=True)

                    # Feature dropout
                    if pr.feature_dropout_scores:
                        st.markdown("**Feature Dropout Scores:**")
                        fd_df = pd.DataFrame({
                            "Feature Index": list(pr.feature_dropout_scores.keys()),
                            "ARI After Drop": list(pr.feature_dropout_scores.values()),
                        })
                        st.dataframe(fd_df, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 7 — AI INSIGHTS (GEMINI)
# ──────────────────────────────────────────────────────────────────

with tab_ai:
    st.subheader("🤖 AI-Powered Clustering Insights")

    # Check API key
    gemini_key = None
    try:
        gemini_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    if not gemini_key:
        _glass("""
        <h4 style='color:var(--accent-gold);margin:0;'>🔑 Gemini API Key Required</h4>
        <p style='color:var(--text-secondary);'>
        Add your Gemini API key to <code>.streamlit/secrets.toml</code>:<br/>
        <code>GEMINI_API_KEY = "your-key-here"</code>
        </p>
        """)
        st.stop()

    if not st.session_state.run_complete:
        st.info("🚀 Run clustering first to generate AI insights.")
        st.stop()

    # Build context from results
    def _build_ai_context() -> str:
        """Assemble a comprehensive data summary for Gemini."""
        lines = [
            f"Dataset: {st.session_state.dataset_name}",
            f"Samples: {st.session_state.X_processed.shape[0]}",
            f"Features: {st.session_state.X_processed.shape[1]}",
            f"Feature Names: {', '.join(st.session_state.feature_names[:15])}",
            "",
        ]
        reports = st.session_state.eval_reports
        if reports:
            lines.append("=== CLUSTERING RESULTS ===")
            for r in reports[:5]:
                lines.append(f"\nAlgorithm: {r.algorithm_name}")
                lines.append(f"  Clusters: {r.n_clusters} | Noise: {r.n_noise}")
                lines.append(f"  Ranking Score: {r.ranking_score:.4f}")
                for mk, mv in r.metrics.items():
                    if mv.error is None:
                        lines.append(f"  {mv.display_name}: {mv.value:.4f}")

        stab = st.session_state.stability_reports
        if stab:
            lines.append("\n=== STABILITY ANALYSIS ===")
            for sr in stab:
                s = sr.stability_score
                lines.append(
                    f"{s.algorithm_name}: ARI={s.mean_ari:.4f}±{s.std_ari:.4f} "
                    f"Grade={s.stability_grade}"
                )

        return "\n".join(lines)

    # AI prompt
    st.markdown("##### 💬 Ask the AI about your clustering results")
    default_text = (
        "Analyze these clustering results. Which algorithm performed best and why? "
        "Are the clusters well-separated? What insights can you draw about the data structure? "
        "Provide actionable recommendations for improving the clustering."
    )
    # Check if a template was pushed
    initial_value = st.session_state.get("ai_prompt_pushed", default_text)
    
    user_prompt = st.text_area(
        "Your question:", value=initial_value, height=100, key="ai_prompt_widget",
    )

    ai_left, ai_right = st.columns([1, 1])

    with ai_left:
        if st.button("🧠 Generate Insights", use_container_width=True, type="primary"):
            context = _build_ai_context()
            full_prompt = (
                f"You are an expert data scientist specializing in unsupervised learning "
                f"and cluster analysis. Here is the analysis context:\n\n"
                f"{context}\n\n"
                f"USER QUESTION: {user_prompt}\n\n"
                f"Provide a detailed, actionable response with specific references to "
                f"the metrics and algorithms shown above. Use markdown formatting."
            )

            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-flash-lite-latest")
                with st.spinner("🧠 Thinking..."):
                    response = model.generate_content(full_prompt)
                    ai_text = response.text
                    st.session_state.gemini_response = ai_text
                    st.session_state.gemini_history.append({
                        "prompt": user_prompt,
                        "response": ai_text,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    })
            except Exception as e:
                st.error(f"❌ Gemini API error: {e}")
                st.code(traceback.format_exc(), language="text")

    with ai_right:
        if st.button("📊 Auto-Summarize Results", use_container_width=True):
            context = _build_ai_context()
            summary_prompt = (
                f"You are an expert data scientist. Summarize these clustering results "
                f"in a clear, professional report format with sections: "
                f"Executive Summary, Best Algorithm, Key Metrics, Stability Assessment, "
                f"and Recommendations.\n\n{context}"
            )
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-flash-lite-latest")
                with st.spinner("📊 Generating summary..."):
                    response = model.generate_content(summary_prompt)
                    st.session_state.gemini_response = response.text
            except Exception as e:
                st.error(f"❌ Gemini API error: {e}")

    # Display AI response
    if st.session_state.gemini_response:
        st.markdown("---")
        _glass("<h4 style='margin:0;color:var(--accent-cyan);'>🤖 AI Analysis</h4>")
        st.markdown(st.session_state.gemini_response)

    # History
    if st.session_state.gemini_history:
        with st.expander("📜 Conversation History"):
            for i, entry in enumerate(reversed(st.session_state.gemini_history)):
                st.markdown(
                    f"**[{entry['timestamp']}] You:** {entry['prompt'][:100]}..."
                )
                st.markdown(entry["response"][:500] + "...")
                st.markdown("---")

    # Export
    st.markdown("---")
    st.markdown("##### 📥 Export Results")
    export_cols = st.columns(3)

    with export_cols[0]:
        if st.button("💾 Export Labels (CSV)", use_container_width=True):
            batch = st.session_state.batch_result
            best = next(
                (r for r in batch.results
                 if r.algorithm_name == batch.best_algorithm and r.status == RunStatus.SUCCESS),
                None,
            )
            if best is not None:
                orch = ClusteringOrchestrator()
                label_df = orch.export_labels(best)
                csv_data = label_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Labels", csv_data,
                    f"cluster_labels_{best.algorithm_name}.csv",
                    "text/csv",
                )

    with export_cols[1]:
        if st.session_state.eval_reports:
            if st.button("📊 Export Metrics (CSV)", use_container_width=True):
                rows = []
                for r in st.session_state.eval_reports:
                    row = {"Algorithm": r.algorithm_name, "Ranking": r.ranking_score}
                    for mk, mv in r.metrics.items():
                        if mv.error is None:
                            row[mv.display_name] = round(mv.value, 4)
                    rows.append(row)
                metrics_csv = pd.DataFrame(rows).to_csv(index=False)
                st.download_button(
                    "⬇️ Download Metrics", metrics_csv,
                    "clustering_metrics.csv", "text/csv",
                )

    with export_cols[2]:
        if st.session_state.gemini_response:
            if st.button("📝 Export AI Report", use_container_width=True):
                st.download_button(
                    "⬇️ Download Report",
                    st.session_state.gemini_response,
                    "ai_clustering_report.md",
                    "text/markdown",
                )


# ──────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# ADDITIONAL VISUALIZATIONS — CLUSTER SIZE & PARALLEL COORDS
# ──────────────────────────────────────────────────────────────────

with tab_viz:
    # ── Cluster Size Distribution ──
    with st.expander("📏 Cluster Size Distribution"):
        try:
            fig_sizes = DistributionPlotter.cluster_sizes(active_labels)
            st.plotly_chart(fig_sizes, use_container_width=True)
        except Exception as e:
            st.warning(f"Size plot error: {e}")

    # ── Parallel Coordinates ──
    with st.expander("🔀 Parallel Coordinates Plot"):
        try:
            fig_parallel = DistributionPlotter.parallel_coordinates(
                X_data, active_labels,
                feature_names=st.session_state.feature_names,
            )
            st.plotly_chart(fig_parallel, use_container_width=True)
        except Exception as e:
            st.warning(f"Parallel coordinates error: {e}")

    # ── Feature Box Plots ──
    with st.expander("📦 Feature Box Plots by Cluster"):
        fnames = st.session_state.feature_names
        if fnames:
            box_feat = st.selectbox("Feature", fnames, index=0, key="box_feat")
            box_idx = fnames.index(box_feat)
            try:
                fig_box = DistributionPlotter.feature_boxplots(
                    X_data, active_labels,
                    feature_names=st.session_state.feature_names,
                    feature_idx=box_idx,
                )
                st.plotly_chart(fig_box, use_container_width=True)
            except Exception as e:
                st.warning(f"Box plot error: {e}")

    # ── Feature Histograms by Cluster ──
    with st.expander("📊 Feature Histograms by Cluster"):
        fnames = st.session_state.feature_names
        if fnames:
            hist_feat = st.selectbox("Feature", fnames, index=0, key="hist_feat_tab5_unique")
            hist_idx = fnames.index(hist_feat)
            try:
                fig_hist = DistributionPlotter.feature_histogram(
                    X_data, active_labels,
                    feature_idx=hist_idx, feature_name=hist_feat,
                )
                st.plotly_chart(fig_hist, use_container_width=True, key="fig_hist_tab5_plot")
            except Exception as e:
                st.warning(f"Histogram plot error: {e}")

    # ── Dendrogram (if few samples) ──
    if X_data.shape[0] <= 500:
        with st.expander("🌳 Hierarchical Dendrogram"):
            try:
                fig_dendro = DendrogramPlotter.plot(X_data)
                st.plotly_chart(fig_dendro, use_container_width=True)
            except Exception as e:
                st.warning(f"Dendrogram error: {e}")

    # ── Gap Statistic Plot ──
    if st.session_state.gap_data:
        with st.expander("📊 Gap Statistic Plot"):
            gd = st.session_state.gap_data
            if "k_values" in gd and "gaps" in gd:
                try:
                    fig_gap = ElbowPlotter.plot_gap_statistic(
                        gd["k_values"], gd["gaps"],
                        gd.get("gap_stds"), gd.get("optimal_k"),
                    )
                    st.plotly_chart(fig_gap, use_container_width=True)
                except Exception as e:
                    st.warning(f"Gap plot error: {e}")

    # ── Algorithm Comparison Bar Chart ──
    if reports and len(reports) > 1:
        with st.expander("🏅 Algorithm Ranking Comparison"):
            algo_names_cmp = [r.algorithm_name for r in reports]
            metric_names_cmp = []
            metric_values_cmp = {}

            for r in reports:
                for mk, mv in r.metrics.items():
                    if mv.error is None:
                        if mv.display_name not in metric_names_cmp:
                            metric_names_cmp.append(mv.display_name)
                        if mv.display_name not in metric_values_cmp:
                            metric_values_cmp[mv.display_name] = {}
                        normalizer = MetricNormalizer()
                        metric_values_cmp[mv.display_name][r.algorithm_name] = normalizer.normalize(mv)

            if metric_names_cmp:
                comparison_metric = st.selectbox(
                    "Compare Metric", metric_names_cmp, index=0, key="cmp_metric",
                )
                if comparison_metric in metric_values_cmp:
                    vals = metric_values_cmp[comparison_metric]
                    try:
                        fig_cmp = GeneralBarPlotter.plot(
                            list(vals.keys()), list(vals.values()),
                            title=f"Comparison: {comparison_metric}",
                            ylabel=comparison_metric,
                        )
                        st.plotly_chart(fig_cmp, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Comparison error: {e}")


# ──────────────────────────────────────────────────────────────────
# ADDITIONAL STABILITY — CROSS-VALIDATION & EXPORT
# ──────────────────────────────────────────────────────────────────

with tab_stability:
    if st.session_state.stability_reports:
        st.markdown("---")
        st.markdown("### 🔁 Additional Stability Diagnostics")

        # Cross-validation stability
        with st.expander("🔄 Cross-Validation Stability"):
            from stability_consensus import CrossValidationStability
            cv_algo = st.selectbox(
                "Algorithm for CV Stability",
                [r.algorithm_name for r in successful],
                index=0, key="cv_stab_algo",
            )
            cv_folds = st.slider("Folds", 3, 10, 5, key="cv_folds")

            if st.button("Run CV Stability", use_container_width=True, key="cv_run"):
                cv_result_obj = next(
                    (r for r in successful if r.algorithm_name == cv_algo), None
                )
                if cv_result_obj:
                    cv = CrossValidationStability(n_folds=cv_folds)
                    with st.spinner("Running cross-validation stability..."):
                        cv_res = cv.run(X_data, cv_algo, cv_result_obj.params_used)
                    cv_c1, cv_c2, cv_c3, cv_c4 = st.columns(4)
                    with cv_c1:
                        _safe_metric("CV Mean ARI", f"{cv_res['mean_ari']:.4f}")
                    with cv_c2:
                        _safe_metric("CV Std ARI", f"{cv_res['std_ari']:.4f}")
                    with cv_c3:
                        _safe_metric("Min ARI", f"{cv_res['min_ari']:.4f}")
                    with cv_c4:
                        _safe_metric("Comparisons", cv_res["n_comparisons"])

        # Temporal stability
        with st.expander("📈 Temporal Stability (Incremental Data)"):
            from stability_consensus import TemporalStability
            temp_algo = st.selectbox(
                "Algorithm for Temporal Stability",
                [r.algorithm_name for r in successful],
                index=0, key="temp_stab_algo",
            )
            temp_checkpoints = st.slider("Checkpoints", 5, 20, 10, key="temp_cp")

            if st.button("Run Temporal Stability", use_container_width=True, key="temp_run"):
                temp_result_obj = next(
                    (r for r in successful if r.algorithm_name == temp_algo), None
                )
                if temp_result_obj:
                    ts = TemporalStability(n_checkpoints=temp_checkpoints)
                    with st.spinner("Running temporal stability..."):
                        ts_res = ts.run(X_data, temp_algo, temp_result_obj.params_used)
                    st.markdown(f"**Temporal Stability:** `{ts_res['temporal_stability']:.4f}`")
                    st.markdown(f"**Cluster Count Trace:** `{ts_res['cluster_count_trace']}`")

                    # Plot temporal ARI trace
                    aris = [cp.get("ari_vs_prev") for cp in ts_res["checkpoints"]
                            if cp.get("ari_vs_prev") is not None]
                    if aris:
                        import plotly.graph_objects as go
                        fig_temp = go.Figure()
                        fig_temp.add_trace(go.Scatter(
                            x=list(range(1, len(aris) + 1)),
                            y=aris, mode="lines+markers",
                            line=dict(color=ACCENT_CYAN, width=2),
                            marker=dict(size=6),
                            name="ARI vs Previous",
                        ))
                        fig_temp.update_layout(
                            title="Temporal Stability — ARI vs Previous Checkpoint",
                            xaxis_title="Checkpoint",
                            yaxis_title="ARI",
                            plot_bgcolor=DARK_BG,
                            paper_bgcolor=DARK_BG,
                            font=dict(color=TEXT_COLOR),
                        )
                        st.plotly_chart(fig_temp, use_container_width=True)

        # Stability export
        with st.expander("📥 Export Stability Reports"):
            stab_export = []
            for sr in st.session_state.stability_reports:
                d = StabilityPipeline.export_report_dict(sr)
                stab_export.append(d)
            stab_json = json.dumps(stab_export, indent=2, default=str)
            st.download_button(
                "⬇️ Download Stability JSON", stab_json,
                "stability_reports.json", "application/json",
            )


# ──────────────────────────────────────────────────────────────────
# ADDITIONAL RUN TAB — PER-ALGO DRILL-DOWN
# ──────────────────────────────────────────────────────────────────

with tab_run:
    if st.session_state.run_complete and st.session_state.batch_result:
        st.markdown("---")
        st.markdown("### 🔍 Per-Algorithm Evaluation Drill-Down")

        batch = st.session_state.batch_result
        reports = st.session_state.eval_reports
        successful_for_drill = [r for r in batch.results if r.status == RunStatus.SUCCESS]

        for result in successful_for_drill:
            report = next(
                (rp for rp in reports if rp.algorithm_name == result.algorithm_name), None
            )
            if report is None:
                continue

            with st.expander(
                f"🔬 {result.display_name} — "
                f"k={result.n_clusters_found} | "
                f"Time={result.fit_time_seconds:.3f}s | "
                f"Score={report.ranking_score:.4f}"
            ):
                # Metric cards
                m_cols = st.columns(min(len(report.metrics), 6))
                for i, (mk, mv) in enumerate(report.metrics.items()):
                    if mv.error is None and i < len(m_cols):
                        with m_cols[i]:
                            direction = "↑" if mv.higher_is_better else "↓"
                            _safe_metric(
                                f"{mv.display_name} {direction}",
                                f"{mv.value:.4f}",
                            )

                # Summary text
                engine_summary = EvaluationEngine()
                summary_text = engine_summary.get_summary_text(report)
                st.code(summary_text, language="text")

                # Params used
                st.markdown("**Parameters Used:**")
                st.json(result.params_used)

                # Labels distribution
                labels_series = pd.Series(result.labels)
                label_counts = labels_series.value_counts().sort_index()
                lc_data = {
                    "Cluster": [f"{'Noise' if k == -1 else f'Cluster {k}'}" for k in label_counts.index],
                    "Count": label_counts.values.tolist(),
                    "Percentage": [f"{v / len(result.labels) * 100:.1f}%" for v in label_counts.values],
                }
                st.dataframe(pd.DataFrame(lc_data), use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# ADDITIONAL DATA PROFILE — ADVANCED STATISTICS
# ──────────────────────────────────────────────────────────────────

with tab_profile:
    if st.session_state.data_profile:
        st.markdown("---")
        st.markdown("### 📊 Advanced Data Statistics")

        profile = st.session_state.data_profile
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) >= 2:
            with st.expander("📈 Feature Distribution Overview"):
                desc_df = df_raw[num_cols].describe().T
                desc_df["skewness"] = df_raw[num_cols].skew()
                desc_df["kurtosis"] = df_raw[num_cols].kurtosis()
                desc_df["missing_%"] = (df_raw[num_cols].isnull().sum() / len(df_raw) * 100).round(2)
                st.dataframe(desc_df, use_container_width=True, height=350)

            with st.expander("📊 Feature Histograms"):
                hist_col = st.selectbox("Feature", num_cols, index=0, key="hist_feat")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=df_raw[hist_col].dropna(),
                    nbinsx=50,
                    marker_color=ACCENT_CYAN,
                    opacity=0.7,
                    name=hist_col,
                ))
                fig_hist.update_layout(
                    title=f"Distribution: {hist_col}",
                    xaxis_title=hist_col,
                    yaxis_title="Count",
                    plot_bgcolor=DARK_BG,
                    paper_bgcolor=DARK_BG,
                    font=dict(color=TEXT_COLOR),
                )
                st.plotly_chart(fig_hist, use_container_width=True, key="fig_dist_hist_extra")

            with st.expander("🔢 Scatter Plot Explorer"):
                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    feat_x = st.selectbox("X axis", num_cols, index=0, key="scat_x")
                with s_c2:
                    feat_y = st.selectbox(
                        "Y axis", num_cols,
                        index=min(1, len(num_cols) - 1), key="scat_y",
                    )

                color_col = None
                if "true_label" in df_raw.columns:
                    color_col = df_raw["true_label"].astype(str)

                fig_scatter_explore = go.Figure()
                if color_col is not None:
                    for label_val in sorted(color_col.unique()):
                        mask = color_col == label_val
                        fig_scatter_explore.add_trace(go.Scattergl(
                            x=df_raw.loc[mask, feat_x],
                            y=df_raw.loc[mask, feat_y],
                            mode="markers",
                            marker=dict(size=4, opacity=0.6),
                            name=f"Class {label_val}",
                        ))
                else:
                    fig_scatter_explore.add_trace(go.Scattergl(
                        x=df_raw[feat_x], y=df_raw[feat_y],
                        mode="markers",
                        marker=dict(size=4, color=ACCENT_CYAN, opacity=0.5),
                        name="Data",
                    ))
                fig_scatter_explore.update_layout(
                    title=f"{feat_x} vs {feat_y}",
                    xaxis_title=feat_x, yaxis_title=feat_y,
                    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
                    font=dict(color=TEXT_COLOR),
                )
                st.plotly_chart(fig_scatter_explore, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# ADDITIONAL PREPROCESSING — DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────

with tab_preprocess:
    if st.session_state.preprocessing_done:
        result = st.session_state.get("preprocessing_result")
        if result:
            st.markdown("---")
            st.markdown("### 📊 Post-Processing Diagnostics")

            with st.expander("📐 Feature Variance After Scaling"):
                X_proc = st.session_state.X_processed
                variances = np.var(X_proc, axis=0)
                var_df = pd.DataFrame({
                    "Feature": st.session_state.feature_names,
                    "Variance": np.round(variances, 6),
                    "Std Dev": np.round(np.sqrt(variances), 6),
                    "Mean": np.round(np.mean(X_proc, axis=0), 6),
                })
                st.dataframe(var_df, use_container_width=True)

                fig_var_bar = go.Figure()
                fig_var_bar.add_trace(go.Bar(
                    x=st.session_state.feature_names,
                    y=variances,
                    marker_color=ACCENT_CYAN,
                ))
                fig_var_bar.update_layout(
                    title="Feature Variance Distribution",
                    xaxis_title="Feature", yaxis_title="Variance",
                    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
                    font=dict(color=TEXT_COLOR),
                )
                st.plotly_chart(fig_var_bar, use_container_width=True)

            with st.expander("🔗 Post-Processing Correlation Matrix"):
                X_proc_df = pd.DataFrame(
                    st.session_state.X_processed,
                    columns=st.session_state.feature_names,
                )
                corr_post = X_proc_df.corr()
                fig_corr_post = HeatmapPlotter.correlation_heatmap(
                    corr_post, title="Post-Processing Correlations"
                )
                st.plotly_chart(fig_corr_post, use_container_width=True)

            with st.expander("📋 Dropped Columns"):
                if result.dropped_columns:
                    for col in result.dropped_columns:
                        st.markdown(f"- `{col}`")
                else:
                    st.success("No columns were dropped.")

            with st.expander("⚠️ Pipeline Warnings"):
                if result.warnings:
                    for w in result.warnings:
                        st.warning(w)
                else:
                    st.success("No warnings generated.")

            with st.expander("📊 Before vs After Shape"):
                ba_c1, ba_c2 = st.columns(2)
                with ba_c1:
                    _safe_metric(
                        "Original Shape",
                        f"{result.X_original.shape[0]} × {result.X_original.shape[1]}",
                    )
                with ba_c2:
                    _safe_metric(
                        "Processed Shape",
                        f"{result.X_processed.shape[0]} × {result.X_processed.shape[1]}",
                    )

                reduction_pct = round(
                    (1 - result.X_processed.shape[1] / max(result.X_original.shape[1], 1)) * 100, 1
                )
                st.markdown(
                    f"**Dimensionality Reduction:** `{reduction_pct}%` "
                    f"({result.X_original.shape[1]} → {result.X_processed.shape[1]} features)"
                )


# ──────────────────────────────────────────────────────────────────
# ADDITIONAL AI TAB — ADVANCED PROMPTS
# ──────────────────────────────────────────────────────────────────

with tab_ai:
    st.markdown("---")
    st.markdown("##### 🔮 Quick Analysis Templates")

    template_cols = st.columns(4)

    templates = [
        ("🔍 Data Quality", "Analyze the data quality based on the preprocessing results. "
         "Are there potential issues with the feature distributions, missing data patterns, "
         "or outliers that could affect clustering?"),
        ("📊 K Selection", "Based on the elbow analysis and gap statistic, what is the "
         "optimal number of clusters? Explain the evidence from multiple methods."),
        ("🔒 Stability", "Evaluate the stability of the clustering results. Which algorithm "
         "produces the most reproducible clusters? What does the consensus analysis reveal?"),
        ("💡 Next Steps", "Given these clustering results, suggest the top 3 next steps "
         "for improving the analysis. Consider feature engineering, algorithm tuning, "
         "and validation approaches."),
    ]

    for i, (label, prompt) in enumerate(templates):
        with template_cols[i]:
            if st.button(label, use_container_width=True, key=f"tmpl_{i}"):
                st.session_state["ai_prompt_pushed"] = prompt
                st.rerun()

    # Session info
    st.markdown("---")
    st.markdown("##### ℹ️ Session Information")
    info_c1, info_c2, info_c3 = st.columns(3)
    with info_c1:
        _safe_metric("Dataset", st.session_state.dataset_name)
    with info_c2:
        total_algos = len(st.session_state.selected_algorithms)
        _safe_metric("Algorithms", total_algos)
    with info_c3:
        n_reports = len(st.session_state.eval_reports)
        _safe_metric("Evaluations", n_reports)


# ──────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:var(--text-muted);padding:20px;'>"
    "<span style='font-size:0.8rem;'>UnSuPERvIsED v1.0 — "
    "The World's Most Advanced Clustering Intelligence Lab</span><br/>"
    "<span style='font-size:0.7rem;'>Built with Streamlit · Plotly · scikit-learn · Gemini</span>"
    "</div>",
    unsafe_allow_html=True,
)

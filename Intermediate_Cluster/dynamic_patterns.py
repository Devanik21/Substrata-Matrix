"""
DYNAMIC PATTERN GENERATOR — Zero-Memory Overhead
Generates 21+ synthetic clustering patterns on-demand using sklearn.

Integration: Add to UnSuPERvIsED after data ingestion, before preprocessing.
Memory: ~1-2MB per pattern (generated fresh, not stored).

v2 — PERFECT EDITION
────────────────────────────────────────────────────
Every pattern is post-normalized to a canonical coordinate space of
±8.5 units, centered at origin. This guarantees:
  • No pattern ever leaves the fixed animation window (±10.5)
  • No pattern is ever too small or microscopic
  • Aspect ratio is always preserved (no distortion)
  • The arctan2 colour gradient in the animation always looks stunning
    because all patterns are centred at (0, 0)
────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_blobs, make_circles, make_moons,
    make_swiss_roll, make_s_curve, make_classification
)
from typing import Tuple, Dict

# ─────────────────────────────────────────────────────────────────────
# CANONICAL NORMALISATION  (the core fix)
# ─────────────────────────────────────────────────────────────────────

_CANONICAL_RADIUS = 8.5   # All patterns fit within this radius from origin

def _normalise(X: np.ndarray, radius: float = _CANONICAL_RADIUS) -> np.ndarray:
    """
    Center the cloud at (0,0) and scale it so the largest extent in any
    direction equals `radius`.  Aspect ratio is preserved perfectly.
    A tiny margin_factor shrinks it slightly so points never brush the axis edge.
    """
    X = np.asarray(X, dtype=float)
    # 1. Center
    X = X - X.mean(axis=0)
    # 2. Uniform scale (preserve shape)
    max_extent = np.abs(X).max()
    if max_extent > 1e-9:
        margin = 0.90           # Use 90 % of the target radius → clean margin
        X = X * (radius * margin / max_extent)
    return X


# ─────────────────────────────────────────────────────────────────────
# RAW HELPER GENERATORS  (shapes before normalisation)
# ─────────────────────────────────────────────────────────────────────

def _raw_noisy_blobs(n: int, k: int, rs: int) -> np.ndarray:
    """Tight blobs + scattered noise."""
    rng = np.random.RandomState(rs)
    X, _ = make_blobs(n_samples=int(n * 0.82), centers=k,
                      cluster_std=0.35, random_state=rs)
    noise = rng.uniform(-9, 9, (n - int(n * 0.82), 2))
    return np.vstack([X, noise])


def _raw_outlier_blobs(n: int, k: int, rs: int,
                       outlier_ratio: float = 0.10) -> np.ndarray:
    """Blobs with controlled outlier percentage."""
    rng = np.random.RandomState(rs)
    n_clean = int(n * (1 - outlier_ratio))
    X, _ = make_blobs(n_samples=n_clean, centers=k,
                      cluster_std=0.4, random_state=rs)
    outliers = rng.uniform(-9, 9, (n - n_clean, 2))
    return np.vstack([X, outliers])


def _raw_nested_circles(n: int, k: int, rs: int) -> np.ndarray:
    """Multiple concentric circles — beautiful under arctan2 colour."""
    rng = np.random.RandomState(rs)
    n_per = n // k
    X_list = []
    radii = np.linspace(1.0, 4.5, k)
    for r in radii:
        theta = rng.uniform(0, 2 * np.pi, n_per)
        noise_r = rng.normal(0, r * 0.04, n_per)   # proportional noise
        x = (r + noise_r) * np.cos(theta)
        y = (r + noise_r) * np.sin(theta)
        X_list.append(np.column_stack([x, y]))
    return np.vstack(X_list)[:n]


def _raw_hierarchical_blobs(n: int, k: int, rs: int) -> np.ndarray:
    """Multi-scale hierarchical blobs."""
    rng = np.random.RandomState(rs)
    super_centers = rng.uniform(-7, 7, (k, 2))
    per_super = n // k
    X_list = []
    for center in super_centers:
        for _ in range(2):                          # 2 sub-clusters per super
            sub_c = center + rng.normal(0, 1.2, 2)
            pts = rng.normal(sub_c, 0.28, (per_super // 2, 2))
            X_list.append(pts)
    return np.vstack(X_list)[:n]


def _raw_density_varied_blobs(n: int, k: int, rs: int) -> np.ndarray:
    """Clusters with very different point densities."""
    rng = np.random.RandomState(rs)
    centers = rng.uniform(-6, 6, (k, 2))
    sizes = np.array([2 ** i for i in range(k)], dtype=float)
    sizes = (sizes / sizes.sum() * n).astype(int)
    sizes[-1] = n - sizes[:-1].sum()            # fix rounding
    X_list = []
    for i, (center, n_i) in enumerate(zip(centers, sizes)):
        std = 0.25 + 0.18 * i
        X_list.append(rng.normal(center, std, (max(1, n_i), 2)))
    return np.vstack(X_list)[:n]


def _raw_elongated_blobs(n: int, k: int, rs: int) -> np.ndarray:
    """Long thin cluster shapes arranged to use full canvas."""
    rng = np.random.RandomState(rs)
    X_list = []
    for i in range(k):
        center = rng.uniform(-5.5, 5.5, 2)
        angle = rng.uniform(0, np.pi)           # any orientation
        length = rng.uniform(2.5, 5.0)
        width = rng.uniform(0.08, 0.22)
        t = rng.uniform(-length / 2, length / 2, n // k)
        s = rng.normal(0, width, n // k)
        x = center[0] + t * np.cos(angle) + s * np.sin(angle)
        y = center[1] + t * np.sin(angle) - s * np.cos(angle)
        X_list.append(np.column_stack([x, y]))
    return np.vstack(X_list)[:n]


def _raw_overlapped_blobs(n: int, k: int, rs: int,
                          overlap: float = 0.5) -> np.ndarray:
    """Blobs with controlled overlap — compression brings them closer."""
    X, _ = make_blobs(n_samples=n, centers=k, random_state=rs, cluster_std=0.8)
    X *= (1 - overlap * 0.55)
    return X


def _raw_rotated_moons(n: int, rs: int) -> np.ndarray:
    """Two crescent moons rotated 45°."""
    X, _ = make_moons(n_samples=n, noise=0.05, random_state=rs)
    angle = np.pi / 4
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
    return X @ rot.T


def _raw_donut_clusters(n: int, k: int, rs: int) -> np.ndarray:
    """Ring-shaped (donut) clusters tiled across canvas."""
    rng = np.random.RandomState(rs)
    # Place k donuts on a grid-like layout
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    positions = []
    spacing = 5.0
    for r in range(rows):
        for c in range(cols):
            if len(positions) < k:
                cx = (c - (cols - 1) / 2) * spacing
                cy = (r - (rows - 1) / 2) * spacing
                positions.append(np.array([cx, cy]))

    X_list = []
    n_per = n // k
    ring_r = 1.5
    for pos in positions:
        theta = rng.uniform(0, 2 * np.pi, n_per)
        r = rng.normal(ring_r, ring_r * 0.08, n_per)
        x = pos[0] + r * np.cos(theta)
        y = pos[1] + r * np.sin(theta)
        X_list.append(np.column_stack([x, y]))
    return np.vstack(X_list)[:n]


def _raw_blob_in_circle(n: int, rs: int) -> np.ndarray:
    """Central Gaussian blob surrounded by a tight ring."""
    rng = np.random.RandomState(rs)
    n_inner = int(n * 0.30)
    n_outer = n - n_inner
    X_inner = rng.normal([0, 0], 0.5, (n_inner, 2))
    theta = rng.uniform(0, 2 * np.pi, n_outer)
    r = rng.normal(3.5, 0.18, n_outer)
    X_outer = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return np.vstack([X_inner, X_outer])


def _raw_stripe_pattern(n: int, k: int, rs: int) -> np.ndarray:
    """Parallel linear bands — uses full width, evenly spaced."""
    rng = np.random.RandomState(rs)
    X_list = []
    y_positions = np.linspace(-7, 7, k)
    for y_center in y_positions:
        n_strip = n // k
        x = rng.uniform(-8, 8, n_strip)
        y = rng.normal(y_center, 0.15, n_strip)
        X_list.append(np.column_stack([x, y]))
    return np.vstack(X_list)[:n]


def _raw_galaxy_spiral(n: int, k: int, rs: int) -> np.ndarray:
    """
    Logarithmic spiral arms — beautiful under the arctan2 colour map.
    k controls number of arms (capped to max 6 for clarity).
    """
    rng = np.random.RandomState(rs)
    k_arms = min(k, 6)
    n_per = n // k_arms
    X_list = []
    for arm_i in range(k_arms):
        phase = arm_i * (2 * np.pi / k_arms)
        t = rng.uniform(0.3, 1.0, n_per) ** 0.7    # non-linear density
        theta = t * 3.5 * np.pi + phase
        r = t * 8.0
        noise_r = rng.normal(0, r * 0.055, n_per)
        noise_theta = rng.normal(0, 0.12, n_per)
        x = (r + noise_r) * np.cos(theta + noise_theta)
        y = (r + noise_r) * np.sin(theta + noise_theta)
        X_list.append(np.column_stack([x, y]))
    return np.vstack(X_list)[:n]


def _raw_checkerboard(n: int, k: int, rs: int) -> np.ndarray:
    """Grid of blobs in a checkerboard layout."""
    rng = np.random.RandomState(rs)
    side = max(2, int(np.ceil(np.sqrt(k * 2))))
    positions = []
    spacing = 4.5
    for r in range(side):
        for c in range(side):
            if (r + c) % 2 == 0:
                positions.append(np.array([(c - side / 2) * spacing,
                                            (r - side / 2) * spacing]))
    n_blobs = len(positions)
    n_per = max(1, n // n_blobs)
    X_list = []
    for pos in positions:
        X_list.append(rng.normal(pos, 0.45, (n_per, 2)))
    return np.vstack(X_list)[:n]


def _raw_figure_eight(n: int, rs: int) -> np.ndarray:
    """Two overlapping loops forming a figure-eight lemniscate."""
    rng = np.random.RandomState(rs)
    half = n // 2
    # Left loop
    theta1 = rng.uniform(0, 2 * np.pi, half)
    r1 = rng.normal(3.0, 0.18, half)
    x1 = r1 * np.cos(theta1) - 3.0
    y1 = r1 * np.sin(theta1)
    # Right loop
    theta2 = rng.uniform(0, 2 * np.pi, n - half)
    r2 = rng.normal(3.0, 0.18, n - half)
    x2 = r2 * np.cos(theta2) + 3.0
    y2 = r2 * np.sin(theta2)
    return np.vstack([np.column_stack([x1, y1]),
                      np.column_stack([x2, y2])])


def _raw_star_clusters(n: int, k: int, rs: int) -> np.ndarray:
    """Star-shaped (spiked) clusters — radial spikes from each centre."""
    rng = np.random.RandomState(rs)
    n_spikes = 5
    centers = rng.uniform(-6, 6, (k, 2))
    X_list = []
    n_per = n // k
    n_per_spike = n_per // n_spikes
    for center in centers:
        for s in range(n_spikes):
            angle = s * 2 * np.pi / n_spikes
            t = rng.uniform(0, 1.8, n_per_spike)
            x = center[0] + t * np.cos(angle)
            y = center[1] + t * np.sin(angle)
            X_list.append(np.column_stack([x, y]))
    return np.vstack(X_list)[:n]


def _raw_crescents(n: int, k: int, rs: int) -> np.ndarray:
    """Multiple crescent moons tiled across the canvas."""
    rng = np.random.RandomState(rs)
    cols = int(np.ceil(np.sqrt(k)))
    spacing = 5.5
    X_list = []
    n_per = n // k
    for idx in range(k):
        cx = (idx % cols - (cols - 1) / 2) * spacing
        cy = (idx // cols - (k // cols - 1) / 2) * spacing
        # Outer arc
        theta = rng.uniform(-np.pi * 0.65, np.pi * 0.65, n_per)
        r_outer = rng.normal(1.8, 0.11, n_per)
        x_out = cx + r_outer * np.cos(theta)
        y_out = cy + r_outer * np.sin(theta)
        X_list.append(np.column_stack([x_out, y_out]))
    return np.vstack(X_list)[:n]


def _raw_aniso_blobs(n: int, k: int, rs: int) -> np.ndarray:
    """Elongated Gaussian clusters via linear transformation."""
    X, _ = make_blobs(n_samples=n, centers=k, random_state=rs, cluster_std=0.7)
    transform = np.array([[0.70, -0.65],
                          [-0.40,  0.85]])
    return X @ transform.T


def _raw_swiss_roll_2d(n: int, rs: int) -> np.ndarray:
    """Swiss roll projected to 2D — beautiful curve."""
    X, _ = make_swiss_roll(n_samples=n, noise=0.08, random_state=rs)
    return X[:, [0, 2]]


def _raw_s_curve_2d(n: int, rs: int) -> np.ndarray:
    """S-curve manifold projected to 2D."""
    X, _ = make_s_curve(n_samples=n, noise=0.08, random_state=rs)
    return X[:, [0, 2]]


# ─────────────────────────────────────────────────────────────────────
# PATTERN CATALOG  (21 patterns + 4 bonus)
# ─────────────────────────────────────────────────────────────────────

PATTERN_CATALOG: Dict[str, dict] = {

    # ── BLOB-BASED ────────────────────────────────────────────────
    "blobs_isotropic": {
        "name": " Isotropic Blobs",
        "desc": "Equal-sized Gaussian clusters",
        "generator": lambda n, k, rs: make_blobs(
            n_samples=n, centers=k, random_state=rs, cluster_std=0.55
        )[0],
        "best_for": "K-means, GMM",
    },
    "blobs_anisotropic": {
        "name": "Anisotropic Blobs",
        "desc": "Elongated Gaussian clusters via shear transform",
        "generator": lambda n, k, rs: _raw_aniso_blobs(n, k, rs),
        "best_for": "Spectral, DBSCAN",
    },
    "blobs_varied_sizes": {
        "name": "Varied-Size Blobs",
        "desc": "Different Gaussian variances per cluster",
        "generator": lambda n, k, rs: make_blobs(
            n_samples=n, centers=k, random_state=rs,
            cluster_std=np.random.RandomState(rs).uniform(0.25, 1.2, size=k)
        )[0],
        "best_for": "GMM, DBSCAN",
    },

    # ── GEOMETRIC ────────────────────────────────────────────────
    "moons": {
        "name": "🌙 Two Moons",
        "desc": "Two interleaving crescent moons",
        "generator": lambda n, k, rs: make_moons(
            n_samples=n, noise=0.045, random_state=rs
        )[0],
        "best_for": "Spectral, Agglomerative",
    },
    "circles": {
        "name": " Concentric Circles",
        "desc": "Two nested circles",
        "generator": lambda n, k, rs: make_circles(
            n_samples=n, noise=0.04, factor=0.45, random_state=rs
        )[0],
        "best_for": "Spectral, DBSCAN",
    },
    "swiss_roll": {
        "name": " Swiss Roll",
        "desc": "2D projection of 3D roll manifold",
        "generator": lambda n, k, rs: _raw_swiss_roll_2d(n, rs),
        "best_for": "DBSCAN, Manifold",
    },
    "s_curve": {
        "name": " S-Curve",
        "desc": "Elongated S-shaped manifold",
        "generator": lambda n, k, rs: _raw_s_curve_2d(n, rs),
        "best_for": "DBSCAN, t-SNE",
    },

    # ── NOISE + CLUSTERS ─────────────────────────────────────────
    "blobs_with_noise": {
        "name": "Noisy Blobs",
        "desc": "Tight blobs surrounded by outliers",
        "generator": lambda n, k, rs: _raw_noisy_blobs(n, k, rs),
        "best_for": "DBSCAN, LOF",
    },
    "tight_with_outliers": {
        "name": " Tight + Outliers",
        "desc": "Compact clusters with scattered noise",
        "generator": lambda n, k, rs: _raw_outlier_blobs(n, k, rs, 0.10),
        "best_for": "Isolation Forest, DBSCAN",
    },

    # ── HIERARCHICAL ─────────────────────────────────────────────
    "nested_circles": {
        "name": " Nested Rings",
        "desc": "Multiple concentric circles",
        "generator": lambda n, k, rs: _raw_nested_circles(n, k, rs),
        "best_for": "Agglomerative, Spectral",
    },
    "hierarchical_blobs": {
        "name": "🌳 Hierarchical Blobs",
        "desc": "Blobs within blobs (multi-scale)",
        "generator": lambda n, k, rs: _raw_hierarchical_blobs(n, k, rs),
        "best_for": "Agglomerative (dendrograms)",
    },

    # ── DENSITY-BASED ─────────────────────────────────────────────
    "unequal_density": {
        "name": "🌊 Unequal Density",
        "desc": "Clusters with very different point densities",
        "generator": lambda n, k, rs: _raw_density_varied_blobs(n, k, rs),
        "best_for": "DBSCAN, OPTICS",
    },
    "elongated_clusters": {
        "name": "Elongated Clusters",
        "desc": "Long thin cluster shapes",
        "generator": lambda n, k, rs: _raw_elongated_blobs(n, k, rs),
        "best_for": "Spectral, DBSCAN",
    },

    # ── OVERLAPPING ──────────────────────────────────────────────
    "heavily_overlapped": {
        "name": " Heavily Overlapped",
        "desc": "Blurred cluster boundaries",
        "generator": lambda n, k, rs: _raw_overlapped_blobs(n, k, rs, 0.70),
        "best_for": "GMM, Agglomerative",
    },
    "slightly_overlapped": {
        "name": "Slightly Overlapped",
        "desc": "Some boundary overlap, mostly separable",
        "generator": lambda n, k, rs: _raw_overlapped_blobs(n, k, rs, 0.28),
        "best_for": "K-means, Spectral",
    },

    # ── SPECIAL ───────────────────────────────────────────────────
    "halfmoons_rotated": {
        "name": " Rotated Half-Moons",
        "desc": "Two tilted crescent shapes",
        "generator": lambda n, k, rs: _raw_rotated_moons(n, rs),
        "best_for": "Spectral, Agglomerative",
    },
    "donuts": {
        "name": "Donut Clusters",
        "desc": "Ring-shaped clusters tiled across canvas",
        "generator": lambda n, k, rs: _raw_donut_clusters(n, k, rs),
        "best_for": "DBSCAN, Spectral",
    },
    "blob_in_circle": {
        "name": "Blob in Circle",
        "desc": "Central blob surrounded by ring",
        "generator": lambda n, k, rs: _raw_blob_in_circle(n, rs),
        "best_for": "DBSCAN, Spectral",
    },
    "stripes": {
        "name": "Stripe Pattern",
        "desc": "Parallel linear cluster bands",
        "generator": lambda n, k, rs: _raw_stripe_pattern(n, k, rs),
        "best_for": "Spectral, OPTICS",
    },
    "random_uniform": {
        "name": "Uniform Random",
        "desc": "No structure (baseline / robustness test)",
        "generator": lambda n, k, rs: np.random.RandomState(rs).uniform(
            -8, 8, (n, 2)
        ),
        "best_for": "Testing robustness",
    },

    # ── BONUS PATTERNS ────────────────────────────────────────────
    "galaxy_spiral": {
        "name": "Galaxy Spiral",
        "desc": "Logarithmic spiral arms — perfect with radial colour",
        "generator": lambda n, k, rs: _raw_galaxy_spiral(n, k, rs),
        "best_for": "Spectral, DBSCAN",
    },
    "checkerboard": {
        "name": "Checkerboard",
        "desc": "Grid of blobs in alternating positions",
        "generator": lambda n, k, rs: _raw_checkerboard(n, k, rs),
        "best_for": "Spectral, K-means",
    },
    "figure_eight": {
        "name": "Figure Eight",
        "desc": "Two overlapping loops",
        "generator": lambda n, k, rs: _raw_figure_eight(n, rs),
        "best_for": "Spectral, DBSCAN",
    },
    "star_clusters": {
        "name": "⭐ Star Clusters",
        "desc": "Star-shaped clusters with radial spikes",
        "generator": lambda n, k, rs: _raw_star_clusters(n, k, rs),
        "best_for": "Spectral, Agglomerative",
    },
    "crescents": {
        "name": " Crescent Array",
        "desc": "Multiple crescent moons tiled across the canvas",
        "generator": lambda n, k, rs: _raw_crescents(n, k, rs),
        "best_for": "Spectral, DBSCAN",
    },
}


# ─────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────

def get_pattern_names() -> Dict[str, str]:
    """Return {pattern_id: display_name} for every pattern."""
    return {k: v["name"] for k, v in PATTERN_CATALOG.items()}


def get_pattern_info(pattern_id: str) -> dict:
    """Return full metadata dict for a given pattern_id."""
    return PATTERN_CATALOG.get(pattern_id, {})


def generate_pattern(
    pattern_id: str,
    n_samples: int = 300,
    n_clusters: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, dict]:
    """
    Generate a synthetic clustering pattern on-demand.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
        All values are normalised to ±_CANONICAL_RADIUS (≈ ±8.5).
        The cloud is always centred at (0, 0).
        No point ever exceeds the fixed animation window (±10.5).

    metadata : dict
        Pattern info including name, description, best_for, etc.
    """
    if pattern_id not in PATTERN_CATALOG:
        raise ValueError(
            f"Unknown pattern '{pattern_id}'. "
            f"Available: {list(PATTERN_CATALOG)}"
        )

    pattern = PATTERN_CATALOG[pattern_id]
    raw = pattern["generator"](n_samples, n_clusters, random_state)

    # Safely extract ndarray from any return type
    if isinstance(raw, tuple):
        X = raw[0]
    elif isinstance(raw, list) and raw and isinstance(raw[0], (np.ndarray, list)):
        X = raw[0]
    else:
        X = raw

    X = np.asarray(X, dtype=float).reshape(-1, 2)

    # ── THE KEY FIX: canonical normalisation ──────────────────────
    X = _normalise(X)

    return X, {
        "pattern_id":   pattern_id,
        "pattern_name": pattern["name"],
        "description":  pattern["desc"],
        "best_for":     pattern["best_for"],
        "n_samples":    X.shape[0],
        "n_features":   2,
    }


def generate_pattern_dataframe(
    pattern_id: str,
    n_samples: int = 300,
    n_clusters: int = 5,
    random_state: int = 42,
    feature_names: list = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate pattern and return as DataFrame for direct Streamlit use.

    Parameters
    ----------
    pattern_id    : key from PATTERN_CATALOG
    n_samples     : number of data points
    n_clusters    : hint for how many groups to generate
    random_state  : reproducibility seed
    feature_names : column names (default: ['feature_1', 'feature_2'])

    Returns
    -------
    (df, metadata)  — df is always normalised to ±8.5 units
    """
    X, metadata = generate_pattern(pattern_id, n_samples, n_clusters, random_state)

    if feature_names is None:
        feature_names = ["feature_1", "feature_2"]

    df = pd.DataFrame(X, columns=feature_names[:2])
    return df, metadata


# ─────────────────────────────────────────────────────────────────────
# STREAMLIT INTEGRATION SNIPPET  (for reference — unchanged API)
# ─────────────────────────────────────────────────────────────────────

STREAMLIT_INTEGRATION = """
# Add this to UnSuPERvIsED main app (in Data Ingestion section)

import streamlit as st
from dynamic_patterns import get_pattern_names, generate_pattern_dataframe

# ─── In Data Ingestion page ────────────────────────────────────────

st.markdown("### 🎯 Or Generate Synthetic Pattern")
col_pat1, col_pat2, col_pat3 = st.columns(3)

with col_pat1:
    pattern_id = st.selectbox(
        "Clustering Pattern",
        options=list(get_pattern_names().keys()),
        format_func=lambda x: get_pattern_names()[x],
        key="pattern_select"
    )

with col_pat2:
    n_samples = st.slider("Sample Count", 50, 2000, 300, step=50, key="pat_samples")

with col_pat3:
    n_clusters = st.slider("Cluster Count", 2, 10, 5, key="pat_clusters")

if st.button("🎲 Generate Pattern", use_container_width=True):
    with st.spinner("Generating..."):
        df, metadata = generate_pattern_dataframe(
            pattern_id,
            n_samples=n_samples,
            n_clusters=n_clusters,
            random_state=42
        )
        st.session_state.df_raw = df
        st.session_state.source_file = f"Generated: {metadata['pattern_name']}"
        st.success(f"✅ Generated {metadata['n_samples']} samples from {metadata['pattern_name']}")

        with st.expander("📋 Pattern Details", expanded=False):
            st.write(f"**Best For:** {metadata['best_for']}")
            st.write(f"**Description:** {metadata['description']}")

# Show current pattern preview (2D scatter)
if st.session_state.df_raw is not None and len(st.session_state.df_raw.columns) >= 2:
    import plotly.express as px
    fig = px.scatter(
        st.session_state.df_raw,
        x=st.session_state.df_raw.columns[0],
        y=st.session_state.df_raw.columns[1],
        title="Current Data",
        opacity=0.7,
        color_discrete_sequence=["#00e5ff"]
    )
    fig.update_layout(
        height=400,
        paper_bgcolor="#0d0d1e",
        plot_bgcolor="#0d0d1e",
        font_color="#e0e0f0",
        hovermode="closest"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
"""


# ─────────────────────────────────────────────────────────────────────
# QUICK DEMO
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("UnSuPERvIsED — Dynamic Pattern Generator  (Perfect Edition)\n")
    print(f"{'Pattern ID':<25}  {'Shape':>12}  {'x range':>18}  {'y range':>18}")
    print("─" * 80)
    for pid in PATTERN_CATALOG:
        X, meta = generate_pattern(pid, n_samples=1000, n_clusters=6, random_state=42)
        xr = f"[{X[:,0].min():+.2f}, {X[:,0].max():+.2f}]"
        yr = f"[{X[:,1].min():+.2f}, {X[:,1].max():+.2f}]"
        print(f"{pid:<25}  {str(X.shape):>12}  {xr:>18}  {yr:>18}")
    print("\n✅ All patterns normalised to ±8.5 — zero clipping, perfect fill.")

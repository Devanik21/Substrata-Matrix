"""
DYNAMIC PATTERN GENERATOR — Zero-Memory Overhead
Generates 21+ synthetic clustering patterns on-demand using sklearn.

Integration: Add to UnSuPERvIsED after data ingestion, before preprocessing.
Memory: ~1-2MB per pattern (generated fresh, not stored).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_blobs, make_circles, make_moons, 
    make_swiss_roll, make_s_curve, make_classification
)
from typing import Tuple, Dict, Callable

# ─────────────────────────────────────────────────────────────────────
# PATTERN DEFINITIONS (21 VARIANTS)
# ─────────────────────────────────────────────────────────────────────

PATTERN_CATALOG = {
    # BLOB-BASED (centroid clustering)
    "blobs_isotropic": {
        "name": "📍 Isotropic Blobs",
        "desc": "Equal-sized Gaussian clusters",
        "generator": lambda n, k, rs: make_blobs(
            n_samples=n, centers=k, random_state=rs,
            cluster_std=0.5, return_centers=False
        ),
        "best_for": "K-means, GMM"
    },
    "blobs_anisotropic": {
        "name": "🔀 Anisotropic Blobs",
        "desc": "Elongated Gaussian clusters",
        "generator": lambda n, k, rs: make_blobs(
            n_samples=n, centers=k, random_state=rs,
            cluster_std=[0.3, 1.2], return_centers=False
        ),
        "best_for": "Spectral, DBSCAN"
    },
    "blobs_varied_sizes": {
        "name": "📊 Varied-Size Blobs",
        "desc": "Different Gaussian variances per cluster",
        "generator": lambda n, k, rs: make_blobs(
            n_samples=n, centers=k, random_state=rs,
            cluster_std=np.linspace(0.2, 1.5, k), 
            return_centers=False
        ),
        "best_for": "HDBSCAN, GMM"
    },
    
    # GEOMETRIC SHAPES
    "moons": {
        "name": "🌙 Two Moons",
        "desc": "Two interleaving crescent moons",
        "generator": lambda n, k, rs: make_moons(
            n_samples=n, noise=0.05, random_state=rs
        )[0],
        "best_for": "Spectral, Agglomerative"
    },
    "circles": {
        "name": "⭕ Concentric Circles",
        "desc": "Two nested circles",
        "generator": lambda n, k, rs: make_circles(
            n_samples=n, noise=0.05, random_state=rs
        )[0],
        "best_for": "Spectral, DBSCAN"
    },
    "swiss_roll": {
        "name": "🌀 Swiss Roll",
        "desc": "2D manifold from 3D roll",
        "generator": lambda n, k, rs: make_swiss_roll(
            n_samples=n, noise=0.1, random_state=rs
        )[0][:, [0, 2]],  # Project to 2D
        "best_for": "DBSCAN, Manifold"
    },
    "s_curve": {
        "name": "S️ S-Curve",
        "desc": "Elongated S-shaped manifold",
        "generator": lambda n, k, rs: make_s_curve(
            n_samples=n, noise=0.1, random_state=rs
        )[0][:, [0, 2]],  # Project to 2D
        "best_for": "DBSCAN, t-SNE"
    },
    
    # NOISE + CLUSTERS
    "blobs_with_noise": {
        "name": "🔊 Noisy Blobs",
        "desc": "Tight blobs surrounded by outliers",
        "generator": lambda n, k, rs: _noisy_blobs(n, k, rs),
        "best_for": "DBSCAN, LOF"
    },
    "tight_with_outliers": {
        "name": "🎯 Tight + Outliers",
        "desc": "Compact clusters with scattered noise",
        "generator": lambda n, k, rs: _outlier_blobs(n, k, rs, outlier_ratio=0.1),
        "best_for": "Isolation Forest, DBSCAN"
    },
    
    # HIERARCHICAL PATTERNS
    "nested_circles": {
        "name": "🎪 Nested Rings",
        "desc": "Multiple concentric circles",
        "generator": lambda n, k, rs: _nested_circles(n, k, rs),
        "best_for": "Agglomerative, Spectral"
    },
    "hierarchical_blobs": {
        "name": "🌳 Hierarchical Blobs",
        "desc": "Blobs within blobs (multi-scale)",
        "generator": lambda n, k, rs: _hierarchical_blobs(n, k, rs),
        "best_for": "Agglomerative (dendrograms)"
    },
    
    # DENSITY-BASED PATTERNS
    "unequal_density": {
        "name": "🌊 Unequal Density",
        "desc": "Clusters with different point densities",
        "generator": lambda n, k, rs: _density_varied_blobs(n, k, rs),
        "best_for": "DBSCAN, OPTICS"
    },
    "elongated_clusters": {
        "name": "📏 Elongated Clusters",
        "desc": "Long thin cluster shapes",
        "generator": lambda n, k, rs: _elongated_blobs(n, k, rs),
        "best_for": "Spectral, DBSCAN"
    },
    
    # OVERLAPPING PATTERNS
    "heavily_overlapped": {
        "name": "🔗 Heavily Overlapped",
        "desc": "Blurred cluster boundaries",
        "generator": lambda n, k, rs: _overlapped_blobs(n, k, rs, overlap=0.7),
        "best_for": "GMM, Agglomerative"
    },
    "slightly_overlapped": {
        "name": "↔️ Slightly Overlapped",
        "desc": "Some boundary overlap, mostly separable",
        "generator": lambda n, k, rs: _overlapped_blobs(n, k, rs, overlap=0.3),
        "best_for": "K-means, Spectral"
    },
    
    # SPECIAL CASES
    "halfmoons_rotated": {
        "name": "🌗 Rotated Half-Moons",
        "desc": "Two tilted crescent shapes",
        "generator": lambda n, k, rs: _rotated_moons(n, rs),
        "best_for": "Spectral, Agglomerative"
    },
    "donuts": {
        "name": "🍩 Donut Clusters",
        "desc": "Ring-shaped clusters",
        "generator": lambda n, k, rs: _donut_clusters(n, k, rs),
        "best_for": "DBSCAN, Spectral"
    },
    "blob_in_circle": {
        "name": "🎯 Blob in Circle",
        "desc": "Central blob surrounded by ring",
        "generator": lambda n, k, rs: _blob_in_circle(n, rs),
        "best_for": "DBSCAN, Spectral"
    },
    "stripes": {
        "name": "📊 Stripe Pattern",
        "desc": "Parallel linear cluster bands",
        "generator": lambda n, k, rs: _stripe_pattern(n, k, rs),
        "best_for": "Spectral, OPTICS"
    },
    "random_uniform": {
        "name": "🎲 Uniform Random",
        "desc": "No structure (baseline)",
        "generator": lambda n, k, rs: np.random.RandomState(rs).uniform(-5, 5, (n, 2)),
        "best_for": "Testing robustness"
    },
}

# ─────────────────────────────────────────────────────────────────────
# HELPER GENERATORS (Custom patterns)
# ─────────────────────────────────────────────────────────────────────

def _noisy_blobs(n: int, k: int, rs: int) -> np.ndarray:
    """Tight blobs + scattered noise."""
    X, _ = make_blobs(n_samples=int(n*0.8), centers=k, cluster_std=0.3, random_state=rs)
    noise = np.random.RandomState(rs+1).uniform(-5, 5, (int(n*0.2), 2))
    return np.vstack([X, noise])

def _outlier_blobs(n: int, k: int, rs: int, outlier_ratio: float = 0.1) -> np.ndarray:
    """Blobs with controlled outlier percentage."""
    X, _ = make_blobs(n_samples=int(n*(1-outlier_ratio)), centers=k, 
                      cluster_std=0.4, random_state=rs)
    outliers = np.random.RandomState(rs+1).uniform(-8, 8, (int(n*outlier_ratio), 2))
    return np.vstack([X, outliers])

def _nested_circles(n: int, k: int, rs: int) -> np.ndarray:
    """Multiple concentric circles."""
    rs_gen = np.random.RandomState(rs)
    theta = rs_gen.uniform(0, 2*np.pi, n)
    radius_idx = rs_gen.randint(0, k, n)
    radii = np.linspace(1, 4, k)[radius_idx]
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    noise = rs_gen.normal(0, 0.15, (n, 2))
    return np.column_stack([x, y]) + noise

def _hierarchical_blobs(n: int, k: int, rs: int) -> np.ndarray:
    """Multi-scale hierarchical blobs."""
    rs_gen = np.random.RandomState(rs)
    # Super-clusters
    super_centers = rs_gen.uniform(-8, 8, (k, 2))
    per_super = n // k
    
    X_list = []
    for center in super_centers:
        # Sub-clusters within each super-cluster
        X_sub, _ = make_blobs(
            n_samples=per_super, centers=2, cluster_std=0.3,
            center_box=np.column_stack([center-1.5, center+1.5]), 
            random_state=rs_gen.randint(0, 10000)
        )
        X_list.append(X_sub)
    return np.vstack(X_list)[:n]

def _density_varied_blobs(n: int, k: int, rs: int) -> np.ndarray:
    """Clusters with different point densities."""
    rs_gen = np.random.RandomState(rs)
    X_list = []
    for i in range(k):
        center = rs_gen.uniform(-6, 6, 2)
        n_i = int(n * (2**(i-k/2)) / (2**(k/2)))  # Exponential distribution
        std = 0.3 + 0.2*i
        X_i = rs_gen.normal(center, std, (n_i, 2))
        X_list.append(X_i)
    return np.vstack(X_list)[:n]

def _elongated_blobs(n: int, k: int, rs: int) -> np.ndarray:
    """Long thin cluster shapes."""
    rs_gen = np.random.RandomState(rs)
    X_list = []
    for i in range(k):
        center = rs_gen.uniform(-6, 6, 2)
        angle = rs_gen.uniform(0, 2*np.pi)
        # Stretched along direction
        t = rs_gen.uniform(-1, 1, n//k)
        s = rs_gen.normal(0, 0.15, n//k)
        x = center[0] + 2*t*np.cos(angle) + 0.2*s*np.sin(angle)
        y = center[1] + 2*t*np.sin(angle) - 0.2*s*np.cos(angle)
        X_list.append(np.column_stack([x, y]))
    return np.vstack(X_list)[:n]

def _overlapped_blobs(n: int, k: int, rs: int, overlap: float = 0.5) -> np.ndarray:
    """Blobs with controlled overlap."""
    X, _ = make_blobs(n_samples=n, centers=k, random_state=rs)
    # Compress to increase overlap
    X *= (1 - overlap*0.5)
    return X

def _rotated_moons(n: int, rs: int) -> np.ndarray:
    """Rotated crescent moons."""
    X, _ = make_moons(n_samples=n, noise=0.05, random_state=rs)
    # Rotate 45 degrees
    angle = np.pi/4
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    return X @ rot.T

def _donut_clusters(n: int, k: int, rs: int) -> np.ndarray:
    """Ring-shaped clusters."""
    rs_gen = np.random.RandomState(rs)
    centers = rs_gen.uniform(-6, 6, (k, 2))
    X_list = []
    for center in centers:
        theta = rs_gen.uniform(0, 2*np.pi, n//k)
        r = rs_gen.normal(2, 0.3, n//k)
        x = center[0] + r*np.cos(theta) + rs_gen.normal(0, 0.1, n//k)
        y = center[1] + r*np.sin(theta) + rs_gen.normal(0, 0.1, n//k)
        X_list.append(np.column_stack([x, y]))
    return np.vstack(X_list)[:n]

def _blob_in_circle(n: int, rs: int) -> np.ndarray:
    """Central blob surrounded by ring."""
    rs_gen = np.random.RandomState(rs)
    # Inner blob
    X_inner = rs_gen.normal([0, 0], 0.4, (int(n*0.3), 2))
    # Outer ring
    theta = rs_gen.uniform(0, 2*np.pi, int(n*0.7))
    r = rs_gen.normal(3, 0.3, int(n*0.7))
    x = r*np.cos(theta) + rs_gen.normal(0, 0.15, int(n*0.7))
    y = r*np.sin(theta) + rs_gen.normal(0, 0.15, int(n*0.7))
    X_outer = np.column_stack([x, y])
    return np.vstack([X_inner, X_outer])[:n]

def _stripe_pattern(n: int, k: int, rs: int) -> np.ndarray:
    """Parallel linear cluster bands."""
    rs_gen = np.random.RandomState(rs)
    X_list = []
    for i in range(k):
        y = np.ones(n//k) * (i - k/2) * 0.8
        x = rs_gen.uniform(-5, 5, n//k)
        noise = rs_gen.normal(0, 0.2, (n//k, 2))
        X_list.append(np.column_stack([x, y]) + noise)
    return np.vstack(X_list)[:n]

# ─────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────

def get_pattern_names() -> Dict[str, str]:
    """Get all available pattern IDs and display names."""
    return {k: v["name"] for k, v in PATTERN_CATALOG.items()}

def get_pattern_info(pattern_id: str) -> Dict:
    """Get full info about a pattern (name, description, best-for)."""
    return PATTERN_CATALOG.get(pattern_id, {})

def generate_pattern(
    pattern_id: str,
    n_samples: int = 300,
    n_clusters: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Generate a synthetic clustering pattern on-demand.
    
    Args:
        pattern_id: Key from PATTERN_CATALOG
        n_samples: Number of points to generate
        n_clusters: Number of clusters (ignored for some patterns)
        random_state: Random seed for reproducibility
    
    Returns:
        (X, metadata) where X is (n_samples, 2) array, metadata is pattern info
    """
    if pattern_id not in PATTERN_CATALOG:
        raise ValueError(f"Unknown pattern: {pattern_id}")
    
    pattern = PATTERN_CATALOG[pattern_id]
    generator = pattern["generator"]
    
    # Call generator (some return just X, others return (X, y))
    raw_output = generator(n_samples, n_clusters, random_state)
    
    # Extract X if the generator returned a tuple
    if isinstance(raw_output, tuple):
        X = raw_output[0]
    else:
        X = raw_output
    
    # Ensure output shape
    X = np.array(X).reshape(-1, 2)
    
    # Return with metadata
    return X, {
        "pattern_id": pattern_id,
        "pattern_name": pattern["name"],
        "description": pattern["desc"],
        "best_for": pattern["best_for"],
        "n_samples": X.shape[0],
        "n_features": 2,
    }

def generate_pattern_dataframe(
    pattern_id: str,
    n_samples: int = 300,
    n_clusters: int = 5,
    random_state: int = 42,
    feature_names: list = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate pattern and return as DataFrame for direct Streamlit use.
    
    Args:
        pattern_id: Pattern key
        n_samples: Number of samples
        n_clusters: Number of clusters
        random_state: Random seed
        feature_names: Column names (default: ['feature_1', 'feature_2'])
    
    Returns:
        (df, metadata)
    """
    X, metadata = generate_pattern(pattern_id, n_samples, n_clusters, random_state)
    
    if feature_names is None:
        feature_names = ['feature_1', 'feature_2']
    
    df = pd.DataFrame(X, columns=feature_names[:2])
    return df, metadata

# ─────────────────────────────────────────────────────────────────────
# STREAMLIT INTEGRATION SNIPPET
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
        
        # Show pattern info
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

if __name__ == "__main__":
    # Quick demo
    X, meta = generate_pattern("moons", n_samples=200)
    print(f"✅ Generated {meta['pattern_name']}")
    print(f"Shape: {X.shape}, Memory: ~{X.nbytes/1024:.1f} KB")

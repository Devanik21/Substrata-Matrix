"""
visualization.py — ClusterX Advanced Visualization Engine
============================================================
Handles: dimensionality reduction (PCA/UMAP/t-SNE), 2D/3D scatter,
         silhouette plots, elbow curves, radar charts, heatmaps,
         dendrograms, parallel coordinates, cluster distributions,
         consensus matrix heatmaps, and gap statistic plots.
Author: ClusterX Intelligence Lab
"""

from __future__ import annotations

import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# COLOR PALETTES
# ──────────────────────────────────────────────────────────────────

DARK_BG = "#0a0a0f"
CARD_BG = "#12121a"
GRID_COLOR = "#1e1e2e"
TEXT_COLOR = "#e0e0e8"
ACCENT_CYAN = "#00f0ff"
ACCENT_MAGENTA = "#ff00aa"
ACCENT_GOLD = "#ffd700"

CLUSTER_PALETTE = [
    "#00f0ff", "#ff00aa", "#ffd700", "#00ff88", "#ff6633",
    "#aa77ff", "#ff4466", "#33ddff", "#ffaa00", "#88ff44",
    "#ff77cc", "#44ffdd", "#ff9955", "#77aaff", "#ddff33",
    "#ff5577", "#55ffaa", "#ffcc33", "#9966ff", "#33ff99",
    "#ff3388", "#66ddff", "#ffdd55", "#7744ff", "#44ff77",
    "#ff2266", "#22ffcc", "#ff8844", "#5533ff", "#33ffbb",
]

NOISE_COLOR = "#333344"

DARK_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="Space Grotesk, sans-serif", color=TEXT_COLOR, size=12),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=GRID_COLOR),
    )
)


def _apply_dark(fig: go.Figure) -> go.Figure:
    fig.update_layout(**DARK_TEMPLATE["layout"])
    return fig


def _get_color(idx: int) -> str:
    if idx < 0:
        return NOISE_COLOR
    return CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]


# ──────────────────────────────────────────────────────────────────
# DIMENSIONALITY REDUCTION
# ──────────────────────────────────────────────────────────────────

class DimReducer:
    """Reduces data to 2D or 3D for visualization."""

    @staticmethod
    def pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, float]:
        n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        X_red = pca.fit_transform(X)
        var_explained = float(pca.explained_variance_ratio_.sum()) * 100
        return X_red, var_explained

    @staticmethod
    def tsne(X: np.ndarray, n_components: int = 2,
             perplexity: float = 30.0) -> np.ndarray:
        n = len(X)
        perp = min(perplexity, max(5, n // 5))
        tsne = TSNE(n_components=n_components, perplexity=perp,
                     random_state=42, n_iter=1000, init="pca",
                     learning_rate="auto")
        return tsne.fit_transform(X)

    @staticmethod
    def umap(X: np.ndarray, n_components: int = 2,
             n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components,
                                n_neighbors=min(n_neighbors, len(X) - 1),
                                min_dist=min_dist, random_state=42)
            return reducer.fit_transform(X)
        except ImportError:
            logger.warning("UMAP not installed, falling back to PCA")
            return DimReducer.pca(X, n_components)[0]

    @staticmethod
    def reduce(X: np.ndarray, method: str = "pca",
               n_components: int = 2, **kwargs) -> np.ndarray:
        if method == "pca":
            return DimReducer.pca(X, n_components)[0]
        elif method == "tsne":
            return DimReducer.tsne(X, n_components, **kwargs)
        elif method == "umap":
            return DimReducer.umap(X, n_components, **kwargs)
        return DimReducer.pca(X, n_components)[0]


# ──────────────────────────────────────────────────────────────────
# SCATTER PLOTS (2D & 3D)
# ──────────────────────────────────────────────────────────────────

class ScatterPlotter:
    """Creates 2D and 3D scatter plots with cluster coloring."""

    @staticmethod
    def scatter_2d(X_2d: np.ndarray, labels: np.ndarray,
                   title: str = "Cluster Scatter (2D)",
                   axis_labels: Tuple[str, str] = ("Component 1", "Component 2"),
                   point_size: int = 5, opacity: float = 0.7,
                   show_centroids: bool = True) -> go.Figure:
        fig = go.Figure()
        unique = sorted(set(labels))
        for c in unique:
            mask = labels == c
            color = _get_color(c)
            name = "Noise" if c == -1 else f"Cluster {c}"
            fig.add_trace(go.Scattergl(
                x=X_2d[mask, 0], y=X_2d[mask, 1],
                mode="markers",
                marker=dict(size=point_size, color=color, opacity=opacity,
                            line=dict(width=0.3, color="rgba(255,255,255,0.2)")),
                name=name,
                hovertemplate=f"{name}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>",
            ))
            if show_centroids and c >= 0:
                cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
                fig.add_trace(go.Scatter(
                    x=[cx], y=[cy], mode="markers",
                    marker=dict(size=14, color=color, symbol="x",
                                line=dict(width=2, color="white")),
                    name=f"Centroid {c}", showlegend=False,
                ))
        fig.update_layout(title=title, xaxis_title=axis_labels[0],
                          yaxis_title=axis_labels[1])
        return _apply_dark(fig)

    @staticmethod
    def scatter_3d(X_3d: np.ndarray, labels: np.ndarray,
                   title: str = "Cluster Scatter (3D)",
                   point_size: int = 3, opacity: float = 0.6) -> go.Figure:
        fig = go.Figure()
        unique = sorted(set(labels))
        for c in unique:
            mask = labels == c
            color = _get_color(c)
            name = "Noise" if c == -1 else f"Cluster {c}"
            fig.add_trace(go.Scatter3d(
                x=X_3d[mask, 0], y=X_3d[mask, 1], z=X_3d[mask, 2],
                mode="markers",
                marker=dict(size=point_size, color=color, opacity=opacity),
                name=name,
            ))
        fig.update_layout(title=title,
                          scene=dict(
                              xaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID_COLOR),
                              yaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID_COLOR),
                              zaxis=dict(backgroundcolor=CARD_BG, gridcolor=GRID_COLOR),
                          ))
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# SILHOUETTE PLOT
# ──────────────────────────────────────────────────────────────────

class SilhouettePlotter:

    @staticmethod
    def plot(X: np.ndarray, labels: np.ndarray,
             title: str = "Silhouette Analysis",
             sample_size: int = 5000) -> go.Figure:
        n = min(sample_size, len(X))
        if n < len(X):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), n, replace=False)
            X_s, labels_s = X[idx], labels[idx]
        else:
            X_s, labels_s = X, labels
        try:
            sil_vals = silhouette_samples(X_s, labels_s)
            sil_avg = float(sil_vals.mean())
        except Exception:
            return go.Figure().update_layout(title="Silhouette computation failed")

        fig = go.Figure()
        unique = sorted([l for l in set(labels_s) if l >= 0])
        y_lower = 0
        for c in unique:
            c_sil = np.sort(sil_vals[labels_s == c])
            y_upper = y_lower + len(c_sil)
            color = _get_color(c)
            fig.add_trace(go.Bar(
                x=c_sil, y=list(range(y_lower, y_upper)),
                orientation="h", marker_color=color,
                name=f"Cluster {c} ({len(c_sil)})",
                hovertemplate="Silhouette: %{x:.3f}<extra></extra>",
            ))
            y_lower = y_upper + 2

        fig.add_vline(x=sil_avg, line_dash="dash", line_color=ACCENT_GOLD,
                       annotation_text=f"Avg: {sil_avg:.3f}")
        fig.update_layout(title=title, xaxis_title="Silhouette Coefficient",
                          yaxis_title="Sample", showlegend=True,
                          bargap=0, yaxis=dict(showticklabels=False))
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# ELBOW & GAP STATISTIC PLOTS
# ──────────────────────────────────────────────────────────────────

class ElbowPlotter:

    @staticmethod
    def plot_elbow(k_values: List[int], inertias: List[float],
                   optimal_k: Optional[int] = None,
                   title: str = "Elbow Method") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=inertias, mode="lines+markers",
            line=dict(color=ACCENT_CYAN, width=2),
            marker=dict(size=8, color=ACCENT_CYAN),
            name="Inertia",
        ))
        if optimal_k is not None:
            idx = k_values.index(optimal_k) if optimal_k in k_values else 0
            fig.add_trace(go.Scatter(
                x=[optimal_k], y=[inertias[idx]], mode="markers",
                marker=dict(size=16, color=ACCENT_GOLD, symbol="star"),
                name=f"Optimal k={optimal_k}",
            ))
        fig.update_layout(title=title, xaxis_title="Number of Clusters (k)",
                          yaxis_title="Inertia (WCSS)")
        return _apply_dark(fig)

    @staticmethod
    def plot_silhouette_curve(k_values: List[int], scores: List[float],
                               optimal_k: Optional[int] = None,
                               title: str = "Silhouette vs k") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=scores, mode="lines+markers",
            line=dict(color=ACCENT_MAGENTA, width=2),
            marker=dict(size=8, color=ACCENT_MAGENTA),
            name="Silhouette",
        ))
        if optimal_k is not None and optimal_k in k_values:
            idx = k_values.index(optimal_k)
            fig.add_trace(go.Scatter(
                x=[optimal_k], y=[scores[idx]], mode="markers",
                marker=dict(size=16, color=ACCENT_GOLD, symbol="star"),
                name=f"Optimal k={optimal_k}",
            ))
        fig.update_layout(title=title, xaxis_title="Number of Clusters (k)",
                          yaxis_title="Silhouette Score")
        return _apply_dark(fig)

    @staticmethod
    def plot_gap_statistic(k_values: List[int], gaps: List[float],
                           gap_stds: List[float], optimal_k: int,
                           title: str = "Gap Statistic") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=gaps, mode="lines+markers",
            line=dict(color=ACCENT_CYAN, width=2),
            marker=dict(size=8), name="Gap",
            error_y=dict(type="data", array=gap_stds, visible=True,
                         color="rgba(0,240,255,0.3)"),
        ))
        if optimal_k in k_values:
            idx = k_values.index(optimal_k)
            fig.add_trace(go.Scatter(
                x=[optimal_k], y=[gaps[idx]], mode="markers",
                marker=dict(size=16, color=ACCENT_GOLD, symbol="star"),
                name=f"Optimal k={optimal_k}",
            ))
        fig.update_layout(title=title, xaxis_title="k",
                          yaxis_title="Gap Statistic")
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# RADAR / COMPARISON CHART
# ──────────────────────────────────────────────────────────────────

class RadarPlotter:

    @staticmethod
    def plot_comparison(algo_names: List[str],
                        metrics_dict: Dict[str, Dict[str, float]],
                        title: str = "Algorithm Comparison") -> go.Figure:
        if not metrics_dict:
            return go.Figure().update_layout(title="No data")
        metric_names = list(next(iter(metrics_dict.values())).keys())
        fig = go.Figure()
        for i, algo in enumerate(algo_names):
            if algo not in metrics_dict:
                continue
            vals = [metrics_dict[algo].get(m, 0) for m in metric_names]
            vals.append(vals[0])
            cats = metric_names + [metric_names[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                name=algo, line=dict(color=_get_color(i), width=2),
                fillcolor=f"rgba({int(_get_color(i)[1:3],16)},{int(_get_color(i)[3:5],16)},{int(_get_color(i)[5:7],16)},0.15)",
            ))
        fig.update_layout(
            title=title,
            polar=dict(
                bgcolor=CARD_BG,
                radialaxis=dict(gridcolor=GRID_COLOR, color=TEXT_COLOR),
                angularaxis=dict(gridcolor=GRID_COLOR, color=TEXT_COLOR),
            ),
        )
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# HEATMAPS
# ──────────────────────────────────────────────────────────────────

class HeatmapPlotter:

    @staticmethod
    def correlation_heatmap(corr: pd.DataFrame,
                            title: str = "Feature Correlation") -> go.Figure:
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0, "#0a0a3a"], [0.5, "#1a1a4a"], [1, ACCENT_CYAN]],
            zmin=-1, zmax=1,
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(title=title, height=max(400, len(corr) * 20))
        return _apply_dark(fig)

    @staticmethod
    def consensus_heatmap(consensus: np.ndarray,
                          title: str = "Consensus Matrix") -> go.Figure:
        fig = go.Figure(data=go.Heatmap(
            z=consensus,
            colorscale=[[0, "#0a0a1a"], [0.5, "#1a1a5a"], [1, ACCENT_MAGENTA]],
            zmin=0, zmax=1,
            hovertemplate="i=%{x}, j=%{y}: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(title=title, xaxis_title="Sample", yaxis_title="Sample",
                          height=600, width=600)
        return _apply_dark(fig)

    @staticmethod
    def missing_value_heatmap(df: pd.DataFrame,
                              title: str = "Missing Values") -> go.Figure:
        missing = df.isnull().astype(int)
        if len(missing) > 500:
            missing = missing.sample(500, random_state=42).sort_index()
        fig = go.Figure(data=go.Heatmap(
            z=missing.values.T,
            x=list(range(len(missing))),
            y=missing.columns.tolist(),
            colorscale=[[0, CARD_BG], [1, "#ff4466"]],
            showscale=False,
        ))
        fig.update_layout(title=title, xaxis_title="Row", yaxis_title="Feature")
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# DENDROGRAM
# ──────────────────────────────────────────────────────────────────

class DendrogramPlotter:

    @staticmethod
    def plot(X: np.ndarray, labels: Optional[np.ndarray] = None,
             method: str = "ward", max_samples: int = 200,
             title: str = "Hierarchical Dendrogram") -> go.Figure:
        from scipy.cluster.hierarchy import linkage, dendrogram
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(max_samples, len(X))
        if n < len(X):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), n, replace=False)
            X_s = X[idx]
        else:
            X_s = X

        Z = linkage(X_s, method=method)
        fig_mpl, ax = plt.subplots(figsize=(12, 4))
        dend = dendrogram(Z, ax=ax, no_labels=True, color_threshold=0,
                          above_threshold_color=ACCENT_CYAN)
        plt.close(fig_mpl)

        fig = go.Figure()
        for xs, ys, color in zip(dend["icoord"], dend["dcoord"],
                                  dend["color_list"]):
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=ACCENT_CYAN, width=1),
                showlegend=False, hoverinfo="skip",
            ))
        fig.update_layout(title=title, xaxis_title="Samples",
                          yaxis_title="Distance", height=350)
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# DISTRIBUTION PLOTS
# ──────────────────────────────────────────────────────────────────

class DistributionPlotter:

    @staticmethod
    def cluster_sizes(labels: np.ndarray,
                      title: str = "Cluster Sizes") -> go.Figure:
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        sizes = [(labels == l).sum() for l in unique]
        names = [f"Cluster {l}" for l in unique]
        colors = [_get_color(l) for l in unique]

        noise_n = int((labels == -1).sum())
        if noise_n > 0:
            unique.append(-1)
            sizes.append(noise_n)
            names.append("Noise")
            colors.append(NOISE_COLOR)

        fig = go.Figure(go.Bar(
            x=names, y=sizes, marker_color=colors,
            text=sizes, textposition="auto",
            hovertemplate="%{x}: %{y}<extra></extra>",
        ))
        fig.update_layout(title=title, xaxis_title="Cluster",
                          yaxis_title="Count")
        return _apply_dark(fig)

    @staticmethod
    def feature_boxplots(X: np.ndarray, labels: np.ndarray,
                         feature_names: List[str],
                         feature_idx: int = 0,
                         title: str = "Feature Distribution") -> go.Figure:
        fig = go.Figure()
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        fname = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
        for c in unique:
            vals = X[labels == c, feature_idx]
            fig.add_trace(go.Box(
                y=vals, name=f"Cluster {c}",
                marker_color=_get_color(c),
                boxmean="sd",
            ))
        fig.update_layout(title=f"{title}: {fname}",
                          yaxis_title=fname)
        return _apply_dark(fig)

    @staticmethod
    def parallel_coordinates(X: np.ndarray, labels: np.ndarray,
                             feature_names: List[str],
                             max_features: int = 10,
                             title: str = "Parallel Coordinates") -> go.Figure:
        n_feat = min(max_features, X.shape[1])
        dims = []
        for i in range(n_feat):
            fname = feature_names[i] if i < len(feature_names) else f"F{i}"
            dims.append(dict(label=fname, values=X[:, i]))

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=labels,
                colorscale=[[i / max(len(CLUSTER_PALETTE) - 1, 1), c]
                            for i, c in enumerate(CLUSTER_PALETTE[:len(set(labels))])],
                showscale=False,
            ),
            dimensions=dims,
        ))
        fig.update_layout(title=title)
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# STABILITY VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────

class StabilityPlotter:

    @staticmethod
    def perturbation_curve(noise_levels: List[float],
                           mean_aris: List[float],
                           std_aris: List[float],
                           title: str = "Perturbation Robustness") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=noise_levels, y=mean_aris, mode="lines+markers",
            line=dict(color=ACCENT_CYAN, width=2),
            marker=dict(size=8), name="Mean ARI",
            error_y=dict(type="data", array=std_aris, visible=True,
                         color="rgba(0,240,255,0.3)"),
        ))
        fig.add_hline(y=0.8, line_dash="dash", line_color=ACCENT_GOLD,
                       annotation_text="Stability threshold")
        fig.update_layout(title=title, xaxis_title="Noise Level (σ)",
                          yaxis_title="ARI vs Original")
        return _apply_dark(fig)

    @staticmethod
    def consensus_cdf(cdf_x: np.ndarray, cdf_values: np.ndarray,
                      title: str = "Consensus CDF") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cdf_x, y=cdf_values, mode="lines",
            line=dict(color=ACCENT_MAGENTA, width=2),
            fill="tozeroy", fillcolor="rgba(255,0,170,0.1)",
            name="CDF",
        ))
        fig.update_layout(title=title, xaxis_title="Consensus Value",
                          yaxis_title="CDF")
        return _apply_dark(fig)

    @staticmethod
    def stability_bars(algo_names: List[str], mean_aris: List[float],
                       grades: List[str],
                       title: str = "Stability Leaderboard") -> go.Figure:
        colors = []
        for ari in mean_aris:
            if ari >= 0.9:
                colors.append("#00ff88")
            elif ari >= 0.7:
                colors.append(ACCENT_CYAN)
            elif ari >= 0.5:
                colors.append(ACCENT_GOLD)
            else:
                colors.append("#ff4466")
        fig = go.Figure(go.Bar(
            x=algo_names, y=mean_aris,
            marker_color=colors,
            text=[f"{a:.3f} ({g})" for a, g in zip(mean_aris, grades)],
            textposition="auto",
        ))
        fig.add_hline(y=0.7, line_dash="dash", line_color=ACCENT_GOLD,
                       annotation_text="Stable threshold")
        fig.update_layout(title=title, xaxis_title="Algorithm",
                          yaxis_title="Mean Bootstrap ARI")
        return _apply_dark(fig)

    @staticmethod
    def feature_importance_bars(feature_names: List[str],
                                dropout_scores: Dict[int, float],
                                title: str = "Feature Importance (Dropout)") -> go.Figure:
        indices = sorted(dropout_scores.keys())
        names = [feature_names[i] if i < len(feature_names) else f"F{i}"
                 for i in indices]
        scores = [1.0 - dropout_scores[i] for i in indices]
        sorted_pairs = sorted(zip(names, scores), key=lambda x: -x[1])
        names_s = [p[0] for p in sorted_pairs]
        scores_s = [p[1] for p in sorted_pairs]

        fig = go.Figure(go.Bar(
            x=scores_s, y=names_s, orientation="h",
            marker_color=ACCENT_CYAN,
        ))
        fig.update_layout(title=title, xaxis_title="Importance (1 - ARI drop)",
                          yaxis_title="Feature", height=max(300, len(names) * 25))
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# METRIC GAUGE
# ──────────────────────────────────────────────────────────────────

class GaugePlotter:

    @staticmethod
    def metric_gauge(value: float, title: str = "Score",
                     min_val: float = -1, max_val: float = 1,
                     thresholds: Optional[List[float]] = None) -> go.Figure:
        if thresholds is None:
            thresholds = [0.3, 0.6, 0.8]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"color": TEXT_COLOR}},
            gauge={
                "axis": {"range": [min_val, max_val],
                         "tickcolor": TEXT_COLOR},
                "bar": {"color": ACCENT_CYAN},
                "bgcolor": CARD_BG,
                "steps": [
                    {"range": [min_val, thresholds[0]], "color": "#331111"},
                    {"range": [thresholds[0], thresholds[1]], "color": "#333311"},
                    {"range": [thresholds[1], thresholds[2]], "color": "#113311"},
                    {"range": [thresholds[2], max_val], "color": "#115511"},
                ],
                "threshold": {
                    "line": {"color": ACCENT_GOLD, "width": 3},
                    "thickness": 0.8, "value": value,
                },
            },
            number={"font": {"color": TEXT_COLOR}},
        ))
        fig.update_layout(height=250)
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# PCA VARIANCE EXPLAINED
# ──────────────────────────────────────────────────────────────────

class PCAPlotter:
    """Plotly charts for PCA analysis."""

    @staticmethod
    def variance_explained(X: np.ndarray, max_components: int = 20,
                           title: str = "PCA Variance Explained") -> go.Figure:
        """Scree plot of PCA explained variance."""
        n_comp = min(max_components, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(X)
        var_ratio = pca.explained_variance_ratio_
        cumulative = np.cumsum(var_ratio)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=list(range(1, n_comp + 1)),
            y=var_ratio * 100,
            marker_color=ACCENT_CYAN,
            name="Individual %",
            opacity=0.7,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=list(range(1, n_comp + 1)),
            y=cumulative * 100,
            mode="lines+markers",
            line=dict(color=ACCENT_MAGENTA, width=2),
            marker=dict(size=6),
            name="Cumulative %",
        ), secondary_y=True)
        fig.add_hline(y=90, line_dash="dash", line_color=ACCENT_GOLD,
                       annotation_text="90% threshold", secondary_y=True)
        fig.update_layout(
            title=title,
            xaxis_title="Principal Component",
            yaxis_title="Variance Explained (%)",
            yaxis2_title="Cumulative (%)",
        )
        return _apply_dark(fig)

    @staticmethod
    def biplot(X: np.ndarray, labels: np.ndarray,
               feature_names: Optional[List[str]] = None,
               title: str = "PCA Biplot") -> go.Figure:
        """PCA biplot showing samples and feature loading vectors."""
        n_comp = min(2, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        fig = go.Figure()
        unique = sorted(set(labels))
        for c in unique:
            mask = labels == c
            color = _get_color(c)
            name = "Noise" if c == -1 else f"Cluster {c}"
            fig.add_trace(go.Scattergl(
                x=X_pca[mask, 0], y=X_pca[mask, 1],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.5),
                name=name,
            ))

        scale_factor = np.abs(X_pca).max() * 0.8 / max(np.abs(loadings).max(), 1e-16)
        n_feat = min(X.shape[1], 15)
        for i in range(n_feat):
            fname = feature_names[i] if feature_names and i < len(feature_names) else f"F{i}"
            fig.add_annotation(
                x=loadings[i, 0] * scale_factor,
                y=loadings[i, 1] * scale_factor,
                ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor=ACCENT_GOLD,
                text=fname,
                font=dict(size=9, color=ACCENT_GOLD),
            )

        var_exp = pca.explained_variance_ratio_ * 100
        fig.update_layout(
            title=title,
            xaxis_title=f"PC1 ({var_exp[0]:.1f}%)",
            yaxis_title=f"PC2 ({var_exp[1]:.1f}%)" if n_comp > 1 else "PC2",
        )
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# VIOLIN PLOTS
# ──────────────────────────────────────────────────────────────────

class ViolinPlotter:
    """Violin plots for feature distributions across clusters."""

    @staticmethod
    def plot(X: np.ndarray, labels: np.ndarray,
             feature_names: List[str],
             feature_idx: int = 0,
             title: str = "Feature Violin Plot") -> go.Figure:
        fig = go.Figure()
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        fname = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
        for c in unique:
            vals = X[labels == c, feature_idx]
            fig.add_trace(go.Violin(
                y=vals, name=f"Cluster {c}",
                line_color=_get_color(c),
                fillcolor=f"rgba({int(_get_color(c)[1:3],16)},{int(_get_color(c)[3:5],16)},{int(_get_color(c)[5:7],16)},0.3)",
                box_visible=True,
                meanline_visible=True,
                points="outliers",
            ))
        fig.update_layout(title=f"{title}: {fname}", yaxis_title=fname)
        return _apply_dark(fig)

    @staticmethod
    def multi_feature(X: np.ndarray, labels: np.ndarray,
                      feature_names: List[str],
                      max_features: int = 6,
                      title: str = "Multi-Feature Violins") -> go.Figure:
        """Side-by-side violin plots for multiple features."""
        n_feat = min(max_features, X.shape[1])
        fig = make_subplots(rows=1, cols=n_feat,
                            subplot_titles=[feature_names[i] if i < len(feature_names) else f"F{i}"
                                            for i in range(n_feat)])
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        for fi in range(n_feat):
            for c in unique:
                vals = X[labels == c, fi]
                fig.add_trace(go.Violin(
                    y=vals, name=f"C{c}",
                    line_color=_get_color(c),
                    showlegend=(fi == 0),
                    scalemode="width",
                    meanline_visible=True,
                ), row=1, col=fi + 1)
        fig.update_layout(title=title, height=400)
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# PAIR PLOT (Scatter Matrix)
# ──────────────────────────────────────────────────────────────────

class PairPlotter:
    """Creates scatter matrix / pair plot for selected features."""

    @staticmethod
    def plot(X: np.ndarray, labels: np.ndarray,
             feature_names: List[str],
             max_features: int = 5,
             sample_size: int = 1000,
             title: str = "Pair Plot") -> go.Figure:
        n_feat = min(max_features, X.shape[1])
        n = min(sample_size, len(X))
        if n < len(X):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), n, replace=False)
            X_s, labels_s = X[idx], labels[idx]
        else:
            X_s, labels_s = X, labels

        fnames = [feature_names[i] if i < len(feature_names) else f"F{i}"
                  for i in range(n_feat)]
        fig = make_subplots(rows=n_feat, cols=n_feat,
                            shared_xaxes=True, shared_yaxes=True,
                            vertical_spacing=0.02, horizontal_spacing=0.02)

        unique = sorted([l for l in np.unique(labels_s) if l >= 0])
        for row in range(n_feat):
            for col in range(n_feat):
                for c in unique:
                    mask = labels_s == c
                    color = _get_color(c)
                    if row == col:
                        fig.add_trace(go.Histogram(
                            x=X_s[mask, col], marker_color=color,
                            opacity=0.5, showlegend=False, nbinsx=20,
                        ), row=row + 1, col=col + 1)
                    else:
                        fig.add_trace(go.Scattergl(
                            x=X_s[mask, col], y=X_s[mask, row],
                            mode="markers",
                            marker=dict(size=2, color=color, opacity=0.4),
                            showlegend=False,
                        ), row=row + 1, col=col + 1)

        for i, fn in enumerate(fnames):
            fig.update_xaxes(title_text=fn, row=n_feat, col=i + 1)
            fig.update_yaxes(title_text=fn, row=i + 1, col=1)

        fig.update_layout(title=title, height=150 * n_feat, width=150 * n_feat)
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# SUNBURST CHART
# ──────────────────────────────────────────────────────────────────

class SunburstPlotter:
    """Sunburst chart showing hierarchical cluster composition."""

    @staticmethod
    def plot(labels: np.ndarray, secondary_labels: Optional[np.ndarray] = None,
             title: str = "Cluster Composition") -> go.Figure:
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        ids = ["All"]
        parents = [""]
        values = [len(labels)]
        colors = [CARD_BG]

        for c in unique:
            c_name = f"Cluster {c}"
            ids.append(c_name)
            parents.append("All")
            values.append(int((labels == c).sum()))
            colors.append(_get_color(c))

            if secondary_labels is not None:
                mask = labels == c
                sec_unique = sorted(np.unique(secondary_labels[mask]))
                for s in sec_unique:
                    sub_name = f"C{c}-S{s}"
                    ids.append(sub_name)
                    parents.append(c_name)
                    values.append(int(((labels == c) & (secondary_labels == s)).sum()))
                    colors.append(_get_color(int(s)))

        noise_n = int((labels == -1).sum())
        if noise_n > 0:
            ids.append("Noise")
            parents.append("All")
            values.append(noise_n)
            colors.append(NOISE_COLOR)

        fig = go.Figure(go.Sunburst(
            ids=ids, labels=ids, parents=parents, values=values,
            marker=dict(colors=colors),
            branchvalues="total",
        ))
        fig.update_layout(title=title, height=500)
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# CONVERGENCE TRACE PLOT
# ──────────────────────────────────────────────────────────────────

class ConvergencePlotter:
    """Plots convergence diagnostics from KMeans step-by-step analysis."""

    @staticmethod
    def inertia_trace(inertia_values: List[float],
                      title: str = "KMeans Convergence") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(inertia_values) + 1)),
            y=inertia_values,
            mode="lines+markers",
            line=dict(color=ACCENT_CYAN, width=2),
            marker=dict(size=6),
            name="Inertia",
        ))
        fig.update_layout(title=title, xaxis_title="Iteration",
                          yaxis_title="Inertia (WCSS)")
        return _apply_dark(fig)

    @staticmethod
    def center_drift_trace(drift_values: List[float],
                           title: str = "Center Drift") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(drift_values) + 1)),
            y=drift_values,
            mode="lines+markers",
            line=dict(color=ACCENT_MAGENTA, width=2),
            marker=dict(size=6),
            name="Center Drift",
            fill="tozeroy",
            fillcolor="rgba(255,0,170,0.1)",
        ))
        fig.add_hline(y=1e-6, line_dash="dash", line_color=ACCENT_GOLD,
                       annotation_text="Convergence threshold")
        fig.update_layout(title=title, xaxis_title="Iteration",
                          yaxis_title="Total Center Displacement",
                          yaxis_type="log")
        return _apply_dark(fig)

    @staticmethod
    def label_changes_trace(changes: List[int],
                            title: str = "Label Reassignments") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, len(changes) + 1)),
            y=changes,
            marker_color=ACCENT_CYAN,
            name="Samples Reassigned",
        ))
        fig.update_layout(title=title, xaxis_title="Iteration",
                          yaxis_title="# Reassigned")
        return _apply_dark(fig)

    @staticmethod
    def multi_init_inertias(inertias: List[float],
                            title: str = "Multi-Init Inertia Distribution") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Init {i+1}" for i in range(len(inertias))],
            y=inertias,
            marker_color=[ACCENT_CYAN if v == min(inertias) else GRID_COLOR
                          for v in inertias],
        ))
        fig.add_hline(y=min(inertias), line_dash="dash", line_color=ACCENT_GOLD,
                       annotation_text=f"Best: {min(inertias):.1f}")
        fig.update_layout(title=title, yaxis_title="Inertia")
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# METRICS TABLE PLOT
# ──────────────────────────────────────────────────────────────────

class MetricsTablePlotter:
    """Renders a metrics comparison as a styled Plotly table."""

    @staticmethod
    def plot(metrics_df: pd.DataFrame,
             title: str = "Metrics Comparison") -> go.Figure:
        if metrics_df.empty:
            return go.Figure().update_layout(title="No data available")

        header_colors = [ACCENT_CYAN] * len(metrics_df.columns)
        cell_colors = [[CARD_BG] * len(metrics_df)] * len(metrics_df.columns)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(metrics_df.columns),
                fill_color="#1a1a2e",
                font=dict(color=ACCENT_CYAN, size=12),
                align="center",
                line=dict(color=GRID_COLOR, width=1),
            ),
            cells=dict(
                values=[metrics_df[col] for col in metrics_df.columns],
                fill_color="#12121a",
                font=dict(color=TEXT_COLOR, size=11),
                align="center",
                line=dict(color=GRID_COLOR, width=1),
                height=28,
            ),
        )])
        fig.update_layout(title=title,
                          height=max(250, 50 + 30 * len(metrics_df)))
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# NN DISTANCE PROFILE PLOT
# ──────────────────────────────────────────────────────────────────

class NNDistancePlotter:
    """Plots k-NN distance profiles for DBSCAN epsilon estimation."""

    @staticmethod
    def plot(kth_distances: np.ndarray, suggested_eps: float,
             k: int = 5,
             title: str = "k-NN Distance Profile") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(kth_distances))),
            y=kth_distances,
            mode="lines",
            line=dict(color=ACCENT_CYAN, width=1.5),
            name=f"{k}-NN Distance",
            fill="tozeroy",
            fillcolor="rgba(0,240,255,0.05)",
        ))
        fig.add_hline(y=suggested_eps, line_dash="dash", line_color=ACCENT_GOLD,
                       annotation_text=f"Suggested ε = {suggested_eps:.4f}")
        fig.update_layout(
            title=title,
            xaxis_title="Points (sorted)",
            yaxis_title=f"{k}-NN Distance",
        )
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# HOPKINS GAUGE
# ──────────────────────────────────────────────────────────────────

class HopkinsPlotter:
    """Gauge chart for Hopkins clusterability statistic."""

    @staticmethod
    def plot(hopkins_value: float,
             title: str = "Hopkins Clusterability") -> go.Figure:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hopkins_value,
            title={"text": title, "font": {"color": TEXT_COLOR}},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": TEXT_COLOR},
                "bar": {"color": ACCENT_CYAN},
                "bgcolor": CARD_BG,
                "steps": [
                    {"range": [0, 0.45], "color": "#331111"},
                    {"range": [0.45, 0.6], "color": "#333311"},
                    {"range": [0.6, 0.75], "color": "#113311"},
                    {"range": [0.75, 1.0], "color": "#115511"},
                ],
            },
            number={"font": {"color": TEXT_COLOR}},
        ))
        fig.update_layout(height=250)
        return _apply_dark(fig)


# ──────────────────────────────────────────────────────────────────
# CLUSTER OVERLAP HEATMAP
# ──────────────────────────────────────────────────────────────────

class OverlapHeatmapPlotter:
    """Heatmap showing pairwise cluster overlap scores."""

    @staticmethod
    def plot(overlap_matrix: np.ndarray,
             cluster_ids: Optional[List[int]] = None,
             title: str = "Cluster Overlap") -> go.Figure:
        k = overlap_matrix.shape[0]
        labels = [f"C{i}" for i in (cluster_ids or range(k))]
        fig = go.Figure(data=go.Heatmap(
            z=overlap_matrix,
            x=labels, y=labels,
            colorscale=[[0, "#0a0a1a"], [0.5, "#3a1a4a"], [1, "#ff4466"]],
            zmin=0, zmax=1,
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(title=title, height=450, width=500)
        return _apply_dark(fig)

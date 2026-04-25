"""
visualization.py — ClusterX Master Visualization Engine
=========================================================
Produces production-quality Plotly figures for every stage
of the clustering pipeline:

  • 2D/3D scatter plots: PCA, UMAP, t-SNE, ISOMAP
  • Metric comparison charts: radar, bar, heatmap, violin
  • Stability visualisations: ARI surfaces, box plots, timelines
  • Consensus visualisations: co-association heatmap, weight bars
  • Algorithm ranking tables: annotated scorecards
  • k-sweep elbow curves
  • Cluster profile charts: size distribution, silhouette bars
  • Feature importance for clustering
  • Correlation matrices and PairGrid analogues
  • Interactive dendrogram
  • Dark-mode design system throughout

Author: ClusterX Intelligence Lab
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — Dark Neon Theme
# ──────────────────────────────────────────────────────────────────

class Theme:
    BG_DARK       = "#0a0a0f"
    BG_CARD       = "#10101a"
    BG_PANEL      = "#14141e"
    BORDER        = "#1e1e2e"
    TEXT_PRIMARY  = "#e8e8f0"
    TEXT_DIM      = "#8888aa"
    ACCENT_CYAN   = "#00e5ff"
    ACCENT_VIOLET = "#9b59ff"
    ACCENT_GREEN  = "#00ff88"
    ACCENT_PINK   = "#ff4daa"
    ACCENT_ORANGE = "#ff8c00"
    ACCENT_YELLOW = "#ffd700"
    GRID_COLOR    = "#1e1e2e"

    # 24-color palette for clusters (neon-friendly)
    CLUSTER_PALETTE = [
        "#00e5ff", "#9b59ff", "#00ff88", "#ff4daa",
        "#ff8c00", "#ffd700", "#ff3366", "#33ffcc",
        "#aa55ff", "#ff6633", "#00ccff", "#ff99cc",
        "#66ff66", "#ffaa00", "#cc44ff", "#44ccff",
        "#ff5555", "#55ff99", "#ffcc44", "#cc99ff",
        "#ff77bb", "#77ffdd", "#ff9944", "#9977ff",
    ]

    @classmethod
    def cluster_color(cls, idx: int) -> str:
        return cls.CLUSTER_PALETTE[idx % len(cls.CLUSTER_PALETTE)]

    @classmethod
    def plotly_layout(cls, title: str = "", height: int = 550,
                      showlegend: bool = True) -> Dict[str, Any]:
        return dict(
            title=dict(text=title, font=dict(color=cls.TEXT_PRIMARY,
                                              size=16, family="Inter, sans-serif")),
            paper_bgcolor=cls.BG_DARK,
            plot_bgcolor=cls.BG_CARD,
            font=dict(color=cls.TEXT_PRIMARY, family="Inter, sans-serif", size=12),
            height=height,
            showlegend=showlegend,
            legend=dict(
                bgcolor=cls.BG_PANEL,
                bordercolor=cls.BORDER,
                borderwidth=1,
                font=dict(size=11),
            ),
            xaxis=dict(
                gridcolor=cls.GRID_COLOR,
                zerolinecolor=cls.GRID_COLOR,
                linecolor=cls.BORDER,
                tickfont=dict(color=cls.TEXT_DIM),
            ),
            yaxis=dict(
                gridcolor=cls.GRID_COLOR,
                zerolinecolor=cls.GRID_COLOR,
                linecolor=cls.BORDER,
                tickfont=dict(color=cls.TEXT_DIM),
            ),
            margin=dict(l=60, r=40, t=60, b=60),
        )


# ──────────────────────────────────────────────────────────────────
# EMBEDDING ENGINE
# ──────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """Computes 2D/3D embeddings for cluster visualisation."""

    METHODS = ["PCA", "UMAP", "t-SNE", "ISOMAP", "LLE"]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._cache: Dict[str, np.ndarray] = {}

    def embed(self, X: np.ndarray, method: str = "PCA",
              n_components: int = 2, **kwargs) -> np.ndarray:
        cache_key = f"{method}_{n_components}_{X.shape}_{id(X)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        n_comp = min(n_components, X.shape[1])
        if n_comp < 2:
            # Pad with zeros if needed
            Z = np.hstack([X, np.zeros((len(X), 2 - X.shape[1]))])
            self._cache[cache_key] = Z
            return Z

        try:
            if method == "PCA":
                Z = self._pca(X, n_comp, **kwargs)
            elif method == "UMAP":
                Z = self._umap(X, n_comp, **kwargs)
            elif method == "t-SNE":
                Z = self._tsne(X, n_comp, **kwargs)
            elif method == "ISOMAP":
                Z = self._isomap(X, n_comp, **kwargs)
            elif method == "LLE":
                Z = self._lle(X, n_comp, **kwargs)
            else:
                Z = self._pca(X, n_comp)
        except Exception as e:
            logger.warning(f"Embedding {method} failed: {e}; falling back to PCA")
            Z = self._pca(X, n_comp)

        self._cache[cache_key] = Z
        return Z

    def _pca(self, X, n_comp, **kwargs):
        from sklearn.decomposition import PCA
        return PCA(n_components=n_comp, random_state=self.random_state).fit_transform(X)

    def _umap(self, X, n_comp, **kwargs):
        try:
            import umap
            n_neighbors = min(kwargs.get("n_neighbors", 15), len(X) - 1)
            reducer = umap.UMAP(
                n_components=n_comp,
                n_neighbors=n_neighbors,
                min_dist=kwargs.get("min_dist", 0.1),
                random_state=self.random_state,
            )
            return reducer.fit_transform(X)
        except ImportError:
            logger.warning("UMAP not installed; falling back to t-SNE")
            return self._tsne(X, n_comp)

    def _tsne(self, X, n_comp, **kwargs):
        from sklearn.manifold import TSNE
        # Pre-reduce with PCA if high-dim
        Xin = X
        if X.shape[1] > 50:
            from sklearn.decomposition import PCA
            Xin = PCA(n_components=50, random_state=self.random_state).fit_transform(X)
        perplexity = min(kwargs.get("perplexity", 30), max(5, len(X) // 5))
        return TSNE(
            n_components=min(n_comp, 3),
            perplexity=perplexity,
            random_state=self.random_state,
            n_iter=kwargs.get("n_iter", 1000),
        ).fit_transform(Xin)

    def _isomap(self, X, n_comp, **kwargs):
        from sklearn.manifold import Isomap
        n_neighbors = min(kwargs.get("n_neighbors", 10), len(X) - 1)
        return Isomap(n_components=n_comp, n_neighbors=n_neighbors).fit_transform(X)

    def _lle(self, X, n_comp, **kwargs):
        from sklearn.manifold import LocallyLinearEmbedding
        n_neighbors = min(kwargs.get("n_neighbors", 10), len(X) - 1)
        return LocallyLinearEmbedding(
            n_components=n_comp, n_neighbors=n_neighbors,
            random_state=self.random_state,
        ).fit_transform(X)

    def variance_explained(self, X: np.ndarray,
                            max_components: int = 20) -> Dict[str, Any]:
        """PCA scree data for variance-explained chart."""
        from sklearn.decomposition import PCA
        n = min(max_components, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n, random_state=self.random_state)
        pca.fit(X)
        return {
            "components": list(range(1, n + 1)),
            "explained_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative": np.cumsum(pca.explained_variance_ratio_).tolist(),
        }

    def clear_cache(self):
        self._cache.clear()


# ──────────────────────────────────────────────────────────────────
# SCATTER PLOT BUILDER
# ──────────────────────────────────────────────────────────────────

class ScatterPlotBuilder:
    """Builds 2D and 3D cluster scatter plots."""

    MAX_POINTS = 15_000

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        self._emb = embedding_engine or EmbeddingEngine()

    def build_2d(self,
                  X: np.ndarray,
                  labels: np.ndarray,
                  method: str = "PCA",
                  title: str = "",
                  algorithm_name: str = "",
                  show_noise: bool = True,
                  marker_size: int = 5,
                  opacity: float = 0.80,
                  **embed_kwargs) -> Any:
        import plotly.graph_objects as go

        # Subsample if needed
        X_plot, labels_plot = self._maybe_sample(X, labels)
        Z = self._emb.embed(X_plot, method=method, n_components=2, **embed_kwargs)

        fig = go.Figure()
        unique = sorted(set(labels_plot.tolist()))

        for c in unique:
            mask = labels_plot == c
            if c == -1:
                if not show_noise:
                    continue
                color = Theme.TEXT_DIM
                name  = "Noise"
                sym   = "x"
                sz    = marker_size - 1
                op    = 0.35
            else:
                color = Theme.cluster_color(c)
                name  = f"Cluster {c}"
                sym   = "circle"
                sz    = marker_size
                op    = opacity

            fig.add_trace(go.Scatter(
                x=Z[mask, 0], y=Z[mask, 1],
                mode="markers",
                name=name,
                marker=dict(
                    color=color, size=sz, opacity=op, symbol=sym,
                    line=dict(width=0.3, color="#000000"),
                ),
                text=[f"Cluster {c}" if c != -1 else "Noise"] * mask.sum(),
                hovertemplate=f"<b>{name}</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>",
            ))

        layout = Theme.plotly_layout(
            title=title or f"{algorithm_name} — {method} Projection",
            height=520,
        )
        layout["xaxis"]["title"] = f"{method} 1"
        layout["yaxis"]["title"] = f"{method} 2"
        fig.update_layout(**layout)
        return fig

    def build_3d(self,
                  X: np.ndarray,
                  labels: np.ndarray,
                  method: str = "PCA",
                  title: str = "",
                  algorithm_name: str = "",
                  marker_size: int = 3,
                  opacity: float = 0.75,
                  **embed_kwargs) -> Any:
        import plotly.graph_objects as go

        X_plot, labels_plot = self._maybe_sample(X, labels)
        Z = self._emb.embed(X_plot, method=method, n_components=3, **embed_kwargs)
        if Z.shape[1] < 3:
            Z = np.hstack([Z, np.zeros((len(Z), 3 - Z.shape[1]))])

        fig = go.Figure()
        for c in sorted(set(labels_plot.tolist())):
            mask = labels_plot == c
            color = Theme.TEXT_DIM if c == -1 else Theme.cluster_color(c)
            name  = "Noise" if c == -1 else f"Cluster {c}"
            op    = 0.2 if c == -1 else opacity

            fig.add_trace(go.Scatter3d(
                x=Z[mask, 0], y=Z[mask, 1], z=Z[mask, 2],
                mode="markers", name=name,
                marker=dict(color=color, size=marker_size, opacity=op),
            ))

        layout = Theme.plotly_layout(
            title=title or f"{algorithm_name} — {method} 3D Projection",
            height=600,
        )
        layout["scene"] = dict(
            xaxis=dict(backgroundcolor=Theme.BG_CARD, gridcolor=Theme.GRID_COLOR,
                       title=f"{method} 1"),
            yaxis=dict(backgroundcolor=Theme.BG_CARD, gridcolor=Theme.GRID_COLOR,
                       title=f"{method} 2"),
            zaxis=dict(backgroundcolor=Theme.BG_CARD, gridcolor=Theme.GRID_COLOR,
                       title=f"{method} 3"),
            bgcolor=Theme.BG_DARK,
        )
        fig.update_layout(**layout)
        return fig

    def build_multi_embedding(self,
                               X: np.ndarray,
                               labels: np.ndarray,
                               methods: List[str],
                               algorithm_name: str = "") -> Any:
        """Subplot grid with multiple embeddings side by side."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n_methods = len(methods)
        cols = min(n_methods, 3)
        rows = (n_methods + cols - 1) // cols
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{m} Projection" for m in methods],
        )

        X_plot, labels_plot = self._maybe_sample(X, labels)
        unique = sorted(set(labels_plot.tolist()))
        legend_shown = set()

        for mi, method in enumerate(methods):
            r, c = divmod(mi, cols)
            try:
                Z = self._emb.embed(X_plot, method=method, n_components=2)
            except Exception:
                continue
            for cl in unique:
                mask = labels_plot == cl
                color = Theme.TEXT_DIM if cl == -1 else Theme.cluster_color(cl)
                name  = "Noise" if cl == -1 else f"Cluster {cl}"
                show  = name not in legend_shown
                legend_shown.add(name)
                fig.add_trace(
                    go.Scatter(
                        x=Z[mask, 0], y=Z[mask, 1],
                        mode="markers", name=name,
                        legendgroup=name,
                        showlegend=show,
                        marker=dict(color=color, size=4, opacity=0.7),
                    ),
                    row=r + 1, col=c + 1,
                )

        layout = Theme.plotly_layout(
            title=f"{algorithm_name} — Multi-Embedding View",
            height=max(400, 350 * rows),
        )
        fig.update_layout(**layout)
        return fig

    def _maybe_sample(self, X: np.ndarray,
                       labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) <= self.MAX_POINTS:
            return X, labels
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), self.MAX_POINTS, replace=False)
        return X[idx], labels[idx]


# ──────────────────────────────────────────────────────────────────
# METRIC CHARTS
# ──────────────────────────────────────────────────────────────────

class MetricChartBuilder:
    """Builds charts for evaluation metrics comparison."""

    def bar_metric_comparison(self,
                               eval_results_list: List[Any],
                               metric_id: str,
                               metric_name: str,
                               title: str = "") -> Any:
        import plotly.graph_objects as go

        names  = [r.algorithm_name[:30] for r in eval_results_list]
        values = [r.metric_value(metric_id) or 0.0 for r in eval_results_list]
        colors = [Theme.cluster_color(i) for i in range(len(names))]

        # Sort descending
        paired = sorted(zip(names, values, colors), key=lambda x: -x[1])
        names, values, colors = zip(*paired) if paired else ([], [], [])

        fig = go.Figure(go.Bar(
            x=list(values), y=list(names),
            orientation="h",
            marker=dict(color=list(colors), opacity=0.85),
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            textfont=dict(color=Theme.TEXT_PRIMARY, size=10),
        ))
        layout = Theme.plotly_layout(
            title=title or f"{metric_name} — Algorithm Comparison", height=max(400, 28 * len(names))
        )
        layout["xaxis"]["title"] = metric_name
        layout["yaxis"]["autorange"] = "reversed"
        fig.update_layout(**layout)
        return fig

    def composite_score_bar(self, eval_results_list: List[Any]) -> Any:
        import plotly.graph_objects as go

        items = [(r.algorithm_name[:30], r.composite_score,
                  r.algorithm_family) for r in eval_results_list]
        items = sorted(items, key=lambda x: -x[1])

        fig = go.Figure(go.Bar(
            x=[v for _, v, _ in items],
            y=[n for n, _, _ in items],
            orientation="h",
            marker=dict(
                color=[v / 100.0 for _, v, _ in items],
                colorscale=[
                    [0, "#1a0030"], [0.3, "#4a00aa"],
                    [0.6, "#00aaff"], [0.85, "#00ffcc"],
                    [1, "#ffffff"]
                ],
                showscale=True,
                colorbar=dict(
                    title="Score/100",
                    tickfont=dict(color=Theme.TEXT_DIM),
                    titlefont=dict(color=Theme.TEXT_DIM),
                ),
                opacity=0.90,
            ),
            text=[f"{v:.1f}" for _, v, _ in items],
            textposition="outside",
            textfont=dict(color=Theme.TEXT_PRIMARY, size=10),
        ))
        layout = Theme.plotly_layout("Composite Clustering Score — All Algorithms",
                                      height=max(450, 28 * len(items)))
        layout["xaxis"]["range"] = [0, 105]
        layout["xaxis"]["title"] = "Composite Score (0–100)"
        layout["yaxis"]["autorange"] = "reversed"
        fig.update_layout(**layout)
        return fig

    def radar_chart(self,
                    eval_results_list: List[Any],
                    top_n: int = 6) -> Any:
        import plotly.graph_objects as go

        metrics = [
            ("Silhouette", "silhouette"),
            ("1–DB Index", "davies_bouldin"),
            ("log(CH) Score", "calinski_harabasz"),
            ("Dunn Index", "dunn_index"),
            ("Cluster Balance", "cluster_balance"),
            ("Low Noise", "noise_ratio"),
        ]
        categories = [m[0] for m in metrics] + [metrics[0][0]]  # Close radar

        top = eval_results_list[:top_n]
        fig = go.Figure()

        for i, r in enumerate(top):
            values = []
            for name, mid in metrics:
                mv = r.metrics.get(mid)
                val = float(mv.normalised or 0) if mv and mv.normalised is not None else 0.0
                values.append(val)
            values += [values[0]]  # Close

            fig.add_trace(go.Scatterpolar(
                r=values, theta=categories,
                fill="toself", name=r.algorithm_name[:25],
                opacity=0.65,
                line=dict(color=Theme.cluster_color(i), width=2),
                fillcolor=Theme.cluster_color(i),
            ))

        layout = Theme.plotly_layout("Metric Radar — Top Algorithms", height=550)
        layout["polar"] = dict(
            bgcolor=Theme.BG_CARD,
            radialaxis=dict(
                range=[0, 1],
                gridcolor=Theme.GRID_COLOR,
                linecolor=Theme.BORDER,
                tickfont=dict(color=Theme.TEXT_DIM),
            ),
            angularaxis=dict(
                gridcolor=Theme.GRID_COLOR,
                linecolor=Theme.BORDER,
                tickfont=dict(color=Theme.TEXT_PRIMARY),
            ),
        )
        fig.update_layout(**layout)
        return fig

    def metric_heatmap(self, heatmap_df: pd.DataFrame, title: str = "") -> Any:
        import plotly.graph_objects as go

        fig = go.Figure(go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns.tolist(),
            y=heatmap_df.index.tolist(),
            colorscale=[
                [0.0, "#1a0030"], [0.2, "#4a00aa"],
                [0.5, "#0044ff"], [0.75, "#00ccff"],
                [1.0, "#00ffcc"],
            ],
            zmin=0, zmax=1,
            text=heatmap_df.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=9, color=Theme.TEXT_PRIMARY),
            colorbar=dict(
                tickfont=dict(color=Theme.TEXT_DIM),
                title=dict(text="Normalised Score",
                           font=dict(color=Theme.TEXT_DIM)),
            ),
        ))
        layout = Theme.plotly_layout(
            title=title or "Metric Heatmap — All Algorithms × Metrics",
            height=max(400, 28 * len(heatmap_df)),
            showlegend=False,
        )
        layout["xaxis"]["tickangle"] = -30
        fig.update_layout(**layout)
        return fig

    def violin_metric(self,
                       eval_results_list: List[Any],
                       metric_ids: Optional[List[str]] = None) -> Any:
        import plotly.graph_objects as go

        if metric_ids is None:
            metric_ids = ["silhouette", "davies_bouldin", "calinski_harabasz"]

        fig = go.Figure()
        for i, mid in enumerate(metric_ids):
            vals = [r.metric_value(mid) for r in eval_results_list
                    if r.metric_value(mid) is not None]
            if not vals:
                continue
            fig.add_trace(go.Violin(
                y=vals, name=mid.replace("_", " ").title(),
                box_visible=True, meanline_visible=True,
                fillcolor=Theme.cluster_color(i),
                opacity=0.7,
                line_color=Theme.TEXT_DIM,
            ))
        layout = Theme.plotly_layout("Metric Distribution — Violin Plot", height=460)
        layout["yaxis"]["title"] = "Metric Value"
        fig.update_layout(**layout)
        return fig

    def k_sweep_elbow(self,
                       k_values: List[int],
                       sil_values: List[float],
                       optimal_k: Optional[int] = None,
                       algorithm_name: str = "") -> Any:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=sil_values, mode="lines+markers",
            name="Silhouette Score",
            line=dict(color=Theme.ACCENT_CYAN, width=2),
            marker=dict(color=Theme.ACCENT_CYAN, size=7),
        ))
        if optimal_k is not None:
            y_opt = sil_values[k_values.index(optimal_k)] if optimal_k in k_values else None
            if y_opt is not None:
                fig.add_vline(
                    x=optimal_k,
                    line_dash="dash",
                    line_color=Theme.ACCENT_GREEN,
                    annotation_text=f"Optimal k={optimal_k}",
                    annotation_font_color=Theme.ACCENT_GREEN,
                )
        layout = Theme.plotly_layout(
            f"k-Sweep Elbow — {algorithm_name}", height=400)
        layout["xaxis"]["title"] = "Number of Clusters (k)"
        layout["yaxis"]["title"] = "Silhouette Score"
        fig.update_layout(**layout)
        return fig

    def scree_plot(self, variance_explained: Dict[str, Any]) -> Any:
        import plotly.graph_objects as go

        comps = variance_explained["components"]
        indiv = [v * 100 for v in variance_explained["explained_ratio"]]
        cumul = [v * 100 for v in variance_explained["cumulative"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comps, y=indiv, name="Individual %",
            marker_color=Theme.ACCENT_VIOLET, opacity=0.8,
        ))
        fig.add_trace(go.Scatter(
            x=comps, y=cumul, name="Cumulative %",
            mode="lines+markers",
            line=dict(color=Theme.ACCENT_CYAN, width=2),
            yaxis="y2",
        ))
        layout = Theme.plotly_layout("PCA Scree — Variance Explained", height=420)
        layout["xaxis"]["title"] = "Principal Component"
        layout["yaxis"]["title"] = "Explained Variance %"
        layout["yaxis2"] = dict(
            title="Cumulative %", overlaying="y", side="right",
            gridcolor="transparent", tickfont=dict(color=Theme.TEXT_DIM),
        )
        fig.update_layout(**layout)
        return fig


# ──────────────────────────────────────────────────────────────────
# CLUSTER PROFILE CHARTS
# ──────────────────────────────────────────────────────────────────

class ClusterProfileBuilder:
    """Charts showing internal cluster structure."""

    def cluster_size_distribution(self,
                                   labels: np.ndarray,
                                   algorithm_name: str = "") -> Any:
        import plotly.graph_objects as go
        from collections import Counter

        counts = Counter(labels.tolist())
        clusters = sorted([c for c in counts if c != -1])
        sizes = [counts[c] for c in clusters]
        colors = [Theme.cluster_color(c) for c in clusters]

        fig = go.Figure(go.Bar(
            x=[f"C{c}" for c in clusters],
            y=sizes,
            marker_color=colors,
            marker_line_width=0,
            text=sizes,
            textposition="outside",
            textfont=dict(color=Theme.TEXT_DIM, size=10),
        ))
        if -1 in counts:
            fig.add_hline(
                y=counts[-1], line_dash="dot",
                line_color=Theme.TEXT_DIM,
                annotation_text=f"Noise: {counts[-1]}",
                annotation_font_color=Theme.TEXT_DIM,
            )
        layout = Theme.plotly_layout(
            f"Cluster Size Distribution — {algorithm_name}", height=380)
        layout["xaxis"]["title"] = "Cluster"
        layout["yaxis"]["title"] = "Size (points)"
        layout["showlegend"] = False
        fig.update_layout(**layout)
        return fig

    def silhouette_bar(self,
                        silhouette_values: np.ndarray,
                        labels: np.ndarray,
                        algorithm_name: str = "") -> Any:
        """Classic silhouette bar plot, colour-coded by cluster."""
        import plotly.graph_objects as go

        if silhouette_values is None or len(silhouette_values) == 0:
            return go.Figure()

        valid_mask = labels != -1
        sil_v = silhouette_values
        lv    = labels[valid_mask] if len(silhouette_values) == valid_mask.sum() else labels

        fig = go.Figure()
        y_offset = 0
        for c in sorted(np.unique(lv)):
            mask = lv == c
            cluster_sil = sil_v[mask]
            cluster_sil_sorted = np.sort(cluster_sil)[::-1]
            n = len(cluster_sil_sorted)
            fig.add_trace(go.Bar(
                x=cluster_sil_sorted,
                y=list(range(y_offset, y_offset + n)),
                orientation="h",
                name=f"Cluster {c}",
                marker_color=Theme.cluster_color(c),
                marker_line_width=0,
                showlegend=True,
            ))
            y_offset += n + 5

        mean_sil = float(sil_v.mean())
        fig.add_vline(x=mean_sil, line_dash="dash",
                       line_color=Theme.ACCENT_YELLOW,
                       annotation_text=f"Mean={mean_sil:.3f}",
                       annotation_font_color=Theme.ACCENT_YELLOW)
        layout = Theme.plotly_layout(
            f"Silhouette Plot — {algorithm_name}", height=max(400, y_offset * 2))
        layout["xaxis"]["title"] = "Silhouette Coefficient"
        layout["yaxis"]["title"] = "Sample"
        layout["yaxis"]["showticklabels"] = False
        fig.update_layout(**layout)
        return fig

    def cluster_centroid_heatmap(self,
                                  X: np.ndarray,
                                  labels: np.ndarray,
                                  feature_names: List[str],
                                  algorithm_name: str = "") -> Any:
        """Shows normalised centroid feature values as a heatmap."""
        import plotly.graph_objects as go

        unique = [c for c in np.unique(labels) if c != -1]
        centroids = np.array([X[labels == c].mean(axis=0) for c in unique])
        # Z-score normalise columns
        centroids_norm = (centroids - centroids.mean(axis=0)) / (centroids.std(axis=0) + 1e-8)

        short_features = [f[:20] for f in feature_names]
        cluster_labels = [f"Cluster {c}" for c in unique]

        fig = go.Figure(go.Heatmap(
            z=centroids_norm,
            x=short_features,
            y=cluster_labels,
            colorscale=[
                [0, "#1a0030"], [0.25, "#4a00aa"],
                [0.5, Theme.BG_CARD], [0.75, "#00aaff"],
                [1.0, "#00ffcc"],
            ],
            zmid=0,
            text=np.round(centroids_norm, 2),
            texttemplate="%{text}",
            textfont=dict(size=8),
            colorbar=dict(
                tickfont=dict(color=Theme.TEXT_DIM),
                title=dict(text="z-score", font=dict(color=Theme.TEXT_DIM)),
            ),
        ))
        layout = Theme.plotly_layout(
            f"Centroid Heatmap — {algorithm_name}",
            height=max(350, 50 * len(unique)),
            showlegend=False,
        )
        layout["xaxis"]["tickangle"] = -35
        fig.update_layout(**layout)
        return fig

    def feature_importance_bar(self,
                                X: np.ndarray,
                                labels: np.ndarray,
                                feature_names: List[str],
                                top_n: int = 20) -> Any:
        """Feature importance via between-cluster variance ratio."""
        import plotly.graph_objects as go

        unique = [c for c in np.unique(labels) if c != -1]
        if len(unique) < 2 or X.shape[1] == 0:
            return go.Figure()

        # Between-cluster / total variance
        total_var = X.var(axis=0) + 1e-10
        sizes = np.array([(labels == c).sum() for c in unique])
        centroids = np.array([X[labels == c].mean(axis=0) for c in unique])
        grand_mean = X.mean(axis=0)
        between_var = (sizes[:, None] * (centroids - grand_mean) ** 2).sum(axis=0) / len(X)
        importance = between_var / total_var

        top_idx = np.argsort(importance)[::-1][:top_n]
        top_imp  = importance[top_idx]
        top_feat = [feature_names[i] if i < len(feature_names) else f"F{i}"
                    for i in top_idx]
        colors = [Theme.cluster_color(i) for i in range(len(top_feat))]

        fig = go.Figure(go.Bar(
            x=top_imp, y=top_feat, orientation="h",
            marker_color=colors, marker_line_width=0,
        ))
        layout = Theme.plotly_layout(
            f"Feature Importance (Between-Cluster Variance Ratio) — Top {top_n}",
            height=max(380, 24 * top_n),
            showlegend=False,
        )
        layout["xaxis"]["title"] = "Importance Score"
        layout["yaxis"]["autorange"] = "reversed"
        fig.update_layout(**layout)
        return fig

    def pair_scatter_matrix(self,
                             X: np.ndarray,
                             labels: np.ndarray,
                             feature_names: List[str],
                             max_features: int = 6) -> Any:
        """Plotly splom (scatter matrix) for top features."""
        import plotly.graph_objects as go

        n_feat = min(max_features, X.shape[1])
        top_feat_names = feature_names[:n_feat]
        colors = [Theme.cluster_color(int(l)) if l != -1 else Theme.TEXT_DIM
                  for l in labels]

        dims = [dict(label=n[:20], values=X[:, i])
                for i, n in enumerate(top_feat_names)]

        fig = go.Figure(go.Splom(
            dimensions=dims,
            marker=dict(color=colors, size=3, opacity=0.6,
                        line=dict(width=0)),
            diagonal_visible=True,
            showupperhalf=False,
        ))
        layout = Theme.plotly_layout(
            "Pair Scatter Matrix — Cluster Assignments",
            height=max(500, 120 * n_feat),
            showlegend=False,
        )
        fig.update_layout(**layout)
        return fig


# ──────────────────────────────────────────────────────────────────
# STABILITY CHARTS
# ──────────────────────────────────────────────────────────────────

class StabilityChartBuilder:
    """Visualises stability analysis outputs."""

    def ari_boxplot(self, ari_data: Dict[str, List[float]],
                    title: str = "") -> Any:
        import plotly.graph_objects as go

        fig = go.Figure()
        for i, (alg_name, ari_vals) in enumerate(ari_data.items()):
            if not ari_vals:
                continue
            fig.add_trace(go.Box(
                y=ari_vals,
                name=alg_name[:25],
                boxpoints="outliers",
                marker=dict(color=Theme.cluster_color(i), size=4),
                line=dict(color=Theme.cluster_color(i)),
                fillcolor=Theme.cluster_color(i),
                opacity=0.7,
            ))
        layout = Theme.plotly_layout(
            title or "ARI Distribution — Stability Analysis", height=480)
        layout["yaxis"]["title"] = "Adjusted Rand Index (ARI)"
        layout["yaxis"]["range"] = [-0.05, 1.05]
        fig.update_layout(**layout)
        return fig

    def noise_degradation_curves(self,
                                   noise_profiles: Dict[str, Dict]) -> Any:
        import plotly.graph_objects as go

        fig = go.Figure()
        for i, (alg_name, profile) in enumerate(noise_profiles.items()):
            levels = profile.get("noise_levels", [])
            mean_aris = profile.get("mean_ari", [])
            std_aris  = profile.get("std_ari", [])
            if not levels or not mean_aris:
                continue
            color = Theme.cluster_color(i)

            # Confidence band
            upper = [min(1.0, m + s) for m, s in zip(mean_aris, std_aris)]
            lower = [max(0.0, m - s) for m, s in zip(mean_aris, std_aris)]
            fig.add_trace(go.Scatter(
                x=levels + levels[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(width=0),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=levels, y=mean_aris, mode="lines+markers",
                name=alg_name[:25],
                line=dict(color=color, width=2),
                marker=dict(color=color, size=6),
            ))

        layout = Theme.plotly_layout(
            "Noise Robustness — ARI vs Gaussian Noise σ", height=450)
        layout["xaxis"]["title"] = "Noise Level (σ relative to feature std)"
        layout["yaxis"]["title"] = "Mean ARI"
        fig.update_layout(**layout)
        return fig

    def stability_scorecard(self, scorecard_df: pd.DataFrame) -> Any:
        """Colour-coded stability scorecard table."""
        import plotly.graph_objects as go

        if scorecard_df.empty:
            return go.Figure()

        grade_colors = {
            "Highly Stable":    Theme.ACCENT_GREEN,
            "Stable":           "#88ff44",
            "Moderately Stable": Theme.ACCENT_YELLOW,
            "Unstable":         Theme.ACCENT_ORANGE,
            "Highly Unstable":  "#ff3333",
        }

        cell_colors = []
        for col in scorecard_df.columns:
            col_colors = []
            for val in scorecard_df[col]:
                if col == "Grade":
                    col_colors.append(grade_colors.get(str(val), Theme.BG_PANEL))
                else:
                    col_colors.append(Theme.BG_PANEL)
            cell_colors.append(col_colors)

        fig = go.Figure(go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in scorecard_df.columns],
                fill_color=Theme.BG_DARK,
                font=dict(color=Theme.ACCENT_CYAN, size=12),
                line_color=Theme.BORDER,
                align="center",
            ),
            cells=dict(
                values=[scorecard_df[c].tolist() for c in scorecard_df.columns],
                fill_color=cell_colors,
                font=dict(color=Theme.TEXT_PRIMARY, size=11),
                line_color=Theme.BORDER,
                align="center",
            ),
        ))
        layout = Theme.plotly_layout("Stability Scorecard", height=max(380, 35 * len(scorecard_df)))
        layout["showlegend"] = False
        fig.update_layout(**layout)
        return fig

    def persistence_bar(self, persistence_list: List[Any],
                         algorithm_name: str = "") -> Any:
        import plotly.graph_objects as go

        cids  = [f"C{p.cluster_id}" for p in persistence_list]
        jacs  = [p.mean_jaccard for p in persistence_list]
        stds  = [p.std_jaccard for p in persistence_list]
        colors = [Theme.ACCENT_GREEN if p.is_stable else Theme.ACCENT_ORANGE
                  for p in persistence_list]

        fig = go.Figure(go.Bar(
            x=cids, y=jacs, name="Mean Jaccard",
            error_y=dict(array=stds, color=Theme.TEXT_DIM),
            marker_color=colors,
        ))
        layout = Theme.plotly_layout(
            f"Cluster Persistence — {algorithm_name}", height=400)
        layout["yaxis"]["title"] = "Mean Jaccard (vs Reference)"
        layout["yaxis"]["range"] = [0, 1.05]
        layout["xaxis"]["title"] = "Cluster"
        layout["showlegend"] = False
        fig.update_layout(**layout)
        return fig

    def ari_matrix_heatmap(self, ari_matrix_df: pd.DataFrame) -> Any:
        import plotly.graph_objects as go

        fig = go.Figure(go.Heatmap(
            z=ari_matrix_df.values,
            x=ari_matrix_df.columns.tolist(),
            y=ari_matrix_df.index.tolist(),
            colorscale=[
                [0.0, "#1a0030"], [0.25, "#4a00aa"],
                [0.5, "#0044ff"], [0.75, "#00ccff"],
                [1.0, "#00ffcc"],
            ],
            zmin=0, zmax=1,
            text=ari_matrix_df.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(tickfont=dict(color=Theme.TEXT_DIM)),
        ))
        layout = Theme.plotly_layout(
            "Pairwise ARI Matrix — Algorithm Agreement",
            height=max(450, 25 * len(ari_matrix_df)),
            showlegend=False,
        )
        layout["xaxis"]["tickangle"] = -40
        fig.update_layout(**layout)
        return fig


# ──────────────────────────────────────────────────────────────────
# CONSENSUS CHARTS
# ──────────────────────────────────────────────────────────────────

class ConsensusChartBuilder:
    """Visualises consensus clustering outputs."""

    def coassoc_heatmap(self,
                         coassoc: np.ndarray,
                         labels: np.ndarray,
                         title: str = "Co-Association Matrix") -> Any:
        import plotly.graph_objects as go

        # Sort by cluster
        sort_idx = np.argsort(labels)
        co_sorted = coassoc[np.ix_(sort_idx, sort_idx)]

        fig = go.Figure(go.Heatmap(
            z=co_sorted,
            colorscale=[
                [0.0, Theme.BG_DARK], [0.2, "#1a0050"],
                [0.5, "#4400bb"], [0.8, "#0088ff"],
                [1.0, Theme.ACCENT_CYAN],
            ],
            zmin=0, zmax=1,
            colorbar=dict(
                title=dict(text="Co-Assoc.", font=dict(color=Theme.TEXT_DIM)),
                tickfont=dict(color=Theme.TEXT_DIM),
            ),
            showscale=True,
        ))
        layout = Theme.plotly_layout(title, height=520, showlegend=False)
        layout["xaxis"]["title"] = "Sample (sorted by cluster)"
        layout["yaxis"]["title"] = "Sample (sorted by cluster)"
        layout["xaxis"]["showticklabels"] = False
        layout["yaxis"]["showticklabels"] = False
        fig.update_layout(**layout)
        return fig

    def weight_barchart(self, weight_df: pd.DataFrame) -> Any:
        import plotly.graph_objects as go

        colors = [Theme.cluster_color(i) for i in range(len(weight_df))]
        fig = go.Figure(go.Bar(
            x=weight_df["Weight"].values,
            y=weight_df["Algorithm"].values,
            orientation="h",
            marker_color=colors,
            text=[f"{w:.3f}" for w in weight_df["Weight"]],
            textposition="outside",
            textfont=dict(color=Theme.TEXT_PRIMARY, size=10),
        ))
        layout = Theme.plotly_layout("Algorithm Weights — Consensus", height=max(350, 28 * len(weight_df)))
        layout["xaxis"]["title"] = "Normalised Weight"
        layout["yaxis"]["autorange"] = "reversed"
        layout["showlegend"] = False
        fig.update_layout(**layout)
        return fig

    def method_comparison(self, comparison_df: pd.DataFrame) -> Any:
        import plotly.graph_objects as go

        if comparison_df.empty:
            return go.Figure()

        fig = go.Figure(go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in comparison_df.columns],
                fill_color=Theme.BG_DARK,
                font=dict(color=Theme.ACCENT_VIOLET, size=12),
                line_color=Theme.BORDER,
                align="center",
            ),
            cells=dict(
                values=[comparison_df[c].tolist() for c in comparison_df.columns],
                fill_color=Theme.BG_PANEL,
                font=dict(color=Theme.TEXT_PRIMARY, size=11),
                line_color=Theme.BORDER,
                align="center",
            ),
        ))
        layout = Theme.plotly_layout("Consensus Method Comparison",
                                      height=max(380, 35 * len(comparison_df)))
        layout["showlegend"] = False
        fig.update_layout(**layout)
        return fig

    def diversity_radar(self, diversity_data: Dict[str, Dict[str, float]]) -> Any:
        import plotly.graph_objects as go

        if not diversity_data:
            return go.Figure()

        categories = list(list(diversity_data.values())[0].keys())
        categories += [categories[0]]
        fig = go.Figure()
        for i, (alg, metrics) in enumerate(diversity_data.items()):
            vals = list(metrics.values()) + [list(metrics.values())[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=categories, fill="toself",
                name=alg[:25],
                opacity=0.6,
                line=dict(color=Theme.cluster_color(i), width=2),
            ))
        layout = Theme.plotly_layout("Stability Radar — Diversity Breakdown", height=520)
        layout["polar"] = dict(
            bgcolor=Theme.BG_CARD,
            radialaxis=dict(range=[0, 1], gridcolor=Theme.GRID_COLOR,
                            tickfont=dict(color=Theme.TEXT_DIM)),
            angularaxis=dict(gridcolor=Theme.GRID_COLOR,
                             tickfont=dict(color=Theme.TEXT_PRIMARY)),
        )
        fig.update_layout(**layout)
        return fig


# ──────────────────────────────────────────────────────────────────
# DATA PROFILE CHARTS
# ──────────────────────────────────────────────────────────────────

class DataProfileChartBuilder:
    """Charts for the data profiling / EDA section."""

    def missing_values_bar(self, profile: Any) -> Any:
        import plotly.graph_objects as go

        cols  = []
        pcts  = []
        for col, cp in profile.column_profiles.items():
            if cp.missing_pct > 0:
                cols.append(col[:30])
                pcts.append(cp.missing_pct)

        if not cols:
            return go.Figure()

        paired = sorted(zip(pcts, cols), reverse=True)
        pcts, cols = zip(*paired)

        fig = go.Figure(go.Bar(
            x=list(pcts), y=list(cols), orientation="h",
            marker_color=Theme.ACCENT_ORANGE,
            text=[f"{p:.1f}%" for p in pcts],
            textposition="outside",
        ))
        layout = Theme.plotly_layout("Missing Value % by Column", height=max(350, 22 * len(cols)))
        layout["xaxis"]["title"] = "Missing %"
        layout["xaxis"]["range"] = [0, 105]
        layout["showlegend"] = False
        fig.update_layout(**layout)
        return fig

    def correlation_heatmap(self, corr_matrix: pd.DataFrame,
                             max_cols: int = 30) -> Any:
        import plotly.graph_objects as go

        corr = corr_matrix.iloc[:max_cols, :max_cols]
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale=[
                [0.0, "#aa0000"], [0.5, Theme.BG_DARK], [1.0, "#00aaff"]
            ],
            zmin=-1, zmax=1, zmid=0,
            colorbar=dict(tickfont=dict(color=Theme.TEXT_DIM)),
        ))
        layout = Theme.plotly_layout("Feature Correlation Matrix",
                                      height=max(450, 22 * len(corr)),
                                      showlegend=False)
        layout["xaxis"]["tickangle"] = -40
        fig.update_layout(**layout)
        return fig

    def distribution_histogram(self, series: pd.Series,
                                col_name: str = "") -> Any:
        import plotly.graph_objects as go

        fig = go.Figure(go.Histogram(
            x=series.dropna().tolist(),
            nbinsx=40,
            marker_color=Theme.ACCENT_CYAN,
            opacity=0.8,
        ))
        layout = Theme.plotly_layout(f"Distribution — {col_name}", height=340)
        layout["xaxis"]["title"] = col_name
        layout["yaxis"]["title"] = "Count"
        layout["showlegend"] = False
        fig.update_layout(**layout)
        return fig

    def outlier_scatter(self, X: np.ndarray,
                         outlier_mask: np.ndarray,
                         feature_x: int = 0,
                         feature_y: int = 1) -> Any:
        import plotly.graph_objects as go

        inlier_mask = ~outlier_mask
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X[inlier_mask, feature_x], y=X[inlier_mask, feature_y],
            mode="markers", name="Inlier",
            marker=dict(color=Theme.ACCENT_CYAN, size=4, opacity=0.6),
        ))
        if outlier_mask.any():
            fig.add_trace(go.Scatter(
                x=X[outlier_mask, feature_x], y=X[outlier_mask, feature_y],
                mode="markers", name="Outlier",
                marker=dict(color=Theme.ACCENT_PINK, size=6, symbol="x", opacity=0.9),
            ))
        layout = Theme.plotly_layout("Outlier Detection Scatter", height=420)
        fig.update_layout(**layout)
        return fig


# ──────────────────────────────────────────────────────────────────
# MASTER VISUALISATION FACADE
# ──────────────────────────────────────────────────────────────────

class VisualisationEngine:
    """
    Unified entry point for all ClusterX visualisations.
    Encapsulates all sub-builders with a clean API.
    """

    def __init__(self, random_state: int = 42):
        self._emb    = EmbeddingEngine(random_state=random_state)
        self._scatter = ScatterPlotBuilder(self._emb)
        self._metric  = MetricChartBuilder()
        self._profile = ClusterProfileBuilder()
        self._stability = StabilityChartBuilder()
        self._consensus = ConsensusChartBuilder()
        self._data      = DataProfileChartBuilder()

    # ── Scatter ──────────────────────────────────────────────────

    def scatter_2d(self, X, labels, method="PCA", title="", algorithm_name="",
                   **kw):
        return self._scatter.build_2d(X, labels, method, title, algorithm_name, **kw)

    def scatter_3d(self, X, labels, method="PCA", title="", algorithm_name="",
                   **kw):
        return self._scatter.build_3d(X, labels, method, title, algorithm_name, **kw)

    def multi_embedding(self, X, labels, methods=None, algorithm_name=""):
        methods = methods or ["PCA", "UMAP", "t-SNE"]
        return self._scatter.build_multi_embedding(X, labels, methods, algorithm_name)

    # ── Metrics ───────────────────────────────────────────────────

    def composite_score_bar(self, eval_results):
        return self._metric.composite_score_bar(eval_results)

    def metric_bar(self, eval_results, metric_id, metric_name, title=""):
        return self._metric.bar_metric_comparison(eval_results, metric_id, metric_name, title)

    def radar_chart(self, eval_results, top_n=6):
        return self._metric.radar_chart(eval_results, top_n)

    def metric_heatmap(self, heatmap_df, title=""):
        return self._metric.metric_heatmap(heatmap_df, title)

    def violin_metric(self, eval_results, metric_ids=None):
        return self._metric.violin_metric(eval_results, metric_ids)

    def k_sweep_elbow(self, k_values, sil_values, optimal_k=None, algorithm_name=""):
        return self._metric.k_sweep_elbow(k_values, sil_values, optimal_k, algorithm_name)

    def scree_plot(self, variance_explained):
        return self._metric.scree_plot(variance_explained)

    # ── Cluster Profile ───────────────────────────────────────────

    def cluster_size_distribution(self, labels, algorithm_name=""):
        return self._profile.cluster_size_distribution(labels, algorithm_name)

    def silhouette_bar(self, sil_vals, labels, algorithm_name=""):
        return self._profile.silhouette_bar(sil_vals, labels, algorithm_name)

    def centroid_heatmap(self, X, labels, feature_names, algorithm_name=""):
        return self._profile.cluster_centroid_heatmap(X, labels, feature_names, algorithm_name)

    def feature_importance(self, X, labels, feature_names, top_n=20):
        return self._profile.feature_importance_bar(X, labels, feature_names, top_n)

    def pair_scatter_matrix(self, X, labels, feature_names, max_features=6):
        return self._profile.pair_scatter_matrix(X, labels, feature_names, max_features)

    # ── Stability ────────────────────────────────────────────────

    def ari_boxplot(self, ari_data, title=""):
        return self._stability.ari_boxplot(ari_data, title)

    def noise_degradation(self, noise_profiles):
        return self._stability.noise_degradation_curves(noise_profiles)

    def stability_scorecard_table(self, scorecard_df):
        return self._stability.stability_scorecard(scorecard_df)

    def persistence_bar(self, persistence_list, algorithm_name=""):
        return self._stability.persistence_bar(persistence_list, algorithm_name)

    def ari_matrix_heatmap(self, ari_matrix_df):
        return self._stability.ari_matrix_heatmap(ari_matrix_df)

    # ── Consensus ─────────────────────────────────────────────────

    def coassoc_heatmap(self, coassoc, labels, title=""):
        return self._consensus.coassoc_heatmap(coassoc, labels, title)

    def weight_barchart(self, weight_df):
        return self._consensus.weight_barchart(weight_df)

    def consensus_method_comparison(self, comparison_df):
        return self._consensus.method_comparison(comparison_df)

    def diversity_radar(self, diversity_data):
        return self._consensus.diversity_radar(diversity_data)

    # ── Data Profile ─────────────────────────────────────────────

    def missing_values_bar(self, profile):
        return self._data.missing_values_bar(profile)

    def correlation_heatmap(self, corr_matrix, max_cols=30):
        return self._data.correlation_heatmap(corr_matrix, max_cols)

    def distribution_histogram(self, series, col_name=""):
        return self._data.distribution_histogram(series, col_name)

    def outlier_scatter(self, X, outlier_mask, fx=0, fy=1):
        return self._data.outlier_scatter(X, outlier_mask, fx, fy)

    # ── Variance ─────────────────────────────────────────────────

    def variance_explained(self, X, max_components=20):
        return self._emb.variance_explained(X, max_components)

    def clear_embedding_cache(self):
        self._emb.clear_cache()


# ──────────────────────────────────────────────────────────────────
# SINGLETON
# ──────────────────────────────────────────────────────────────────

_VIS_ENGINE: Optional[VisualisationEngine] = None


def get_vis_engine(random_state: int = 42) -> VisualisationEngine:
    global _VIS_ENGINE
    if _VIS_ENGINE is None:
        _VIS_ENGINE = VisualisationEngine(random_state=random_state)
    return _VIS_ENGINE


# ──────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def get_theme() -> Theme:
    return Theme()


def available_embedding_methods() -> List[str]:
    return EmbeddingEngine.METHODS


def label_colormap(labels: np.ndarray) -> List[str]:
    """Returns hex color list aligned to label array."""
    return [
        Theme.TEXT_DIM if l == -1 else Theme.cluster_color(int(l))
        for l in labels
    ]


def empty_figure(message: str = "No data to display") -> Any:
    """Placeholder figure with a message."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=Theme.TEXT_DIM),
    )
    fig.update_layout(
        paper_bgcolor=Theme.BG_DARK,
        plot_bgcolor=Theme.BG_CARD,
        font=dict(color=Theme.TEXT_PRIMARY),
        height=350,
    )
    return fig


def make_subplots_dark(rows: int, cols: int,
                        subplot_titles: Optional[List[str]] = None,
                        height: int = 500) -> Any:
    """Helper to create dark-themed multi-panel figures."""
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles or [],
    )
    fig.update_layout(
        paper_bgcolor=Theme.BG_DARK,
        plot_bgcolor=Theme.BG_CARD,
        font=dict(color=Theme.TEXT_PRIMARY, family="Inter, sans-serif"),
        height=height,
    )
    return fig

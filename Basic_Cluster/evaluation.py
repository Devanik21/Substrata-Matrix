"""
evaluation.py — Substrata-Matrix Evaluation & Ranking Module

Comprehensive metric computation, multi-dimensional ranking,
and interpretability for clustering quality assessment. Covers
internal indices, label-based measures, geometry diagnostics,
and natural-language interpretation.
"""

from __future__ import annotations

import warnings
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import entropy

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ──────────────────────────────────────────────────────────────────

class MetricDirection(str, Enum):
    HIGHER_BETTER = "higher_is_better"
    LOWER_BETTER  = "lower_is_better"


class MetricCategory(str, Enum):
    COMPACTNESS   = "Compactness"
    SEPARATION    = "Separation"
    CONNECTIVITY  = "Connectivity"
    DENSITY       = "Density"
    GEOMETRIC     = "Geometric"
    BALANCE       = "Balance"


# ──────────────────────────────────────────────────────────────────
# METRIC DESCRIPTORS
# ──────────────────────────────────────────────────────────────────

@dataclass
class MetricSpec:
    id: str
    name: str
    category: MetricCategory
    direction: MetricDirection
    ideal_value: Optional[float]
    value_range: Tuple[Optional[float], Optional[float]]
    description: str
    interpretation: str
    weight: float = 1.0              # Default weighting in composite score


METRIC_REGISTRY: Dict[str, MetricSpec] = {
    "silhouette": MetricSpec(
        id="silhouette",
        name="Silhouette Score",
        category=MetricCategory.COMPACTNESS,
        direction=MetricDirection.HIGHER_BETTER,
        ideal_value=1.0,
        value_range=(-1.0, 1.0),
        description="Measures cohesion vs separation. Mean of per-sample silhouette widths.",
        interpretation=(
            "[-1,0): Incorrect labelling. "
            "[0, 0.25): No substantial structure. "
            "[0.25, 0.5): Weak structure. "
            "[0.5, 0.7): Reasonable structure. "
            "[0.7, 1.0]: Strong structure."
        ),
        weight=3.0,
    ),
    "davies_bouldin": MetricSpec(
        id="davies_bouldin",
        name="Davies-Bouldin Index",
        category=MetricCategory.SEPARATION,
        direction=MetricDirection.LOWER_BETTER,
        ideal_value=0.0,
        value_range=(0.0, None),
        description="Ratio of within-cluster scatter to between-cluster separation.",
        interpretation="Lower is better. Values < 1.0 indicate well-separated clusters.",
        weight=2.0,
    ),
    "calinski_harabasz": MetricSpec(
        id="calinski_harabasz",
        name="Calinski-Harabász Score",
        category=MetricCategory.SEPARATION,
        direction=MetricDirection.HIGHER_BETTER,
        ideal_value=None,
        value_range=(0.0, None),
        description="Ratio of between-cluster dispersion to within-cluster dispersion.",
        interpretation="Higher is better. No absolute scale — use for comparison.",
        weight=2.0,
    ),
    "dunn_index": MetricSpec(
        id="dunn_index",
        name="Dunn Index",
        category=MetricCategory.SEPARATION,
        direction=MetricDirection.HIGHER_BETTER,
        ideal_value=None,
        value_range=(0.0, None),
        description="Min inter-cluster distance / max intra-cluster distance.",
        interpretation="Higher means better-separated, more compact clusters.",
        weight=2.0,
    ),
    "xb_index": MetricSpec(
        id="xb_index",
        name="Xie-Beni Index",
        category=MetricCategory.COMPACTNESS,
        direction=MetricDirection.LOWER_BETTER,
        ideal_value=0.0,
        value_range=(0.0, None),
        description="Total compactness / n × min inter-centroid distance².",
        interpretation="Lower means more compact, better-separated clusters.",
        weight=1.5,
    ),
    "inertia": MetricSpec(
        id="inertia",
        name="Inertia (WCSS)",
        category=MetricCategory.COMPACTNESS,
        direction=MetricDirection.LOWER_BETTER,
        ideal_value=0.0,
        value_range=(0.0, None),
        description="Within-cluster sum of squared distances to centroids.",
        interpretation="Lower = more compact. Not comparable across k or datasets.",
        weight=1.0,
    ),
    "noise_ratio": MetricSpec(
        id="noise_ratio",
        name="Noise Ratio",
        category=MetricCategory.DENSITY,
        direction=MetricDirection.LOWER_BETTER,
        ideal_value=0.0,
        value_range=(0.0, 1.0),
        description="Fraction of points labelled as noise (-1).",
        interpretation="High noise ratio indicates sparse data or tight eps.",
        weight=1.5,
    ),
    "cluster_balance": MetricSpec(
        id="cluster_balance",
        name="Cluster Size Balance",
        category=MetricCategory.BALANCE,
        direction=MetricDirection.HIGHER_BETTER,
        ideal_value=1.0,
        value_range=(0.0, 1.0),
        description="Min cluster size / max cluster size ratio.",
        interpretation="1.0 = perfectly balanced. Low = highly imbalanced.",
        weight=1.0,
    ),
    "size_entropy": MetricSpec(
        id="size_entropy",
        name="Cluster Size Entropy",
        category=MetricCategory.BALANCE,
        direction=MetricDirection.HIGHER_BETTER,
        ideal_value=None,
        value_range=(0.0, None),
        description="Entropy of the cluster size distribution.",
        interpretation="Higher = more uniform cluster sizes.",
        weight=0.5,
    ),
    "avg_intra_dist": MetricSpec(
        id="avg_intra_dist",
        name="Avg Intra-Cluster Distance",
        category=MetricCategory.COMPACTNESS,
        direction=MetricDirection.LOWER_BETTER,
        ideal_value=0.0,
        value_range=(0.0, None),
        description="Mean pairwise Euclidean distance within clusters.",
        interpretation="Lower = more compact clusters.",
        weight=1.0,
    ),
    "avg_inter_dist": MetricSpec(
        id="avg_inter_dist",
        name="Avg Inter-Cluster Distance",
        category=MetricCategory.SEPARATION,
        direction=MetricDirection.HIGHER_BETTER,
        ideal_value=None,
        value_range=(0.0, None),
        description="Mean pairwise distance between cluster centroids.",
        interpretation="Higher = better-separated clusters.",
        weight=1.0,
    ),
}


# ──────────────────────────────────────────────────────────────────
# METRIC VALUE CONTAINERS
# ──────────────────────────────────────────────────────────────────

@dataclass
class MetricValue:
    metric_id: str
    value: Optional[float]
    normalised: Optional[float]   # 0–1 normalised (higher always better)
    error: Optional[str] = None
    compute_time: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.value is not None and not np.isnan(self.value)

    @property
    def grade(self) -> str:
        """Human-readable quality grade based on normalised score."""
        if self.normalised is None:
            return "N/A"
        if self.normalised >= 0.85:
            return "Excellent"
        elif self.normalised >= 0.70:
            return "Good"
        elif self.normalised >= 0.50:
            return "Fair"
        elif self.normalised >= 0.30:
            return "Poor"
        else:
            return "Very Poor"


@dataclass
class EvaluationResult:
    algorithm_id: str
    algorithm_name: str
    algorithm_family: str
    n_clusters: int
    n_noise: int
    noise_ratio: float
    metrics: Dict[str, MetricValue]
    composite_score: float           # Weighted aggregate 0–100
    rank: int = 0                    # Set by ranker
    runtime_seconds: float = 0.0
    interpretation: str = ""
    warnings: List[str] = field(default_factory=list)

    def metric_value(self, metric_id: str) -> Optional[float]:
        mv = self.metrics.get(metric_id)
        return mv.value if mv and mv.is_valid else None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "algorithm_id": self.algorithm_id,
            "algorithm_name": self.algorithm_name,
            "algorithm_family": self.algorithm_family,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "noise_ratio": round(self.noise_ratio, 4),
            "composite_score": round(self.composite_score, 2),
            "rank": self.rank,
            "runtime_seconds": round(self.runtime_seconds, 4),
            "interpretation": self.interpretation,
        }
        for mid, mv in self.metrics.items():
            d[mid] = round(mv.value, 4) if mv.is_valid else None
        return d

    def to_dataframe_row(self) -> Dict[str, Any]:
        row = {
            "Algorithm": self.algorithm_name,
            "Family": self.algorithm_family,
            "k": self.n_clusters,
            "Noise%": round(self.noise_ratio * 100, 1),
            "Score": round(self.composite_score, 1),
            "Rank": self.rank,
            "Time(s)": round(self.runtime_seconds, 3),
        }
        for mid in ["silhouette", "davies_bouldin", "calinski_harabasz", "dunn_index"]:
            mv = self.metrics.get(mid)
            row[METRIC_REGISTRY[mid].name if mid in METRIC_REGISTRY else mid] = (
                round(mv.value, 4) if mv and mv.is_valid else None
            )
        return row


# ──────────────────────────────────────────────────────────────────
# INDIVIDUAL METRIC COMPUTERS
# ──────────────────────────────────────────────────────────────────

class SilhouetteComputer:
    """Computes silhouette with fast sampling for large datasets."""

    MAX_SAMPLES_EXACT   = 10_000
    MAX_SAMPLES_APPROX  = 50_000

    def compute(self, X: np.ndarray, labels: np.ndarray) -> MetricValue:
        valid_mask = labels != -1
        Xv, lv = X[valid_mask], labels[valid_mask]
        n_cls = len(np.unique(lv))
        if n_cls < 2 or len(Xv) < 4:
            return MetricValue("silhouette", None, None,
                               error="Too few valid clusters or samples")
        t = time.perf_counter()
        try:
            from sklearn.metrics import silhouette_score
            if len(Xv) > self.MAX_SAMPLES_APPROX:
                # Large: sample down
                idx = np.random.default_rng(42).choice(len(Xv), self.MAX_SAMPLES_APPROX,
                                                        replace=False)
                score = float(silhouette_score(Xv[idx], lv[idx],
                                               metric="euclidean", sample_size=None))
            elif len(Xv) > self.MAX_SAMPLES_EXACT:
                score = float(silhouette_score(Xv, lv, metric="euclidean",
                                               sample_size=self.MAX_SAMPLES_EXACT,
                                               random_state=42))
            else:
                score = float(silhouette_score(Xv, lv, metric="euclidean"))
        except Exception as e:
            return MetricValue("silhouette", None, None, error=str(e))
        # Normalise: map [-1,1] → [0,1]
        norm = (score + 1) / 2
        return MetricValue("silhouette", score, norm,
                           compute_time=time.perf_counter() - t)

    def per_sample(self, X: np.ndarray, labels: np.ndarray,
                   max_samples: int = 5_000) -> Optional[np.ndarray]:
        """Returns per-sample silhouette values for visualisation."""
        valid_mask = labels != -1
        Xv, lv = X[valid_mask], labels[valid_mask]
        if len(np.unique(lv)) < 2 or len(Xv) < 4:
            return None
        try:
            from sklearn.metrics import silhouette_samples
            if len(Xv) > max_samples:
                idx = np.random.default_rng(42).choice(len(Xv), max_samples, replace=False)
                vals = silhouette_samples(Xv[idx], lv[idx])
            else:
                vals = silhouette_samples(Xv, lv)
            return vals
        except Exception:
            return None


class DaviesBouldinComputer:
    def compute(self, X: np.ndarray, labels: np.ndarray) -> MetricValue:
        valid_mask = labels != -1
        Xv, lv = X[valid_mask], labels[valid_mask]
        if len(np.unique(lv)) < 2:
            return MetricValue("davies_bouldin", None, None, error="< 2 clusters")
        t = time.perf_counter()
        try:
            from sklearn.metrics import davies_bouldin_score
            score = float(davies_bouldin_score(Xv, lv))
        except Exception as e:
            return MetricValue("davies_bouldin", None, None, error=str(e))
        # Normalise: higher is worse, cap at 5.0 for normalisation
        norm = max(0.0, 1 - score / 5.0)
        return MetricValue("davies_bouldin", score, norm,
                           compute_time=time.perf_counter() - t)


class CalinskiHarabaszComputer:
    def compute(self, X: np.ndarray, labels: np.ndarray) -> MetricValue:
        valid_mask = labels != -1
        Xv, lv = X[valid_mask], labels[valid_mask]
        if len(np.unique(lv)) < 2:
            return MetricValue("calinski_harabasz", None, None, error="< 2 clusters")
        t = time.perf_counter()
        try:
            from sklearn.metrics import calinski_harabasz_score
            score = float(calinski_harabasz_score(Xv, lv))
        except Exception as e:
            return MetricValue("calinski_harabasz", None, None, error=str(e))
        # Normalise: log-scale, cap at 10000
        norm = float(np.log1p(score) / np.log1p(10_000))
        norm = min(1.0, norm)
        return MetricValue("calinski_harabasz", score, norm,
                           compute_time=time.perf_counter() - t)


class DunnIndexComputer:
    MAX_SAMPLES = 5_000

    def compute(self, X: np.ndarray, labels: np.ndarray) -> MetricValue:
        valid_mask = labels != -1
        Xv, lv = X[valid_mask], labels[valid_mask]
        unique_labels = np.unique(lv)
        if len(unique_labels) < 2:
            return MetricValue("dunn_index", None, None, error="< 2 clusters")

        # Sample for speed
        if len(Xv) > self.MAX_SAMPLES:
            idx = np.random.default_rng(42).choice(len(Xv), self.MAX_SAMPLES, replace=False)
            Xv, lv = Xv[idx], lv[idx]
            unique_labels = np.unique(lv)

        t = time.perf_counter()
        try:
            clusters = {c: Xv[lv == c] for c in unique_labels}
            # Min inter-cluster distance
            min_inter = np.inf
            cluster_list = list(clusters.values())
            for i in range(len(cluster_list)):
                for j in range(i + 1, len(cluster_list)):
                    if len(cluster_list[i]) == 0 or len(cluster_list[j]) == 0:
                        continue
                    ci = cluster_list[i][:100]
                    cj = cluster_list[j][:100]
                    dists = cdist(ci, cj, metric="euclidean")
                    min_inter = min(min_inter, float(dists.min()))

            # Max intra-cluster diameter
            max_intra = 0.0
            for pts in cluster_list:
                if len(pts) < 2:
                    continue
                sub = pts[:200]
                dists = cdist(sub, sub, metric="euclidean")
                max_intra = max(max_intra, float(dists.max()))

            if max_intra < 1e-12:
                return MetricValue("dunn_index", None, None, error="Zero intra-distance")

            score = float(min_inter / max_intra)
            norm = min(1.0, score)
        except Exception as e:
            return MetricValue("dunn_index", None, None, error=str(e))

        return MetricValue("dunn_index", score, norm,
                           compute_time=time.perf_counter() - t)


class XieBeniComputer:
    MAX_SAMPLES = 5_000

    def compute(self, X: np.ndarray, labels: np.ndarray) -> MetricValue:
        valid_mask = labels != -1
        Xv, lv = X[valid_mask], labels[valid_mask]
        unique_labels = np.unique(lv)
        if len(unique_labels) < 2:
            return MetricValue("xb_index", None, None, error="< 2 clusters")

        if len(Xv) > self.MAX_SAMPLES:
            idx = np.random.default_rng(42).choice(len(Xv), self.MAX_SAMPLES, replace=False)
            Xv, lv = Xv[idx], lv[idx]
            unique_labels = np.unique(lv)

        t = time.perf_counter()
        try:
            n = len(Xv)
            # Centroids
            centroids = np.array([Xv[lv == c].mean(axis=0) for c in unique_labels])
            # Compactness
            compactness = sum(
                float(np.sum((Xv[lv == c] - centroids[i]) ** 2))
                for i, c in enumerate(unique_labels)
            )
            # Min inter-centroid distance^2
            centroid_dists = cdist(centroids, centroids)
            np.fill_diagonal(centroid_dists, np.inf)
            min_sep2 = float(centroid_dists.min()) ** 2
            if min_sep2 < 1e-12:
                return MetricValue("xb_index", None, None, error="Identical centroids")
            score = compactness / (n * min_sep2)
            # Normalise: lower is better, map to [0,1] with soft cap
            norm = max(0.0, 1 - min(score / 10.0, 1.0))
        except Exception as e:
            return MetricValue("xb_index", None, None, error=str(e))

        return MetricValue("xb_index", score, norm,
                           compute_time=time.perf_counter() - t)


class InertiaComputer:
    def compute(self, X: np.ndarray, labels: np.ndarray) -> MetricValue:
        valid_mask = labels != -1
        Xv, lv = X[valid_mask], labels[valid_mask]
        unique_labels = np.unique(lv)
        if len(unique_labels) < 1:
            return MetricValue("inertia", None, None, error="No clusters")
        t = time.perf_counter()
        try:
            inertia = float(sum(
                np.sum((Xv[lv == c] - Xv[lv == c].mean(axis=0)) ** 2)
                for c in unique_labels
            ))
            # Normalise: log scale
            total_var = float(np.sum((Xv - Xv.mean(axis=0)) ** 2)) + 1e-10
            norm = max(0.0, 1 - inertia / total_var)
        except Exception as e:
            return MetricValue("inertia", None, None, error=str(e))
        return MetricValue("inertia", inertia, norm,
                           compute_time=time.perf_counter() - t)


class GeometricStatsComputer:
    MAX_SAMPLES = 3_000

    def compute(self, X: np.ndarray,
                labels: np.ndarray) -> Tuple[MetricValue, MetricValue,
                                              MetricValue, MetricValue,
                                              MetricValue]:
        """Returns: noise_ratio, balance, entropy, avg_intra, avg_inter"""
        n_total = len(labels)
        n_noise = int((labels == -1).sum())
        noise_ratio = n_noise / max(n_total, 1)

        valid_mask = labels != -1
        Xv, lv = X[valid_mask], labels[valid_mask]
        unique_labels = np.unique(lv)
        n_cls = len(unique_labels)

        # Balance
        sizes = [int((lv == c).sum()) for c in unique_labels] if n_cls > 0 else [0]
        balance = float(min(sizes) / max(sizes)) if max(sizes) > 0 else 0.0
        size_probs = np.array(sizes) / max(sum(sizes), 1)
        ent = float(entropy(size_probs + 1e-12))

        # Avg intra / inter distances
        avg_intra, avg_inter = None, None
        try:
            if n_cls >= 2 and len(Xv) > 0:
                # Sample for speed
                if len(Xv) > self.MAX_SAMPLES:
                    idx = np.random.default_rng(42).choice(
                        len(Xv), self.MAX_SAMPLES, replace=False)
                    Xvs, lvs = Xv[idx], lv[idx]
                    unique_s = np.unique(lvs)
                else:
                    Xvs, lvs, unique_s = Xv, lv, unique_labels

                centroids = {c: Xvs[lvs == c].mean(axis=0) for c in unique_s}
                intra_dists = []
                for c in unique_s:
                    pts = Xvs[lvs == c]
                    if len(pts) >= 2:
                        intra_dists.append(
                            float(np.mean(cdist(pts[:100], pts[:100])))
                        )
                avg_intra = float(np.mean(intra_dists)) if intra_dists else None

                cen_arr = np.array(list(centroids.values()))
                inter_dists = cdist(cen_arr, cen_arr)
                np.fill_diagonal(inter_dists, 0)
                avg_inter = float(inter_dists[inter_dists > 0].mean()) if len(cen_arr) > 1 else None
        except Exception:
            pass

        mv_noise   = MetricValue("noise_ratio", noise_ratio,
                                  max(0.0, 1 - noise_ratio))
        mv_balance = MetricValue("cluster_balance", balance, balance)
        mv_entropy = MetricValue("size_entropy", ent, min(1.0, ent / 3.0))
        mv_intra   = MetricValue("avg_intra_dist", avg_intra,
                                  max(0.0, 1 - min(avg_intra / 5.0, 1.0))
                                  if avg_intra is not None else None)
        mv_inter   = MetricValue("avg_inter_dist", avg_inter,
                                  min(1.0, avg_inter / 5.0)
                                  if avg_inter is not None else None)
        return mv_noise, mv_balance, mv_entropy, mv_intra, mv_inter


# ──────────────────────────────────────────────────────────────────
# COMPOSITE SCORER
# ──────────────────────────────────────────────────────────────────

class CompositeScorer:
    """
    Combines multiple normalised metric values into a single
    0–100 composite score using configurable weights.
    """

    DEFAULT_WEIGHTS = {
        "silhouette":        3.0,
        "davies_bouldin":    2.0,
        "calinski_harabasz": 2.0,
        "dunn_index":        2.0,
        "xb_index":          1.5,
        "noise_ratio":       1.5,
        "cluster_balance":   1.0,
        "avg_intra_dist":    1.0,
        "avg_inter_dist":    1.0,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def score(self, metrics: Dict[str, MetricValue]) -> float:
        """Returns 0–100 composite score."""
        total_weight = 0.0
        weighted_sum = 0.0
        for mid, mv in metrics.items():
            if not mv.is_valid or mv.normalised is None:
                continue
            w = self.weights.get(mid, 1.0)
            weighted_sum += mv.normalised * w
            total_weight += w
        if total_weight == 0:
            return 0.0
        return round((weighted_sum / total_weight) * 100, 2)


# ──────────────────────────────────────────────────────────────────
# NATURAL LANGUAGE INTERPRETER
# ──────────────────────────────────────────────────────────────────

class ClusteringInterpreter:
    """Generates concise natural-language summaries of clustering results."""

    def interpret(self, result: EvaluationResult) -> str:
        parts = []
        sil = result.metric_value("silhouette")
        db  = result.metric_value("davies_bouldin")
        ch  = result.metric_value("calinski_harabasz")

        score = result.composite_score
        if score >= 80:
            quality = "excellent"
        elif score >= 65:
            quality = "good"
        elif score >= 45:
            quality = "moderate"
        else:
            quality = "poor"

        parts.append(
            f"**{result.algorithm_name}** found {result.n_clusters} clusters "
            f"with a composite score of {result.composite_score:.1f}/100 ({quality} quality)."
        )

        if sil is not None:
            if sil > 0.7:
                parts.append(f"Silhouette = {sil:.3f}: clusters are dense and well-separated.")
            elif sil > 0.5:
                parts.append(f"Silhouette = {sil:.3f}: reasonably compact clusters.")
            elif sil > 0.25:
                parts.append(f"Silhouette = {sil:.3f}: some structure, but clusters overlap.")
            else:
                parts.append(f"Silhouette = {sil:.3f}: weak or inconsistent structure.")

        if db is not None:
            if db < 0.5:
                parts.append(f"Davies-Bouldin = {db:.3f}: excellent cluster separation.")
            elif db < 1.0:
                parts.append(f"Davies-Bouldin = {db:.3f}: acceptable separation.")
            else:
                parts.append(f"Davies-Bouldin = {db:.3f}: clusters may be overlapping.")

        if result.noise_ratio > 0.15:
            parts.append(
                f"⚠ {result.noise_ratio:.1%} of points labelled as noise — "
                "consider adjusting density parameters."
            )
        elif result.noise_ratio > 0.0:
            parts.append(f"{result.noise_ratio:.1%} noise points identified.")

        balance = result.metric_value("cluster_balance")
        if balance is not None and balance < 0.1:
            parts.append(
                "⚠ Very imbalanced clusters detected — one dominant cluster likely."
            )

        return " ".join(parts)


# ──────────────────────────────────────────────────────────────────
# MAIN EVALUATOR
# ──────────────────────────────────────────────────────────────────

class ClusteringEvaluator:
    """
    Evaluates a single clustering result against all metrics.
    Thread-safe; stateless after construction.
    """

    def __init__(self,
                 metric_weights: Optional[Dict[str, float]] = None,
                 fast_mode: bool = False):
        self._sil   = SilhouetteComputer()
        self._db    = DaviesBouldinComputer()
        self._ch    = CalinskiHarabaszComputer()
        self._dunn  = DunnIndexComputer()
        self._xb    = XieBeniComputer()
        self._inert = InertiaComputer()
        self._geo   = GeometricStatsComputer()
        self._score = CompositeScorer(weights=metric_weights)
        self._interp = ClusteringInterpreter()
        self.fast_mode = fast_mode

    def evaluate(self,
                 X: np.ndarray,
                 labels: np.ndarray,
                 algorithm_id: str,
                 algorithm_name: str,
                 algorithm_family: str,
                 runtime: float = 0.0) -> EvaluationResult:
        """Compute all metrics for one clustering result."""
        metrics: Dict[str, MetricValue] = {}
        warnings_list: List[str] = []

        # Validity check
        valid_labels = labels[labels != -1]
        n_clusters = len(np.unique(valid_labels)) if len(valid_labels) > 0 else 0
        n_noise = int((labels == -1).sum())
        noise_ratio = n_noise / max(len(labels), 1)

        if n_clusters < 2:
            warnings_list.append(f"Only {n_clusters} cluster(s) found — most metrics inapplicable")

        # --- Core metrics ---
        metrics["silhouette"]        = self._sil.compute(X, labels)
        metrics["davies_bouldin"]    = self._db.compute(X, labels)
        metrics["calinski_harabasz"] = self._ch.compute(X, labels)

        if not self.fast_mode:
            metrics["dunn_index"] = self._dunn.compute(X, labels)
            metrics["xb_index"]   = self._xb.compute(X, labels)

        metrics["inertia"]    = self._inert.compute(X, labels)

        # --- Geometric / balance metrics ---
        mv_noise, mv_bal, mv_ent, mv_intra, mv_inter = self._geo.compute(X, labels)
        metrics["noise_ratio"]      = mv_noise
        metrics["cluster_balance"]  = mv_bal
        metrics["size_entropy"]     = mv_ent
        metrics["avg_intra_dist"]   = mv_intra
        metrics["avg_inter_dist"]   = mv_inter

        # Collect errors as warnings
        for mid, mv in metrics.items():
            if mv.error:
                warnings_list.append(f"{mid}: {mv.error}")

        # Composite score
        composite = self._score.score(metrics)

        result = EvaluationResult(
            algorithm_id=algorithm_id,
            algorithm_name=algorithm_name,
            algorithm_family=algorithm_family,
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_ratio=noise_ratio,
            metrics=metrics,
            composite_score=composite,
            runtime_seconds=runtime,
            warnings=warnings_list,
        )
        result.interpretation = self._interp.interpret(result)
        return result

    def evaluate_batch(self,
                       X: np.ndarray,
                       clustering_results: List[Any]   # List[ClusteringResult]
                       ) -> List[EvaluationResult]:
        """Evaluate a batch of ClusteringResult objects."""
        eval_results = []
        for cr in clustering_results:
            if not cr.succeeded or len(cr.labels) != len(X):
                continue
            er = self.evaluate(
                X, cr.labels,
                algorithm_id=cr.algorithm_id,
                algorithm_name=cr.algorithm_name,
                algorithm_family=cr.algorithm_family,
                runtime=cr.runtime_seconds,
            )
            eval_results.append(er)
        return eval_results


# ──────────────────────────────────────────────────────────────────
# RANKER
# ──────────────────────────────────────────────────────────────────

class AlgorithmRanker:
    """
    Ranks evaluated algorithms using multiple strategies.
    """

    def rank_by_composite(self, results: List[EvaluationResult]
                          ) -> List[EvaluationResult]:
        """Sort by composite score descending, assign ranks."""
        ranked = sorted(results, key=lambda r: r.composite_score, reverse=True)
        for i, r in enumerate(ranked, 1):
            r.rank = i
        return ranked

    def rank_by_metric(self, results: List[EvaluationResult],
                       metric_id: str) -> List[EvaluationResult]:
        spec = METRIC_REGISTRY.get(metric_id)
        reverse = spec.direction == MetricDirection.HIGHER_BETTER if spec else True
        valid   = [r for r in results if r.metric_value(metric_id) is not None]
        invalid = [r for r in results if r.metric_value(metric_id) is None]
        ranked  = sorted(valid,
                         key=lambda r: r.metric_value(metric_id),
                         reverse=reverse)
        for i, r in enumerate(ranked + invalid, 1):
            r.rank = i
        return ranked + invalid

    def pareto_front(self, results: List[EvaluationResult],
                     objectives: Optional[List[str]] = None
                     ) -> List[EvaluationResult]:
        """Returns Pareto-optimal algorithms (non-dominated set)."""
        if objectives is None:
            objectives = ["silhouette", "davies_bouldin", "calinski_harabasz"]

        # Convert to maximisation (flip lower-is-better)
        def to_max(r: EvaluationResult, mid: str) -> Optional[float]:
            mv = r.metrics.get(mid)
            if mv is None or not mv.is_valid:
                return None
            spec = METRIC_REGISTRY.get(mid)
            if spec and spec.direction == MetricDirection.LOWER_BETTER:
                return -(mv.value or 0)
            return mv.value

        valid = [r for r in results
                 if all(to_max(r, m) is not None for m in objectives)]

        pareto = []
        for i, ri in enumerate(valid):
            dominated = False
            for j, rj in enumerate(valid):
                if i == j:
                    continue
                rj_better_all = all(
                    (to_max(rj, m) or -np.inf) >= (to_max(ri, m) or -np.inf)
                    for m in objectives
                )
                rj_better_one = any(
                    (to_max(rj, m) or -np.inf) > (to_max(ri, m) or -np.inf)
                    for m in objectives
                )
                if rj_better_all and rj_better_one:
                    dominated = True
                    break
            if not dominated:
                pareto.append(ri)
        return pareto

    def elbow_optimal_k(self, k_sweep_results: List[EvaluationResult]
                        ) -> Optional[int]:
        """Find optimal k from a k-sweep via silhouette elbow."""
        valid = sorted(
            [(r.n_clusters, r.metric_value("silhouette"))
             for r in k_sweep_results
             if r.metric_value("silhouette") is not None],
            key=lambda x: x[0]
        )
        if len(valid) < 3:
            return valid[0][0] if valid else None
        ks    = [v[0] for v in valid]
        scores = [v[1] for v in valid]
        # Best silhouette
        best_idx = int(np.argmax(scores))
        return ks[best_idx]


# ──────────────────────────────────────────────────────────────────
# RESULTS TABLE BUILDER
# ──────────────────────────────────────────────────────────────────

class ResultsTableBuilder:
    """Converts evaluation results to pandas DataFrames for display."""

    COLUMN_DISPLAY_ORDER = [
        "Rank", "Algorithm", "Family", "k", "Score",
        "Silhouette Score", "Davies-Bouldin Index",
        "Calinski-Harabász Score", "Dunn Index",
        "Noise%", "Time(s)",
    ]

    def build(self, results: List[EvaluationResult],
              include_all_metrics: bool = False) -> pd.DataFrame:
        rows = [r.to_dataframe_row() for r in results]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Reorder columns
        existing = [c for c in self.COLUMN_DISPLAY_ORDER if c in df.columns]
        extra = [c for c in df.columns if c not in existing]
        df = df[existing + (extra if include_all_metrics else [])]
        return df

    def build_metric_comparison(self,
                                 results: List[EvaluationResult],
                                 metric_ids: Optional[List[str]] = None
                                 ) -> pd.DataFrame:
        if metric_ids is None:
            metric_ids = list(METRIC_REGISTRY.keys())

        rows = []
        for r in results:
            row = {"Algorithm": r.algorithm_name}
            for mid in metric_ids:
                mv = r.metrics.get(mid)
                row[METRIC_REGISTRY.get(mid, MetricSpec(mid, mid,
                    MetricCategory.COMPACTNESS, MetricDirection.HIGHER_BETTER,
                    None, (None, None), "", "")).name] = (
                    round(mv.value, 4) if mv and mv.is_valid else None
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def build_score_heatmap_data(self,
                                  results: List[EvaluationResult]
                                  ) -> pd.DataFrame:
        """Normalised scores [0,1] for heatmap display."""
        metric_ids = [
            "silhouette", "davies_bouldin", "calinski_harabasz",
            "dunn_index", "xb_index", "noise_ratio", "cluster_balance",
        ]
        rows = []
        for r in results:
            row = {"Algorithm": r.algorithm_name[:30]}
            for mid in metric_ids:
                mv = r.metrics.get(mid)
                row[mid] = round(mv.normalised, 3) if mv and mv.normalised is not None else 0.0
            rows.append(row)
        df = pd.DataFrame(rows).set_index("Algorithm")
        return df


# ──────────────────────────────────────────────────────────────────
# K-SWEEP ANALYSER
# ──────────────────────────────────────────────────────────────────

class KSweepAnalyser:
    """Runs and analyses a k-sweep for any algorithm."""

    def __init__(self, evaluator: Optional[ClusteringEvaluator] = None):
        self._eval = evaluator or ClusteringEvaluator(fast_mode=True)
        self._ranker = AlgorithmRanker()

    def run_sweep(self,
                  algorithm_id: str,
                  X: np.ndarray,
                  k_range: range,
                  extra_params: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
        from clustering_runner import SingleAlgorithmExecutor, RunnerConfig
        config = RunnerConfig(n_clusters=2)
        executor = SingleAlgorithmExecutor(config)
        extra_params = extra_params or {}

        metrics_by_k: Dict[str, List] = {
            "k": [], "silhouette": [], "davies_bouldin": [],
            "calinski_harabasz": [], "composite": [],
        }

        for k in k_range:
            params = {"n_clusters": k, **extra_params}
            cr = executor.run(algorithm_id, X, params)
            if not cr.succeeded or len(cr.labels) != len(X):
                continue
            er = self._eval.evaluate(
                X, cr.labels, algorithm_id, algorithm_id, "unknown")
            metrics_by_k["k"].append(k)
            metrics_by_k["silhouette"].append(er.metric_value("silhouette"))
            metrics_by_k["davies_bouldin"].append(er.metric_value("davies_bouldin"))
            metrics_by_k["calinski_harabasz"].append(er.metric_value("calinski_harabasz"))
            metrics_by_k["composite"].append(er.composite_score)

        optimal_k = None
        sil_vals = [v for v in metrics_by_k["silhouette"] if v is not None]
        ks = metrics_by_k["k"]
        if sil_vals and ks:
            best_idx = int(np.argmax(sil_vals))
            optimal_k = ks[best_idx] if best_idx < len(ks) else None

        return {
            "metrics_by_k": metrics_by_k,
            "optimal_k": optimal_k,
            "algorithm_id": algorithm_id,
        }


# ──────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def evaluate_all(X: np.ndarray,
                 batch_result: Any,
                 fast_mode: bool = False) -> List[EvaluationResult]:
    """Top-level: evaluate all successful results in a BatchRunResult."""
    evaluator = ClusteringEvaluator(fast_mode=fast_mode)
    ranker = AlgorithmRanker()
    results_list = list(batch_result.successful().values())
    eval_results = evaluator.evaluate_batch(X, results_list)
    eval_results = ranker.rank_by_composite(eval_results)
    return eval_results


def build_results_dataframe(eval_results: List[EvaluationResult]) -> pd.DataFrame:
    builder = ResultsTableBuilder()
    return builder.build(eval_results)


def get_best_algorithm(eval_results: List[EvaluationResult]
                       ) -> Optional[EvaluationResult]:
    if not eval_results:
        return None
    return max(eval_results, key=lambda r: r.composite_score)


def get_metric_description(metric_id: str) -> Optional[MetricSpec]:
    return METRIC_REGISTRY.get(metric_id)


def metric_names() -> List[str]:
    return list(METRIC_REGISTRY.keys())


def silhouette_grade(score: float) -> str:
    if score >= 0.7:   return "Strong structure"
    if score >= 0.5:   return "Reasonable structure"
    if score >= 0.25:  return "Weak structure"
    if score >= 0.0:   return "No substantial structure"
    return "Possibly incorrect labelling"


def format_metric_value(metric_id: str, value: float) -> str:
    """Format a metric value with appropriate precision and unit."""
    if value is None:
        return "N/A"
    spec = METRIC_REGISTRY.get(metric_id)
    if metric_id == "noise_ratio":
        return f"{value * 100:.1f}%"
    if metric_id in ("silhouette", "cluster_balance"):
        return f"{value:.4f}"
    if metric_id == "calinski_harabasz":
        return f"{value:.1f}"
    return f"{value:.4f}"


# ══════════════════════════════════════════════════════════════════
# FINAL POLISH — WORLD-CLASS ADDITIONS
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# HOPKINS STATISTIC — CLUSTERABILITY TEST
# ──────────────────────────────────────────────────────────────────

class HopkinsStatistic:
    """
    Measures clustering tendency BEFORE running any algorithm.
    H ≈ 0.5  → random (no cluster structure)
    H → 1.0  → highly clusterable
    H → 0.0  → uniformly distributed

    Essential first step — if Hopkins < 0.5, clustering is meaningless.
    """
    def __init__(self, n_samples: int = 150, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state

    def compute(self, X: np.ndarray) -> Dict[str, Any]:
        n, d = X.shape
        m = min(self.n_samples, max(n // 5, 5), 200)
        if m < 5:
            return {"hopkins": None, "interpretation": "Too few samples", "is_clusterable": None}
        rng = np.random.default_rng(self.random_state)
        try:
            from sklearn.neighbors import NearestNeighbors
            idx = rng.choice(n, m, replace=False)
            X_sample = X[idx]
            X_min, X_max = X.min(axis=0), X.max(axis=0)
            X_range = np.where((X_max - X_min) < 1e-10, 1.0, X_max - X_min)
            X_rand = rng.uniform(0, 1, (m, d)) * X_range + X_min
            nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(X)
            u = nbrs.kneighbors(X_rand)[0][:, 0]
            w = nbrs.kneighbors(X_sample)[0][:, 1]
            sum_u = float(np.sum(u ** d))
            sum_w = float(np.sum(w ** d))
            denom = sum_u + sum_w
            if denom < 1e-15:
                return {"hopkins": None, "interpretation": "Degenerate", "is_clusterable": None}
            H = sum_u / denom
        except Exception as e:
            return {"hopkins": None, "interpretation": str(e), "is_clusterable": None}
        interp, clusterable = self._interpret(H)
        return {"hopkins": round(float(H), 4), "interpretation": interp,
                "is_clusterable": clusterable, "m_used": m,
                "recommendation": self._recommend(H)}

    @staticmethod
    def _interpret(H: float) -> tuple:
        if H >= 0.75: return "Highly clusterable — strong spatial structure.", True
        if H >= 0.60: return "Moderately clusterable — reasonable structure.", True
        if H >= 0.50: return "Weakly clusterable — proceed with caution.", False
        if H >= 0.40: return "Near-random — clustering may not be meaningful.", False
        return "Uniformly distributed — clustering likely meaningless.", False

    @staticmethod
    def _recommend(H: float) -> str:
        if H >= 0.75: return "Proceed confidently. Most algorithms will reveal structure."
        if H >= 0.60: return "Viable. Prefer robust algorithms (HDBSCAN, Spectral, GMM)."
        if H >= 0.50: return "Use density-based algorithms and validate carefully."
        return "Consider dimensionality reduction or domain filtering before clustering."


# ──────────────────────────────────────────────────────────────────
# GAP STATISTIC — OPTIMAL K SELECTOR
# ──────────────────────────────────────────────────────────────────

class GapStatistic:
    """
    Tibshirani et al. (2001) Gap Statistic.
    Compares within-cluster dispersion against reference uniform distribution.
    Gold standard for optimal k selection.
    """
    def __init__(self, k_range=None, n_refs: int = 10, random_state: int = 42):
        self.k_range = k_range or range(1, 11)
        self.n_refs = n_refs
        self.random_state = random_state

    def compute(self, X: np.ndarray) -> Dict[str, Any]:
        from sklearn.cluster import KMeans
        rng = np.random.default_rng(self.random_state)
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_range = np.where((X_max - X_min) < 1e-10, 1.0, X_max - X_min)

        gaps, sks, log_wks, log_wk_refs_mean = [], [], [], []
        for k in self.k_range:
            km = KMeans(n_clusters=k, n_init=3, random_state=self.random_state, max_iter=200)
            km.fit(X)
            wk = self._wcd(X, km.labels_, km.cluster_centers_)
            log_wk = np.log(max(wk, 1e-15))
            log_wks.append(log_wk)
            ref_lwks = []
            for b in range(self.n_refs):
                Xr = rng.uniform(0, 1, X.shape) * X_range + X_min
                km_r = KMeans(n_clusters=k, n_init=1, random_state=b, max_iter=100)
                km_r.fit(Xr)
                wkr = self._wcd(Xr, km_r.labels_, km_r.cluster_centers_)
                ref_lwks.append(np.log(max(wkr, 1e-15)))
            ra = np.array(ref_lwks)
            gap_k = float(ra.mean() - log_wk)
            sdk = float(ra.std() * np.sqrt(1 + 1 / self.n_refs))
            gaps.append(gap_k); sks.append(sdk)
            log_wk_refs_mean.append(float(ra.mean()))

        # Optimal k
        optimal_k = list(self.k_range)[0]
        ks = list(self.k_range)
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - sks[i + 1]:
                optimal_k = ks[i]; break
        else:
            optimal_k = ks[int(np.argmax(gaps))]

        return {"k_values": ks, "gaps": [round(g,4) for g in gaps],
                "sk": [round(s,4) for s in sks],
                "log_wk": [round(w,4) for w in log_wks],
                "log_wk_ref": [round(w,4) for w in log_wk_refs_mean],
                "optimal_k": int(optimal_k)}

    @staticmethod
    def _wcd(X, labels, centers) -> float:
        total = 0.0
        for k, c in enumerate(centers):
            pts = X[labels == k]
            if len(pts) > 1:
                total += float(np.sum((pts - c) ** 2)) / (2 * len(pts))
        return total


# ──────────────────────────────────────────────────────────────────
# DENSITY-BASED CLUSTER VALIDITY (DBCV)
# ──────────────────────────────────────────────────────────────────

class DBCVIndex:
    """
    Moulavi et al. (2014) DBCV — proper validity for density-based clusters.
    Range [-1, 1]. Unlike silhouette, correctly handles arbitrary shapes.
    """
    MAX_SAMPLES = 3000

    def compute(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        valid = labels != -1
        Xv, lv = X[valid], labels[valid]
        unique = np.unique(lv)
        if len(unique) < 2:
            return {"dbcv": None, "error": "< 2 clusters"}
        if len(Xv) > self.MAX_SAMPLES:
            idx = np.random.default_rng(42).choice(len(Xv), self.MAX_SAMPLES, replace=False)
            Xv, lv = Xv[idx], lv[idx]
            unique = np.unique(lv)
        try:
            clusters = {c: Xv[lv == c] for c in unique}
            vc_list = []
            for c in unique:
                pts = clusters[c]
                if len(pts) < 2: vc_list.append(0.0); continue
                int_s = self._internal(pts)
                ext_s = self._external(pts, clusters, c)
                vc = (ext_s - int_s) / max(ext_s, int_s, 1e-10)
                vc_list.append(float(np.clip(vc, -1, 1)))
            sizes = np.array([len(clusters[c]) for c in unique])
            dbcv = float(np.average(vc_list, weights=sizes))
            return {"dbcv": round(dbcv, 4),
                    "per_cluster": {int(c): round(v, 4) for c, v in zip(unique, vc_list)},
                    "interpretation": self._interp(dbcv)}
        except Exception as e:
            return {"dbcv": None, "error": str(e)}

    @staticmethod
    def _core_dist(pts, min_pts=5):
        from sklearn.neighbors import NearestNeighbors
        k = min(min_pts, len(pts)-1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(pts)
        return nbrs.kneighbors(pts)[0][:, -1]

    def _internal(self, pts):
        cd = self._core_dist(pts)
        mr = np.maximum(cd[:, None], cd[None, :])
        np.fill_diagonal(mr, 0)
        return float(mr.max())

    def _external(self, pts_c, clusters, cid):
        from scipy.spatial.distance import cdist
        min_sep = np.inf
        for c2, pts2 in clusters.items():
            if c2 == cid or len(pts2) < 1: continue
            d = cdist(pts_c[:40], pts2[:40]).min()
            min_sep = min(min_sep, float(d))
        return min_sep if min_sep != np.inf else 0.0

    @staticmethod
    def _interp(v):
        if v >= 0.6:  return "Excellent density-based structure"
        if v >= 0.35: return "Good density-based structure"
        if v >= 0.1:  return "Moderate structure"
        if v >= -0.1: return "Weak structure"
        return "Poor density-based structure"


# ──────────────────────────────────────────────────────────────────
# BOOTSTRAP CONFIDENCE INTERVALS FOR METRICS
# ──────────────────────────────────────────────────────────────────

class MetricBootstrapCI:
    """Bootstrap CI for silhouette score — quantifies metric uncertainty."""
    def __init__(self, n_bootstrap=50, confidence=0.95, max_sample=5000, random_state=42):
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.max_sample = max_sample
        self.rng = np.random.default_rng(random_state)

    def silhouette_ci(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        from sklearn.metrics import silhouette_score
        valid = labels != -1
        Xv, lv = X[valid], labels[valid]
        if len(np.unique(lv)) < 2 or len(Xv) < 10:
            return {"mean": None, "ci_lower": None, "ci_upper": None}
        n = len(Xv); sn = min(n, self.max_sample)
        scores = []
        for _ in range(self.n_bootstrap):
            idx = self.rng.choice(n, sn, replace=True)
            Xs, ls = Xv[idx], lv[idx]
            if len(np.unique(ls)) < 2: continue
            try:
                scores.append(float(silhouette_score(Xs, ls)))
            except Exception:
                pass
        if not scores:
            return {"mean": None, "ci_lower": None, "ci_upper": None}
        a = 1 - self.confidence
        return {"mean": round(float(np.mean(scores)), 4),
                "ci_lower": round(float(np.percentile(scores, 100*a/2)), 4),
                "ci_upper": round(float(np.percentile(scores, 100*(1-a/2))), 4),
                "std": round(float(np.std(scores)), 4),
                "n_bootstrap": len(scores)}


# ──────────────────────────────────────────────────────────────────
# CLUSTER SEPARABILITY MATRIX
# ──────────────────────────────────────────────────────────────────

class ClusterSeparabilityMatrix:
    """
    Computes pairwise separability between clusters using
    Bhattacharyya distance and Mahalanobis distance.
    Reveals which cluster pairs are confused.
    """
    def compute(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        unique = [c for c in np.unique(labels) if c != -1]
        n_cls = len(unique)
        if n_cls < 2:
            return {"matrix": None, "worst_pair": None}
        clusters = {c: X[labels == c] for c in unique}
        means = {c: pts.mean(axis=0) for c, pts in clusters.items()}
        sep_mat = np.zeros((n_cls, n_cls))
        from scipy.spatial.distance import mahalanobis
        from scipy.linalg import inv
        try:
            cov_pooled = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
            cov_inv = inv(cov_pooled)
        except Exception:
            cov_inv = np.eye(X.shape[1])
        worst_dist, worst_pair = np.inf, None
        for i, ci in enumerate(unique):
            for j, cj in enumerate(unique):
                if i == j: continue
                try:
                    d = float(mahalanobis(means[ci], means[cj], cov_inv))
                except Exception:
                    d = float(np.linalg.norm(means[ci] - means[cj]))
                sep_mat[i, j] = round(d, 4)
                if i < j and d < worst_dist:
                    worst_dist = d; worst_pair = (int(ci), int(cj))
        df = pd.DataFrame(sep_mat,
                          index=[f"C{c}" for c in unique],
                          columns=[f"C{c}" for c in unique])
        return {"matrix": df, "worst_pair": worst_pair,
                "min_separation": round(worst_dist, 4) if worst_dist != np.inf else None}


# ──────────────────────────────────────────────────────────────────
# FULL CLUSTERABILITY REPORT
# ──────────────────────────────────────────────────────────────────

def run_clusterability_analysis(X: np.ndarray) -> Dict[str, Any]:
    """Hopkins statistic + PCA intrinsic dim + variance analysis."""
    hop = HopkinsStatistic().compute(X)
    n, d = X.shape
    from sklearn.decomposition import PCA
    nc = min(d, n-1, 30)
    pca = PCA(n_components=nc).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    intrinsic = int(np.searchsorted(cum, 0.90)) + 1
    low_var = int((X.var(axis=0) < 0.001).sum())
    return {"n_samples": n, "n_features": d,
            "hopkins": hop, "intrinsic_dim_90pct": intrinsic,
            "low_var_features": low_var,
            "is_clusterable": hop.get("is_clusterable"),
            "recommendation": hop.get("recommendation", "")}


def run_gap_statistic(X: np.ndarray, k_range=None, n_refs: int = 10) -> Dict[str, Any]:
    return GapStatistic(k_range=k_range or range(1, 11), n_refs=n_refs).compute(X)


def compute_dbcv(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    return DBCVIndex().compute(X, labels)

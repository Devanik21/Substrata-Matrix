"""
evaluation.py — ClusterX Comprehensive Clustering Evaluation Engine
=====================================================================
Handles: internal metrics, external metrics, cluster statistics, gap statistic,
         per-cluster analysis, ranking, and JSON-exportable report generation.
Author: ClusterX Intelligence Lab
"""

from __future__ import annotations

import time
import warnings
import logging
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
    fowlkes_mallows_score, v_measure_score,
    adjusted_mutual_info_score, homogeneity_score,
    completeness_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    """Single metric computation result."""
    name: str
    value: float
    display_name: str = ""
    higher_is_better: bool = True
    category: str = "internal"
    description: str = ""
    computation_time: float = 0.0
    error: Optional[str] = None


@dataclass
class ClusterStats:
    """Statistics for a single cluster."""
    cluster_id: int
    size: int
    pct_of_total: float
    centroid: Optional[np.ndarray]
    mean_intra_distance: float
    max_intra_distance: float
    std_intra_distance: float
    compactness: float
    nearest_cluster: int
    nearest_cluster_distance: float
    feature_means: Optional[Dict[str, float]] = None
    feature_stds: Optional[Dict[str, float]] = None


@dataclass
class ClusteringReport:
    """Full evaluation report for one clustering result."""
    algorithm_name: str
    n_clusters: int
    n_noise: int
    n_samples: int
    metrics: Dict[str, MetricResult]
    cluster_stats: List[ClusterStats]
    ranking_score: float = 0.0
    total_eval_time: float = 0.0

    def to_dict(self) -> Dict:
        d = {
            "algorithm": self.algorithm_name,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "n_samples": self.n_samples,
            "ranking_score": self.ranking_score,
            "total_eval_time": self.total_eval_time,
            "metrics": {},
            "cluster_stats": [],
        }
        for k, m in self.metrics.items():
            d["metrics"][k] = {
                "value": m.value, "higher_is_better": m.higher_is_better,
                "category": m.category,
            }
        for cs in self.cluster_stats:
            d["cluster_stats"].append({
                "cluster_id": cs.cluster_id, "size": cs.size,
                "pct": cs.pct_of_total, "compactness": cs.compactness,
                "nearest_cluster": cs.nearest_cluster,
                "nearest_distance": cs.nearest_cluster_distance,
            })
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ──────────────────────────────────────────────────────────────────
# INTERNAL METRICS
# ──────────────────────────────────────────────────────────────────

class InternalMetrics:
    """Computes clustering quality metrics that don't require ground truth."""

    def __init__(self, sample_size: int = 5000):
        self.sample_size = sample_size

    def silhouette(self, X: np.ndarray, labels: np.ndarray) -> MetricResult:
        t0 = time.perf_counter()
        try:
            n = min(self.sample_size, len(X))
            val = silhouette_score(X, labels, sample_size=n)
            return MetricResult("silhouette", float(val), "Silhouette Score",
                                True, "internal", "Mean silhouette coefficient [-1,1]",
                                time.perf_counter() - t0)
        except Exception as e:
            return MetricResult("silhouette", -1.0, "Silhouette Score",
                                error=str(e), computation_time=time.perf_counter() - t0)

    def silhouette_samples_array(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        try:
            n = min(self.sample_size, len(X))
            if n < len(X):
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X), n, replace=False)
                return silhouette_samples(X[idx], labels[idx])
            return silhouette_samples(X, labels)
        except Exception:
            return np.zeros(min(self.sample_size, len(X)))

    def davies_bouldin(self, X: np.ndarray, labels: np.ndarray) -> MetricResult:
        t0 = time.perf_counter()
        try:
            val = davies_bouldin_score(X, labels)
            return MetricResult("davies_bouldin", float(val), "Davies-Bouldin Index",
                                False, "internal", "Average cluster similarity ratio (lower=better)",
                                time.perf_counter() - t0)
        except Exception as e:
            return MetricResult("davies_bouldin", 999.0, "Davies-Bouldin Index",
                                error=str(e), computation_time=time.perf_counter() - t0)

    def calinski_harabasz(self, X: np.ndarray, labels: np.ndarray) -> MetricResult:
        t0 = time.perf_counter()
        try:
            val = calinski_harabasz_score(X, labels)
            return MetricResult("calinski_harabasz", float(val), "Calinski-Harabasz Index",
                                True, "internal", "Ratio of between/within cluster dispersion",
                                time.perf_counter() - t0)
        except Exception as e:
            return MetricResult("calinski_harabasz", 0.0, "Calinski-Harabasz Index",
                                error=str(e), computation_time=time.perf_counter() - t0)

    def dunn_index(self, X: np.ndarray, labels: np.ndarray) -> MetricResult:
        t0 = time.perf_counter()
        try:
            unique = [l for l in np.unique(labels) if l >= 0]
            if len(unique) < 2:
                return MetricResult("dunn_index", 0.0, "Dunn Index",
                                    computation_time=time.perf_counter() - t0)
            n_sample = min(500, len(X))
            if n_sample < len(X):
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X), n_sample, replace=False)
                X_s, labels_s = X[idx], labels[idx]
            else:
                X_s, labels_s = X, labels
            max_intra = 0.0
            for l in unique:
                pts = X_s[labels_s == l]
                if len(pts) > 1:
                    d = pdist(pts)
                    if len(d) > 0:
                        max_intra = max(max_intra, d.max())
            if max_intra == 0:
                max_intra = 1e-16
            min_inter = np.inf
            for i, l1 in enumerate(unique):
                for l2 in unique[i + 1:]:
                    pts1 = X_s[labels_s == l1]
                    pts2 = X_s[labels_s == l2]
                    if len(pts1) > 0 and len(pts2) > 0:
                        d = cdist(pts1[:50], pts2[:50]).min()
                        min_inter = min(min_inter, d)
            val = float(min_inter / max_intra) if np.isfinite(min_inter) else 0.0
            return MetricResult("dunn_index", val, "Dunn Index",
                                True, "internal", "Ratio of min inter-cluster to max intra-cluster distance",
                                time.perf_counter() - t0)
        except Exception as e:
            return MetricResult("dunn_index", 0.0, "Dunn Index",
                                error=str(e), computation_time=time.perf_counter() - t0)

    def xie_beni(self, X: np.ndarray, labels: np.ndarray,
                 centers: Optional[np.ndarray] = None) -> MetricResult:
        t0 = time.perf_counter()
        try:
            unique = [l for l in np.unique(labels) if l >= 0]
            if centers is None:
                centers = np.array([X[labels == l].mean(axis=0) for l in unique])
            n = len(X)
            total_var = 0.0
            for i, l in enumerate(unique):
                pts = X[labels == l]
                total_var += np.sum((pts - centers[i]) ** 2)
            if len(centers) >= 2:
                min_sep = cdist(centers, centers)
                np.fill_diagonal(min_sep, np.inf)
                min_sep = min_sep.min() ** 2
            else:
                min_sep = 1.0
            val = total_var / (n * max(min_sep, 1e-16))
            return MetricResult("xie_beni", float(val), "Xie-Beni Index",
                                False, "internal", "Compactness vs separation ratio (lower=better)",
                                time.perf_counter() - t0)
        except Exception as e:
            return MetricResult("xie_beni", 999.0, "Xie-Beni Index",
                                error=str(e), computation_time=time.perf_counter() - t0)

    def inertia(self, X: np.ndarray, labels: np.ndarray,
                centers: Optional[np.ndarray] = None) -> MetricResult:
        t0 = time.perf_counter()
        try:
            unique = [l for l in np.unique(labels) if l >= 0]
            if centers is None:
                centers = np.array([X[labels == l].mean(axis=0) for l in unique])
            total = 0.0
            for i, l in enumerate(unique):
                pts = X[labels == l]
                total += np.sum((pts - centers[i]) ** 2)
            return MetricResult("inertia", float(total), "Inertia (WCSS)",
                                False, "internal", "Within-cluster sum of squares",
                                time.perf_counter() - t0)
        except Exception as e:
            return MetricResult("inertia", 0.0, "Inertia (WCSS)",
                                error=str(e), computation_time=time.perf_counter() - t0)

    def s_dbw(self, X: np.ndarray, labels: np.ndarray) -> MetricResult:
        """S_Dbw validity index (simplified)."""
        t0 = time.perf_counter()
        try:
            unique = [l for l in np.unique(labels) if l >= 0]
            if len(unique) < 2:
                return MetricResult("s_dbw", 999.0, "S_Dbw Index",
                                    False, "internal", computation_time=time.perf_counter() - t0)
            centers = np.array([X[labels == l].mean(axis=0) for l in unique])
            scatters = []
            for i, l in enumerate(unique):
                pts = X[labels == l]
                if len(pts) > 1:
                    var_c = np.mean(np.var(pts, axis=0))
                else:
                    var_c = 0.0
                scatters.append(var_c)
            total_var = np.mean(np.var(X, axis=0))
            scat = np.mean(scatters) / max(total_var, 1e-16)
            dens_sum = 0.0
            count_pairs = 0
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    mid = (centers[i] + centers[j]) / 2.0
                    pts_i = X[labels == unique[i]]
                    pts_j = X[labels == unique[j]]
                    stdev = np.sqrt(np.mean(scatters))
                    if stdev == 0:
                        stdev = 1e-16
                    d_ij = np.sum(np.linalg.norm(pts_i - mid, axis=1) <= stdev)
                    d_ij += np.sum(np.linalg.norm(pts_j - mid, axis=1) <= stdev)
                    d_i = np.sum(np.linalg.norm(pts_i - centers[i], axis=1) <= stdev)
                    d_j = np.sum(np.linalg.norm(pts_j - centers[j], axis=1) <= stdev)
                    denom = max(d_i, d_j, 1)
                    dens_sum += d_ij / denom
                    count_pairs += 1
            dens = dens_sum / max(count_pairs, 1)
            val = scat + dens
            return MetricResult("s_dbw", float(val), "S_Dbw Index",
                                False, "internal", "Scatter + density between clusters (lower=better)",
                                time.perf_counter() - t0)
        except Exception as e:
            return MetricResult("s_dbw", 999.0, "S_Dbw Index",
                                error=str(e), computation_time=time.perf_counter() - t0)


# ──────────────────────────────────────────────────────────────────
# EXTERNAL METRICS (require ground truth)
# ──────────────────────────────────────────────────────────────────

class ExternalMetrics:
    """Metrics that compare clustering to known ground truth labels."""

    @staticmethod
    def compute_all(labels_true: np.ndarray,
                    labels_pred: np.ndarray) -> Dict[str, MetricResult]:
        results = {}
        metric_funcs = [
            ("ari", "Adjusted Rand Index", adjusted_rand_score,
             "Similarity adjusted for chance [-1,1]"),
            ("nmi", "Normalized Mutual Information", normalized_mutual_info_score,
             "Mutual information normalized to [0,1]"),
            ("ami", "Adjusted Mutual Information", adjusted_mutual_info_score,
             "AMI adjusted for chance"),
            ("fmi", "Fowlkes-Mallows Index", fowlkes_mallows_score,
             "Geometric mean of precision and recall [0,1]"),
            ("v_measure", "V-Measure", v_measure_score,
             "Harmonic mean of homogeneity and completeness"),
            ("homogeneity", "Homogeneity", homogeneity_score,
             "Each cluster contains only members of a single class"),
            ("completeness", "Completeness", completeness_score,
             "All members of a given class are assigned to the same cluster"),
        ]
        for name, display, func, desc in metric_funcs:
            try:
                t0 = time.perf_counter()
                val = float(func(labels_true, labels_pred))
                results[name] = MetricResult(
                    name, val, display, True, "external", desc,
                    time.perf_counter() - t0,
                )
            except Exception as e:
                results[name] = MetricResult(name, 0.0, display, error=str(e))
        return results

    @staticmethod
    def purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> MetricResult:
        try:
            t0 = time.perf_counter()
            cm = contingency_matrix(labels_true, labels_pred)
            val = float(cm.max(axis=0).sum()) / len(labels_true)
            return MetricResult("purity", val, "Cluster Purity", True, "external",
                                "Fraction of correctly assigned samples",
                                time.perf_counter() - t0)
        except Exception as e:
            return MetricResult("purity", 0.0, "Cluster Purity", error=str(e))


# ──────────────────────────────────────────────────────────────────
# GAP STATISTIC
# ──────────────────────────────────────────────────────────────────

class GapStatistic:
    """Full Monte Carlo Gap Statistic implementation."""

    def __init__(self, n_references: int = 10, random_state: int = 42):
        self.n_refs = n_references
        self.random_state = random_state

    def compute(self, X: np.ndarray, k_range: Tuple[int, int] = (2, 10),
                callback=None) -> Dict[str, Any]:
        rng = np.random.RandomState(self.random_state)
        k_values = list(range(k_range[0], k_range[1] + 1))
        n, d = X.shape
        gaps = []
        gap_stds = []
        inertias_real = []

        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        for idx, k in enumerate(k_values):
            km = KMeans(n_clusters=k, n_init=5, max_iter=200,
                        random_state=self.random_state)
            km.fit(X)
            wk = np.log(max(km.inertia_, 1e-16))
            inertias_real.append(float(km.inertia_))

            ref_wks = []
            for _ in range(self.n_refs):
                X_ref = rng.uniform(mins, maxs, size=(n, d))
                km_ref = KMeans(n_clusters=k, n_init=3, max_iter=100,
                                random_state=self.random_state)
                km_ref.fit(X_ref)
                ref_wks.append(np.log(max(km_ref.inertia_, 1e-16)))

            gap = np.mean(ref_wks) - wk
            sdk = np.std(ref_wks) * np.sqrt(1 + 1.0 / self.n_refs)
            gaps.append(float(gap))
            gap_stds.append(float(sdk))

            if callback:
                callback(f"Gap stat k={k}", (idx + 1) / len(k_values))

        optimal_k = k_values[0]
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - gap_stds[i + 1]:
                optimal_k = k_values[i]
                break
        else:
            optimal_k = k_values[int(np.argmax(gaps))]

        return {
            "k_values": k_values,
            "gaps": gaps,
            "gap_stds": gap_stds,
            "inertias": inertias_real,
            "optimal_k": optimal_k,
        }


# ──────────────────────────────────────────────────────────────────
# CLUSTER-LEVEL STATISTICS
# ──────────────────────────────────────────────────────────────────

class ClusterStatisticsComputer:
    """Computes detailed per-cluster statistics."""

    def compute(self, X: np.ndarray, labels: np.ndarray,
                feature_names: Optional[List[str]] = None) -> List[ClusterStats]:
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        n_total = len(labels)
        stats_list: List[ClusterStats] = []

        centroids = {}
        for l in unique:
            centroids[l] = X[labels == l].mean(axis=0)

        for l in unique:
            pts = X[labels == l]
            n_pts = len(pts)
            centroid = centroids[l]
            dists_to_center = np.linalg.norm(pts - centroid, axis=1)
            mean_d = float(dists_to_center.mean()) if n_pts > 0 else 0.0
            max_d = float(dists_to_center.max()) if n_pts > 0 else 0.0
            std_d = float(dists_to_center.std()) if n_pts > 1 else 0.0
            compactness = mean_d

            nearest_c = -1
            nearest_dist = np.inf
            for l2 in unique:
                if l2 != l:
                    d = float(np.linalg.norm(centroid - centroids[l2]))
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_c = l2

            feat_means = None
            feat_stds = None
            if feature_names and len(feature_names) == X.shape[1]:
                feat_means = {fn: float(pts[:, i].mean()) for i, fn in enumerate(feature_names)}
                feat_stds = {fn: float(pts[:, i].std()) for i, fn in enumerate(feature_names)}

            stats_list.append(ClusterStats(
                cluster_id=int(l), size=n_pts,
                pct_of_total=round(n_pts / max(n_total, 1) * 100, 2),
                centroid=centroid, mean_intra_distance=mean_d,
                max_intra_distance=max_d, std_intra_distance=std_d,
                compactness=compactness, nearest_cluster=nearest_c,
                nearest_cluster_distance=float(nearest_dist),
                feature_means=feat_means, feature_stds=feat_stds,
            ))
        return stats_list

    def between_cluster_distances(self, X: np.ndarray,
                                  labels: np.ndarray) -> np.ndarray:
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        centroids = np.array([X[labels == l].mean(axis=0) for l in unique])
        return cdist(centroids, centroids)

    def separation_matrix(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Min distance between any two points in different clusters."""
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        k = len(unique)
        sep = np.full((k, k), np.inf)
        for i in range(k):
            sep[i, i] = 0.0
            pts_i = X[labels == unique[i]][:100]
            for j in range(i + 1, k):
                pts_j = X[labels == unique[j]][:100]
                d = cdist(pts_i, pts_j).min()
                sep[i, j] = d
                sep[j, i] = d
        return sep


# ──────────────────────────────────────────────────────────────────
# EVALUATION ENGINE
# ──────────────────────────────────────────────────────────────────

class EvaluationEngine:
    """Orchestrates all evaluation metrics and generates reports."""

    def __init__(self, sample_size: int = 5000):
        self.internal = InternalMetrics(sample_size)
        self.external = ExternalMetrics()
        self.gap_stat = GapStatistic()
        self.stats_computer = ClusterStatisticsComputer()

    def evaluate(self, X: np.ndarray, labels: np.ndarray,
                 algorithm_name: str = "unknown",
                 centers: Optional[np.ndarray] = None,
                 labels_true: Optional[np.ndarray] = None,
                 feature_names: Optional[List[str]] = None) -> ClusteringReport:
        t0 = time.perf_counter()
        metrics: Dict[str, MetricResult] = {}
        clean_mask = labels >= 0
        X_clean = X[clean_mask]
        labels_clean = labels[clean_mask]
        n_clusters = len(set(labels_clean))
        n_noise = int((labels == -1).sum())

        if n_clusters >= 2 and len(X_clean) >= n_clusters + 1:
            metrics["silhouette"] = self.internal.silhouette(X_clean, labels_clean)
            metrics["davies_bouldin"] = self.internal.davies_bouldin(X_clean, labels_clean)
            metrics["calinski_harabasz"] = self.internal.calinski_harabasz(X_clean, labels_clean)
            metrics["dunn_index"] = self.internal.dunn_index(X_clean, labels_clean)
            metrics["xie_beni"] = self.internal.xie_beni(X_clean, labels_clean, centers)
            metrics["inertia"] = self.internal.inertia(X_clean, labels_clean, centers)
            metrics["s_dbw"] = self.internal.s_dbw(X_clean, labels_clean)

        if labels_true is not None:
            lt_clean = labels_true[clean_mask]
            ext = self.external.compute_all(lt_clean, labels_clean)
            metrics.update(ext)
            metrics["purity"] = self.external.purity(lt_clean, labels_clean)

        cluster_stats = self.stats_computer.compute(X_clean, labels_clean, feature_names)
        ranking = self._compute_ranking_score(metrics)
        total_time = time.perf_counter() - t0

        return ClusteringReport(
            algorithm_name=algorithm_name, n_clusters=n_clusters,
            n_noise=n_noise, n_samples=len(X),
            metrics=metrics, cluster_stats=cluster_stats,
            ranking_score=ranking, total_eval_time=round(total_time, 4),
        )

    def evaluate_batch(self, X: np.ndarray,
                       results: List,
                       labels_true: Optional[np.ndarray] = None,
                       feature_names: Optional[List[str]] = None) -> List[ClusteringReport]:
        reports = []
        for r in results:
            if hasattr(r, "status") and r.status.value == "success":
                report = self.evaluate(
                    X, r.labels, r.algorithm_name,
                    centers=r.centers, labels_true=labels_true,
                    feature_names=feature_names,
                )
                reports.append(report)
        return reports

    def rank_results(self, reports: List[ClusteringReport]) -> List[ClusteringReport]:
        return sorted(reports, key=lambda r: -r.ranking_score)

    def _compute_ranking_score(self, metrics: Dict[str, MetricResult]) -> float:
        score = 0.0
        weights = {
            "silhouette": 30.0, "davies_bouldin": 20.0,
            "calinski_harabasz": 15.0, "dunn_index": 15.0,
            "xie_beni": 10.0, "s_dbw": 10.0,
        }
        if "silhouette" in metrics and metrics["silhouette"].error is None:
            score += weights["silhouette"] * (metrics["silhouette"].value + 1) / 2
        if "davies_bouldin" in metrics and metrics["davies_bouldin"].error is None:
            dbi = metrics["davies_bouldin"].value
            score += weights["davies_bouldin"] * max(0, 1 - dbi / 5.0)
        if "calinski_harabasz" in metrics and metrics["calinski_harabasz"].error is None:
            ch = metrics["calinski_harabasz"].value
            score += weights["calinski_harabasz"] * min(ch / 1000, 1.0)
        if "dunn_index" in metrics and metrics["dunn_index"].error is None:
            score += weights["dunn_index"] * min(metrics["dunn_index"].value, 1.0)
        if "xie_beni" in metrics and metrics["xie_beni"].error is None:
            xb = metrics["xie_beni"].value
            score += weights["xie_beni"] * max(0, 1 - xb / 10.0)
        if "s_dbw" in metrics and metrics["s_dbw"].error is None:
            sd = metrics["s_dbw"].value
            score += weights["s_dbw"] * max(0, 1 - sd / 5.0)
        return round(score, 4)

    def get_metrics_dataframe(self, reports: List[ClusteringReport]) -> pd.DataFrame:
        rows = []
        for r in reports:
            row = {"Algorithm": r.algorithm_name, "Clusters": r.n_clusters,
                   "Noise": r.n_noise, "Ranking": r.ranking_score}
            for k, m in r.metrics.items():
                row[m.display_name] = round(m.value, 4) if m.error is None else None
            rows.append(row)
        return pd.DataFrame(rows)

    def get_cluster_stats_dataframe(self, report: ClusteringReport) -> pd.DataFrame:
        rows = []
        for cs in report.cluster_stats:
            rows.append({
                "Cluster": cs.cluster_id, "Size": cs.size,
                "% Total": cs.pct_of_total, "Compactness": round(cs.compactness, 4),
                "Mean Intra-Dist": round(cs.mean_intra_distance, 4),
                "Max Intra-Dist": round(cs.max_intra_distance, 4),
                "Nearest Cluster": cs.nearest_cluster,
                "Nearest Dist": round(cs.nearest_cluster_distance, 4),
            })
        return pd.DataFrame(rows)

    def get_summary_text(self, report: ClusteringReport) -> str:
        """Generate a human-readable summary of the evaluation report."""
        lines = [
            f"Algorithm: {report.algorithm_name}",
            f"Clusters: {report.n_clusters} | Noise points: {report.n_noise}",
            f"Samples: {report.n_samples}",
            f"Ranking Score: {report.ranking_score:.4f}",
            f"Evaluation Time: {report.total_eval_time:.4f}s",
            "", "--- Metrics ---",
        ]
        for k, m in report.metrics.items():
            direction = "↑" if m.higher_is_better else "↓"
            err_str = f" (ERROR: {m.error})" if m.error else ""
            lines.append(f"  {m.display_name}: {m.value:.4f} {direction}{err_str}")
        lines.append("")
        lines.append("--- Cluster Stats ---")
        for cs in report.cluster_stats:
            lines.append(f"  Cluster {cs.cluster_id}: {cs.size} pts ({cs.pct_of_total}%), "
                         f"compactness={cs.compactness:.4f}, nearest=C{cs.nearest_cluster} "
                         f"(d={cs.nearest_cluster_distance:.4f})")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# HOPKINS STATISTIC — Clusterability test
# ──────────────────────────────────────────────────────────────────

class HopkinsStatistic:
    """Tests whether data has meaningful cluster structure (vs uniform random)."""

    def __init__(self, n_samples: int = 100, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state

    def compute(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Compute the Hopkins statistic.
        H ≈ 0.5 means uniform random; H → 1.0 means highly clustered.
        """
        rng = np.random.RandomState(self.random_state)
        n, d = X.shape
        m = min(self.n_samples, n // 2)
        if m < 5:
            return {"hopkins": 0.5, "is_clusterable": False, "interpretation": "Too few samples"}

        # Sample m random points from X
        sample_idx = rng.choice(n, m, replace=False)
        X_sample = X[sample_idx]

        # Generate m random points in the data bounding box
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        X_random = rng.uniform(mins, maxs, size=(m, d))

        # Find nearest neighbor distances for real and random samples
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X)

        # For the sampled real points, NN distance (excluding self)
        dists_real, _ = nn.kneighbors(X_sample)
        u_dists = dists_real[:, 1]  # second neighbor (first is self if in dataset)

        # For random points, NN distance to real data
        dists_random, _ = nn.kneighbors(X_random)
        w_dists = dists_random[:, 0]

        u_sum = np.sum(u_dists ** d)
        w_sum = np.sum(w_dists ** d)

        hopkins = float(w_sum / max(u_sum + w_sum, 1e-16))

        if hopkins > 0.75:
            interp = "Highly clusterable (strong cluster structure)"
        elif hopkins > 0.6:
            interp = "Moderately clusterable"
        elif hopkins > 0.45:
            interp = "Weakly clusterable (near random)"
        else:
            interp = "Not clusterable (uniform-like distribution)"

        return {
            "hopkins": round(hopkins, 6),
            "is_clusterable": hopkins > 0.6,
            "interpretation": interp,
            "n_sampled": m,
        }


# ──────────────────────────────────────────────────────────────────
# NEAREST-NEIGHBOR DISTANCE PROFILE
# ──────────────────────────────────────────────────────────────────

class NNDistanceProfile:
    """Computes k-NN distance profile for DBSCAN eps estimation."""

    def __init__(self, k: int = 5, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def compute(self, X: np.ndarray,
                sample_size: int = 5000) -> Dict[str, Any]:
        """Compute sorted k-NN distance array and suggest eps."""
        n_sample = min(sample_size, len(X))
        if n_sample < len(X):
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X), n_sample, replace=False)
            X_s = X[idx]
        else:
            X_s = X

        k_use = min(self.k, len(X_s) - 1)
        nn = NearestNeighbors(n_neighbors=k_use)
        nn.fit(X_s)
        dists, _ = nn.kneighbors(X_s)

        kth_dists = np.sort(dists[:, -1])[::-1]
        mean_d = float(kth_dists.mean())
        median_d = float(np.median(kth_dists))
        std_d = float(kth_dists.std())
        q25 = float(np.percentile(kth_dists, 25))
        q75 = float(np.percentile(kth_dists, 75))

        # Kneedle for eps suggestion
        x_norm = np.linspace(0, 1, len(kth_dists))
        y_norm = (kth_dists - kth_dists.min()) / max(kth_dists.max() - kth_dists.min(), 1e-16)
        diff = y_norm - x_norm
        knee_idx = int(np.argmax(diff))
        suggested_eps = float(kth_dists[knee_idx])

        return {
            "k": k_use,
            "kth_distances": kth_dists,
            "mean": mean_d,
            "median": median_d,
            "std": std_d,
            "q25": q25,
            "q75": q75,
            "suggested_eps": round(suggested_eps, 6),
            "knee_index": knee_idx,
            "n_points": len(kth_dists),
        }


# ──────────────────────────────────────────────────────────────────
# CONNECTIVITY INDEX
# ──────────────────────────────────────────────────────────────────

class ConnectivityIndex:
    """Connectivity index: measures how well clusters respect local neighborhoods."""

    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors

    def compute(self, X: np.ndarray, labels: np.ndarray) -> MetricResult:
        t0 = time.perf_counter()
        try:
            n = len(X)
            k = min(self.n_neighbors, n - 1)
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(X)
            _, indices = nn.kneighbors(X)

            connectivity = 0.0
            for i in range(n):
                if labels[i] < 0:
                    continue
                for j_rank, j in enumerate(indices[i]):
                    if labels[j] != labels[i]:
                        connectivity += 1.0 / (j_rank + 1)

            return MetricResult(
                "connectivity", float(connectivity), "Connectivity Index",
                False, "internal",
                "Sum of fractional penalties for cross-cluster NN (lower=better)",
                time.perf_counter() - t0,
            )
        except Exception as e:
            return MetricResult("connectivity", 999.0, "Connectivity Index",
                                error=str(e), computation_time=time.perf_counter() - t0)


# ──────────────────────────────────────────────────────────────────
# TREND ANALYSIS — Metrics over k-range
# ──────────────────────────────────────────────────────────────────

class TrendAnalyzer:
    """Compute and analyze metric trends across different k values."""

    def __init__(self, engine: EvaluationEngine):
        self.engine = engine

    def compute_trends(self, sweep_points: List,
                       X: np.ndarray) -> pd.DataFrame:
        """Evaluate full metric suite for each sweep point."""
        rows = []
        for sp in sweep_points:
            if not hasattr(sp, "labels") or sp.labels is None:
                continue
            try:
                report = self.engine.evaluate(X, sp.labels, sp.algorithm)
                row = {
                    "k": sp.k,
                    "algorithm": sp.algorithm,
                    "fit_time": sp.fit_time,
                    "ranking": report.ranking_score,
                }
                for mk, mv in report.metrics.items():
                    if mv.error is None:
                        row[mv.display_name] = round(mv.value, 4)
                rows.append(row)
            except Exception:
                pass
        return pd.DataFrame(rows)

    def find_optimal_k(self, trends_df: pd.DataFrame,
                       metric: str = "Silhouette Score") -> int:
        """Find k that maximizes the given metric in the trends table."""
        if trends_df.empty or metric not in trends_df.columns:
            return 2
        best_idx = trends_df[metric].idxmax()
        return int(trends_df.loc[best_idx, "k"])

    def compute_stability_across_k(self, trends_df: pd.DataFrame,
                                   metric: str = "Silhouette Score") -> Dict[str, float]:
        """Measure how stable a metric is across k values."""
        if trends_df.empty or metric not in trends_df.columns:
            return {"mean": 0.0, "std": 0.0, "cv": 0.0}
        vals = trends_df[metric].dropna()
        mean_v = float(vals.mean())
        std_v = float(vals.std())
        cv = std_v / max(abs(mean_v), 1e-16)
        return {"mean": round(mean_v, 4), "std": round(std_v, 4), "cv": round(cv, 4)}


# ──────────────────────────────────────────────────────────────────
# CLUSTER OVERLAP ANALYSIS
# ──────────────────────────────────────────────────────────────────

class ClusterOverlapAnalyzer:
    """Measures overlap between clusters using distance-based and density measures."""

    def compute_pairwise_overlap(self, X: np.ndarray,
                                 labels: np.ndarray) -> Dict[str, Any]:
        """Compute overlap score between all cluster pairs."""
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        k = len(unique)
        overlap_matrix = np.zeros((k, k))
        centroids = {l: X[labels == l].mean(axis=0) for l in unique}
        radii = {}
        for l in unique:
            pts = X[labels == l]
            dists = np.linalg.norm(pts - centroids[l], axis=1)
            radii[l] = float(np.percentile(dists, 95)) if len(pts) > 0 else 0.0

        for i, l1 in enumerate(unique):
            for j, l2 in enumerate(unique):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                    continue
                d_centers = float(np.linalg.norm(centroids[l1] - centroids[l2]))
                sum_radii = radii[l1] + radii[l2]
                if sum_radii == 0:
                    overlap_matrix[i, j] = 0.0
                else:
                    overlap_matrix[i, j] = max(0, 1 - d_centers / sum_radii)

        total_overlap = float(overlap_matrix[np.triu_indices(k, k=1)].mean())
        max_overlap = float(overlap_matrix[np.triu_indices(k, k=1)].max()) if k > 1 else 0.0
        worst_pair = None
        if k > 1:
            upper = overlap_matrix.copy()
            np.fill_diagonal(upper, -1)
            idx_flat = np.argmax(upper)
            r, c = divmod(idx_flat, k)
            worst_pair = (unique[r], unique[c])

        return {
            "overlap_matrix": overlap_matrix,
            "total_overlap": round(total_overlap, 4),
            "max_overlap": round(max_overlap, 4),
            "worst_pair": worst_pair,
            "cluster_radii": radii,
        }


# ──────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE PER CLUSTER
# ──────────────────────────────────────────────────────────────────

class FeatureImportanceAnalyzer:
    """Ranks features by discriminative power across clusters."""

    def compute_anova_importance(self, X: np.ndarray, labels: np.ndarray,
                                 feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute F-statistic (one-way ANOVA) per feature across clusters."""
        from scipy.stats import f_oneway
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        if len(unique) < 2:
            return pd.DataFrame()
        results = []
        n_features = X.shape[1]
        for f_idx in range(n_features):
            groups = [X[labels == l, f_idx] for l in unique if (labels == l).sum() > 0]
            if len(groups) < 2:
                continue
            try:
                f_stat, p_val = f_oneway(*groups)
                fname = feature_names[f_idx] if feature_names and f_idx < len(feature_names) else f"feature_{f_idx}"
                results.append({
                    "Feature": fname,
                    "F-Statistic": round(float(f_stat), 4),
                    "p-value": float(p_val),
                    "Significant": p_val < 0.05,
                    "feature_idx": f_idx,
                })
            except Exception:
                pass
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("F-Statistic", ascending=False).reset_index(drop=True)
        return df

    def compute_kruskal_importance(self, X: np.ndarray, labels: np.ndarray,
                                   feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Non-parametric Kruskal-Wallis test per feature."""
        from scipy.stats import kruskal
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        if len(unique) < 2:
            return pd.DataFrame()
        results = []
        for f_idx in range(X.shape[1]):
            groups = [X[labels == l, f_idx] for l in unique if (labels == l).sum() > 0]
            if len(groups) < 2:
                continue
            try:
                h_stat, p_val = kruskal(*groups)
                fname = feature_names[f_idx] if feature_names and f_idx < len(feature_names) else f"feature_{f_idx}"
                results.append({
                    "Feature": fname,
                    "H-Statistic": round(float(h_stat), 4),
                    "p-value": float(p_val),
                    "Significant": p_val < 0.05,
                })
            except Exception:
                pass
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("H-Statistic", ascending=False).reset_index(drop=True)
        return df


# ──────────────────────────────────────────────────────────────────
# METRIC NORMALIZER
# ──────────────────────────────────────────────────────────────────

class MetricNormalizer:
    """Normalizes diverse metrics to [0,1] for fair comparison."""

    KNOWN_RANGES = {
        "silhouette": (-1.0, 1.0),
        "davies_bouldin": (0.0, 10.0),
        "calinski_harabasz": (0.0, 5000.0),
        "dunn_index": (0.0, 2.0),
        "xie_beni": (0.0, 50.0),
        "s_dbw": (0.0, 5.0),
        "ari": (-1.0, 1.0),
        "nmi": (0.0, 1.0),
        "purity": (0.0, 1.0),
    }

    def normalize(self, metric: MetricResult) -> float:
        """Normalize a metric value to [0,1] where 1 is always best."""
        r = self.KNOWN_RANGES.get(metric.name, (0.0, 1.0))
        val = np.clip(metric.value, r[0], r[1])
        normalized = (val - r[0]) / max(r[1] - r[0], 1e-16)
        if not metric.higher_is_better:
            normalized = 1.0 - normalized
        return round(float(normalized), 4)

    def normalize_report(self, report: ClusteringReport) -> Dict[str, float]:
        """Normalize all metrics in a report."""
        normalized = {}
        for k, m in report.metrics.items():
            if m.error is None:
                normalized[k] = self.normalize(m)
        return normalized


# ──────────────────────────────────────────────────────────────────
# COMPARISON MATRIX BUILDER
# ──────────────────────────────────────────────────────────────────

class ComparisonMatrixBuilder:
    """Builds pairwise algorithm comparison matrices."""

    def __init__(self, normalizer: Optional[MetricNormalizer] = None):
        self.normalizer = normalizer or MetricNormalizer()

    def build(self, reports: List[ClusteringReport]) -> pd.DataFrame:
        """Build a normalized comparison matrix: algos × metrics."""
        rows = []
        for r in reports:
            row = {"Algorithm": r.algorithm_name}
            for k, m in r.metrics.items():
                if m.error is None:
                    row[m.display_name] = self.normalizer.normalize(m)
            row["Overall Ranking"] = r.ranking_score
            rows.append(row)
        df = pd.DataFrame(rows)
        if "Overall Ranking" in df.columns:
            df = df.sort_values("Overall Ranking", ascending=False).reset_index(drop=True)
        return df

    def pairwise_win_matrix(self, reports: List[ClusteringReport]) -> pd.DataFrame:
        """Count how many metrics each algorithm wins vs each other."""
        names = [r.algorithm_name for r in reports]
        n = len(names)
        wins = np.zeros((n, n), dtype=int)
        all_metrics = set()
        for r in reports:
            for k in r.metrics:
                if r.metrics[k].error is None:
                    all_metrics.add(k)

        for metric_key in all_metrics:
            values = []
            for r in reports:
                m = r.metrics.get(metric_key)
                if m and m.error is None:
                    values.append((self.normalizer.normalize(m), r.algorithm_name))
                else:
                    values.append((-1, r.algorithm_name))
            values.sort(key=lambda x: -x[0])
            if values:
                winner_name = values[0][1]
                winner_idx = names.index(winner_name)
                for j in range(n):
                    if j != winner_idx:
                        wins[winner_idx, j] += 1

        return pd.DataFrame(wins, index=names, columns=names)


# ──────────────────────────────────────────────────────────────────
# CLUSTER PROFILER — per-cluster centroid summary
# ──────────────────────────────────────────────────────────────────

class ClusterProfiler:
    """Generates a rich per-cluster profile summary for reporting."""

    @staticmethod
    def profile(X: np.ndarray, labels: np.ndarray,
                feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Build a DataFrame with mean, std, min, max per feature per cluster."""
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        rows = []
        for l in unique:
            pts = X[labels == l]
            for fi in range(X.shape[1]):
                fname = feature_names[fi] if feature_names and fi < len(feature_names) else f"feature_{fi}"
                rows.append({
                    "Cluster": l,
                    "Feature": fname,
                    "Mean": round(float(pts[:, fi].mean()), 4),
                    "Std": round(float(pts[:, fi].std()), 4),
                    "Min": round(float(pts[:, fi].min()), 4),
                    "Max": round(float(pts[:, fi].max()), 4),
                    "Median": round(float(np.median(pts[:, fi])), 4),
                })
        return pd.DataFrame(rows)

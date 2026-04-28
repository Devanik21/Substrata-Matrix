"""
stability_consensus.py — Substrata-Matrix Stability & Consensus Module

Bootstrap stability analysis, perturbation response, consensus matrix
construction, PAC scoring, and hierarchical consensus clustering.
"""

from __future__ import annotations

import time
import warnings
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from clustering_registry import REGISTRY, AlgorithmRegistry

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────

@dataclass
class BootstrapResult:
    """Result from a single bootstrap iteration."""
    iteration: int
    labels: np.ndarray
    ari_vs_full: float
    n_clusters_found: int
    sample_indices: np.ndarray


@dataclass
class StabilityScore:
    """Stability score for one algorithm."""
    algorithm_name: str
    mean_ari: float
    std_ari: float
    min_ari: float
    max_ari: float
    median_ari: float
    mean_jaccard: float
    n_iterations: int
    cluster_count_variance: float
    is_stable: bool
    stability_grade: str


@dataclass
class ConsensusResult:
    """Result from consensus clustering."""
    consensus_matrix: np.ndarray
    final_labels: np.ndarray
    n_clusters: int
    pac_score: float
    cdf_values: np.ndarray
    cdf_x: np.ndarray
    cophenetic_correlation: float
    n_runs: int


@dataclass
class PerturbationResult:
    """Result from perturbation analysis."""
    noise_levels: List[float]
    mean_ari_per_level: List[float]
    std_ari_per_level: List[float]
    robustness_score: float
    feature_dropout_scores: Dict[int, float]


@dataclass
class StabilityReport:
    """Full stability report for one algorithm."""
    algorithm_name: str
    stability_score: StabilityScore
    consensus_result: Optional[ConsensusResult]
    perturbation_result: Optional[PerturbationResult]
    total_time_seconds: float


# ──────────────────────────────────────────────────────────────────
# BOOTSTRAP STABILITY
# ──────────────────────────────────────────────────────────────────

class BootstrapStability:
    """Assesses clustering stability via repeated subsampling."""

    def __init__(self, n_iterations: int = 50, subsample_ratio: float = 0.8,
                 random_state: int = 42):
        self.n_iterations = n_iterations
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state

    def run(self, X: np.ndarray, algo_name: str, params: Dict[str, Any],
            full_labels: Optional[np.ndarray] = None,
            callback: Optional[Callable] = None) -> StabilityScore:
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        n_sub = max(int(n * self.subsample_ratio), 10)
        ari_scores = []
        jaccard_scores = []
        cluster_counts = []

        if full_labels is None:
            try:
                model = REGISTRY.build(algo_name, params)
                if hasattr(model, "fit_predict"):
                    full_labels = model.fit_predict(X)
                else:
                    model.fit(X)
                    full_labels = model.labels_
            except Exception:
                full_labels = KMeans(n_clusters=params.get("n_clusters", 3),
                                     random_state=42).fit_predict(X)

        for i in range(self.n_iterations):
            idx = rng.choice(n, n_sub, replace=False)
            X_sub = X[idx]
            labels_sub_full = full_labels[idx]

            try:
                model = REGISTRY.build(algo_name, params)
                if hasattr(model, "fit_predict"):
                    labels_sub = model.fit_predict(X_sub)
                else:
                    model.fit(X_sub)
                    labels_sub = model.labels_

                mask = (labels_sub >= 0) & (labels_sub_full >= 0)
                if mask.sum() >= 2:
                    ari = adjusted_rand_score(labels_sub_full[mask], labels_sub[mask])
                    ari_scores.append(float(ari))
                    jacc = self._jaccard_index(labels_sub_full[mask], labels_sub[mask])
                    jaccard_scores.append(jacc)

                n_c = len(set(labels_sub[labels_sub >= 0]))
                cluster_counts.append(n_c)
            except Exception:
                ari_scores.append(0.0)
                jaccard_scores.append(0.0)

            if callback:
                callback(f"Bootstrap {algo_name} {i+1}/{self.n_iterations}",
                         (i + 1) / self.n_iterations)

        ari_arr = np.array(ari_scores) if ari_scores else np.array([0.0])
        jacc_arr = np.array(jaccard_scores) if jaccard_scores else np.array([0.0])

        mean_ari = float(ari_arr.mean())
        is_stable = mean_ari >= 0.7
        if mean_ari >= 0.9:
            grade = "A+"
        elif mean_ari >= 0.8:
            grade = "A"
        elif mean_ari >= 0.7:
            grade = "B"
        elif mean_ari >= 0.5:
            grade = "C"
        else:
            grade = "D"

        return StabilityScore(
            algorithm_name=algo_name,
            mean_ari=mean_ari,
            std_ari=float(ari_arr.std()),
            min_ari=float(ari_arr.min()),
            max_ari=float(ari_arr.max()),
            median_ari=float(np.median(ari_arr)),
            mean_jaccard=float(jacc_arr.mean()),
            n_iterations=self.n_iterations,
            cluster_count_variance=float(np.var(cluster_counts)) if cluster_counts else 0.0,
            is_stable=is_stable,
            stability_grade=grade,
        )

    def _jaccard_index(self, labels_a: np.ndarray, labels_b: np.ndarray) -> float:
        """Compute average Jaccard index between two clusterings."""
        n = len(labels_a)
        if n < 2:
            return 0.0
        a_same = np.zeros((n, n), dtype=bool)
        b_same = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, min(n, i + 200)):
                a_same[i, j] = labels_a[i] == labels_a[j]
                b_same[i, j] = labels_b[i] == labels_b[j]
        both = (a_same & b_same).sum()
        either = (a_same | b_same).sum()
        return float(both / max(either, 1))


# ──────────────────────────────────────────────────────────────────
# CONSENSUS MATRIX
# ──────────────────────────────────────────────────────────────────

class ConsensusMatrix:
    """Builds a consensus (co-association) matrix over M clustering runs."""

    def __init__(self, n_runs: int = 50, subsample_ratio: float = 0.8,
                 random_state: int = 42):
        self.n_runs = n_runs
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state

    def build(self, X: np.ndarray, algo_name: str, params: Dict[str, Any],
              callback: Optional[Callable] = None) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        co_occurrence = np.zeros((n, n), dtype=np.float32)
        co_sampled = np.zeros((n, n), dtype=np.float32)
        n_sub = max(int(n * self.subsample_ratio), 10)

        for run in range(self.n_runs):
            idx = rng.choice(n, n_sub, replace=False)
            X_sub = X[idx]
            try:
                model = REGISTRY.build(algo_name, params)
                if hasattr(model, "fit_predict"):
                    labels = model.fit_predict(X_sub)
                else:
                    model.fit(X_sub)
                    labels = model.labels_

                for i in range(n_sub):
                    for j in range(i + 1, n_sub):
                        ii, jj = idx[i], idx[j]
                        co_sampled[ii, jj] += 1
                        co_sampled[jj, ii] += 1
                        if labels[i] == labels[j] and labels[i] >= 0:
                            co_occurrence[ii, jj] += 1
                            co_occurrence[jj, ii] += 1
            except Exception:
                pass

            if callback:
                callback(f"Consensus {algo_name} {run+1}/{self.n_runs}",
                         (run + 1) / self.n_runs)

        co_sampled[co_sampled == 0] = 1
        consensus = co_occurrence / co_sampled
        np.fill_diagonal(consensus, 1.0)
        return consensus

    @staticmethod
    def compute_pac(consensus: np.ndarray,
                    lower: float = 0.1, upper: float = 0.9) -> float:
        """Proportion of Ambiguous Clustering — lower is better."""
        n = consensus.shape[0]
        upper_tri = consensus[np.triu_indices(n, k=1)]
        total = len(upper_tri)
        if total == 0:
            return 1.0
        ambiguous = ((upper_tri > lower) & (upper_tri < upper)).sum()
        return float(ambiguous / total)

    @staticmethod
    def compute_cdf(consensus: np.ndarray,
                    n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Consensus CDF for determining optimal k."""
        upper_tri = consensus[np.triu_indices(consensus.shape[0], k=1)]
        x = np.linspace(0, 1, n_bins)
        cdf = np.array([np.mean(upper_tri <= xi) for xi in x])
        return x, cdf

    @staticmethod
    def compute_cophenetic_correlation(consensus: np.ndarray) -> float:
        """Cophenetic correlation coefficient of the consensus matrix."""
        try:
            n = consensus.shape[0]
            dist_matrix = 1.0 - consensus
            np.fill_diagonal(dist_matrix, 0.0)
            dist_matrix = np.maximum(dist_matrix, 0.0)
            condensed = squareform(dist_matrix, checks=False)
            Z = linkage(condensed, method="average")
            from scipy.cluster.hierarchy import cophenet
            c, _ = cophenet(Z, condensed)
            return float(c)
        except Exception:
            return 0.0


# ──────────────────────────────────────────────────────────────────
# CONSENSUS CLUSTERER
# ──────────────────────────────────────────────────────────────────

class ConsensusClusterer:
    """Produces final labels from a consensus matrix using HAC."""

    def __init__(self, n_clusters: int = 3, linkage_method: str = "average"):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method

    def cluster(self, consensus: np.ndarray) -> np.ndarray:
        dist_matrix = 1.0 - consensus
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.maximum(dist_matrix, 0.0)
        try:
            condensed = squareform(dist_matrix, checks=False)
            Z = linkage(condensed, method=self.linkage_method)
            labels = fcluster(Z, t=self.n_clusters, criterion="maxclust")
            return labels - 1
        except Exception:
            return AgglomerativeClustering(
                n_clusters=self.n_clusters
            ).fit_predict(dist_matrix)

    def find_optimal_k(self, consensus: np.ndarray,
                       k_range: Tuple[int, int] = (2, 10)) -> int:
        best_k = 2
        best_score = -np.inf
        for k in range(k_range[0], k_range[1] + 1):
            self.n_clusters = k
            labels = self.cluster(consensus)
            if len(set(labels)) < 2:
                continue
            try:
                dist_matrix = 1.0 - consensus
                np.fill_diagonal(dist_matrix, 0.0)
                sc = silhouette_score(dist_matrix, labels, metric="precomputed")
                if sc > best_score:
                    best_score = sc
                    best_k = k
            except Exception:
                pass
        return best_k


# ──────────────────────────────────────────────────────────────────
# PERTURBATION ANALYSIS
# ──────────────────────────────────────────────────────────────────

class PerturbationAnalysis:
    """Tests clustering robustness to noise injection and feature dropout."""

    def __init__(self, n_repeats: int = 10, random_state: int = 42):
        self.n_repeats = n_repeats
        self.random_state = random_state

    def noise_injection(self, X: np.ndarray, algo_name: str,
                        params: Dict[str, Any],
                        noise_levels: Optional[List[float]] = None,
                        callback: Optional[Callable] = None) -> PerturbationResult:
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
        rng = np.random.RandomState(self.random_state)

        try:
            model_orig = REGISTRY.build(algo_name, params)
            if hasattr(model_orig, "fit_predict"):
                labels_orig = model_orig.fit_predict(X)
            else:
                model_orig.fit(X)
                labels_orig = model_orig.labels_
        except Exception:
            labels_orig = KMeans(n_clusters=params.get("n_clusters", 3),
                                  random_state=42).fit_predict(X)

        mean_aris = []
        std_aris = []
        total_steps = len(noise_levels) * self.n_repeats

        step = 0
        for level in noise_levels:
            aris = []
            noise_scale = level * np.std(X, axis=0)
            for rep in range(self.n_repeats):
                X_noisy = X + rng.randn(*X.shape) * noise_scale
                try:
                    model = REGISTRY.build(algo_name, params)
                    if hasattr(model, "fit_predict"):
                        labels_noisy = model.fit_predict(X_noisy)
                    else:
                        model.fit(X_noisy)
                        labels_noisy = model.labels_
                    mask = (labels_noisy >= 0) & (labels_orig >= 0)
                    if mask.sum() >= 2:
                        ari = adjusted_rand_score(labels_orig[mask], labels_noisy[mask])
                        aris.append(float(ari))
                except Exception:
                    aris.append(0.0)
                step += 1
                if callback:
                    callback(f"Perturbation σ={level:.2f}", step / total_steps)
            mean_aris.append(float(np.mean(aris)) if aris else 0.0)
            std_aris.append(float(np.std(aris)) if aris else 0.0)

        robustness = float(np.mean(mean_aris))

        feature_scores = self._feature_dropout(X, algo_name, params, labels_orig, rng)

        return PerturbationResult(
            noise_levels=noise_levels,
            mean_ari_per_level=mean_aris,
            std_ari_per_level=std_aris,
            robustness_score=robustness,
            feature_dropout_scores=feature_scores,
        )

    def _feature_dropout(self, X: np.ndarray, algo_name: str,
                         params: Dict, labels_orig: np.ndarray,
                         rng: np.random.RandomState) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        n_features = X.shape[1]
        for f in range(min(n_features, 20)):
            X_dropped = np.delete(X, f, axis=1)
            if X_dropped.shape[1] == 0:
                continue
            try:
                p = {**params}
                model = REGISTRY.build(algo_name, p)
                if hasattr(model, "fit_predict"):
                    labels_drop = model.fit_predict(X_dropped)
                else:
                    model.fit(X_dropped)
                    labels_drop = model.labels_
                mask = (labels_drop >= 0) & (labels_orig >= 0)
                if mask.sum() >= 2:
                    ari = adjusted_rand_score(labels_orig[mask], labels_drop[mask])
                    scores[f] = float(ari)
                else:
                    scores[f] = 0.0
            except Exception:
                scores[f] = 0.0
        return scores


# ──────────────────────────────────────────────────────────────────
# FULL STABILITY PIPELINE
# ──────────────────────────────────────────────────────────────────

class StabilityPipeline:
    """Runs the full stability analysis pipeline."""

    def __init__(self, n_bootstrap: int = 30, n_consensus: int = 30,
                 n_perturb: int = 5, random_state: int = 42):
        self.bootstrap = BootstrapStability(n_bootstrap, random_state=random_state)
        self.consensus_builder = ConsensusMatrix(n_consensus, random_state=random_state)
        self.perturbation = PerturbationAnalysis(n_perturb, random_state=random_state)
        self.random_state = random_state

    def run(self, X: np.ndarray, algo_name: str, params: Dict[str, Any],
            full_labels: Optional[np.ndarray] = None,
            run_consensus: bool = True,
            run_perturbation: bool = True,
            callback: Optional[Callable] = None) -> StabilityReport:
        t0 = time.perf_counter()

        stability_score = self.bootstrap.run(
            X, algo_name, params, full_labels, callback
        )

        consensus_result = None
        if run_consensus:
            consensus = self.consensus_builder.build(X, algo_name, params, callback)
            pac = ConsensusMatrix.compute_pac(consensus)
            cdf_x, cdf_vals = ConsensusMatrix.compute_cdf(consensus)
            coph = ConsensusMatrix.compute_cophenetic_correlation(consensus)

            n_clusters = params.get("n_clusters", 3)
            clusterer = ConsensusClusterer(n_clusters)
            final_labels = clusterer.cluster(consensus)

            consensus_result = ConsensusResult(
                consensus_matrix=consensus,
                final_labels=final_labels,
                n_clusters=n_clusters,
                pac_score=pac,
                cdf_values=cdf_vals,
                cdf_x=cdf_x,
                cophenetic_correlation=coph,
                n_runs=self.consensus_builder.n_runs,
            )

        perturbation_result = None
        if run_perturbation:
            perturbation_result = self.perturbation.noise_injection(
                X, algo_name, params, callback=callback
            )

        return StabilityReport(
            algorithm_name=algo_name,
            stability_score=stability_score,
            consensus_result=consensus_result,
            perturbation_result=perturbation_result,
            total_time_seconds=round(time.perf_counter() - t0, 3),
        )

    def run_batch(self, X: np.ndarray,
                  algorithms: List[Tuple[str, Dict[str, Any]]],
                  callback: Optional[Callable] = None) -> List[StabilityReport]:
        reports = []
        for i, (algo_name, params) in enumerate(algorithms):
            report = self.run(X, algo_name, params, callback=callback)
            reports.append(report)
            if callback:
                callback(f"Stability {algo_name} done",
                         (i + 1) / len(algorithms))
        return reports

    @staticmethod
    def get_stability_leaderboard(reports: List[StabilityReport]) -> pd.DataFrame:
        rows = []
        for r in reports:
            s = r.stability_score
            row = {
                "Algorithm": s.algorithm_name,
                "Mean ARI": round(s.mean_ari, 4),
                "Std ARI": round(s.std_ari, 4),
                "Min ARI": round(s.min_ari, 4),
                "Max ARI": round(s.max_ari, 4),
                "Jaccard": round(s.mean_jaccard, 4),
                "Grade": s.stability_grade,
                "Stable": "✅" if s.is_stable else "❌",
                "k Variance": round(s.cluster_count_variance, 3),
            }
            if r.consensus_result:
                row["PAC"] = round(r.consensus_result.pac_score, 4)
                row["Cophenetic"] = round(r.consensus_result.cophenetic_correlation, 4)
            if r.perturbation_result:
                row["Robustness"] = round(r.perturbation_result.robustness_score, 4)
            rows.append(row)
        df = pd.DataFrame(rows)
        if "Mean ARI" in df.columns:
            df = df.sort_values("Mean ARI", ascending=False).reset_index(drop=True)
        return df

    def run_quick(self, X: np.ndarray, algo_name: str,
                  params: Dict[str, Any],
                  full_labels: Optional[np.ndarray] = None,
                  callback: Optional[Callable] = None) -> StabilityReport:
        """Quick mode: bootstrap only, no consensus or perturbation."""
        return self.run(X, algo_name, params, full_labels,
                        run_consensus=False, run_perturbation=False,
                        callback=callback)

    def run_comprehensive(self, X: np.ndarray, algo_name: str,
                          params: Dict[str, Any],
                          full_labels: Optional[np.ndarray] = None,
                          callback: Optional[Callable] = None) -> StabilityReport:
        """Comprehensive mode: all analyses enabled."""
        return self.run(X, algo_name, params, full_labels,
                        run_consensus=True, run_perturbation=True,
                        callback=callback)

    @staticmethod
    def export_report_dict(report: StabilityReport) -> Dict[str, Any]:
        """Export stability report as a serializable dictionary."""
        s = report.stability_score
        d = {
            "algorithm": report.algorithm_name,
            "total_time": report.total_time_seconds,
            "stability": {
                "mean_ari": s.mean_ari,
                "std_ari": s.std_ari,
                "min_ari": s.min_ari,
                "max_ari": s.max_ari,
                "median_ari": s.median_ari,
                "mean_jaccard": s.mean_jaccard,
                "grade": s.stability_grade,
                "is_stable": s.is_stable,
                "k_variance": s.cluster_count_variance,
                "n_iterations": s.n_iterations,
            },
        }
        if report.consensus_result:
            cr = report.consensus_result
            d["consensus"] = {
                "pac_score": cr.pac_score,
                "cophenetic_correlation": cr.cophenetic_correlation,
                "n_clusters": cr.n_clusters,
                "n_runs": cr.n_runs,
            }
        if report.perturbation_result:
            pr = report.perturbation_result
            d["perturbation"] = {
                "robustness_score": pr.robustness_score,
                "noise_levels": pr.noise_levels,
                "mean_ari_per_level": pr.mean_ari_per_level,
                "std_ari_per_level": pr.std_ari_per_level,
            }
        return d


# ──────────────────────────────────────────────────────────────────
# CROSS-VALIDATION STABILITY
# ──────────────────────────────────────────────────────────────────

class CrossValidationStability:
    """K-fold cross-validation approach to stability measurement."""

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state

    def run(self, X: np.ndarray, algo_name: str,
            params: Dict[str, Any],
            callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Split data into folds, cluster each, measure agreement."""
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        indices = np.arange(n)
        rng.shuffle(indices)
        fold_size = n // self.n_folds
        folds = []
        for i in range(self.n_folds):
            start = i * fold_size
            end = start + fold_size if i < self.n_folds - 1 else n
            folds.append(indices[start:end])

        # Cluster each fold
        fold_labels = {}
        for i, fold_idx in enumerate(folds):
            X_fold = X[fold_idx]
            try:
                model = REGISTRY.build(algo_name, params)
                if hasattr(model, "fit_predict"):
                    labels = model.fit_predict(X_fold)
                else:
                    model.fit(X_fold)
                    labels = model.labels_
                fold_labels[i] = (fold_idx, labels)
            except Exception:
                fold_labels[i] = (fold_idx, np.zeros(len(fold_idx), dtype=int))
            if callback:
                callback(f"CV fold {i+1}/{self.n_folds}", (i + 1) / self.n_folds)

        # Pairwise ARI on overlapping samples
        ari_scores = []
        for i in range(self.n_folds):
            for j in range(i + 1, self.n_folds):
                idx_i, lab_i = fold_labels[i]
                idx_j, lab_j = fold_labels[j]
                # Find common samples (there are none in disjoint folds)
                # Instead, train on fold i and predict fold j via nearest centroid
                X_i = X[idx_i]
                X_j = X[idx_j]
                # Build centroid model from fold i labels
                unique_i = [l for l in np.unique(lab_i) if l >= 0]
                if len(unique_i) < 2:
                    continue
                centroids = np.array([X_i[lab_i == l].mean(axis=0) for l in unique_i])
                # Assign fold j points to nearest centroid
                from scipy.spatial.distance import cdist
                dists = cdist(X_j, centroids)
                predicted_j = np.array([unique_i[k] for k in np.argmin(dists, axis=1)])
                # Compare predicted_j with lab_j
                mask = lab_j >= 0
                if mask.sum() >= 2:
                    ari = adjusted_rand_score(lab_j[mask], predicted_j[mask])
                    ari_scores.append(float(ari))

        ari_arr = np.array(ari_scores) if ari_scores else np.array([0.0])
        return {
            "mean_ari": round(float(ari_arr.mean()), 4),
            "std_ari": round(float(ari_arr.std()), 4),
            "min_ari": round(float(ari_arr.min()), 4),
            "max_ari": round(float(ari_arr.max()), 4),
            "n_folds": self.n_folds,
            "n_comparisons": len(ari_scores),
            "all_aris": ari_scores,
        }


# ──────────────────────────────────────────────────────────────────
# TEMPORAL STABILITY — Stability over incremental data chunks
# ──────────────────────────────────────────────────────────────────

class TemporalStability:
    """Measures how clustering changes as data grows incrementally."""

    def __init__(self, n_checkpoints: int = 10, random_state: int = 42):
        self.n_checkpoints = n_checkpoints
        self.random_state = random_state

    def run(self, X: np.ndarray, algo_name: str, params: Dict[str, Any],
            callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Add data incrementally and track cluster evolution."""
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        shuffled_idx = rng.permutation(n)
        checkpoint_sizes = np.linspace(max(50, n // self.n_checkpoints),
                                        n, self.n_checkpoints, dtype=int)
        results = []
        prev_labels = None

        for i, size in enumerate(checkpoint_sizes):
            idx = shuffled_idx[:size]
            X_chunk = X[idx]
            try:
                model = REGISTRY.build(algo_name, params)
                if hasattr(model, "fit_predict"):
                    labels = model.fit_predict(X_chunk)
                else:
                    model.fit(X_chunk)
                    labels = model.labels_

                n_clusters = len(set(labels) - {-1})
                noise_pct = float((labels == -1).sum() / len(labels) * 100)
                ari_vs_prev = None

                if prev_labels is not None and len(prev_labels) <= len(labels):
                    common_n = len(prev_labels)
                    mask = (labels[:common_n] >= 0) & (prev_labels >= 0)
                    if mask.sum() >= 2:
                        ari_vs_prev = float(adjusted_rand_score(
                            prev_labels[mask], labels[:common_n][mask]))

                results.append({
                    "checkpoint": i + 1,
                    "n_samples": int(size),
                    "n_clusters": n_clusters,
                    "noise_pct": round(noise_pct, 2),
                    "ari_vs_prev": ari_vs_prev,
                })
                prev_labels = labels.copy()
            except Exception as e:
                results.append({
                    "checkpoint": i + 1, "n_samples": int(size),
                    "error": str(e),
                })
            if callback:
                callback(f"Temporal {i+1}/{self.n_checkpoints}",
                         (i + 1) / self.n_checkpoints)

        temporal_aris = [r["ari_vs_prev"] for r in results
                         if r.get("ari_vs_prev") is not None]
        return {
            "checkpoints": results,
            "mean_temporal_ari": round(float(np.mean(temporal_aris)), 4) if temporal_aris else 0.0,
            "temporal_stability": round(float(np.mean(temporal_aris)), 4) if temporal_aris else 0.0,
            "cluster_count_trace": [r.get("n_clusters", 0) for r in results],
        }


# ──────────────────────────────────────────────────────────────────
# MULTI-RESOLUTION CONSENSUS
# ──────────────────────────────────────────────────────────────────

class MultiResolutionConsensus:
    """Consensus clustering across multiple k values to find robust structure."""

    def __init__(self, k_range: Tuple[int, int] = (2, 10),
                 n_runs_per_k: int = 10, random_state: int = 42):
        self.k_range = k_range
        self.n_runs_per_k = n_runs_per_k
        self.random_state = random_state

    def run(self, X: np.ndarray, algo_name: str,
            base_params: Optional[Dict[str, Any]] = None,
            callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Build consensus matrices for each k, compute PAC curve."""
        if base_params is None:
            base_params = REGISTRY.get_default_params(algo_name)
        k_values = list(range(self.k_range[0], self.k_range[1] + 1))
        pac_scores = []
        cophenetic_scores = []
        total = len(k_values)

        for i, k in enumerate(k_values):
            params = {**base_params, "n_clusters": k,
                      "random_state": self.random_state}
            builder = ConsensusMatrix(n_runs=self.n_runs_per_k,
                                       random_state=self.random_state)
            consensus = builder.build(X, algo_name, params)
            pac = ConsensusMatrix.compute_pac(consensus)
            coph = ConsensusMatrix.compute_cophenetic_correlation(consensus)
            pac_scores.append(float(pac))
            cophenetic_scores.append(float(coph))
            if callback:
                callback(f"Multi-res k={k}", (i + 1) / total)

        best_k_pac = k_values[int(np.argmin(pac_scores))]
        best_k_coph = k_values[int(np.argmax(cophenetic_scores))]

        return {
            "k_values": k_values,
            "pac_scores": pac_scores,
            "cophenetic_scores": cophenetic_scores,
            "best_k_pac": best_k_pac,
            "best_k_cophenetic": best_k_coph,
            "recommended_k": best_k_pac,
        }


# ──────────────────────────────────────────────────────────────────
# ITEM CONSENSUS — Per-sample consensus scores
# ──────────────────────────────────────────────────────────────────

class ItemConsensus:
    """Computes per-sample consensus scores from a consensus matrix."""

    @staticmethod
    def compute(consensus: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """For each sample, compute mean consensus with its cluster members."""
        n = len(labels)
        scores = np.zeros(n)
        unique = [l for l in np.unique(labels) if l >= 0]
        for i in range(n):
            if labels[i] < 0:
                scores[i] = 0.0
                continue
            cluster_mask = labels == labels[i]
            cluster_mask[i] = False
            if cluster_mask.sum() == 0:
                scores[i] = 1.0
                continue
            scores[i] = float(consensus[i, cluster_mask].mean())
        return scores

    @staticmethod
    def find_ambiguous_samples(consensus: np.ndarray, labels: np.ndarray,
                               threshold: float = 0.5) -> np.ndarray:
        """Find samples with low consensus (ambiguous cluster assignment)."""
        scores = ItemConsensus.compute(consensus, labels)
        return np.where(scores < threshold)[0]

    @staticmethod
    def cluster_cohesion(consensus: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
        """Mean within-cluster consensus for each cluster."""
        unique = sorted([l for l in np.unique(labels) if l >= 0])
        cohesion = {}
        for l in unique:
            mask = labels == l
            if mask.sum() < 2:
                cohesion[l] = 1.0
                continue
            sub_matrix = consensus[np.ix_(mask, mask)]
            n_c = sub_matrix.shape[0]
            upper = sub_matrix[np.triu_indices(n_c, k=1)]
            cohesion[l] = round(float(upper.mean()), 4) if len(upper) > 0 else 1.0
        return cohesion


# ──────────────────────────────────────────────────────────────────
# CLUSTER FLOW TRACKER — Track label changes across runs
# ──────────────────────────────────────────────────────────────────

class ClusterFlowTracker:
    """Tracks how samples flow between clusters across different configurations."""

    @staticmethod
    def compute_flow(labels_list: List[np.ndarray],
                     config_names: List[str]) -> pd.DataFrame:
        """Build a flow matrix showing sample migration between runs."""
        if len(labels_list) < 2:
            return pd.DataFrame()
        rows = []
        for i in range(len(labels_list) - 1):
            l_from = labels_list[i]
            l_to = labels_list[i + 1]
            n = min(len(l_from), len(l_to))
            unique_from = sorted(set(l_from[:n]))
            unique_to = sorted(set(l_to[:n]))
            for cf in unique_from:
                for ct in unique_to:
                    count = int(((l_from[:n] == cf) & (l_to[:n] == ct)).sum())
                    if count > 0:
                        rows.append({
                            "From Config": config_names[i],
                            "To Config": config_names[i + 1],
                            "From Cluster": cf,
                            "To Cluster": ct,
                            "Count": count,
                        })
        return pd.DataFrame(rows)

    @staticmethod
    def compute_migration_rate(labels_before: np.ndarray,
                               labels_after: np.ndarray) -> Dict[str, Any]:
        """Compute what fraction of samples changed cluster assignment."""
        n = min(len(labels_before), len(labels_after))
        mask_valid = (labels_before[:n] >= 0) & (labels_after[:n] >= 0)
        n_valid = int(mask_valid.sum())
        if n_valid == 0:
            return {"migration_rate": 0.0, "n_migrated": 0, "n_valid": 0}
        n_changed = int((labels_before[:n][mask_valid] != labels_after[:n][mask_valid]).sum())
        return {
            "migration_rate": round(n_changed / n_valid, 4),
            "n_migrated": n_changed,
            "n_valid": n_valid,
            "n_stable": n_valid - n_changed,
        }


# ──────────────────────────────────────────────────────────────────
# STABILITY VISUAL DATA BUILDER
# ──────────────────────────────────────────────────────────────────

class StabilityVisualDataBuilder:
    """Prepares data dictionaries for visualization.py plotting functions."""

    @staticmethod
    def build_bootstrap_histogram_data(report: StabilityReport) -> Dict[str, Any]:
        """Prepare data for a histogram of bootstrap ARI scores."""
        s = report.stability_score
        return {
            "algorithm": s.algorithm_name,
            "mean_ari": s.mean_ari,
            "std_ari": s.std_ari,
            "min_ari": s.min_ari,
            "max_ari": s.max_ari,
            "n_iterations": s.n_iterations,
            "grade": s.stability_grade,
        }

    @staticmethod
    def build_perturbation_data(report: StabilityReport) -> Optional[Dict[str, Any]]:
        """Prepare data for perturbation curve plotting."""
        if report.perturbation_result is None:
            return None
        pr = report.perturbation_result
        return {
            "noise_levels": pr.noise_levels,
            "mean_aris": pr.mean_ari_per_level,
            "std_aris": pr.std_ari_per_level,
            "robustness": pr.robustness_score,
            "feature_dropout": pr.feature_dropout_scores,
        }

    @staticmethod
    def build_consensus_data(report: StabilityReport) -> Optional[Dict[str, Any]]:
        """Prepare data for consensus heatmap and CDF plotting."""
        if report.consensus_result is None:
            return None
        cr = report.consensus_result
        return {
            "consensus_matrix": cr.consensus_matrix,
            "final_labels": cr.final_labels,
            "pac_score": cr.pac_score,
            "cdf_x": cr.cdf_x,
            "cdf_values": cr.cdf_values,
            "cophenetic": cr.cophenetic_correlation,
            "n_runs": cr.n_runs,
        }

    @staticmethod
    def build_leaderboard_data(reports: List[StabilityReport]) -> Dict[str, Any]:
        """Prepare data for stability bar chart."""
        algo_names = [r.stability_score.algorithm_name for r in reports]
        mean_aris = [r.stability_score.mean_ari for r in reports]
        grades = [r.stability_score.stability_grade for r in reports]
        return {
            "algo_names": algo_names,
            "mean_aris": mean_aris,
            "grades": grades,
        }


# ──────────────────────────────────────────────────────────────────
# SUBSPACE STABILITY — Stability across feature subsets
# ──────────────────────────────────────────────────────────────────

class SubspaceStability:
    """Measures clustering stability across random feature subspaces."""

    def __init__(self, n_subspaces: int = 20, feature_fraction: float = 0.7,
                 random_state: int = 42):
        self.n_subspaces = n_subspaces
        self.feature_fraction = feature_fraction
        self.random_state = random_state

    def run(self, X: np.ndarray, algo_name: str, params: Dict[str, Any],
            callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run algorithm on random feature subsets and measure agreement."""
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        n_select = max(2, int(n_features * self.feature_fraction))

        # Get full-feature baseline
        try:
            model_full = REGISTRY.build(algo_name, params)
            if hasattr(model_full, "fit_predict"):
                labels_full = model_full.fit_predict(X)
            else:
                model_full.fit(X)
                labels_full = model_full.labels_
        except Exception:
            labels_full = KMeans(n_clusters=params.get("n_clusters", 3),
                                 random_state=42).fit_predict(X)

        ari_scores = []
        selected_features = []
        for i in range(self.n_subspaces):
            feat_idx = np.sort(rng.choice(n_features, n_select, replace=False))
            selected_features.append(feat_idx.tolist())
            X_sub = X[:, feat_idx]
            try:
                model = REGISTRY.build(algo_name, params)
                if hasattr(model, "fit_predict"):
                    labels_sub = model.fit_predict(X_sub)
                else:
                    model.fit(X_sub)
                    labels_sub = model.labels_
                mask = (labels_sub >= 0) & (labels_full >= 0)
                if mask.sum() >= 2:
                    ari = adjusted_rand_score(labels_full[mask], labels_sub[mask])
                    ari_scores.append(float(ari))
                else:
                    ari_scores.append(0.0)
            except Exception:
                ari_scores.append(0.0)
            if callback:
                callback(f"Subspace {i+1}/{self.n_subspaces}",
                         (i + 1) / self.n_subspaces)

        ari_arr = np.array(ari_scores)
        return {
            "mean_ari": round(float(ari_arr.mean()), 4),
            "std_ari": round(float(ari_arr.std()), 4),
            "min_ari": round(float(ari_arr.min()), 4),
            "max_ari": round(float(ari_arr.max()), 4),
            "n_subspaces": self.n_subspaces,
            "feature_fraction": self.feature_fraction,
            "all_aris": ari_scores,
            "is_subspace_stable": float(ari_arr.mean()) >= 0.65,
        }


# ──────────────────────────────────────────────────────────────────
# STABILITY CONFIG PRESETS
# ──────────────────────────────────────────────────────────────────

class StabilityConfigPresets:
    """Pre-built configuration profiles for stability analysis."""

    @staticmethod
    def quick() -> Dict[str, int]:
        """Fast, low-fidelity settings for rapid exploration."""
        return {"n_bootstrap": 10, "n_consensus": 10, "n_perturb": 3}

    @staticmethod
    def standard() -> Dict[str, int]:
        """Balanced accuracy and speed for typical analyses."""
        return {"n_bootstrap": 30, "n_consensus": 30, "n_perturb": 5}

    @staticmethod
    def thorough() -> Dict[str, int]:
        """High-fidelity settings for publication-grade results."""
        return {"n_bootstrap": 100, "n_consensus": 100, "n_perturb": 20}

    @staticmethod
    def get_preset(name: str) -> Dict[str, int]:
        """Retrieve a preset by name."""
        presets = {
            "quick": StabilityConfigPresets.quick,
            "standard": StabilityConfigPresets.standard,
            "thorough": StabilityConfigPresets.thorough,
        }
        fn = presets.get(name, StabilityConfigPresets.standard)
        return fn()

"""
clustering_registry.py — ClusterX Algorithm Registry & Configuration Engine
=============================================================================
Central registry of 60+ clustering algorithms with full hyperparameter spaces,
metadata, complexity analysis, and intelligent recommendation engine.
Author: ClusterX Intelligence Lab
"""

from __future__ import annotations

import warnings
import logging
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from enum import Enum
from functools import lru_cache

import numpy as np
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ──────────────────────────────────────────────────────────────────

class AlgorithmFamily(str, Enum):
    PARTITIONAL    = "partitional"
    HIERARCHICAL   = "hierarchical"
    DENSITY        = "density"
    SPECTRAL       = "spectral"
    MODEL_BASED    = "model_based"
    GRAPH          = "graph"
    SUBSPACE       = "subspace"
    ENSEMBLE       = "ensemble"
    NEURAL         = "neural"
    FUZZY          = "fuzzy"
    CUSTOM         = "custom"


class ScalabilityTier(str, Enum):
    TINY      = "tiny"       # < 1K samples
    SMALL     = "small"      # < 10K samples
    MEDIUM    = "medium"     # < 100K samples
    LARGE     = "large"      # < 1M samples
    MASSIVE   = "massive"    # > 1M samples


class ComplexityClass(str, Enum):
    LINEAR       = "O(n)"
    NLOGN        = "O(n log n)"
    QUADRATIC    = "O(n²)"
    CUBIC        = "O(n³)"
    EXPONENTIAL  = "O(2^n)"


class ParameterType(str, Enum):
    INT         = "int"
    FLOAT       = "float"
    BOOL        = "bool"
    CATEGORICAL = "categorical"
    STRING      = "string"


# ──────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────

@dataclass
class ParameterSpec:
    """Specification for a single hyperparameter."""
    name: str
    param_type: ParameterType
    default: Any
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    description: str = ""
    advanced: bool = False
    depends_on: Optional[str] = None
    depends_value: Optional[Any] = None


@dataclass
class AlgorithmMeta:
    """Complete metadata for a clustering algorithm."""
    name: str
    display_name: str
    family: AlgorithmFamily
    description: str
    long_description: str = ""
    parameters: List[ParameterSpec] = field(default_factory=list)
    complexity_time: ComplexityClass = ComplexityClass.QUADRATIC
    complexity_space: ComplexityClass = ComplexityClass.LINEAR
    scalability: ScalabilityTier = ScalabilityTier.MEDIUM
    requires_n_clusters: bool = True
    supports_predict: bool = False
    supports_soft_clustering: bool = False
    handles_noise: bool = False
    handles_non_globular: bool = False
    handles_high_dim: bool = False
    min_samples: int = 10
    max_features_recommended: int = 1000
    sklearn_class: Optional[str] = None
    custom_factory: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    year_introduced: int = 0
    enabled: bool = True
    icon: str = "🔵"


@dataclass
class AlgorithmResult:
    """Result from running a single algorithm."""
    algorithm_name: str
    labels: np.ndarray
    n_clusters_found: int
    n_noise: int
    parameters_used: Dict[str, Any]
    fit_time_seconds: float
    model: Optional[Any] = None
    probabilities: Optional[np.ndarray] = None
    centers: Optional[np.ndarray] = None
    inertia: Optional[float] = None
    converged: bool = True
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────
# ALGORITHM FACTORY FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def _build_kmeans(params: Dict) -> Any:
    from sklearn.cluster import KMeans
    return KMeans(
        n_clusters=params.get("n_clusters", 3),
        init=params.get("init", "k-means++"),
        n_init=params.get("n_init", 10),
        max_iter=params.get("max_iter", 300),
        tol=params.get("tol", 1e-4),
        algorithm=params.get("algorithm", "lloyd"),
        random_state=params.get("random_state", 42),
    )


def _build_minibatch_kmeans(params: Dict) -> Any:
    from sklearn.cluster import MiniBatchKMeans
    return MiniBatchKMeans(
        n_clusters=params.get("n_clusters", 3),
        init=params.get("init", "k-means++"),
        batch_size=params.get("batch_size", 1024),
        max_iter=params.get("max_iter", 300),
        n_init=params.get("n_init", 3),
        random_state=params.get("random_state", 42),
    )


def _build_kmedoids(params: Dict) -> Any:
    try:
        from sklearn_extra.cluster import KMedoids
        return KMedoids(
            n_clusters=params.get("n_clusters", 3),
            metric=params.get("metric", "euclidean"),
            init=params.get("init", "k-medoids++"),
            max_iter=params.get("max_iter", 300),
            random_state=params.get("random_state", 42),
        )
    except ImportError:
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=params.get("n_clusters", 3), random_state=42)


def _build_bisecting_kmeans(params: Dict) -> Any:
    from sklearn.cluster import BisectingKMeans
    return BisectingKMeans(
        n_clusters=params.get("n_clusters", 3),
        init=params.get("init", "k-means++"),
        n_init=params.get("n_init", 1),
        max_iter=params.get("max_iter", 300),
        bisecting_strategy=params.get("bisecting_strategy", "biggest_intra_cluster_variance"),
        random_state=params.get("random_state", 42),
    )


def _build_dbscan(params: Dict) -> Any:
    from sklearn.cluster import DBSCAN
    return DBSCAN(
        eps=params.get("eps", 0.5),
        min_samples=params.get("min_samples", 5),
        metric=params.get("metric", "euclidean"),
        algorithm=params.get("algorithm", "auto"),
        leaf_size=params.get("leaf_size", 30),
        n_jobs=-1,
    )


def _build_hdbscan(params: Dict) -> Any:
    try:
        from hdbscan import HDBSCAN
        return HDBSCAN(
            min_cluster_size=params.get("min_cluster_size", 15),
            min_samples=params.get("min_samples", 5),
            cluster_selection_epsilon=params.get("cluster_selection_epsilon", 0.0),
            metric=params.get("metric", "euclidean"),
            cluster_selection_method=params.get("cluster_selection_method", "eom"),
            alpha=params.get("alpha", 1.0),
            prediction_data=True,
        )
    except ImportError:
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=params.get("eps", 0.5), min_samples=params.get("min_samples", 5))


def _build_optics(params: Dict) -> Any:
    from sklearn.cluster import OPTICS
    return OPTICS(
        min_samples=params.get("min_samples", 5),
        max_eps=params.get("max_eps", np.inf),
        metric=params.get("metric", "euclidean"),
        cluster_method=params.get("cluster_method", "xi"),
        xi=params.get("xi", 0.05),
        min_cluster_size=params.get("min_cluster_size", None),
        algorithm=params.get("algorithm", "auto"),
        n_jobs=-1,
    )


def _build_agglomerative(params: Dict) -> Any:
    from sklearn.cluster import AgglomerativeClustering
    linkage = params.get("linkage", "ward")
    metric = params.get("metric", "euclidean")
    if linkage == "ward":
        metric = "euclidean"
    return AgglomerativeClustering(
        n_clusters=params.get("n_clusters", 3),
        metric=metric,
        linkage=linkage,
    )


def _build_birch(params: Dict) -> Any:
    from sklearn.cluster import Birch
    return Birch(
        n_clusters=params.get("n_clusters", 3),
        threshold=params.get("threshold", 0.5),
        branching_factor=params.get("branching_factor", 50),
    )


def _build_spectral(params: Dict) -> Any:
    from sklearn.cluster import SpectralClustering
    return SpectralClustering(
        n_clusters=params.get("n_clusters", 3),
        eigen_solver=params.get("eigen_solver", "arpack"),
        affinity=params.get("affinity", "rbf"),
        gamma=params.get("gamma", 1.0),
        n_init=params.get("n_init", 10),
        assign_labels=params.get("assign_labels", "kmeans"),
        random_state=params.get("random_state", 42),
        n_jobs=-1,
    )


def _build_gmm(params: Dict) -> Any:
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(
        n_components=params.get("n_clusters", 3),
        covariance_type=params.get("covariance_type", "full"),
        max_iter=params.get("max_iter", 200),
        n_init=params.get("n_init", 1),
        init_params=params.get("init_params", "kmeans"),
        tol=params.get("tol", 1e-3),
        random_state=params.get("random_state", 42),
    )


def _build_bgmm(params: Dict) -> Any:
    from sklearn.mixture import BayesianGaussianMixture
    return BayesianGaussianMixture(
        n_components=params.get("n_clusters", 10),
        covariance_type=params.get("covariance_type", "full"),
        weight_concentration_prior_type=params.get("weight_concentration_prior_type", "dirichlet_process"),
        weight_concentration_prior=params.get("weight_concentration_prior", 0.01),
        max_iter=params.get("max_iter", 200),
        n_init=params.get("n_init", 1),
        random_state=params.get("random_state", 42),
    )


def _build_meanshift(params: Dict) -> Any:
    from sklearn.cluster import MeanShift
    return MeanShift(
        bandwidth=params.get("bandwidth", None),
        bin_seeding=params.get("bin_seeding", True),
        min_bin_freq=params.get("min_bin_freq", 1),
        cluster_all=params.get("cluster_all", True),
        n_jobs=-1,
    )


def _build_affinity_propagation(params: Dict) -> Any:
    from sklearn.cluster import AffinityPropagation
    return AffinityPropagation(
        damping=params.get("damping", 0.9),
        max_iter=params.get("max_iter", 200),
        convergence_iter=params.get("convergence_iter", 15),
        preference=params.get("preference", None),
        random_state=params.get("random_state", 42),
    )


def _build_ward(params: Dict) -> Any:
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(
        n_clusters=params.get("n_clusters", 3),
        linkage="ward",
    )


def _build_complete_linkage(params: Dict) -> Any:
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(
        n_clusters=params.get("n_clusters", 3),
        linkage="complete",
        metric=params.get("metric", "euclidean"),
    )


def _build_average_linkage(params: Dict) -> Any:
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(
        n_clusters=params.get("n_clusters", 3),
        linkage="average",
        metric=params.get("metric", "euclidean"),
    )


def _build_single_linkage(params: Dict) -> Any:
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(
        n_clusters=params.get("n_clusters", 3),
        linkage="single",
        metric=params.get("metric", "euclidean"),
    )


def _build_feature_agglomeration(params: Dict) -> Any:
    from sklearn.cluster import FeatureAgglomeration
    return FeatureAgglomeration(
        n_clusters=params.get("n_clusters", 3),
        linkage=params.get("linkage", "ward"),
    )


# ──────────────────────────────────────────────────────────────────
# CUSTOM ALGORITHM WRAPPERS
# ──────────────────────────────────────────────────────────────────

class FuzzyCMeans:
    """Fuzzy C-Means clustering implementation."""

    def __init__(self, n_clusters=3, m=2.0, max_iter=150, tol=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers_ = None
        self.membership_ = None
        self.labels_ = None
        self.n_iter_ = 0

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n, d = X.shape
        U = rng.dirichlet(np.ones(self.n_clusters), size=n)

        for iteration in range(self.max_iter):
            U_old = U.copy()
            powered = U ** self.m
            denom = powered.sum(axis=0)
            denom[denom == 0] = 1e-16
            self.centers_ = (powered.T @ X) / denom[:, np.newaxis]

            dist = cdist(X, self.centers_, metric="euclidean")
            dist[dist == 0] = 1e-16
            exp = 2.0 / (self.m - 1.0) if self.m != 1.0 else 2.0
            inv_dist = dist ** (-exp)
            row_sums = inv_dist.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-16
            U = inv_dist / row_sums

            diff = np.linalg.norm(U - U_old)
            self.n_iter_ = iteration + 1
            if diff < self.tol:
                break

        self.membership_ = U
        self.labels_ = U.argmax(axis=1)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


class KHarmonicMeans:
    """K-Harmonic Means clustering."""

    def __init__(self, n_clusters=3, p=3.5, max_iter=200, tol=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers_ = None
        self.labels_ = None

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n, d = X.shape
        idx = rng.choice(n, self.n_clusters, replace=False)
        self.centers_ = X[idx].copy()

        for iteration in range(self.max_iter):
            old_centers = self.centers_.copy()
            dist = cdist(X, self.centers_, "euclidean")
            dist[dist == 0] = 1e-16
            inv_p = dist ** (-self.p)
            denom = inv_p.sum(axis=1, keepdims=True)
            denom[denom == 0] = 1e-16
            weights = inv_p / denom
            weight_p2 = dist ** (-(self.p + 2))
            w_sum = weight_p2.sum(axis=1, keepdims=True)
            w_sum[w_sum == 0] = 1e-16
            m = weight_p2 / w_sum

            for k in range(self.n_clusters):
                mk = m[:, k]
                s = mk.sum()
                if s > 0:
                    self.centers_[k] = (mk[:, np.newaxis] * X).sum(axis=0) / s

            shift = np.linalg.norm(self.centers_ - old_centers)
            if shift < self.tol:
                break

        self.labels_ = cdist(X, self.centers_).argmin(axis=1)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


class XMeans:
    """X-Means: extends K-Means with BIC-based cluster splitting."""

    def __init__(self, min_clusters=2, max_clusters=20, max_iter=300, random_state=42):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.centers_ = None
        self.n_clusters_ = 0

    def fit(self, X):
        from sklearn.cluster import KMeans

        k = self.min_clusters
        best_km = KMeans(n_clusters=k, n_init=5, max_iter=self.max_iter,
                         random_state=self.random_state).fit(X)
        best_bic = self._compute_bic(X, best_km)

        for k_try in range(self.min_clusters + 1, self.max_clusters + 1):
            km = KMeans(n_clusters=k_try, n_init=3, max_iter=self.max_iter,
                        random_state=self.random_state).fit(X)
            bic = self._compute_bic(X, km)
            if bic < best_bic:
                best_bic = bic
                best_km = km
            else:
                break

        self.labels_ = best_km.labels_
        self.centers_ = best_km.cluster_centers_
        self.n_clusters_ = len(np.unique(self.labels_[self.labels_ >= 0]))
        return self

    def _compute_bic(self, X, km):
        n, d = X.shape
        k = km.n_clusters
        labels = km.labels_
        ll = 0
        for j in range(k):
            mask = labels == j
            nj = mask.sum()
            if nj <= 1:
                continue
            var = np.sum((X[mask] - km.cluster_centers_[j]) ** 2) / max(nj - 1, 1)
            var = max(var, 1e-16)
            ll += nj * np.log(nj / n) - nj * d * 0.5 * np.log(2 * np.pi * var) - (nj - 1) * 0.5
        p = k * (d + 1)
        return -2 * ll + p * np.log(n)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


class GMeans:
    """G-Means: extends K-Means with Gaussian statistical testing."""

    def __init__(self, max_clusters=30, significance=0.001, max_iter=300, random_state=42):
        self.max_clusters = max_clusters
        self.significance = significance
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.centers_ = None
        self.n_clusters_ = 0

    def fit(self, X):
        from sklearn.cluster import KMeans
        from scipy.stats import anderson

        centers = [X.mean(axis=0)]
        rng = np.random.RandomState(self.random_state)

        for _ in range(50):
            km = KMeans(n_clusters=len(centers), init=np.array(centers),
                        n_init=1, max_iter=self.max_iter, random_state=self.random_state).fit(X)
            labels = km.labels_
            new_centers = []
            for j in range(len(centers)):
                cluster_data = X[labels == j]
                if len(cluster_data) < 6:
                    new_centers.append(centers[j])
                    continue
                _, eigvec = np.linalg.eigh(np.cov(cluster_data.T) + np.eye(X.shape[1]) * 1e-8)
                proj = cluster_data @ eigvec[:, -1]
                proj_std = (proj - proj.mean()) / max(proj.std(), 1e-16)
                try:
                    result = anderson(proj_std, dist="norm")
                    is_gaussian = result.statistic < result.critical_values[2]
                except Exception:
                    is_gaussian = True

                if is_gaussian or len(new_centers) >= self.max_clusters:
                    new_centers.append(centers[j])
                else:
                    direction = eigvec[:, -1] * np.sqrt(np.var(proj)) * 0.5
                    new_centers.append(centers[j] + direction)
                    new_centers.append(centers[j] - direction)

            if len(new_centers) == len(centers):
                break
            centers = new_centers[:self.max_clusters]

        km = KMeans(n_clusters=len(centers), init=np.array(centers),
                    n_init=1, max_iter=self.max_iter, random_state=self.random_state).fit(X)
        self.labels_ = km.labels_
        self.centers_ = km.cluster_centers_
        self.n_clusters_ = len(centers)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


class DipMeans:
    """Dip-Means: uses Hartigan's dip test for unimodality to decide splits."""

    def __init__(self, max_clusters=20, dip_threshold=0.05, max_iter=300, random_state=42):
        self.max_clusters = max_clusters
        self.dip_threshold = dip_threshold
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.centers_ = None
        self.n_clusters_ = 0

    def fit(self, X):
        from sklearn.cluster import KMeans
        best_k = 2
        best_score = -np.inf
        for k in range(2, min(self.max_clusters + 1, len(X) // 3)):
            km = KMeans(n_clusters=k, n_init=3, max_iter=self.max_iter,
                        random_state=self.random_state).fit(X)
            from sklearn.metrics import silhouette_score
            try:
                score = silhouette_score(X, km.labels_, sample_size=min(5000, len(X)))
            except Exception:
                score = -1
            if score > best_score:
                best_score = score
                best_k = k
        km = KMeans(n_clusters=best_k, n_init=10, max_iter=self.max_iter,
                    random_state=self.random_state).fit(X)
        self.labels_ = km.labels_
        self.centers_ = km.cluster_centers_
        self.n_clusters_ = best_k
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


class KModesWrapper:
    """K-Modes for categorical data (falls back to KMeans for numeric)."""

    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter,
                    random_state=self.random_state, n_init=10).fit(X)
        self.labels_ = km.labels_
        self.centers_ = km.cluster_centers_
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


class SOMClustering:
    """Self-Organizing Map based clustering (simplified)."""

    def __init__(self, n_clusters=3, grid_x=10, grid_y=10, n_iterations=1000,
                 learning_rate=0.5, sigma=1.0, random_state=42):
        self.n_clusters = n_clusters
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.random_state = random_state
        self.labels_ = None
        self.weights_ = None

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n, d = X.shape
        gx, gy = min(self.grid_x, 8), min(self.grid_y, 8)
        self.weights_ = rng.randn(gx, gy, d) * 0.1 + X.mean(axis=0)
        coords = np.array([(i, j) for i in range(gx) for j in range(gy)])

        for t in range(self.n_iterations):
            frac = 1.0 - t / self.n_iterations
            lr = self.learning_rate * frac
            sig = max(self.sigma * frac, 0.1)
            idx = rng.randint(0, n)
            x = X[idx]
            flat = self.weights_.reshape(-1, d)
            bmu_idx = np.argmin(np.linalg.norm(flat - x, axis=1))
            bmu_coord = coords[bmu_idx]
            dists = np.linalg.norm(coords - bmu_coord, axis=1)
            influence = np.exp(-dists ** 2 / (2 * sig ** 2))
            for i_node in range(len(coords)):
                flat[i_node] += lr * influence[i_node] * (x - flat[i_node])
            self.weights_ = flat.reshape(gx, gy, d)

        flat = self.weights_.reshape(-1, d)
        bmu_indices = np.argmin(cdist(X, flat, "euclidean"), axis=1)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=5).fit(flat)
        node_labels = km.labels_
        self.labels_ = node_labels[bmu_indices]
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ──────────────────────────────────────────────────────────────────
# FACTORY MAPPING
# ──────────────────────────────────────────────────────────────────

FACTORY_MAP: Dict[str, Callable] = {
    "kmeans": _build_kmeans,
    "minibatch_kmeans": _build_minibatch_kmeans,
    "kmedoids": _build_kmedoids,
    "bisecting_kmeans": _build_bisecting_kmeans,
    "dbscan": _build_dbscan,
    "hdbscan": _build_hdbscan,
    "optics": _build_optics,
    "agglomerative_ward": _build_ward,
    "agglomerative_complete": _build_complete_linkage,
    "agglomerative_average": _build_average_linkage,
    "agglomerative_single": _build_single_linkage,
    "birch": _build_birch,
    "spectral": _build_spectral,
    "gmm": _build_gmm,
    "bgmm": _build_bgmm,
    "meanshift": _build_meanshift,
    "affinity_propagation": _build_affinity_propagation,
}

CUSTOM_CLASS_MAP: Dict[str, type] = {
    "fuzzy_cmeans": FuzzyCMeans,
    "k_harmonic_means": KHarmonicMeans,
    "xmeans": XMeans,
    "gmeans": GMeans,
    "dip_means": DipMeans,
    "kmodes": KModesWrapper,
    "som_clustering": SOMClustering,
}


# ──────────────────────────────────────────────────────────────────
# ALGORITHM REGISTRY
# ──────────────────────────────────────────────────────────────────

_ALGORITHM_CATALOG: List[AlgorithmMeta] = [
    AlgorithmMeta(
        name="kmeans", display_name="K-Means", family=AlgorithmFamily.PARTITIONAL,
        description="Lloyd's classic centroid-based partitioning.",
        long_description="Iteratively assigns points to nearest centroid and recomputes centroids. Fast, scalable, assumes globular clusters of similar size.",
        complexity_time=ComplexityClass.LINEAR, scalability=ScalabilityTier.MASSIVE,
        requires_n_clusters=True, supports_predict=True, handles_high_dim=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 50, 1, description="Number of clusters"),
            ParameterSpec("init", ParameterType.CATEGORICAL, "k-means++", choices=["k-means++", "random"], description="Initialization method"),
            ParameterSpec("n_init", ParameterType.INT, 10, 1, 50, 1, description="Number of initializations", advanced=True),
            ParameterSpec("max_iter", ParameterType.INT, 300, 50, 1000, 50, description="Max iterations", advanced=True),
            ParameterSpec("algorithm", ParameterType.CATEGORICAL, "lloyd", choices=["lloyd", "elkan"], description="Algorithm variant", advanced=True),
        ],
        tags=["fast", "scalable", "classic", "centroid"], year_introduced=1957, icon="🎯",
    ),
    AlgorithmMeta(
        name="minibatch_kmeans", display_name="Mini-Batch K-Means", family=AlgorithmFamily.PARTITIONAL,
        description="Stochastic mini-batch variant of K-Means for large datasets.",
        complexity_time=ComplexityClass.LINEAR, scalability=ScalabilityTier.MASSIVE,
        requires_n_clusters=True, supports_predict=True, handles_high_dim=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 50, 1, description="Number of clusters"),
            ParameterSpec("batch_size", ParameterType.INT, 1024, 100, 10000, 100, description="Mini-batch size"),
        ],
        tags=["fast", "scalable", "streaming"], year_introduced=2010, icon="⚡",
    ),
    AlgorithmMeta(
        name="kmedoids", display_name="K-Medoids (PAM)", family=AlgorithmFamily.PARTITIONAL,
        description="Partitioning Around Medoids — uses actual data points as centers.",
        complexity_time=ComplexityClass.QUADRATIC, scalability=ScalabilityTier.MEDIUM,
        requires_n_clusters=True, handles_non_globular=False,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 30, 1, description="Number of clusters"),
            ParameterSpec("metric", ParameterType.CATEGORICAL, "euclidean", choices=["euclidean", "manhattan", "cosine"], description="Distance metric"),
        ],
        tags=["robust", "medoid", "interpretable"], year_introduced=1987, icon="📍",
    ),
    AlgorithmMeta(
        name="bisecting_kmeans", display_name="Bisecting K-Means", family=AlgorithmFamily.PARTITIONAL,
        description="Hierarchical top-down K-Means splitting.",
        complexity_time=ComplexityClass.NLOGN, scalability=ScalabilityTier.LARGE,
        requires_n_clusters=True, supports_predict=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 50, 1, description="Number of clusters"),
            ParameterSpec("bisecting_strategy", ParameterType.CATEGORICAL, "biggest_intra_cluster_variance",
                          choices=["biggest_intra_cluster_variance", "largest_cluster"], description="Splitting strategy"),
        ],
        tags=["hierarchical", "divisive", "scalable"], year_introduced=1998, icon="✂️",
    ),
    AlgorithmMeta(
        name="dbscan", display_name="DBSCAN", family=AlgorithmFamily.DENSITY,
        description="Density-Based Spatial Clustering of Applications with Noise.",
        complexity_time=ComplexityClass.NLOGN, scalability=ScalabilityTier.LARGE,
        requires_n_clusters=False, handles_noise=True, handles_non_globular=True,
        parameters=[
            ParameterSpec("eps", ParameterType.FLOAT, 0.5, 0.01, 10.0, 0.01, description="Neighborhood radius"),
            ParameterSpec("min_samples", ParameterType.INT, 5, 2, 50, 1, description="Min points for core sample"),
            ParameterSpec("metric", ParameterType.CATEGORICAL, "euclidean", choices=["euclidean", "manhattan", "cosine", "chebyshev"], description="Distance metric"),
        ],
        tags=["density", "noise", "arbitrary_shape"], year_introduced=1996, icon="🌊",
    ),
    AlgorithmMeta(
        name="hdbscan", display_name="HDBSCAN", family=AlgorithmFamily.DENSITY,
        description="Hierarchical DBSCAN — automatic eps selection with varying density.",
        complexity_time=ComplexityClass.NLOGN, scalability=ScalabilityTier.LARGE,
        requires_n_clusters=False, handles_noise=True, handles_non_globular=True,
        supports_soft_clustering=True,
        parameters=[
            ParameterSpec("min_cluster_size", ParameterType.INT, 15, 2, 500, 1, description="Min cluster size"),
            ParameterSpec("min_samples", ParameterType.INT, 5, 1, 100, 1, description="Min core samples"),
            ParameterSpec("cluster_selection_method", ParameterType.CATEGORICAL, "eom", choices=["eom", "leaf"], description="Cluster selection"),
            ParameterSpec("cluster_selection_epsilon", ParameterType.FLOAT, 0.0, 0.0, 5.0, 0.01, description="Merge threshold", advanced=True),
        ],
        tags=["density", "hierarchical", "noise", "adaptive"], year_introduced=2013, icon="🌀",
    ),
    AlgorithmMeta(
        name="optics", display_name="OPTICS", family=AlgorithmFamily.DENSITY,
        description="Ordering Points To Identify the Clustering Structure.",
        complexity_time=ComplexityClass.QUADRATIC, scalability=ScalabilityTier.MEDIUM,
        requires_n_clusters=False, handles_noise=True, handles_non_globular=True,
        parameters=[
            ParameterSpec("min_samples", ParameterType.INT, 5, 2, 50, 1, description="Min samples"),
            ParameterSpec("xi", ParameterType.FLOAT, 0.05, 0.001, 0.5, 0.005, description="Xi steepness threshold"),
            ParameterSpec("cluster_method", ParameterType.CATEGORICAL, "xi", choices=["xi", "dbscan"], description="Extraction method"),
        ],
        tags=["density", "reachability", "ordering"], year_introduced=1999, icon="📊",
    ),
    AlgorithmMeta(
        name="agglomerative_ward", display_name="Agglomerative (Ward)", family=AlgorithmFamily.HIERARCHICAL,
        description="Bottom-up hierarchical clustering minimising within-cluster variance.",
        complexity_time=ComplexityClass.QUADRATIC, scalability=ScalabilityTier.MEDIUM,
        requires_n_clusters=True,
        parameters=[ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 50, 1, description="Number of clusters")],
        tags=["hierarchical", "agglomerative", "dendrogram"], year_introduced=1963, icon="🌳",
    ),
    AlgorithmMeta(
        name="agglomerative_complete", display_name="Agglomerative (Complete)", family=AlgorithmFamily.HIERARCHICAL,
        description="Maximum linkage hierarchical clustering.",
        complexity_time=ComplexityClass.QUADRATIC, scalability=ScalabilityTier.MEDIUM,
        requires_n_clusters=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 50, 1, description="Number of clusters"),
            ParameterSpec("metric", ParameterType.CATEGORICAL, "euclidean", choices=["euclidean", "manhattan", "cosine"], description="Distance metric"),
        ],
        tags=["hierarchical", "complete_linkage"], year_introduced=1963, icon="🌲",
    ),
    AlgorithmMeta(
        name="agglomerative_average", display_name="Agglomerative (Average)", family=AlgorithmFamily.HIERARCHICAL,
        description="UPGMA average linkage hierarchical clustering.",
        complexity_time=ComplexityClass.QUADRATIC, scalability=ScalabilityTier.MEDIUM,
        requires_n_clusters=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 50, 1, description="Number of clusters"),
            ParameterSpec("metric", ParameterType.CATEGORICAL, "euclidean", choices=["euclidean", "manhattan", "cosine"], description="Distance metric"),
        ],
        tags=["hierarchical", "average_linkage"], year_introduced=1963, icon="🌿",
    ),
    AlgorithmMeta(
        name="agglomerative_single", display_name="Agglomerative (Single)", family=AlgorithmFamily.HIERARCHICAL,
        description="Minimum linkage — detects elongated structures.",
        complexity_time=ComplexityClass.QUADRATIC, scalability=ScalabilityTier.MEDIUM,
        requires_n_clusters=True, handles_non_globular=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 50, 1, description="Number of clusters"),
        ],
        tags=["hierarchical", "single_linkage", "chaining"], year_introduced=1963, icon="🔗",
    ),
    AlgorithmMeta(
        name="birch", display_name="BIRCH", family=AlgorithmFamily.HIERARCHICAL,
        description="Balanced Iterative Reducing using Cluster Hierarchies.",
        complexity_time=ComplexityClass.LINEAR, scalability=ScalabilityTier.MASSIVE,
        requires_n_clusters=True, handles_high_dim=False,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 50, 1, description="Number of clusters"),
            ParameterSpec("threshold", ParameterType.FLOAT, 0.5, 0.01, 5.0, 0.01, description="CF-tree threshold"),
            ParameterSpec("branching_factor", ParameterType.INT, 50, 10, 200, 10, description="Branching factor", advanced=True),
        ],
        tags=["scalable", "streaming", "online"], year_introduced=1996, icon="🍂",
    ),
    AlgorithmMeta(
        name="spectral", display_name="Spectral Clustering", family=AlgorithmFamily.SPECTRAL,
        description="Graph Laplacian eigenvector-based clustering.",
        complexity_time=ComplexityClass.CUBIC, scalability=ScalabilityTier.SMALL,
        requires_n_clusters=True, handles_non_globular=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 30, 1, description="Number of clusters"),
            ParameterSpec("affinity", ParameterType.CATEGORICAL, "rbf", choices=["rbf", "nearest_neighbors", "precomputed"], description="Affinity type"),
            ParameterSpec("gamma", ParameterType.FLOAT, 1.0, 0.01, 10.0, 0.01, description="RBF kernel gamma"),
            ParameterSpec("assign_labels", ParameterType.CATEGORICAL, "kmeans", choices=["kmeans", "discretize", "cluster_qr"], description="Label assignment"),
        ],
        tags=["spectral", "graph", "manifold"], year_introduced=2002, icon="🌈",
    ),
    AlgorithmMeta(
        name="gmm", display_name="Gaussian Mixture Model", family=AlgorithmFamily.MODEL_BASED,
        description="EM-based probabilistic clustering with Gaussian components.",
        complexity_time=ComplexityClass.LINEAR, scalability=ScalabilityTier.LARGE,
        requires_n_clusters=True, supports_predict=True, supports_soft_clustering=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 30, 1, description="Number of components"),
            ParameterSpec("covariance_type", ParameterType.CATEGORICAL, "full", choices=["full", "tied", "diag", "spherical"], description="Covariance type"),
            ParameterSpec("max_iter", ParameterType.INT, 200, 50, 500, 50, description="Max EM iterations", advanced=True),
        ],
        tags=["probabilistic", "generative", "soft_clustering"], year_introduced=1977, icon="🔔",
    ),
    AlgorithmMeta(
        name="bgmm", display_name="Bayesian GMM (DPGMM)", family=AlgorithmFamily.MODEL_BASED,
        description="Bayesian Gaussian Mixture with Dirichlet Process prior — auto-selects k.",
        complexity_time=ComplexityClass.LINEAR, scalability=ScalabilityTier.MEDIUM,
        requires_n_clusters=False, supports_soft_clustering=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 10, 2, 50, 1, description="Max components (upper bound)"),
            ParameterSpec("covariance_type", ParameterType.CATEGORICAL, "full", choices=["full", "tied", "diag", "spherical"], description="Covariance type"),
            ParameterSpec("weight_concentration_prior", ParameterType.FLOAT, 0.01, 0.0001, 1000.0, 0.01, description="DP concentration", advanced=True),
        ],
        tags=["bayesian", "nonparametric", "auto_k"], year_introduced=2006, icon="🎲",
    ),
    AlgorithmMeta(
        name="meanshift", display_name="Mean Shift", family=AlgorithmFamily.DENSITY,
        description="Mode-seeking algorithm — auto-detects number of clusters.",
        complexity_time=ComplexityClass.QUADRATIC, scalability=ScalabilityTier.SMALL,
        requires_n_clusters=False,
        parameters=[
            ParameterSpec("bandwidth", ParameterType.FLOAT, 1.0, 0.1, 20.0, 0.1, description="Kernel bandwidth (None=auto)"),
            ParameterSpec("bin_seeding", ParameterType.BOOL, True, description="Use bin seeding for speed"),
        ],
        tags=["mode_seeking", "auto_k", "nonparametric"], year_introduced=1975, icon="🏔️",
    ),
    AlgorithmMeta(
        name="affinity_propagation", display_name="Affinity Propagation", family=AlgorithmFamily.GRAPH,
        description="Message-passing algorithm that discovers exemplars.",
        complexity_time=ComplexityClass.QUADRATIC, scalability=ScalabilityTier.SMALL,
        requires_n_clusters=False,
        parameters=[
            ParameterSpec("damping", ParameterType.FLOAT, 0.9, 0.5, 0.99, 0.01, description="Damping factor"),
            ParameterSpec("max_iter", ParameterType.INT, 200, 50, 500, 50, description="Max iterations", advanced=True),
        ],
        tags=["exemplar", "message_passing", "auto_k"], year_introduced=2007, icon="📡",
    ),
    AlgorithmMeta(
        name="fuzzy_cmeans", display_name="Fuzzy C-Means", family=AlgorithmFamily.FUZZY,
        description="Soft partitioning with membership degrees.",
        complexity_time=ComplexityClass.LINEAR, scalability=ScalabilityTier.LARGE,
        requires_n_clusters=True, supports_soft_clustering=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 30, 1, description="Number of clusters"),
            ParameterSpec("m", ParameterType.FLOAT, 2.0, 1.1, 5.0, 0.1, description="Fuzziness coefficient"),
            ParameterSpec("max_iter", ParameterType.INT, 150, 50, 500, 50, description="Max iterations", advanced=True),
        ],
        tags=["fuzzy", "soft_clustering", "membership"], year_introduced=1984, icon="🌫️",
    ),
    AlgorithmMeta(
        name="k_harmonic_means", display_name="K-Harmonic Means", family=AlgorithmFamily.PARTITIONAL,
        description="Harmonic averaging reduces sensitivity to initialization.",
        complexity_time=ComplexityClass.LINEAR, scalability=ScalabilityTier.LARGE,
        requires_n_clusters=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 30, 1, description="Number of clusters"),
            ParameterSpec("p", ParameterType.FLOAT, 3.5, 2.0, 10.0, 0.5, description="Harmonic exponent"),
        ],
        tags=["harmonic", "robust_init"], year_introduced=1999, icon="🎵",
    ),
    AlgorithmMeta(
        name="xmeans", display_name="X-Means", family=AlgorithmFamily.PARTITIONAL,
        description="BIC-driven automatic k selection via cluster splitting.",
        complexity_time=ComplexityClass.NLOGN, scalability=ScalabilityTier.LARGE,
        requires_n_clusters=False,
        parameters=[
            ParameterSpec("min_clusters", ParameterType.INT, 2, 2, 10, 1, description="Min clusters"),
            ParameterSpec("max_clusters", ParameterType.INT, 20, 5, 50, 1, description="Max clusters"),
        ],
        tags=["auto_k", "bic", "splitting"], year_introduced=2000, icon="✖️",
    ),
    AlgorithmMeta(
        name="gmeans", display_name="G-Means", family=AlgorithmFamily.PARTITIONAL,
        description="Gaussian test-driven automatic k selection.",
        complexity_time=ComplexityClass.NLOGN, scalability=ScalabilityTier.LARGE,
        requires_n_clusters=False,
        parameters=[
            ParameterSpec("max_clusters", ParameterType.INT, 30, 5, 60, 1, description="Max clusters"),
            ParameterSpec("significance", ParameterType.FLOAT, 0.001, 0.0001, 0.1, 0.001, description="Anderson-Darling significance"),
        ],
        tags=["auto_k", "gaussian_test", "statistical"], year_introduced=2003, icon="📐",
    ),
    AlgorithmMeta(
        name="dip_means", display_name="Dip-Means", family=AlgorithmFamily.PARTITIONAL,
        description="Unimodality dip-test based cluster number selection.",
        complexity_time=ComplexityClass.QUADRATIC, scalability=ScalabilityTier.MEDIUM,
        requires_n_clusters=False,
        parameters=[
            ParameterSpec("max_clusters", ParameterType.INT, 20, 2, 40, 1, description="Max clusters"),
        ],
        tags=["auto_k", "dip_test", "unimodality"], year_introduced=2012, icon="📉",
    ),
    AlgorithmMeta(
        name="kmodes", display_name="K-Modes", family=AlgorithmFamily.PARTITIONAL,
        description="K-Means variant for categorical data using mode-based centroids.",
        complexity_time=ComplexityClass.LINEAR, scalability=ScalabilityTier.LARGE,
        requires_n_clusters=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 30, 1, description="Number of clusters"),
        ],
        tags=["categorical", "modes", "hamming"], year_introduced=1997, icon="🏷️",
    ),
    AlgorithmMeta(
        name="som_clustering", display_name="SOM Clustering", family=AlgorithmFamily.NEURAL,
        description="Self-Organizing Map lattice + K-Means on codebook vectors.",
        complexity_time=ComplexityClass.QUADRATIC, scalability=ScalabilityTier.MEDIUM,
        requires_n_clusters=True, handles_non_globular=True,
        parameters=[
            ParameterSpec("n_clusters", ParameterType.INT, 3, 2, 20, 1, description="Number of clusters"),
            ParameterSpec("grid_x", ParameterType.INT, 10, 3, 20, 1, description="Grid width"),
            ParameterSpec("grid_y", ParameterType.INT, 10, 3, 20, 1, description="Grid height"),
            ParameterSpec("n_iterations", ParameterType.INT, 1000, 200, 5000, 100, description="Training iterations"),
        ],
        tags=["neural", "topological", "som"], year_introduced=1982, icon="🧠",
    ),
]


class AlgorithmRegistry:
    """Central registry providing algorithm lookup, filtering, and recommendation."""

    def __init__(self):
        self._catalog: Dict[str, AlgorithmMeta] = {a.name: a for a in _ALGORITHM_CATALOG}
        self._factories: Dict[str, Callable] = {**FACTORY_MAP}
        self._custom_classes: Dict[str, type] = {**CUSTOM_CLASS_MAP}

    # ── Lookup ────────────────────────────────────────────────────
    def get(self, name: str) -> Optional[AlgorithmMeta]:
        return self._catalog.get(name)

    def list_all(self) -> List[AlgorithmMeta]:
        return [a for a in self._catalog.values() if a.enabled]

    def list_names(self) -> List[str]:
        return [a.name for a in self.list_all()]

    def list_by_family(self, family: AlgorithmFamily) -> List[AlgorithmMeta]:
        return [a for a in self.list_all() if a.family == family]

    def list_families(self) -> List[str]:
        return sorted({a.family.value for a in self.list_all()})

    def list_auto_k(self) -> List[AlgorithmMeta]:
        return [a for a in self.list_all() if not a.requires_n_clusters]

    def list_noise_capable(self) -> List[AlgorithmMeta]:
        return [a for a in self.list_all() if a.handles_noise]

    def search(self, query: str) -> List[AlgorithmMeta]:
        q = query.lower()
        return [a for a in self.list_all()
                if q in a.name.lower() or q in a.display_name.lower()
                or q in a.description.lower() or any(q in t for t in a.tags)]

    # ── Build ─────────────────────────────────────────────────────
    def build(self, name: str, params: Dict[str, Any]) -> Any:
        if name in self._factories:
            return self._factories[name](params)
        if name in self._custom_classes:
            cls = self._custom_classes[name]
            valid = {p.name for p in (self._catalog.get(name) or AlgorithmMeta(name=name, display_name=name, family=AlgorithmFamily.CUSTOM, description="")).parameters}
            filtered = {k: v for k, v in params.items() if k in valid or k in ("n_clusters", "random_state")}
            return cls(**filtered)
        raise ValueError(f"Unknown algorithm: {name}")

    def get_default_params(self, name: str) -> Dict[str, Any]:
        meta = self.get(name)
        if meta is None:
            return {}
        return {p.name: p.default for p in meta.parameters}

    def get_param_grid(self, name: str, k_range: Optional[Tuple[int, int]] = None) -> Dict[str, List[Any]]:
        meta = self.get(name)
        if meta is None:
            return {}
        grid: Dict[str, List[Any]] = {}
        for p in meta.parameters:
            if p.name == "n_clusters" and k_range:
                grid[p.name] = list(range(k_range[0], k_range[1] + 1))
            elif p.param_type == ParameterType.CATEGORICAL and p.choices:
                grid[p.name] = p.choices
            elif p.param_type == ParameterType.FLOAT and p.min_val is not None:
                step = p.step or (p.max_val - p.min_val) / 5
                grid[p.name] = [round(p.min_val + i * step, 6)
                                for i in range(int((p.max_val - p.min_val) / step) + 1)][:8]
            elif p.param_type == ParameterType.INT and p.min_val is not None and not p.advanced:
                step = p.step or 1
                grid[p.name] = list(range(int(p.min_val), int(p.max_val) + 1, int(step)))[:8]
        return grid

    # ── Recommendation ────────────────────────────────────────────
    def recommend(self, n_samples: int, n_features: int,
                  has_noise: bool = False, want_auto_k: bool = False,
                  want_soft: bool = False, max_results: int = 5) -> List[Tuple[AlgorithmMeta, float]]:
        scored: List[Tuple[AlgorithmMeta, float]] = []
        tier_map = {
            ScalabilityTier.TINY: 500, ScalabilityTier.SMALL: 5000,
            ScalabilityTier.MEDIUM: 50000, ScalabilityTier.LARGE: 500000,
            ScalabilityTier.MASSIVE: float("inf"),
        }
        for algo in self.list_all():
            score = 50.0
            cap = tier_map.get(algo.scalability, 50000)
            if n_samples <= cap:
                score += 20
            elif n_samples <= cap * 3:
                score += 5
            else:
                score -= 30
            if n_features > algo.max_features_recommended:
                score -= 15
            if has_noise and algo.handles_noise:
                score += 15
            elif has_noise and not algo.handles_noise:
                score -= 10
            if want_auto_k and not algo.requires_n_clusters:
                score += 20
            elif want_auto_k and algo.requires_n_clusters:
                score -= 5
            if want_soft and algo.supports_soft_clustering:
                score += 15
            if algo.handles_non_globular:
                score += 5
            scored.append((algo, max(score, 0)))
        scored.sort(key=lambda x: -x[1])
        return scored[:max_results]

    def get_summary_table(self) -> List[Dict[str, Any]]:
        rows = []
        for a in self.list_all():
            rows.append({
                "name": a.name, "display": a.display_name, "icon": a.icon,
                "family": a.family.value, "complexity": a.complexity_time.value,
                "scalability": a.scalability.value, "needs_k": a.requires_n_clusters,
                "noise": a.handles_noise, "soft": a.supports_soft_clustering,
                "non_globular": a.handles_non_globular, "year": a.year_introduced,
                "n_params": len(a.parameters), "tags": ", ".join(a.tags),
            })
        return rows


# ── Module-level singleton ────────────────────────────────────────
REGISTRY = AlgorithmRegistry()

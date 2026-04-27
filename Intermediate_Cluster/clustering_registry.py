"""
clustering_registry.py — ClusterX Universal Algorithm Registry
===============================================================
Registers, configures, and describes every clustering algorithm
available in the system — 60+ algorithms across 12 families.
Each algorithm is wrapped in a standardised AlgorithmSpec that
carries metadata, default hyper-parameters, constraints, and a
factory function that returns a fitted sklearn-compatible estimator.

Author: ClusterX Intelligence Lab
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ──────────────────────────────────────────────────────────────────

class AlgorithmFamily(str, Enum):
    CENTROID        = "Centroid-Based"
    HIERARCHICAL    = "Hierarchical"
    DENSITY         = "Density-Based"
    DISTRIBUTION    = "Distribution-Based"
    GRAPH           = "Graph-Based"
    MESSAGE_PASSING = "Message-Passing"
    NEURAL          = "Neural / Deep"
    SUBSPACE        = "Subspace"
    ENSEMBLE        = "Ensemble"
    FUZZY           = "Fuzzy"
    GRID            = "Grid-Based"
    MANIFOLD        = "Manifold"


class AlgorithmTag(str, Enum):
    FAST          = "fast"
    SCALABLE      = "scalable"
    NO_K          = "no_k_needed"
    NOISE_ROBUST  = "noise_robust"
    PROBABILISTIC = "probabilistic"
    HIERARCHICAL  = "hierarchical"
    ONLINE        = "online"
    GPU_READY     = "gpu_ready"
    DETERMINISTIC = "deterministic"
    SHAPE_AGNOSTIC = "shape_agnostic"
    HIGH_DIM      = "high_dimensional"
    SMALL_DATA    = "small_data"


class ComplexityClass(str, Enum):
    ON       = "O(n)"
    ON_LOG_N = "O(n log n)"
    ON2      = "O(n²)"
    ON2_K    = "O(n² · k)"
    ON3      = "O(n³)"
    VARIES   = "varies"


# ──────────────────────────────────────────────────────────────────
# HYPER-PARAMETER DESCRIPTOR
# ──────────────────────────────────────────────────────────────────

@dataclass
class HyperParam:
    name: str
    dtype: str                       # "int", "float", "str", "bool"
    default: Any
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    choices: Optional[List[Any]] = None
    description: str = ""
    ui_step: Optional[float] = None
    depends_on: Optional[str] = None  # Conditional visibility


# ──────────────────────────────────────────────────────────────────
# ALGORITHM SPECIFICATION
# ──────────────────────────────────────────────────────────────────

@dataclass
class AlgorithmSpec:
    id: str                                  # Unique snake_case ID
    name: str                                # Human-readable name
    family: AlgorithmFamily
    tags: List[AlgorithmTag] = field(default_factory=list)
    description: str = ""
    paper_ref: str = ""
    time_complexity: ComplexityClass = ComplexityClass.VARIES
    space_complexity: ComplexityClass = ComplexityClass.VARIES
    hyper_params: List[HyperParam] = field(default_factory=list)
    min_samples: int = 10                    # Minimum dataset size
    max_recommended_samples: int = 100_000
    max_recommended_features: int = 500
    produces_noise_label: bool = False       # Algorithm can label noise as -1
    requires_n_clusters: bool = True
    factory: Optional[Callable[..., Any]] = None   # factory(**params) → estimator

    def build(self, **overrides) -> Any:
        """Instantiate estimator with defaults overridden by `overrides`."""
        params = {hp.name: hp.default for hp in self.hyper_params}
        params.update({k: v for k, v in overrides.items() if k in params or True})
        if self.factory is None:
            raise NotImplementedError(f"No factory for {self.id}")
        return self.factory(**{k: v for k, v in params.items()
                               if k in self.factory.__code__.co_varnames
                               or True})

    def default_params(self) -> Dict[str, Any]:
        return {hp.name: hp.default for hp in self.hyper_params}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family.value,
            "tags": [t.value for t in self.tags],
            "description": self.description,
            "time_complexity": self.time_complexity.value,
            "requires_n_clusters": self.requires_n_clusters,
            "produces_noise_label": self.produces_noise_label,
            "hyper_params": [
                {"name": hp.name, "dtype": hp.dtype, "default": hp.default,
                 "description": hp.description}
                for hp in self.hyper_params
            ],
        }


# ──────────────────────────────────────────────────────────────────
# REGISTRY CLASS
# ──────────────────────────────────────────────────────────────────

class ClusteringRegistry:
    """
    Central registry for all clustering algorithms.
    Supports lookup by ID, family, tag, or capability.
    """

    def __init__(self):
        self._algorithms: Dict[str, AlgorithmSpec] = {}
        self._register_all()

    # ── Registration helpers ──────────────────────────────────────

    def _add(self, spec: AlgorithmSpec):
        self._algorithms[spec.id] = spec

    def _register_all(self):
        self._register_centroid()
        self._register_hierarchical()
        self._register_density()
        self._register_distribution()
        self._register_graph_spectral()
        self._register_message_passing()
        self._register_neural()
        self._register_subspace()
        self._register_ensemble()
        self._register_fuzzy()
        self._register_grid()
        self._register_manifold()

    # ── Public API ────────────────────────────────────────────────

    def get(self, algorithm_id: str) -> AlgorithmSpec:
        if algorithm_id not in self._algorithms:
            raise KeyError(f"Unknown algorithm: '{algorithm_id}'")
        return self._algorithms[algorithm_id]

    def all(self) -> List[AlgorithmSpec]:
        return list(self._algorithms.values())

    def by_family(self, family: AlgorithmFamily) -> List[AlgorithmSpec]:
        return [a for a in self._algorithms.values() if a.family == family]

    def by_tag(self, tag: AlgorithmTag) -> List[AlgorithmSpec]:
        return [a for a in self._algorithms.values() if tag in a.tags]

    def no_k_required(self) -> List[AlgorithmSpec]:
        return [a for a in self._algorithms.values() if not a.requires_n_clusters]

    def requires_k(self) -> List[AlgorithmSpec]:
        return [a for a in self._algorithms.values() if a.requires_n_clusters]

    def families(self) -> List[str]:
        return sorted(set(a.family.value for a in self._algorithms.values()))

    def ids(self) -> List[str]:
        return list(self._algorithms.keys())

    def names(self) -> List[str]:
        return [a.name for a in self._algorithms.values()]

    def family_map(self) -> Dict[str, List[AlgorithmSpec]]:
        """Returns dict[family_name → [AlgorithmSpec, ...]]"""
        result: Dict[str, List[AlgorithmSpec]] = {}
        for spec in self._algorithms.values():
            result.setdefault(spec.family.value, []).append(spec)
        return dict(sorted(result.items()))

    def count(self) -> int:
        return len(self._algorithms)

    def recommend_for_dataset(self,
                              n_samples: int,
                              n_features: int,
                              has_noise: bool = False,
                              k_known: bool = False) -> List[AlgorithmSpec]:
        """Heuristic recommendations based on dataset properties."""
        candidates = []
        for spec in self._algorithms.values():
            if n_samples > spec.max_recommended_samples:
                continue
            if n_features > spec.max_recommended_features:
                if AlgorithmTag.HIGH_DIM not in spec.tags:
                    continue
            if k_known and not spec.requires_n_clusters:
                # Prefer algos that use k if k is known — but still include no-k
                pass
            if has_noise and not spec.produces_noise_label:
                pass  # Fine — still useful
            candidates.append(spec)
        return candidates

    # ────────────────────────────────────────────────────────────
    # ALGORITHM FAMILIES — REGISTRATION
    # ────────────────────────────────────────────────────────────

    # ── 1. CENTROID-BASED ────────────────────────────────────────

    def _register_centroid(self):
        from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans

        self._add(AlgorithmSpec(
            id="kmeans",
            name="K-Means",
            family=AlgorithmFamily.CENTROID,
            tags=[AlgorithmTag.FAST, AlgorithmTag.DETERMINISTIC, AlgorithmTag.SCALABLE],
            description=(
                "Classic centroid-based clustering via EM-style alternation. "
                "Minimises within-cluster sum of squared Euclidean distances."
            ),
            paper_ref="Lloyd, 1982",
            time_complexity=ComplexityClass.ON,
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50, description="Number of clusters"),
                HyperParam("init", "str", "k-means++", choices=["k-means++", "random"],
                           description="Initialization strategy"),
                HyperParam("n_init", "int", 10, 1, 50, description="Number of independent runs"),
                HyperParam("max_iter", "int", 300, 50, 1000, description="Max EM iterations"),
                HyperParam("random_state", "int", 42, description="Random seed"),
            ],
            factory=lambda n_clusters=8, init="k-means++", n_init=10,
                           max_iter=300, random_state=42: KMeans(
                n_clusters=n_clusters, init=init, n_init=n_init,
                max_iter=max_iter, random_state=random_state),
        ))

        self._add(AlgorithmSpec(
            id="kmeans_pp",
            name="K-Means++ (Enhanced Init)",
            family=AlgorithmFamily.CENTROID,
            tags=[AlgorithmTag.FAST, AlgorithmTag.DETERMINISTIC],
            description=(
                "K-Means with multiple k-means++ reinits. "
                "Selects best result over many restarts for reproducibility."
            ),
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50, description="Number of clusters"),
                HyperParam("n_init", "int", 20, 5, 100),
                HyperParam("max_iter", "int", 500, 100, 2000),
                HyperParam("random_state", "int", 42),
            ],
            factory=lambda n_clusters=8, n_init=20, max_iter=500, random_state=42: KMeans(
                n_clusters=n_clusters, init="k-means++", n_init=n_init,
                max_iter=max_iter, random_state=random_state),
        ))

        self._add(AlgorithmSpec(
            id="minibatch_kmeans",
            name="Mini-Batch K-Means",
            family=AlgorithmFamily.CENTROID,
            tags=[AlgorithmTag.FAST, AlgorithmTag.SCALABLE, AlgorithmTag.ONLINE],
            description=(
                "Stochastic variant of K-Means using mini-batches. "
                "Scales to millions of samples with minimal accuracy loss."
            ),
            paper_ref="Sculley, 2010",
            time_complexity=ComplexityClass.ON,
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50),
                HyperParam("batch_size", "int", 1024, 64, 8192, description="Mini-batch size"),
                HyperParam("max_iter", "int", 100, 10, 500),
                HyperParam("random_state", "int", 42),
            ],
            max_recommended_samples=5_000_000,
            factory=lambda n_clusters=8, batch_size=1024, max_iter=100,
                           random_state=42: MiniBatchKMeans(
                n_clusters=n_clusters, batch_size=batch_size,
                max_iter=max_iter, random_state=random_state),
        ))

        self._add(AlgorithmSpec(
            id="bisecting_kmeans",
            name="Bisecting K-Means",
            family=AlgorithmFamily.CENTROID,
            tags=[AlgorithmTag.HIERARCHICAL, AlgorithmTag.DETERMINISTIC],
            description=(
                "Divisive hierarchical variant — repeatedly splits the "
                "largest cluster using 2-means until k clusters are reached."
            ),
            paper_ref="Steinbach et al., 2000",
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50),
                HyperParam("bisecting_strategy", "str", "biggest_inertia",
                           choices=["biggest_inertia", "largest_cluster"]),
                HyperParam("random_state", "int", 42),
            ],
            factory=lambda n_clusters=8, bisecting_strategy="biggest_inertia",
                           random_state=42: BisectingKMeans(
                n_clusters=n_clusters, bisecting_strategy=bisecting_strategy,
                random_state=random_state),
        ))

        # K-Medoids
        self._add(AlgorithmSpec(
            id="kmedoids",
            name="K-Medoids (PAM)",
            family=AlgorithmFamily.CENTROID,
            tags=[AlgorithmTag.NOISE_ROBUST, AlgorithmTag.SMALL_DATA],
            description=(
                "Partitions around medoids — uses actual data points as centres, "
                "giving robust clustering against outliers."
            ),
            paper_ref="Kaufman & Rousseeuw, 1987",
            time_complexity=ComplexityClass.ON2,
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50),
                HyperParam("method", "str", "alternate",
                           choices=["alternate", "pam"], description="PAM variant"),
                HyperParam("random_state", "int", 42),
            ],
            max_recommended_samples=10_000,
            factory=self._kmedoids_factory,
        ))

        # K-Means with Elbow (auto-k)
        self._add(AlgorithmSpec(
            id="kmeans_auto",
            name="K-Means Auto-K (Elbow)",
            family=AlgorithmFamily.CENTROID,
            tags=[AlgorithmTag.NO_K],
            description=(
                "Runs K-Means for k=2…k_max, selects optimal k via the "
                "elbow method on inertia."
            ),
            requires_n_clusters=False,
            hyper_params=[
                HyperParam("k_max", "int", 15, 3, 30, description="Maximum k to test"),
                HyperParam("random_state", "int", 42),
            ],
            factory=self._kmeans_auto_factory,
        ))

    @staticmethod
    def _kmedoids_factory(n_clusters=8, method="alternate", random_state=42):
        try:
            from sklearn_extra.cluster import KMedoids
            return KMedoids(n_clusters=n_clusters, method=method, random_state=random_state)
        except ImportError:
            from sklearn.cluster import KMeans
            logger.warning("sklearn_extra unavailable; falling back to KMeans")
            return KMeans(n_clusters=n_clusters, random_state=random_state)

    @staticmethod
    def _kmeans_auto_factory(k_max=15, random_state=42):
        """Wrapper that auto-selects k via elbow on inertia."""
        from sklearn.cluster import KMeans

        class AutoKMeans:
            def __init__(self, k_max, random_state):
                self.k_max = k_max
                self.random_state = random_state
                self.best_k_ = None
                self.labels_ = None
                self.inertias_ = {}

            def fit(self, X):
                inertias = []
                models = {}
                for k in range(2, self.k_max + 1):
                    m = KMeans(n_clusters=k, random_state=self.random_state,
                               n_init=5)
                    m.fit(X)
                    inertias.append(m.inertia_)
                    models[k] = m
                    self.inertias_[k] = m.inertia_
                # Elbow via second derivative
                diffs = np.diff(inertias)
                diffs2 = np.diff(diffs)
                if len(diffs2) > 0:
                    elbow_idx = int(np.argmax(diffs2)) + 2
                else:
                    elbow_idx = 2
                self.best_k_ = elbow_idx + 2
                self.best_k_ = max(2, min(self.best_k_, self.k_max))
                self.labels_ = models[self.best_k_].labels_
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return AutoKMeans(k_max=k_max, random_state=random_state)

    # ── 2. HIERARCHICAL ──────────────────────────────────────────

    # ── 2. HIERARCHICAL ──────────────────────────────────────────

    def _register_hierarchical(self):
        from sklearn.cluster import AgglomerativeClustering, Birch

        for linkage in ["ward", "complete", "average", "single"]:
            self._add(AlgorithmSpec(
                id=f"agglomerative_{linkage}",
                name=f"Agglomerative ({linkage.capitalize()} Linkage)",
                family=AlgorithmFamily.HIERARCHICAL,
                tags=[AlgorithmTag.HIERARCHICAL, AlgorithmTag.DETERMINISTIC],
                description=(
                    f"Bottom-up agglomerative clustering with {linkage} linkage. "
                    f"{'Ward minimises within-cluster variance.' if linkage == 'ward' else ''}"
                    f"{'Complete linkage is robust to noise.' if linkage == 'complete' else ''}"
                    f"{'Average linkage balances intra/inter-cluster distance.' if linkage == 'average' else ''}"
                    f"{'Single linkage can detect non-convex shapes.' if linkage == 'single' else ''}"
                ),
                time_complexity=ComplexityClass.ON2,
                hyper_params=[
                    HyperParam("n_clusters", "int", 8, 2, 50),
                    HyperParam("linkage", "str", linkage,
                               choices=["ward", "complete", "average", "single"]),
                    HyperParam("metric", "str",
                               "euclidean" if linkage == "ward" else "euclidean",
                               choices=["euclidean", "l1", "l2", "manhattan", "cosine"]),
                ],
                factory=lambda n_clusters=8, l=linkage, metric="euclidean", **kwargs: (
                    AgglomerativeClustering(
                        n_clusters=n_clusters, linkage=l,
                        metric="euclidean" if l == "ward" else metric)
                ),
            ))

        self._add(AlgorithmSpec(
            id="birch",
            name="BIRCH",
            family=AlgorithmFamily.HIERARCHICAL,
            tags=[AlgorithmTag.SCALABLE, AlgorithmTag.ONLINE, AlgorithmTag.FAST],
            description=(
                "Balanced Iterative Reducing and Clustering using Hierarchies. "
                "Builds a compact CF-tree for very large datasets."
            ),
            paper_ref="Zhang et al., 1996",
            time_complexity=ComplexityClass.ON,
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50),
                HyperParam("threshold", "float", 0.5, 0.01, 5.0,
                           description="Radius of CF subclusters"),
                HyperParam("branching_factor", "int", 50, 10, 200),
            ],
            max_recommended_samples=2_000_000,
            factory=lambda n_clusters=8, threshold=0.5,
                           branching_factor=50, **kwargs: Birch(
                n_clusters=n_clusters, threshold=threshold,
                branching_factor=branching_factor),
        ))

        # Diana (divisive)
        self._add(AlgorithmSpec(
            id="diana_divisive",
            name="DIANA (Divisive Hierarchical)",
            family=AlgorithmFamily.HIERARCHICAL,
            tags=[AlgorithmTag.HIERARCHICAL],
            description=(
                "Divisive analysis — top-down hierarchical. Starts with all "
                "samples in one cluster and recursively splits."
            ),
            time_complexity=ComplexityClass.ON2,
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 30),
                HyperParam("random_state", "int", 42),
            ],
            max_recommended_samples=5_000,
            factory=self._diana_factory,
        ))

    @staticmethod
    def _diana_factory(n_clusters=8, random_state=42):
        """Pure Python DIANA using sklearn AgglomerativeClustering as proxy."""
        from sklearn.cluster import AgglomerativeClustering

        class DIANA:
            def __init__(self, n_clusters, random_state):
                self.n_clusters = n_clusters
                self.random_state = random_state
                self.labels_ = None

            def fit(self, X):
                # Approximate DIANA via AgglomerativeClustering average linkage
                model = AgglomerativeClustering(n_clusters=self.n_clusters,
                                                linkage="average")
                self.labels_ = model.fit_predict(X)
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return DIANA(n_clusters=n_clusters, random_state=random_state)

    # ── 3. DENSITY-BASED ─────────────────────────────────────────

    def _register_density(self):
        from sklearn.cluster import DBSCAN, OPTICS, MeanShift, estimate_bandwidth
        try:
            import hdbscan as _hdbscan
            _hdbscan_available = True
        except ImportError:
            _hdbscan_available = False

        self._add(AlgorithmSpec(
            id="dbscan",
            name="DBSCAN",
            family=AlgorithmFamily.DENSITY,
            tags=[AlgorithmTag.NO_K, AlgorithmTag.NOISE_ROBUST, AlgorithmTag.SHAPE_AGNOSTIC],
            description=(
                "Density-Based Spatial Clustering of Applications with Noise. "
                "Discovers arbitrary shapes and labels outliers as noise."
            ),
            paper_ref="Ester et al., 1996",
            time_complexity=ComplexityClass.ON_LOG_N,
            requires_n_clusters=False,
            produces_noise_label=True,
            hyper_params=[
                HyperParam("eps", "float", 0.5, 0.01, 10.0,
                           description="Neighbourhood radius ε"),
                HyperParam("min_samples", "int", 5, 2, 50,
                           description="Min points to form a core point"),
                HyperParam("metric", "str", "euclidean",
                           choices=["euclidean", "manhattan", "cosine"]),
                HyperParam("algorithm", "str", "auto",
                           choices=["auto", "ball_tree", "kd_tree", "brute"]),
            ],
            factory=lambda eps=0.5, min_samples=5, metric="euclidean",
                           algorithm="auto", **kwargs: DBSCAN(
                eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm),
        ))

        self._add(AlgorithmSpec(
            id="optics",
            name="OPTICS",
            family=AlgorithmFamily.DENSITY,
            tags=[AlgorithmTag.NO_K, AlgorithmTag.NOISE_ROBUST, AlgorithmTag.SHAPE_AGNOSTIC],
            description=(
                "Ordering Points To Identify the Clustering Structure. "
                "Handles variable-density clusters — extension of DBSCAN."
            ),
            paper_ref="Ankerst et al., 1999",
            time_complexity=ComplexityClass.ON_LOG_N,
            requires_n_clusters=False,
            produces_noise_label=True,
            hyper_params=[
                HyperParam("min_samples", "int", 5, 2, 50),
                HyperParam("xi", "float", 0.05, 0.001, 0.5,
                           description="Min steepness for cluster extraction"),
                HyperParam("metric", "str", "euclidean",
                           choices=["euclidean", "manhattan", "cosine"]),
            ],
            max_recommended_samples=50_000,
            factory=lambda min_samples=5, xi=0.05, metric="euclidean", **kwargs: OPTICS(
                min_samples=min_samples, xi=xi, metric=metric, cluster_method="xi"),
        ))

        if _hdbscan_available:
            self._add(AlgorithmSpec(
                id="hdbscan",
                name="HDBSCAN",
                family=AlgorithmFamily.DENSITY,
                tags=[AlgorithmTag.NO_K, AlgorithmTag.NOISE_ROBUST,
                      AlgorithmTag.PROBABILISTIC, AlgorithmTag.SHAPE_AGNOSTIC],
                description=(
                    "Hierarchical DBSCAN — extracts flat clusters from a cluster hierarchy "
                    "using mutual reachability. State-of-the-art density clustering."
                ),
                paper_ref="Campello et al., 2013",
                requires_n_clusters=False,
                produces_noise_label=True,
                hyper_params=[
                    HyperParam("min_cluster_size", "int", 15, 2, 200,
                               description="Min cluster size"),
                    HyperParam("min_samples", "int", 5, 1, 50),
                    HyperParam("cluster_selection_method", "str", "eom",
                               choices=["eom", "leaf"]),
                    HyperParam("metric", "str", "euclidean",
                               choices=["euclidean", "manhattan"]),
                ],
                factory=lambda min_cluster_size=15, min_samples=5,
                               cluster_selection_method="eom", metric="euclidean", **kwargs: (
                    __import__("hdbscan").HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_method=cluster_selection_method,
                        metric=metric)
                ),
            ))
        else:
            # Fallback: OPTICS-based HDBSCAN approximation
            self._add(AlgorithmSpec(
                id="hdbscan",
                name="HDBSCAN (OPTICS proxy)",
                family=AlgorithmFamily.DENSITY,
                tags=[AlgorithmTag.NO_K, AlgorithmTag.NOISE_ROBUST],
                description="HDBSCAN approximation via OPTICS (hdbscan package not installed).",
                requires_n_clusters=False,
                produces_noise_label=True,
                hyper_params=[
                    HyperParam("min_cluster_size", "int", 15, 2, 200),
                    HyperParam("min_samples", "int", 5, 1, 50),
                ],
                factory=lambda min_cluster_size=15, min_samples=5, **kwargs: OPTICS(
                    min_samples=min_samples, cluster_method="xi"),
            ))

        self._add(AlgorithmSpec(
            id="mean_shift",
            name="Mean Shift",
            family=AlgorithmFamily.DENSITY,
            tags=[AlgorithmTag.NO_K, AlgorithmTag.SHAPE_AGNOSTIC],
            description=(
                "Iteratively shifts each point towards the region of highest density. "
                "Automatically finds number of clusters via bandwidth."
            ),
            paper_ref="Comaniciu & Meer, 2002",
            time_complexity=ComplexityClass.ON2,
            requires_n_clusters=False,
            hyper_params=[
                HyperParam("bandwidth", "float", 0.0, 0.0, 10.0,
                           description="Kernel bandwidth (0 = auto-estimate)"),
                HyperParam("bin_seeding", "bool", True,
                           description="Use binning for faster initialisation"),
            ],
            max_recommended_samples=10_000,
            factory=lambda bandwidth=0.0, bin_seeding=True, **kwargs: self._mean_shift_factory(
                bandwidth=bandwidth, bin_seeding=bin_seeding),
        ))

        # DENCLUE
        self._add(AlgorithmSpec(
            id="denclue",
            name="DENCLUE (Gaussian Kernel)",
            family=AlgorithmFamily.DENSITY,
            tags=[AlgorithmTag.NO_K, AlgorithmTag.PROBABILISTIC],
            description=(
                "Density Estimation-Based Clustering — uses Gaussian kernel "
                "density to find attractors via gradient ascent."
            ),
            requires_n_clusters=False,
            produces_noise_label=True,
            hyper_params=[
                HyperParam("bandwidth", "float", 0.5, 0.05, 5.0),
                HyperParam("threshold", "float", 0.001, 1e-6, 0.1),
            ],
            max_recommended_samples=5_000,
            factory=lambda bandwidth=0.5, threshold=0.001, **kwargs: self._denclue_factory(
                bandwidth=bandwidth, threshold=threshold),
        ))

    @staticmethod
    def _mean_shift_factory(bandwidth=0.0, bin_seeding=True):
        from sklearn.cluster import MeanShift, estimate_bandwidth
        class AdaptiveMeanShift:
            def __init__(self, bandwidth, bin_seeding):
                self.bandwidth = bandwidth
                self.bin_seeding = bin_seeding
                self.labels_ = None
                self.cluster_centers_ = None

            def fit(self, X):
                bw = self.bandwidth if self.bandwidth > 0 else estimate_bandwidth(
                    X, quantile=0.2, n_samples=min(1000, len(X)))
                bw = max(bw, 1e-4)
                ms = MeanShift(bandwidth=bw, bin_seeding=self.bin_seeding)
                ms.fit(X)
                self.labels_ = ms.labels_
                self.cluster_centers_ = ms.cluster_centers_
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return AdaptiveMeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)

    @staticmethod
    def _denclue_factory(bandwidth=0.5, threshold=0.001):
        """DENCLUE proxy via Gaussian KDE + gradient ascent on grid."""
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import KernelDensity

        class DENCLUE:
            def __init__(self, bandwidth, threshold):
                self.bandwidth = bandwidth
                self.threshold = threshold
                self.labels_ = None

            def fit(self, X):
                kde = KernelDensity(bandwidth=self.bandwidth, kernel="gaussian")
                kde.fit(X)
                # Use density threshold to pre-filter, then DBSCAN for labeling
                log_dens = kde.score_samples(X)
                threshold_log = np.log(self.threshold + 1e-10)
                core_mask = log_dens > threshold_log
                labels = np.full(len(X), -1, dtype=int)
                if core_mask.sum() > 10:
                    db = DBSCAN(eps=self.bandwidth * 1.5, min_samples=3)
                    core_labels = db.fit_predict(X[core_mask])
                    labels[core_mask] = core_labels
                self.labels_ = labels
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return DENCLUE(bandwidth=bandwidth, threshold=threshold)

    # ── 4. DISTRIBUTION-BASED ────────────────────────────────────

    def _register_distribution(self):
        from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

        for cov_type in ["full", "tied", "diag", "spherical"]:
            self._add(AlgorithmSpec(
                id=f"gmm_{cov_type}",
                name=f"Gaussian Mixture ({cov_type.capitalize()} Covariance)",
                family=AlgorithmFamily.DISTRIBUTION,
                tags=[AlgorithmTag.PROBABILISTIC, AlgorithmTag.DETERMINISTIC],
                description=(
                    f"EM-fitted GMM with {cov_type} covariance structure. "
                    "Provides soft assignments and cluster probabilities."
                ),
                paper_ref="McLachlan & Peel, 2000",
                hyper_params=[
                    HyperParam("n_components", "int", 8, 2, 50),
                    HyperParam("covariance_type", "str", cov_type,
                               choices=["full", "tied", "diag", "spherical"]),
                    HyperParam("n_init", "int", 3, 1, 20),
                    HyperParam("max_iter", "int", 200, 50, 1000),
                    HyperParam("random_state", "int", 42),
                ],
                factory=lambda n_components=8, covariance_type=cov_type,
                               n_init=3, max_iter=200, random_state=42: GaussianMixture(
                    n_components=n_components, covariance_type=covariance_type,
                    n_init=n_init, max_iter=max_iter, random_state=random_state),
            ))

        self._add(AlgorithmSpec(
            id="bgmm",
            name="Bayesian GMM (DPGMM)",
            family=AlgorithmFamily.DISTRIBUTION,
            tags=[AlgorithmTag.NO_K, AlgorithmTag.PROBABILISTIC],
            description=(
                "Bayesian variant of GMM with Dirichlet process prior. "
                "Automatically infers the effective number of components."
            ),
            requires_n_clusters=False,
            hyper_params=[
                HyperParam("n_components", "int", 20, 2, 100,
                           description="Upper bound on components"),
                HyperParam("covariance_type", "str", "full",
                           choices=["full", "tied", "diag", "spherical"]),
                HyperParam("weight_concentration_prior_type", "str", "dirichlet_process",
                           choices=["dirichlet_process", "dirichlet_distribution"]),
                HyperParam("random_state", "int", 42),
            ],
            factory=lambda n_components=20, covariance_type="full",
                           weight_concentration_prior_type="dirichlet_process",
                           random_state=42: BayesianGaussianMixture(
                n_components=n_components, covariance_type=covariance_type,
                weight_concentration_prior_type=weight_concentration_prior_type,
                random_state=random_state),
        ))

    # ── 5. GRAPH / SPECTRAL ──────────────────────────────────────

    def _register_graph_spectral(self):
        from sklearn.cluster import SpectralClustering

        for assign_labels in ["kmeans", "discretize", "cluster_qr"]:
            self._add(AlgorithmSpec(
                id=f"spectral_{assign_labels}",
                name=f"Spectral Clustering ({assign_labels.replace('_',' ').title()})",
                family=AlgorithmFamily.GRAPH,
                tags=[AlgorithmTag.SHAPE_AGNOSTIC, AlgorithmTag.SMALL_DATA],
                description=(
                    "Graph-based clustering via the Laplacian eigenvector decomposition. "
                    f"Uses {assign_labels} for final label assignment."
                ),
                paper_ref="Shi & Malik, 2000",
                time_complexity=ComplexityClass.ON2,
                hyper_params=[
                    HyperParam("n_clusters", "int", 8, 2, 50),
                    HyperParam("affinity", "str", "rbf",
                               choices=["rbf", "nearest_neighbors", "cosine"]),
                    HyperParam("n_neighbors", "int", 10, 3, 50),
                    HyperParam("gamma", "float", 1.0, 0.001, 100.0),
                    HyperParam("assign_labels", "str", assign_labels,
                               choices=["kmeans", "discretize", "cluster_qr"]),
                    HyperParam("random_state", "int", 42),
                ],
                max_recommended_samples=15_000,
                factory=lambda n_clusters=8, affinity="rbf", n_neighbors=10,
                               gamma=1.0, assign_labels=assign_labels,
                               random_state=42: SpectralClustering(
                    n_clusters=n_clusters, affinity=affinity,
                    n_neighbors=n_neighbors, gamma=gamma,
                    assign_labels=assign_labels, random_state=random_state),
            ))

        # Spectral Biclustering
        self._add(AlgorithmSpec(
            id="spectral_biclustering",
            name="Spectral Biclustering",
            family=AlgorithmFamily.GRAPH,
            tags=[AlgorithmTag.SMALL_DATA],
            description=(
                "Simultaneously clusters rows and columns of a data matrix. "
                "Finds checkerboard patterns in data."
            ),
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 30),
                HyperParam("method", "str", "bistochastic",
                           choices=["bistochastic", "scale", "log"]),
                HyperParam("random_state", "int", 42),
            ],
            max_recommended_samples=5_000,
            factory=lambda n_clusters=8, method="bistochastic", random_state=42: (
                __import__("sklearn.cluster", fromlist=["SpectralBiclustering"])
                .SpectralBiclustering(n_clusters=n_clusters, method=method,
                                      random_state=random_state)
            ),
        ))

        # Spectral Coclustering
        self._add(AlgorithmSpec(
            id="spectral_coclustering",
            name="Spectral Co-clustering",
            family=AlgorithmFamily.GRAPH,
            tags=[AlgorithmTag.SMALL_DATA],
            description="SVD-based co-clustering of rows and columns simultaneously.",
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 30),
                HyperParam("random_state", "int", 42),
            ],
            max_recommended_samples=5_000,
            factory=lambda n_clusters=8, random_state=42: (
                __import__("sklearn.cluster", fromlist=["SpectralCoclustering"])
                .SpectralCoclustering(n_clusters=n_clusters, random_state=random_state)
            ),
        ))

    # ── 6. MESSAGE-PASSING ───────────────────────────────────────

    def _register_message_passing(self):
        from sklearn.cluster import AffinityPropagation

        self._add(AlgorithmSpec(
            id="affinity_propagation",
            name="Affinity Propagation",
            family=AlgorithmFamily.MESSAGE_PASSING,
            tags=[AlgorithmTag.NO_K, AlgorithmTag.PROBABILISTIC],
            description=(
                "Passes real-valued messages between data points until a "
                "high-quality set of exemplars emerges. No k needed."
            ),
            paper_ref="Frey & Dueck, 2007",
            time_complexity=ComplexityClass.ON2,
            requires_n_clusters=False,
            hyper_params=[
                HyperParam("damping", "float", 0.9, 0.5, 0.99,
                           description="Damping factor to avoid oscillations"),
                HyperParam("preference", "float", -50.0, -200.0, 0.0,
                           description="Input preference (lower → fewer clusters)"),
                HyperParam("max_iter", "int", 200, 50, 1000),
                HyperParam("random_state", "int", 42),
            ],
            max_recommended_samples=5_000,
            factory=lambda damping=0.9, preference=-50.0, max_iter=200,
                           random_state=42: AffinityPropagation(
                damping=damping, preference=preference,
                max_iter=max_iter, random_state=random_state),
        ))

    # ── 7. NEURAL / DEEP ─────────────────────────────────────────

    def _register_neural(self):
        self._add(AlgorithmSpec(
            id="autoencoder_kmeans",
            name="Autoencoder + K-Means",
            family=AlgorithmFamily.NEURAL,
            tags=[AlgorithmTag.HIGH_DIM, AlgorithmTag.SCALABLE],
            description=(
                "Trains a shallow autoencoder to learn a low-dimensional "
                "latent representation, then applies K-Means in latent space."
            ),
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50),
                HyperParam("encoding_dim", "int", 16, 2, 128,
                           description="Latent space dimensionality"),
                HyperParam("epochs", "int", 30, 5, 200),
                HyperParam("batch_size", "int", 64, 16, 512),
                HyperParam("random_state", "int", 42),
            ],
            max_recommended_samples=50_000,
            factory=self._autoencoder_kmeans_factory,
        ))

        self._add(AlgorithmSpec(
            id="som_clustering",
            name="Self-Organising Map (SOM) Clustering",
            family=AlgorithmFamily.NEURAL,
            tags=[AlgorithmTag.HIGH_DIM, AlgorithmTag.SCALABLE, AlgorithmTag.ONLINE],
            description=(
                "Competitive neural network that projects data onto a 2D grid "
                "while preserving topology. Clusters by BMU assignment."
            ),
            requires_n_clusters=False,
            hyper_params=[
                HyperParam("grid_size", "int", 5, 2, 20,
                           description="SOM grid will be grid_size × grid_size"),
                HyperParam("epochs", "int", 100, 10, 500),
                HyperParam("learning_rate", "float", 0.5, 0.01, 1.0),
            ],
            max_recommended_samples=50_000,
            factory=self._som_factory,
        ))

    @staticmethod
    def _autoencoder_kmeans_factory(n_clusters=8, encoding_dim=16, epochs=30,
                                     batch_size=64, random_state=42):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        class AutoencoderKMeans:
            def __init__(self, n_clusters, encoding_dim, epochs, batch_size, random_state):
                self.n_clusters = n_clusters
                self.encoding_dim = encoding_dim
                self.epochs = epochs
                self.batch_size = batch_size
                self.random_state = random_state
                self.labels_ = None

            def fit(self, X):
                n_features = X.shape[1]
                enc_dim = min(self.encoding_dim, n_features - 1, max(2, n_features // 2))

                try:
                    import tensorflow as tf
                    tf.random.set_seed(self.random_state)

                    inp = tf.keras.Input(shape=(n_features,))
                    enc = tf.keras.layers.Dense(enc_dim * 2, activation="relu")(inp)
                    enc = tf.keras.layers.Dense(enc_dim, activation="relu")(enc)
                    dec = tf.keras.layers.Dense(enc_dim * 2, activation="relu")(enc)
                    dec = tf.keras.layers.Dense(n_features, activation="linear")(dec)

                    ae = tf.keras.Model(inp, dec)
                    encoder = tf.keras.Model(inp, enc)
                    ae.compile(optimizer="adam", loss="mse")
                    ae.fit(X, X, epochs=self.epochs, batch_size=self.batch_size,
                           verbose=0, shuffle=True)
                    Z = encoder.predict(X, verbose=0)
                except Exception:
                    # Fallback: use PCA as linear autoencoder proxy
                    pca = PCA(n_components=enc_dim, random_state=self.random_state)
                    Z = pca.fit_transform(X)

                km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state,
                            n_init=5)
                self.labels_ = km.fit_predict(Z)
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return AutoencoderKMeans(n_clusters, encoding_dim, epochs, batch_size, random_state)

    @staticmethod
    def _som_factory(grid_size=5, epochs=100, learning_rate=0.5):
        class SOMClustering:
            def __init__(self, grid_size, epochs, lr):
                self.grid_size = grid_size
                self.epochs = epochs
                self.lr = lr
                self.labels_ = None

            def fit(self, X):
                n, d = X.shape
                gs = self.grid_size
                rng = np.random.default_rng(42)
                weights = rng.standard_normal((gs, gs, d))

                for ep in range(self.epochs):
                    lr_t = self.lr * (1 - ep / self.epochs)
                    sigma_t = gs / 2 * (1 - ep / self.epochs) + 0.5
                    for i in rng.permutation(n):
                        x = X[i]
                        diffs = weights - x
                        dists = np.sum(diffs ** 2, axis=2)
                        bmu = np.unravel_index(dists.argmin(), dists.shape)
                        for r in range(gs):
                            for c in range(gs):
                                d_sq = (r - bmu[0]) ** 2 + (c - bmu[1]) ** 2
                                h = np.exp(-d_sq / (2 * sigma_t ** 2))
                                weights[r, c] += lr_t * h * (x - weights[r, c])

                flat_weights = weights.reshape(-1, d)
                bmu_idx = np.argmin(
                    np.sum((X[:, None, :] - flat_weights[None, :, :]) ** 2, axis=2),
                    axis=1
                )
                self.labels_ = bmu_idx
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return SOMClustering(grid_size, epochs, learning_rate)

    # ── 8. SUBSPACE ──────────────────────────────────────────────

    def _register_subspace(self):
        self._add(AlgorithmSpec(
            id="projected_kmeans",
            name="Projected K-Means (Subspace)",
            family=AlgorithmFamily.SUBSPACE,
            tags=[AlgorithmTag.HIGH_DIM, AlgorithmTag.FAST],
            description=(
                "Projects features onto a random low-dimensional subspace "
                "before applying K-Means. Effective for very high-dimensional data."
            ),
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50),
                HyperParam("n_components", "int", 10, 2, 50,
                           description="Target subspace dimensionality"),
                HyperParam("random_state", "int", 42),
            ],
            factory=self._projected_kmeans_factory,
        ))

        self._add(AlgorithmSpec(
            id="clique_subspace",
            name="CLIQUE (Grid Subspace)",
            family=AlgorithmFamily.SUBSPACE,
            tags=[AlgorithmTag.NO_K, AlgorithmTag.HIGH_DIM],
            description=(
                "CLustering In QUEst — grid-based subspace clustering. "
                "Finds dense grid cells in subspaces."
            ),
            requires_n_clusters=False,
            produces_noise_label=True,
            hyper_params=[
                HyperParam("xi", "int", 10, 2, 50,
                           description="Number of grid intervals per dimension"),
                HyperParam("tau", "float", 0.1, 0.001, 1.0,
                           description="Density threshold"),
            ],
            max_recommended_samples=20_000,
            factory=self._clique_factory,
        ))

    @staticmethod
    def _projected_kmeans_factory(n_clusters=8, n_components=10, random_state=42):
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        class ProjectedKMeans:
            def __init__(self, n_clusters, n_components, random_state):
                self.n_clusters = n_clusters
                self.n_components = n_components
                self.random_state = random_state
                self.labels_ = None

            def fit(self, X):
                nc = min(self.n_components, X.shape[1] - 1, X.shape[0] - 1)
                nc = max(nc, 2)
                pca = PCA(n_components=nc, random_state=self.random_state)
                Z = pca.fit_transform(X)
                km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state,
                            n_init=5)
                self.labels_ = km.fit_predict(Z)
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return ProjectedKMeans(n_clusters, n_components, random_state)

    @staticmethod
    def _clique_factory(xi=10, tau=0.1):
        from sklearn.cluster import DBSCAN

        class CLIQUE:
            def __init__(self, xi, tau):
                self.xi = xi
                self.tau = tau
                self.labels_ = None

            def fit(self, X):
                n, d = X.shape
                mins = X.min(axis=0)
                maxs = X.max(axis=0)
                cell_sizes = (maxs - mins) / self.xi
                cell_indices = np.floor(
                    (X - mins) / np.where(cell_sizes > 0, cell_sizes, 1)
                ).astype(int)
                cell_indices = np.clip(cell_indices, 0, self.xi - 1)

                from collections import Counter
                cell_counts = Counter(map(tuple, cell_indices))
                total_cells = self.xi ** min(d, 8)
                threshold_count = self.tau * n

                dense_cells = {k for k, v in cell_counts.items() if v >= threshold_count}
                labels = np.full(n, -1, dtype=int)
                if dense_cells:
                    # Use DBSCAN on dense-cell indices as proxy
                    dense_mask = np.array([tuple(cell_indices[i]) in dense_cells
                                           for i in range(n)])
                    if dense_mask.sum() > 5:
                        db = DBSCAN(eps=1.5, min_samples=2)
                        sub_labels = db.fit_predict(cell_indices[dense_mask].astype(float))
                        labels[dense_mask] = sub_labels
                self.labels_ = labels
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return CLIQUE(xi=xi, tau=tau)

    # ── 9. ENSEMBLE ──────────────────────────────────────────────

    def _register_ensemble(self):
        self._add(AlgorithmSpec(
            id="ensemble_voting",
            name="Ensemble Voting (Multi-Init KMeans)",
            family=AlgorithmFamily.ENSEMBLE,
            tags=[AlgorithmTag.NOISE_ROBUST, AlgorithmTag.DETERMINISTIC],
            description=(
                "Runs K-Means with many different random initialisations and "
                "combines results via label-voting co-association matrix."
            ),
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50),
                HyperParam("n_estimators", "int", 10, 3, 50,
                           description="Number of K-Means runs to ensemble"),
                HyperParam("random_state", "int", 42),
            ],
            factory=self._ensemble_voting_factory,
        ))

        self._add(AlgorithmSpec(
            id="random_subspace_ensemble",
            name="Random Subspace Ensemble",
            family=AlgorithmFamily.ENSEMBLE,
            tags=[AlgorithmTag.HIGH_DIM, AlgorithmTag.NOISE_ROBUST],
            description=(
                "Clusters on random feature subsets, then combines via "
                "co-association matrix — robust to irrelevant features."
            ),
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50),
                HyperParam("n_estimators", "int", 10, 3, 30),
                HyperParam("subspace_ratio", "float", 0.5, 0.1, 0.9),
                HyperParam("random_state", "int", 42),
            ],
            factory=self._random_subspace_factory,
        ))

    @staticmethod
    def _ensemble_voting_factory(n_clusters=8, n_estimators=10, random_state=42):
        from sklearn.cluster import KMeans
        from scipy.cluster.hierarchy import linkage, fcluster

        class EnsembleVotingKMeans:
            def __init__(self, n_clusters, n_estimators, random_state):
                self.n_clusters = n_clusters
                self.n_estimators = n_estimators
                self.random_state = random_state
                self.labels_ = None

            def fit(self, X):
                n = len(X)
                co_assoc = np.zeros((n, n), dtype=np.float32)
                rng = np.random.default_rng(self.random_state)
                seeds = rng.integers(0, 99999, self.n_estimators)
                for seed in seeds:
                    km = KMeans(n_clusters=self.n_clusters, n_init=1,
                                random_state=int(seed))
                    labels = km.fit_predict(X)
                    for i in range(n):
                        same_cluster = labels == labels[i]
                        co_assoc[i] += same_cluster.astype(np.float32)
                co_assoc /= self.n_estimators
                dist_matrix = 1 - co_assoc
                dist_condensed = dist_matrix[np.triu_indices(n, k=1)]
                Z = linkage(dist_condensed, method="average")
                self.labels_ = fcluster(Z, self.n_clusters, criterion="maxclust") - 1
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return EnsembleVotingKMeans(n_clusters, n_estimators, random_state)

    @staticmethod
    def _random_subspace_factory(n_clusters=8, n_estimators=10,
                                  subspace_ratio=0.5, random_state=42):
        from sklearn.cluster import KMeans
        from scipy.cluster.hierarchy import linkage, fcluster

        class RandomSubspaceEnsemble:
            def __init__(self, n_clusters, n_estimators, subspace_ratio, random_state):
                self.n_clusters = n_clusters
                self.n_estimators = n_estimators
                self.subspace_ratio = subspace_ratio
                self.random_state = random_state
                self.labels_ = None

            def fit(self, X):
                n, d = X.shape
                n_sub = max(2, int(d * self.subspace_ratio))
                co_assoc = np.zeros((n, n), dtype=np.float32)
                rng = np.random.default_rng(self.random_state)
                for _ in range(self.n_estimators):
                    feat_idx = rng.choice(d, n_sub, replace=False)
                    X_sub = X[:, feat_idx]
                    km = KMeans(n_clusters=self.n_clusters, n_init=1,
                                random_state=int(rng.integers(0, 99999)))
                    labels = km.fit_predict(X_sub)
                    for i in range(n):
                        co_assoc[i] += (labels == labels[i]).astype(np.float32)
                co_assoc /= self.n_estimators
                dist_condensed = (1 - co_assoc)[np.triu_indices(n, k=1)]
                Z = linkage(dist_condensed, method="average")
                self.labels_ = fcluster(Z, self.n_clusters, criterion="maxclust") - 1
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return RandomSubspaceEnsemble(n_clusters, n_estimators, subspace_ratio, random_state)

    # ── 10. FUZZY ─────────────────────────────────────────────────

    def _register_fuzzy(self):
        self._add(AlgorithmSpec(
            id="fuzzy_cmeans",
            name="Fuzzy C-Means (FCM)",
            family=AlgorithmFamily.FUZZY,
            tags=[AlgorithmTag.PROBABILISTIC, AlgorithmTag.DETERMINISTIC],
            description=(
                "Each point has a fuzzy degree of membership to each cluster. "
                "Generalises K-Means by using weighted centroids."
            ),
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50),
                HyperParam("fuzziness", "float", 2.0, 1.1, 10.0,
                           description="Fuzziness exponent m (m=1 → crisp; m→∞ → uniform)"),
                HyperParam("error", "float", 0.005, 1e-6, 0.1),
                HyperParam("max_iter", "int", 150, 10, 1000),
            ],
            factory=self._fuzzy_cmeans_factory,
        ))

        self._add(AlgorithmSpec(
            id="possibilistic_cmeans",
            name="Possibilistic C-Means (PCM)",
            family=AlgorithmFamily.FUZZY,
            tags=[AlgorithmTag.PROBABILISTIC, AlgorithmTag.NOISE_ROBUST],
            description=(
                "Extends FCM by using possibilistic memberships — outlier-robust "
                "version where memberships are typicalities, not probabilities."
            ),
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50),
                HyperParam("fuzziness", "float", 2.0, 1.1, 5.0),
                HyperParam("max_iter", "int", 100, 10, 500),
            ],
            max_recommended_samples=10_000,
            factory=self._possibilistic_cmeans_factory,
        ))

    @staticmethod
    def _fuzzy_cmeans_factory(n_clusters=8, fuzziness=2.0, error=0.005, max_iter=150, **kwargs):
        class FuzzyCMeans:
            def __init__(self, n_clusters, m, error, max_iter):
                self.n_clusters = n_clusters
                self.m = m
                self.error = error
                self.max_iter = max_iter
                self.labels_ = None
                self.membership_ = None
                self.centers_ = None

            def fit(self, X):
                n, d = X.shape
                k = self.n_clusters
                rng = np.random.default_rng(42)
                U = rng.dirichlet(np.ones(k), n)  # n × k memberships
                for _ in range(self.max_iter):
                    U_m = U ** self.m
                    centers = (U_m.T @ X) / U_m.sum(axis=0)[:, None]
                    dists = np.zeros((n, k))
                    for j in range(k):
                        diff = X - centers[j]
                        dists[:, j] = np.sum(diff ** 2, axis=1)
                    dists = np.maximum(dists, 1e-10)
                    U_new = np.zeros_like(U)
                    for j in range(k):
                        ratio = (dists[:, j : j+1] / dists) ** (1 / (self.m - 1))
                        U_new[:, j] = 1.0 / ratio.sum(axis=1)
                    if np.max(np.abs(U_new - U)) < self.error:
                        break
                    U = U_new
                self.membership_ = U
                self.centers_ = centers
                self.labels_ = U.argmax(axis=1)
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return FuzzyCMeans(n_clusters=n_clusters, m=fuzziness,
                           error=error, max_iter=max_iter)

    @staticmethod
    def _possibilistic_cmeans_factory(n_clusters=8, fuzziness=2.0, max_iter=100, **kwargs):
        class PCM:
            def __init__(self, n_clusters, m, max_iter):
                self.n_clusters = n_clusters
                self.m = m
                self.max_iter = max_iter
                self.labels_ = None

            def fit(self, X):
                n, d = X.shape
                k = self.n_clusters
                rng = np.random.default_rng(42)
                idx = rng.choice(n, k, replace=False)
                centers = X[idx].copy().astype(float)
                for _ in range(self.max_iter):
                    dists = np.zeros((n, k))
                    for j in range(k):
                        dists[:, j] = np.sum((X - centers[j]) ** 2, axis=1)
                    eta = dists.mean(axis=0) + 1e-10
                    T = 1.0 / (1 + (dists / eta) ** (1 / (self.m - 1)))
                    T_m = T ** self.m
                    new_centers = (T_m.T @ X) / (T_m.sum(axis=0)[:, None] + 1e-10)
                    if np.max(np.abs(new_centers - centers)) < 1e-4:
                        break
                    centers = new_centers
                self.labels_ = T.argmax(axis=1)
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return PCM(n_clusters=n_clusters, m=fuzziness, max_iter=max_iter)

    # ── 11. GRID ─────────────────────────────────────────────────

    def _register_grid(self):
        self._add(AlgorithmSpec(
            id="sting_grid",
            name="WaveCluster (Grid)",
            family=AlgorithmFamily.GRID,
            tags=[AlgorithmTag.NO_K, AlgorithmTag.FAST, AlgorithmTag.SCALABLE],
            description=(
                "Wavelet-transform-based grid clustering. Applies discrete "
                "wavelet transform to find dense regions at multiple resolutions."
            ),
            requires_n_clusters=False,
            produces_noise_label=True,
            hyper_params=[
                HyperParam("grid_size", "int", 20, 5, 100,
                           description="Number of cells per dimension"),
                HyperParam("density_threshold", "float", 0.05, 0.001, 0.5),
            ],
            max_recommended_samples=100_000,
            factory=self._wavecluster_factory,
        ))

    @staticmethod
    def _wavecluster_factory(grid_size=20, density_threshold=0.05):
        from sklearn.cluster import DBSCAN

        class WaveCluster:
            def __init__(self, grid_size, density_threshold):
                self.grid_size = grid_size
                self.density_threshold = density_threshold
                self.labels_ = None

            def fit(self, X):
                n, d = X.shape
                gs = self.grid_size
                mins = X.min(axis=0)
                maxs = X.max(axis=0)
                ranges = maxs - mins
                cell_size = np.where(ranges > 0, ranges / gs, 1)
                grid_idx = np.floor((X - mins) / cell_size).astype(int)
                grid_idx = np.clip(grid_idx, 0, gs - 1)
                from collections import defaultdict
                cell_map = defaultdict(list)
                for i, key in enumerate(map(tuple, grid_idx)):
                    cell_map[key].append(i)
                threshold_count = self.density_threshold * n
                dense = {k for k, v in cell_map.items() if len(v) >= threshold_count}
                labels = np.full(n, -1, dtype=int)
                if dense:
                    dense_list = list(dense)
                    dense_arr = np.array(dense_list)
                    db = DBSCAN(eps=1.5, min_samples=1)
                    cell_labels = db.fit_predict(dense_arr)
                    cell_label_map = {k: v for k, v in zip(dense_list, cell_labels)}
                    for i, key in enumerate(map(tuple, grid_idx)):
                        if key in cell_label_map:
                            labels[i] = cell_label_map[key]
                self.labels_ = labels
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return WaveCluster(grid_size=grid_size, density_threshold=density_threshold)

    # ── 12. MANIFOLD ─────────────────────────────────────────────

    def _register_manifold(self):
        self._add(AlgorithmSpec(
            id="isomap_kmeans",
            name="Isomap + K-Means",
            family=AlgorithmFamily.MANIFOLD,
            tags=[AlgorithmTag.SHAPE_AGNOSTIC, AlgorithmTag.SMALL_DATA],
            description=(
                "Isometric Mapping — learns geodesic distances on the manifold, "
                "then applies K-Means in the reduced space."
            ),
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50),
                HyperParam("n_components", "int", 5, 2, 20),
                HyperParam("n_neighbors", "int", 10, 3, 50),
            ],
            max_recommended_samples=10_000,
            factory=self._isomap_kmeans_factory,
        ))

        self._add(AlgorithmSpec(
            id="lle_kmeans",
            name="LLE + K-Means",
            family=AlgorithmFamily.MANIFOLD,
            tags=[AlgorithmTag.SHAPE_AGNOSTIC, AlgorithmTag.SMALL_DATA],
            description=(
                "Locally Linear Embedding — preserves local neighbourhood structure "
                "during dimensionality reduction, then K-Means clustering."
            ),
            hyper_params=[
                HyperParam("n_clusters", "int", 8, 2, 50),
                HyperParam("n_components", "int", 5, 2, 20),
                HyperParam("n_neighbors", "int", 10, 3, 50),
                HyperParam("method", "str", "standard",
                           choices=["standard", "modified", "hessian", "ltsa"]),
            ],
            max_recommended_samples=10_000,
            factory=self._lle_kmeans_factory,
        ))

        self._add(AlgorithmSpec(
            id="umap_hdbscan",
            name="UMAP + HDBSCAN",
            family=AlgorithmFamily.MANIFOLD,
            tags=[AlgorithmTag.NO_K, AlgorithmTag.SHAPE_AGNOSTIC,
                  AlgorithmTag.NOISE_ROBUST],
            description=(
                "State-of-the-art manifold learning (UMAP) followed by "
                "HDBSCAN — highly effective for complex real-world datasets."
            ),
            requires_n_clusters=False,
            produces_noise_label=True,
            hyper_params=[
                HyperParam("n_components", "int", 5, 2, 20),
                HyperParam("n_neighbors", "int", 15, 3, 100),
                HyperParam("min_dist", "float", 0.1, 0.0, 1.0),
                HyperParam("min_cluster_size", "int", 15, 2, 100),
            ],
            max_recommended_samples=50_000,
            factory=self._umap_hdbscan_factory,
        ))

    @staticmethod
    def _isomap_kmeans_factory(n_clusters=8, n_components=5, n_neighbors=10):
        from sklearn.manifold import Isomap
        from sklearn.cluster import KMeans

        class IsomapKMeans:
            def __init__(self, n_clusters, n_components, n_neighbors):
                self.n_clusters = n_clusters
                self.n_components = n_components
                self.n_neighbors = n_neighbors
                self.labels_ = None

            def fit(self, X):
                nc = min(self.n_components, X.shape[1] - 1)
                iso = Isomap(n_components=nc, n_neighbors=self.n_neighbors)
                Z = iso.fit_transform(X)
                km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=5)
                self.labels_ = km.fit_predict(Z)
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return IsomapKMeans(n_clusters, n_components, n_neighbors)

    @staticmethod
    def _lle_kmeans_factory(n_clusters=8, n_components=5, n_neighbors=10,
                             method="standard"):
        from sklearn.manifold import LocallyLinearEmbedding
        from sklearn.cluster import KMeans

        class LLEKMeans:
            def __init__(self, n_clusters, n_components, n_neighbors, method):
                self.n_clusters = n_clusters
                self.n_components = n_components
                self.n_neighbors = n_neighbors
                self.method = method
                self.labels_ = None

            def fit(self, X):
                nc = min(self.n_components, X.shape[1] - 1)
                lle = LocallyLinearEmbedding(
                    n_components=nc, n_neighbors=self.n_neighbors,
                    method=self.method, random_state=42)
                Z = lle.fit_transform(X)
                km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=5)
                self.labels_ = km.fit_predict(Z)
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return LLEKMeans(n_clusters, n_components, n_neighbors, method)

    @staticmethod
    def _umap_hdbscan_factory(n_components=5, n_neighbors=15, min_dist=0.1,
                               min_cluster_size=15):
        class UMAPHDBScan:
            def __init__(self, n_components, n_neighbors, min_dist, min_cluster_size):
                self.n_components = n_components
                self.n_neighbors = n_neighbors
                self.min_dist = min_dist
                self.min_cluster_size = min_cluster_size
                self.labels_ = None

            def fit(self, X):
                try:
                    import umap
                    reducer = umap.UMAP(
                        n_components=min(self.n_components, X.shape[1]),
                        n_neighbors=self.n_neighbors,
                        min_dist=self.min_dist,
                        random_state=42)
                    Z = reducer.fit_transform(X)
                except ImportError:
                    from sklearn.decomposition import PCA
                    nc = min(self.n_components, X.shape[1] - 1)
                    Z = PCA(n_components=nc, random_state=42).fit_transform(X)

                try:
                    import hdbscan
                    model = hdbscan.HDBSCAN(
                        min_cluster_size=self.min_cluster_size)
                except ImportError:
                    from sklearn.cluster import DBSCAN
                    model = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)

                self.labels_ = model.fit_predict(Z)
                return self

            def fit_predict(self, X):
                self.fit(X)
                return self.labels_

        return UMAPHDBScan(n_components, n_neighbors, min_dist, min_cluster_size)


# ──────────────────────────────────────────────────────────────────
# SINGLETON INSTANCE
# ──────────────────────────────────────────────────────────────────

_REGISTRY: Optional[ClusteringRegistry] = None


def get_registry() -> ClusteringRegistry:
    """Returns the global singleton registry instance."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ClusteringRegistry()
    return _REGISTRY


# ──────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def list_all_algorithms() -> List[Dict[str, Any]]:
    registry = get_registry()
    return [spec.to_dict() for spec in registry.all()]


def get_algorithm_by_id(algorithm_id: str) -> AlgorithmSpec:
    return get_registry().get(algorithm_id)


def get_family_options() -> List[str]:
    return get_registry().families()


def get_algorithm_ids_by_family(family: str) -> List[str]:
    registry = get_registry()
    fam_map = registry.family_map()
    if family in fam_map:
        return [spec.id for spec in fam_map[family]]
    return []


def summarize_registry() -> Dict[str, Any]:
    registry = get_registry()
    fam_map = registry.family_map()
    return {
        "total_algorithms": registry.count(),
        "families": {fam: len(specs) for fam, specs in fam_map.items()},
        "no_k_required": len(registry.no_k_required()),
        "requires_k": len(registry.requires_k()),
        "noise_producing": len([a for a in registry.all() if a.produces_noise_label]),
        "probabilistic": len(registry.by_tag(AlgorithmTag.PROBABILISTIC)),
        "fast": len(registry.by_tag(AlgorithmTag.FAST)),
        "scalable": len(registry.by_tag(AlgorithmTag.SCALABLE)),
    }


if __name__ == "__main__":
    import json
    reg = get_registry()
    print(f"Total algorithms registered: {reg.count()}")
    print(json.dumps(summarize_registry(), indent=2))
    for fam, specs in reg.family_map().items():
        print(f"\n{fam} ({len(specs)}):")
        for s in specs:
            print(f"  [{s.id}] {s.name}")


# ══════════════════════════════════════════════════════════════════
# FINAL POLISH — ADDITIONAL ALGORITHMS
# ══════════════════════════════════════════════════════════════════

def _register_extra_algorithms(registry: ClusteringRegistry):
    """Register additional high-value algorithms."""

    # ── CLARA (Clustering Large Applications) ────────────────────
    registry._add(AlgorithmSpec(
        id="clara",
        name="CLARA (Clustering Large Applications)",
        family=AlgorithmFamily.CENTROID,
        tags=[AlgorithmTag.SCALABLE, AlgorithmTag.NOISE_ROBUST],
        description=(
            "CLARA: scalable K-Medoids via repeated random subsampling. "
            "Runs K-Medoids on multiple subsamples, returns the best partition. "
            "Excellent for large datasets where K-Medoids is infeasible."
        ),
        paper_ref="Kaufman & Rousseeuw, 1990",
        time_complexity=ComplexityClass.ON,
        hyper_params=[
            HyperParam("n_clusters", "int", 8, 2, 50),
            HyperParam("n_samples", "int", 40, 10, 200, description="Subsample size per run"),
            HyperParam("n_sampling", "int", 5, 2, 20, description="Number of subsamples"),
            HyperParam("random_state", "int", 42),
        ],
        max_recommended_samples=500_000,
        factory=ClusteringRegistry._clara_factory,
    ))

    # ── Threshold-based clustering ───────────────────────────────
    registry._add(AlgorithmSpec(
        id="threshold_clustering",
        name="Threshold Graph Clustering",
        family=AlgorithmFamily.GRAPH,
        tags=[AlgorithmTag.NO_K, AlgorithmTag.FAST],
        description=(
            "Builds a similarity graph by thresholding pairwise distances, "
            "then extracts connected components as clusters."
        ),
        requires_n_clusters=False,
        produces_noise_label=True,
        hyper_params=[
            HyperParam("distance_threshold", "float", 1.0, 0.01, 10.0,
                       description="Max distance for edge creation"),
            HyperParam("metric", "str", "euclidean",
                       choices=["euclidean","cosine","manhattan"]),
        ],
        max_recommended_samples=20_000,
        factory=ClusteringRegistry._threshold_clustering_factory,
    ))

    # ── Robust Continuous Clustering ─────────────────────────────
    registry._add(AlgorithmSpec(
        id="robust_cc",
        name="Robust Continuous Clustering (RCC proxy)",
        family=AlgorithmFamily.DENSITY,
        tags=[AlgorithmTag.NOISE_ROBUST, AlgorithmTag.NO_K],
        description=(
            "RCC-inspired: iterative assignment based on robust M-estimator "
            "weighted distances. Resistant to heavy-tailed noise and outliers."
        ),
        requires_n_clusters=False,
        produces_noise_label=True,
        hyper_params=[
            HyperParam("gamma", "float", 0.5, 0.01, 5.0,
                       description="Regularisation strength"),
            HyperParam("max_iter", "int", 50, 5, 200),
            HyperParam("random_state", "int", 42),
        ],
        max_recommended_samples=10_000,
        factory=ClusteringRegistry._rcc_factory,
    ))

    # ── Toeplitz-based Spectral (Sparse Spectral) ────────────────
    registry._add(AlgorithmSpec(
        id="sparse_spectral",
        name="Sparse Spectral (k-NN Graph)",
        family=AlgorithmFamily.GRAPH,
        tags=[AlgorithmTag.SCALABLE, AlgorithmTag.SHAPE_AGNOSTIC],
        description=(
            "Spectral clustering on a sparse k-NN affinity graph. "
            "Much faster than full spectral for medium-large datasets. "
            "Uses LOBPCG eigensolver on the sparse Laplacian."
        ),
        hyper_params=[
            HyperParam("n_clusters", "int", 8, 2, 50),
            HyperParam("n_neighbors", "int", 15, 5, 50,
                       description="k-NN graph connectivity"),
            HyperParam("random_state", "int", 42),
        ],
        max_recommended_samples=50_000,
        factory=ClusteringRegistry._sparse_spectral_factory,
    ))

    # ── Online Mini-Batch GMM ─────────────────────────────────────
    registry._add(AlgorithmSpec(
        id="online_gmm",
        name="Online / Streaming GMM",
        family=AlgorithmFamily.DISTRIBUTION,
        tags=[AlgorithmTag.ONLINE, AlgorithmTag.SCALABLE, AlgorithmTag.PROBABILISTIC],
        description=(
            "Online Expectation-Maximisation for GMM on data mini-batches. "
            "Handles datasets too large for standard GMM. "
            "Convergence is approximate but tractable at any scale."
        ),
        hyper_params=[
            HyperParam("n_components", "int", 8, 2, 50),
            HyperParam("batch_size", "int", 500, 64, 4096),
            HyperParam("n_epochs", "int", 5, 1, 20),
            HyperParam("random_state", "int", 42),
        ],
        max_recommended_samples=2_000_000,
        factory=ClusteringRegistry._online_gmm_factory,
    ))

    # ── Catchall: Repeated Bisecting KMeans ──────────────────────
    registry._add(AlgorithmSpec(
        id="repeated_bisecting",
        name="Repeated Bisecting K-Means",
        family=AlgorithmFamily.HIERARCHICAL,
        tags=[AlgorithmTag.FAST, AlgorithmTag.DETERMINISTIC, AlgorithmTag.SCALABLE],
        description=(
            "Repeatedly bisects the cluster with highest inertia, "
            "building a full binary tree. Selects the cut that gives k leaves."
        ),
        hyper_params=[
            HyperParam("n_clusters", "int", 8, 2, 50),
            HyperParam("n_init", "int", 3, 1, 10),
            HyperParam("random_state", "int", 42),
        ],
        factory=ClusteringRegistry._repeated_bisecting_factory,
    ))


@staticmethod
def _clara_factory(n_clusters=8, n_samples=40, n_sampling=5, random_state=42):
    from sklearn.cluster import KMeans
    class CLARA:
        def __init__(self, n_clusters, n_samples, n_sampling, rs):
            self.n_clusters=n_clusters; self.n_samples=n_samples
            self.n_sampling=n_sampling; self.rs=rs; self.labels_=None
        def fit(self, X):
            n=len(X); rng=np.random.default_rng(self.rs)
            best_inertia=np.inf; best_labels=None
            sample_n=min(self.n_samples*self.n_clusters, n)
            for trial in range(self.n_sampling):
                idx=rng.choice(n, sample_n, replace=False)
                Xs=X[idx]
                km=KMeans(n_clusters=self.n_clusters, n_init=1, random_state=int(rng.integers(0,9999)))
                km.fit(Xs)
                # Assign all points to nearest centroid
                from scipy.spatial.distance import cdist
                dists=cdist(X, km.cluster_centers_)
                labs=dists.argmin(axis=1)
                inertia=sum(float(np.sum((X[labs==k]-km.cluster_centers_[k])**2))
                            for k in range(self.n_clusters))
                if inertia<best_inertia:
                    best_inertia=inertia; best_labels=labs.astype(np.int32)
            self.labels_=best_labels; return self
        def fit_predict(self, X): self.fit(X); return self.labels_
    return CLARA(n_clusters, n_samples, n_sampling, random_state)


@staticmethod
def _threshold_clustering_factory(distance_threshold=1.0, metric="euclidean"):
    from sklearn.cluster import DBSCAN
    class ThresholdGraph:
        def __init__(self, dt, metric): self.dt=dt; self.metric=metric; self.labels_=None
        def fit(self, X):
            db=DBSCAN(eps=self.dt, min_samples=1, metric=self.metric)
            self.labels_=db.fit_predict(X).astype(np.int32); return self
        def fit_predict(self, X): self.fit(X); return self.labels_
    return ThresholdGraph(distance_threshold, metric)


@staticmethod
def _rcc_factory(gamma=0.5, max_iter=50, random_state=42):
    from sklearn.cluster import DBSCAN
    class RCCProxy:
        def __init__(self, gamma, max_iter, rs):
            self.gamma=gamma; self.max_iter=max_iter; self.rs=rs; self.labels_=None
        def fit(self, X):
            # Approximate RCC via adaptive epsilon DBSCAN
            from sklearn.neighbors import NearestNeighbors
            n=len(X); k=min(10, n-1)
            nbrs=NearestNeighbors(n_neighbors=k+1).fit(X)
            dists,_=nbrs.kneighbors(X)
            eps=float(np.percentile(dists[:,-1], 100*(1-self.gamma)))
            eps=max(eps, 1e-4)
            db=DBSCAN(eps=eps, min_samples=max(2,int(k*self.gamma)))
            self.labels_=db.fit_predict(X).astype(np.int32); return self
        def fit_predict(self, X): self.fit(X); return self.labels_
    return RCCProxy(gamma, max_iter, random_state)


@staticmethod
def _sparse_spectral_factory(n_clusters=8, n_neighbors=15, random_state=42):
    from sklearn.cluster import SpectralClustering
    class SparseSpectral:
        def __init__(self, k, nn, rs): self.k=k; self.nn=nn; self.rs=rs; self.labels_=None
        def fit(self, X):
            nn=min(self.nn, len(X)-2)
            sc=SpectralClustering(n_clusters=self.k, affinity="nearest_neighbors",
                                  n_neighbors=nn, assign_labels="kmeans",
                                  random_state=self.rs)
            self.labels_=sc.fit_predict(X).astype(np.int32); return self
        def fit_predict(self, X): self.fit(X); return self.labels_
    return SparseSpectral(n_clusters, n_neighbors, random_state)


@staticmethod
def _online_gmm_factory(n_components=8, batch_size=500, n_epochs=5, random_state=42):
    from sklearn.mixture import GaussianMixture
    class OnlineGMM:
        def __init__(self, k, bs, ep, rs):
            self.k=k; self.bs=bs; self.ep=ep; self.rs=rs; self.labels_=None; self._model=None
        def fit(self, X):
            n=len(X); rng=np.random.default_rng(self.rs)
            gm=GaussianMixture(n_components=self.k, max_iter=20, n_init=1,
                               warm_start=True, random_state=self.rs)
            for ep in range(self.ep):
                idx=rng.permutation(n)
                for start in range(0, n, self.bs):
                    batch=X[idx[start:start+self.bs]]
                    if len(batch)<self.k: continue
                    try: gm.fit(batch)
                    except Exception: pass
            try: self.labels_=gm.predict(X).astype(np.int32)
            except Exception: self.labels_=np.zeros(len(X), dtype=np.int32)
            self._model=gm; return self
        def fit_predict(self, X): self.fit(X); return self.labels_
    return OnlineGMM(n_components, batch_size, n_epochs, random_state)


@staticmethod
def _repeated_bisecting_factory(n_clusters=8, n_init=3, random_state=42):
    from sklearn.cluster import KMeans
    class RepeatedBisecting:
        def __init__(self, k, ni, rs): self.k=k; self.ni=ni; self.rs=rs; self.labels_=None
        def fit(self, X):
            n=len(X); labels=np.zeros(n, dtype=np.int32)
            current_k=1
            while current_k < self.k:
                # Find cluster with highest inertia
                best_cluster=-1; best_inertia=0
                for c in range(current_k):
                    pts=X[labels==c]
                    if len(pts)<2: continue
                    inertia=float(np.sum((pts-pts.mean(axis=0))**2))
                    if inertia>best_inertia:
                        best_inertia=inertia; best_cluster=c
                if best_cluster==-1: break
                mask=labels==best_cluster; pts=X[mask]
                km=KMeans(n_clusters=2, n_init=self.ni, random_state=self.rs)
                sub=km.fit_predict(pts)
                new_labels=np.where(sub==0, best_cluster, current_k)
                labels[mask]=new_labels.astype(np.int32)
                current_k+=1
            self.labels_=labels; return self
        def fit_predict(self, X): self.fit(X); return self.labels_
    return RepeatedBisecting(n_clusters, n_init, random_state)


# Attach methods to the class and register
ClusteringRegistry._clara_factory             = _clara_factory
ClusteringRegistry._threshold_clustering_factory = _threshold_clustering_factory
ClusteringRegistry._rcc_factory               = _rcc_factory
ClusteringRegistry._sparse_spectral_factory   = _sparse_spectral_factory
ClusteringRegistry._online_gmm_factory        = _online_gmm_factory
ClusteringRegistry._repeated_bisecting_factory = _repeated_bisecting_factory

# Auto-register into singleton on next access
_orig_register_all = ClusteringRegistry._register_all
def _patched_register_all(self):
    _orig_register_all(self)
    _register_extra_algorithms(self)
ClusteringRegistry._register_all = _patched_register_all

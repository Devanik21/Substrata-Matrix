"""
clustering_runner.py — ClusterX Parallel Execution Engine
==========================================================
Orchestrates execution of all clustering algorithms:
- Sequential and parallel (joblib) execution
- Per-algorithm timeout and error isolation
- Result caching via hash keys
- Progress callback support
- Adaptive parameter tuning per dataset

Author: ClusterX Intelligence Lab
"""

from __future__ import annotations

import time
import warnings
import logging
import hashlib
import pickle
import traceback
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ──────────────────────────────────────────────────────────────────

class RunStatus(str, Enum):
    SUCCESS    = "success"
    FAILED     = "failed"
    TIMEOUT    = "timeout"
    SKIPPED    = "skipped"
    CACHED     = "cached"


class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL   = "parallel"
    ADAPTIVE   = "adaptive"     # Auto-selects based on dataset size


# ──────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────

@dataclass
class ClusteringResult:
    algorithm_id: str
    algorithm_name: str
    algorithm_family: str
    labels: np.ndarray
    status: RunStatus
    runtime_seconds: float
    n_clusters_found: int
    n_noise_points: int
    noise_ratio: float
    params_used: Dict[str, Any]
    error_message: Optional[str] = None
    model: Optional[Any] = None
    soft_labels: Optional[np.ndarray] = None   # Probabilistic memberships
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "algorithm_name": self.algorithm_name,
            "algorithm_family": self.algorithm_family,
            "status": self.status.value,
            "runtime_seconds": round(self.runtime_seconds, 4),
            "n_clusters_found": self.n_clusters_found,
            "n_noise_points": self.n_noise_points,
            "noise_ratio": round(self.noise_ratio, 4),
            "params_used": self.params_used,
            "error_message": self.error_message,
        }

    @property
    def succeeded(self) -> bool:
        return self.status in (RunStatus.SUCCESS, RunStatus.CACHED)

    @property
    def has_noise(self) -> bool:
        return self.n_noise_points > 0


@dataclass
class RunnerConfig:
    n_clusters: int = 8
    execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE
    max_workers: int = 4
    timeout_seconds: int = 120
    use_cache: bool = True
    progress_callback: Optional[Callable[[str, int, int], None]] = None
    param_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skip_slow_on_large: bool = True
    large_dataset_threshold: int = 20_000
    very_large_threshold: int = 100_000
    random_state: int = 42
    verbose: bool = False


@dataclass
class BatchRunResult:
    results: Dict[str, ClusteringResult]
    total_runtime: float
    n_success: int
    n_failed: int
    n_timeout: int
    n_skipped: int
    n_cached: int
    algorithm_ids: List[str]
    dataset_shape: Tuple[int, int]
    run_config: RunnerConfig
    cache_hit_rate: float

    def successful(self) -> Dict[str, ClusteringResult]:
        return {k: v for k, v in self.results.items() if v.succeeded}

    def failed(self) -> Dict[str, ClusteringResult]:
        return {k: v for k, v in self.results.items()
                if v.status == RunStatus.FAILED}

    def sorted_by_speed(self) -> List[ClusteringResult]:
        return sorted(
            [r for r in self.results.values() if r.succeeded],
            key=lambda r: r.runtime_seconds
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "total_algorithms_run": len(self.algorithm_ids),
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "n_timeout": self.n_timeout,
            "n_skipped": self.n_skipped,
            "n_cached": self.n_cached,
            "cache_hit_rate_pct": round(self.cache_hit_rate * 100, 1),
            "total_runtime_s": round(self.total_runtime, 2),
            "dataset_shape": self.dataset_shape,
        }


# ──────────────────────────────────────────────────────────────────
# RESULT CACHE
# ──────────────────────────────────────────────────────────────────

class ResultCache:
    """Thread-safe in-memory LRU cache for clustering results."""

    def __init__(self, max_entries: int = 200):
        self._store: Dict[str, ClusteringResult] = {}
        self._access_order: List[str] = []
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, algorithm_id: str, X: np.ndarray,
                  params: Dict[str, Any]) -> str:
        X_hash = hashlib.md5(X.data.tobytes()).hexdigest()[:12]
        params_str = str(sorted(params.items()))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"{algorithm_id}_{X_hash}_{params_hash}"

    def get(self, algorithm_id: str, X: np.ndarray,
            params: Dict[str, Any]) -> Optional[ClusteringResult]:
        key = self._make_key(algorithm_id, X, params)
        with self._lock:
            if key in self._store:
                self._hits += 1
                self._access_order.remove(key)
                self._access_order.append(key)
                result = self._store[key]
                result.status = RunStatus.CACHED
                return result
            self._misses += 1
            return None

    def set(self, algorithm_id: str, X: np.ndarray,
            params: Dict[str, Any], result: ClusteringResult):
        key = self._make_key(algorithm_id, X, params)
        with self._lock:
            if len(self._store) >= self._max_entries:
                oldest = self._access_order.pop(0)
                del self._store[oldest]
            self._store[key] = result
            self._access_order.append(key)

    def clear(self):
        with self._lock:
            self._store.clear()
            self._access_order.clear()
            self._hits = 0
            self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
        }


# ──────────────────────────────────────────────────────────────────
# ADAPTIVE PARAMETER TUNER
# ──────────────────────────────────────────────────────────────────

class AdaptiveParameterTuner:
    """
    Adjusts algorithm hyper-parameters based on dataset characteristics
    to prevent failures and improve result quality.
    """

    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def tune(self, algorithm_id: str, params: Dict[str, Any],
             X: np.ndarray) -> Dict[str, Any]:
        n, d = X.shape
        params = params.copy()
        params["random_state"] = self.random_state

        # Universal: cap n_clusters to valid range
        for k in ("n_clusters", "n_components"):
            if k in params:
                params[k] = max(2, min(int(params[k]), n - 1, 50))

        # DBSCAN: auto-estimate eps from kNN distance
        if algorithm_id == "dbscan" and params.get("eps", 0.5) == 0.5:
            params["eps"] = self._estimate_dbscan_eps(X, params.get("min_samples", 5))

        # Spectral: cap samples
        if "spectral" in algorithm_id and n > 15_000:
            params["n_clusters"] = params.get("n_clusters", self.n_clusters)

        # Affinity Propagation: scale preference with data
        if algorithm_id == "affinity_propagation":
            if n > 3000:
                params["preference"] = -200.0  # More negative → fewer clusters

        # Mean Shift bandwidth
        if algorithm_id == "mean_shift":
            if params.get("bandwidth", 0.0) == 0.0:
                params["bandwidth"] = 0.0  # Will be auto-estimated in factory

        # GMM: cap n_init for speed
        if "gmm" in algorithm_id and n > 50_000:
            params["n_init"] = 1
            params["max_iter"] = 100

        # BIRCH: smaller threshold for more clusters
        if algorithm_id == "birch" and d > 50:
            params["threshold"] = min(params.get("threshold", 0.5), 1.0)

        # KNN imputer neighbors
        if "knn" in algorithm_id:
            params["n_neighbors"] = max(3, min(params.get("n_neighbors", 5), n // 10))

        # SOM: reduce grid for speed
        if algorithm_id == "som_clustering" and n > 10_000:
            params["epochs"] = min(params.get("epochs", 100), 50)

        # Autoencoder: reduce epochs for speed
        if algorithm_id == "autoencoder_kmeans" and n > 20_000:
            params["epochs"] = min(params.get("epochs", 30), 15)

        return params

    def _estimate_dbscan_eps(self, X: np.ndarray, min_samples: int) -> float:
        """Estimate DBSCAN eps via the k-distance graph elbow method."""
        try:
            from sklearn.neighbors import NearestNeighbors
            k = min(min_samples, len(X) - 1)
            sample = X if len(X) <= 2000 else X[
                np.random.default_rng(42).choice(len(X), 2000, replace=False)]
            nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
            nbrs.fit(sample)
            distances, _ = nbrs.kneighbors(sample)
            k_distances = np.sort(distances[:, -1])[::-1]
            # Find elbow via max second derivative
            diffs2 = np.diff(np.diff(k_distances))
            if len(diffs2) > 0:
                elbow_idx = int(np.argmax(diffs2))
                eps = float(k_distances[elbow_idx])
            else:
                eps = float(np.percentile(k_distances, 10))
            return max(eps, 0.05)
        except Exception:
            return 0.5


# ──────────────────────────────────────────────────────────────────
# SPEED PROFILER
# ──────────────────────────────────────────────────────────────────

class SpeedProfiler:
    """Tracks per-algorithm runtime history for adaptive scheduling."""

    def __init__(self):
        self._history: Dict[str, List[float]] = {}

    def record(self, algorithm_id: str, runtime: float):
        self._history.setdefault(algorithm_id, []).append(runtime)

    def expected_runtime(self, algorithm_id: str) -> Optional[float]:
        times = self._history.get(algorithm_id, [])
        return float(np.mean(times)) if times else None

    def slowest_algorithms(self, n: int = 5) -> List[Tuple[str, float]]:
        avgs = {k: float(np.mean(v)) for k, v in self._history.items()}
        return sorted(avgs.items(), key=lambda x: -x[1])[:n]


# ──────────────────────────────────────────────────────────────────
# SINGLE-ALGORITHM EXECUTOR
# ──────────────────────────────────────────────────────────────────

class SingleAlgorithmExecutor:
    """Executes one algorithm with full error isolation and timing."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self._tuner = AdaptiveParameterTuner(
            n_clusters=config.n_clusters,
            random_state=config.random_state
        )

    def run(self, algorithm_id: str, X: np.ndarray,
            param_overrides: Optional[Dict[str, Any]] = None) -> ClusteringResult:
        from clustering_registry import get_registry, AlgorithmSpec
        registry = get_registry()

        try:
            spec = registry.get(algorithm_id)
        except KeyError as e:
            return self._make_error_result(
                algorithm_id, str(e), runtime=0.0,
                family="unknown", name=algorithm_id, params={}
            )

        # Build params
        params = spec.default_params()
        if "n_clusters" in params:
            params["n_clusters"] = self.config.n_clusters
        if "n_components" in params and not spec.requires_n_clusters:
            pass  # Leave as default for non-k algorithms
        elif "n_components" in params:
            params["n_components"] = self.config.n_clusters

        if param_overrides:
            params.update(param_overrides)

        # Adaptive tuning
        params = self._tuner.tune(algorithm_id, params, X)

        t_start = time.perf_counter()
        try:
            model = spec.factory(**params)
            labels = self._fit_and_extract(model, X, spec)
            runtime = time.perf_counter() - t_start
            return self._make_success_result(
                spec, labels, model, params, runtime
            )
        except MemoryError:
            runtime = time.perf_counter() - t_start
            return self._make_error_result(
                algorithm_id, "MemoryError: dataset too large", runtime,
                spec.family.value, spec.name, params
            )
        except Exception as exc:
            runtime = time.perf_counter() - t_start
            tb = traceback.format_exc()
            logger.debug(f"[{algorithm_id}] Failed: {exc}\n{tb}")
            return self._make_error_result(
                algorithm_id, f"{type(exc).__name__}: {exc}", runtime,
                spec.family.value, spec.name, params
            )

    def _fit_and_extract(self, model, X: np.ndarray, spec) -> np.ndarray:
        """Calls fit_predict or fit+labels_ uniformly."""
        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X)
        elif hasattr(model, "fit"):
            model.fit(X)
            if hasattr(model, "labels_"):
                labels = model.labels_
            elif hasattr(model, "predict"):
                labels = model.predict(X)
            else:
                raise AttributeError("Model has no labels_ or predict method")
        else:
            raise AttributeError("Model has no fit or fit_predict method")

        labels = np.asarray(labels, dtype=np.int32)

        # Validate
        if len(labels) != len(X):
            raise ValueError(
                f"Label count mismatch: got {len(labels)}, expected {len(X)}"
            )
        return labels

    @staticmethod
    def _make_success_result(spec, labels: np.ndarray, model: Any,
                              params: Dict[str, Any],
                              runtime: float) -> ClusteringResult:
        valid_labels = labels[labels != -1]
        n_clusters = int(len(np.unique(valid_labels))) if len(valid_labels) > 0 else 0
        n_noise = int((labels == -1).sum())
        noise_ratio = n_noise / max(len(labels), 1)

        # Extract soft labels if available
        soft_labels = None
        if hasattr(model, "predict_proba"):
            try:
                soft_labels = model.predict_proba(None)  # Already fitted
            except Exception:
                pass
        if hasattr(model, "membership_") and model.membership_ is not None:
            soft_labels = model.membership_

        return ClusteringResult(
            algorithm_id=spec.id,
            algorithm_name=spec.name,
            algorithm_family=spec.family.value,
            labels=labels,
            status=RunStatus.SUCCESS,
            runtime_seconds=runtime,
            n_clusters_found=n_clusters,
            n_noise_points=n_noise,
            noise_ratio=noise_ratio,
            params_used=params,
            model=model,
            soft_labels=soft_labels,
        )

    @staticmethod
    def _make_error_result(algorithm_id: str, error_msg: str, runtime: float,
                            family: str, name: str,
                            params: Dict[str, Any]) -> ClusteringResult:
        return ClusteringResult(
            algorithm_id=algorithm_id,
            algorithm_name=name,
            algorithm_family=family,
            labels=np.array([], dtype=np.int32),
            status=RunStatus.FAILED,
            runtime_seconds=runtime,
            n_clusters_found=0,
            n_noise_points=0,
            noise_ratio=0.0,
            params_used=params,
            error_message=error_msg,
        )


# ──────────────────────────────────────────────────────────────────
# DATASET SIZE CLASSIFIER
# ──────────────────────────────────────────────────────────────────

class DatasetSizeClass(str, Enum):
    TINY   = "tiny"      # < 500
    SMALL  = "small"     # 500 – 5000
    MEDIUM = "medium"    # 5000 – 20000
    LARGE  = "large"     # 20000 – 100000
    XLARGE = "xlarge"    # > 100000


def classify_dataset(n: int, d: int) -> DatasetSizeClass:
    if n < 500:
        return DatasetSizeClass.TINY
    elif n < 5_000:
        return DatasetSizeClass.SMALL
    elif n < 20_000:
        return DatasetSizeClass.MEDIUM
    elif n < 100_000:
        return DatasetSizeClass.LARGE
    else:
        return DatasetSizeClass.XLARGE


# Per-size blacklist: algorithms that are too slow / memory-intensive
_SIZE_BLACKLIST: Dict[DatasetSizeClass, List[str]] = {
    DatasetSizeClass.TINY: [],
    DatasetSizeClass.SMALL: [],
    DatasetSizeClass.MEDIUM: [
        "affinity_propagation", "diana_divisive",
        "spectral_biclustering", "spectral_coclustering",
        "kmedoids",
    ],
    DatasetSizeClass.LARGE: [
        "affinity_propagation", "diana_divisive",
        "spectral_rbf", "spectral_kmeans", "spectral_discretize", "spectral_cluster_qr",
        "spectral_biclustering", "spectral_coclustering",
        "kmedoids", "denclue", "possibilistic_cmeans",
        "mean_shift", "isomap_kmeans", "lle_kmeans",
        "random_subspace_ensemble", "ensemble_voting",
        "agglomerative_ward", "agglomerative_complete",
        "agglomerative_average", "agglomerative_single",
    ],
    DatasetSizeClass.XLARGE: [
        "affinity_propagation", "diana_divisive", "denclue", "possibilistic_cmeans",
        "spectral_kmeans", "spectral_discretize", "spectral_cluster_qr",
        "spectral_biclustering", "spectral_coclustering",
        "kmedoids", "mean_shift", "isomap_kmeans", "lle_kmeans",
        "random_subspace_ensemble", "ensemble_voting",
        "agglomerative_ward", "agglomerative_complete",
        "agglomerative_average", "agglomerative_single",
        "gmm_full", "gmm_tied", "bgmm",
        "fuzzy_cmeans", "clique_subspace", "som_clustering",
    ],
}


def get_recommended_algorithms(n: int, d: int,
                                all_ids: List[str]) -> Tuple[List[str], List[str]]:
    """Returns (allowed_ids, skipped_ids) based on dataset size."""
    size_class = classify_dataset(n, d)
    blacklisted = set(_SIZE_BLACKLIST.get(size_class, []))
    allowed = [aid for aid in all_ids if aid not in blacklisted]
    skipped = [aid for aid in all_ids if aid in blacklisted]
    return allowed, skipped


# ──────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ──────────────────────────────────────────────────────────────────

class ClusteringRunner:
    """
    Master runner — schedules, executes, and collects results for
    a user-specified list of clustering algorithms.
    """

    def __init__(self, config: Optional[RunnerConfig] = None):
        self.config = config or RunnerConfig()
        self._cache = ResultCache(max_entries=200)
        self._executor = SingleAlgorithmExecutor(self.config)
        self._profiler = SpeedProfiler()
        self._run_log: List[str] = []

    # ── Public API ────────────────────────────────────────────────

    def run_algorithms(self,
                       algorithm_ids: List[str],
                       X: np.ndarray,
                       param_overrides: Optional[Dict[str, Dict[str, Any]]] = None
                       ) -> BatchRunResult:
        """Execute a list of algorithms on dataset X."""
        self._run_log = []
        param_overrides = param_overrides or {}
        t_global = time.perf_counter()

        n, d = X.shape
        self._log(f"Running {len(algorithm_ids)} algorithms on {n}×{d} dataset")

        # Determine execution mode
        mode = self._select_execution_mode(n, d)
        self._log(f"Execution mode: {mode.value}")

        # Filter algorithms by dataset size
        if self.config.skip_slow_on_large:
            allowed, skipped_ids = get_recommended_algorithms(n, d, algorithm_ids)
            if skipped_ids:
                self._log(
                    f"Skipping {len(skipped_ids)} algorithms (too slow for "
                    f"n={n}): {skipped_ids[:5]}{'...' if len(skipped_ids) > 5 else ''}"
                )
        else:
            allowed = algorithm_ids
            skipped_ids = []

        # Execute
        results: Dict[str, ClusteringResult] = {}

        # Pre-fill skipped
        for aid in skipped_ids:
            results[aid] = self._make_skipped(aid)

        if mode == ExecutionMode.PARALLEL:
            run_results = self._run_parallel(allowed, X, param_overrides)
        else:
            run_results = self._run_sequential(allowed, X, param_overrides)

        results.update(run_results)

        total_runtime = time.perf_counter() - t_global
        self._log(f"Total batch runtime: {total_runtime:.2f}s")

        return self._build_batch_result(results, algorithm_ids, X, total_runtime)

    def run_single(self, algorithm_id: str, X: np.ndarray,
                   params: Optional[Dict[str, Any]] = None) -> ClusteringResult:
        """Execute a single algorithm."""
        params = params or {}
        if self.config.use_cache:
            cached = self._cache.get(algorithm_id, X, params)
            if cached is not None:
                self._log(f"Cache hit: {algorithm_id}")
                return cached
        result = self._executor.run(algorithm_id, X, params)
        self._profiler.record(algorithm_id, result.runtime_seconds)
        if self.config.use_cache and result.succeeded:
            self._cache.set(algorithm_id, X, params, result)
        return result

    def run_single_with_timeout(self, algorithm_id: str, X: np.ndarray,
                                params: Optional[Dict[str, Any]] = None
                                ) -> ClusteringResult:
        """Execute with hard timeout using thread-based isolation."""
        params = params or {}
        timeout = self.config.timeout_seconds
        result_holder = {}

        def _worker():
            try:
                result_holder["result"] = self._executor.run(algorithm_id, X, params)
            except Exception as e:
                result_holder["error"] = str(e)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            self._log(f"TIMEOUT: {algorithm_id} exceeded {timeout}s")
            return ClusteringResult(
                algorithm_id=algorithm_id,
                algorithm_name=algorithm_id,
                algorithm_family="unknown",
                labels=np.array([], dtype=np.int32),
                status=RunStatus.TIMEOUT,
                runtime_seconds=float(timeout),
                n_clusters_found=0,
                n_noise_points=0,
                noise_ratio=0.0,
                params_used=params,
                error_message=f"Timeout after {timeout}s",
            )

        if "error" in result_holder:
            return SingleAlgorithmExecutor._make_error_result(
                algorithm_id, result_holder["error"], timeout,
                "unknown", algorithm_id, params
            )

        result = result_holder.get("result")
        if result is None:
            return SingleAlgorithmExecutor._make_error_result(
                algorithm_id, "Unknown error (no result returned)", timeout,
                "unknown", algorithm_id, params
            )
        return result

    # ── Execution Strategies ──────────────────────────────────────

    def _run_sequential(self,
                        algorithm_ids: List[str],
                        X: np.ndarray,
                        param_overrides: Dict[str, Dict[str, Any]]
                        ) -> Dict[str, ClusteringResult]:
        results = {}
        total = len(algorithm_ids)
        for i, aid in enumerate(algorithm_ids, 1):
            if self.config.progress_callback:
                self.config.progress_callback(aid, i, total)

            params = param_overrides.get(aid, {})
            result = self.run_single_with_timeout(aid, X, params)
            results[aid] = result
            self._profiler.record(aid, result.runtime_seconds)

            status_icon = "✓" if result.succeeded else ("⚠" if result.status == RunStatus.TIMEOUT else "✗")
            self._log(
                f"[{i:3d}/{total}] {status_icon} {aid:40s} "
                f"{result.runtime_seconds:6.2f}s  "
                f"k={result.n_clusters_found}"
            )
        return results

    def _run_parallel(self,
                      algorithm_ids: List[str],
                      X: np.ndarray,
                      param_overrides: Dict[str, Dict[str, Any]]
                      ) -> Dict[str, ClusteringResult]:
        results = {}
        total = len(algorithm_ids)
        completed = 0

        # Sort: cheap algorithms first to populate cache quickly
        fast_ids = [aid for aid in algorithm_ids if "kmeans" in aid or "minibatch" in aid]
        slow_ids = [aid for aid in algorithm_ids if aid not in fast_ids]
        ordered_ids = fast_ids + slow_ids

        def _task(aid):
            params = param_overrides.get(aid, {})
            return aid, self.run_single_with_timeout(aid, X, params)

        max_workers = min(self.config.max_workers, len(ordered_ids))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_task, aid): aid for aid in ordered_ids}
            for future in as_completed(futures):
                try:
                    aid, result = future.result()
                    results[aid] = result
                    completed += 1
                    self._profiler.record(aid, result.runtime_seconds)
                    if self.config.progress_callback:
                        self.config.progress_callback(aid, completed, total)
                    status_icon = "✓" if result.succeeded else "✗"
                    self._log(
                        f"[{completed:3d}/{total}] {status_icon} {aid:40s} "
                        f"{result.runtime_seconds:6.2f}s"
                    )
                except Exception as e:
                    aid = futures[future]
                    results[aid] = SingleAlgorithmExecutor._make_error_result(
                        aid, str(e), 0.0, "unknown", aid, {}
                    )
                    completed += 1

        return results

    def _select_execution_mode(self, n: int, d: int) -> ExecutionMode:
        if self.config.execution_mode != ExecutionMode.ADAPTIVE:
            return self.config.execution_mode
        # Adaptive: parallel for medium+ datasets, sequential for tiny/small
        if n >= 5_000:
            return ExecutionMode.PARALLEL
        return ExecutionMode.SEQUENTIAL

    # ── Helpers ───────────────────────────────────────────────────

    def _build_batch_result(self, results: Dict[str, ClusteringResult],
                            all_ids: List[str], X: np.ndarray,
                            total_runtime: float) -> BatchRunResult:
        n_success  = sum(1 for r in results.values() if r.status == RunStatus.SUCCESS)
        n_cached   = sum(1 for r in results.values() if r.status == RunStatus.CACHED)
        n_failed   = sum(1 for r in results.values() if r.status == RunStatus.FAILED)
        n_timeout  = sum(1 for r in results.values() if r.status == RunStatus.TIMEOUT)
        n_skipped  = sum(1 for r in results.values() if r.status == RunStatus.SKIPPED)

        return BatchRunResult(
            results=results,
            total_runtime=total_runtime,
            n_success=n_success,
            n_failed=n_failed,
            n_timeout=n_timeout,
            n_skipped=n_skipped,
            n_cached=n_cached,
            algorithm_ids=all_ids,
            dataset_shape=X.shape,
            run_config=self.config,
            cache_hit_rate=self._cache.hit_rate,
        )

    @staticmethod
    def _make_skipped(algorithm_id: str) -> ClusteringResult:
        return ClusteringResult(
            algorithm_id=algorithm_id,
            algorithm_name=algorithm_id,
            algorithm_family="unknown",
            labels=np.array([], dtype=np.int32),
            status=RunStatus.SKIPPED,
            runtime_seconds=0.0,
            n_clusters_found=0,
            n_noise_points=0,
            noise_ratio=0.0,
            params_used={},
            error_message="Skipped: dataset too large for this algorithm",
        )

    def _log(self, msg: str):
        self._run_log.append(msg)
        if self.config.verbose:
            logger.info(msg)

    # ── Cache API ─────────────────────────────────────────────────

    def clear_cache(self):
        self._cache.clear()
        self._log("Cache cleared")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        return self._cache.stats

    @property
    def run_log(self) -> List[str]:
        return self._run_log


# ──────────────────────────────────────────────────────────────────
# PARAMETER SWEEP RUNNER
# ──────────────────────────────────────────────────────────────────

class ParameterSweepRunner:
    """
    Runs a single algorithm across a grid of hyper-parameter values,
    returning all results for comparison.
    """

    def __init__(self, base_config: Optional[RunnerConfig] = None):
        self._base_config = base_config or RunnerConfig()

    def sweep(self,
              algorithm_id: str,
              X: np.ndarray,
              param_grid: Dict[str, List[Any]],
              progress_callback: Optional[Callable] = None
              ) -> List[ClusteringResult]:
        """
        Executes algorithm for all combinations in param_grid.
        Returns list of ClusteringResults, one per combination.
        """
        import itertools
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        results = []
        total = len(combos)
        executor = SingleAlgorithmExecutor(self._base_config)

        for i, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))
            if progress_callback:
                progress_callback(params, i, total)
            result = executor.run(algorithm_id, X, params)
            result.extra_info["param_combo"] = params
            results.append(result)

        return results

    def k_sweep(self,
                algorithm_id: str,
                X: np.ndarray,
                k_range: range,
                extra_params: Optional[Dict[str, Any]] = None
                ) -> List[ClusteringResult]:
        """Sweep over number of clusters k."""
        extra_params = extra_params or {}
        param_grid = {"n_clusters": list(k_range)}
        param_grid.update({k: [v] for k, v in extra_params.items()})
        return self.sweep(algorithm_id, X, param_grid)


# ──────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def make_runner(n_clusters: int = 8,
                parallel: bool = True,
                timeout: int = 120,
                verbose: bool = False,
                progress_callback: Optional[Callable] = None) -> ClusteringRunner:
    """Convenience factory for creating a pre-configured runner."""
    config = RunnerConfig(
        n_clusters=n_clusters,
        execution_mode=ExecutionMode.PARALLEL if parallel else ExecutionMode.SEQUENTIAL,
        timeout_seconds=timeout,
        verbose=verbose,
        progress_callback=progress_callback,
        use_cache=True,
        skip_slow_on_large=True,
    )
    return ClusteringRunner(config)


def run_all_algorithms(X: np.ndarray,
                       n_clusters: int = 8,
                       algorithm_ids: Optional[List[str]] = None,
                       progress_callback: Optional[Callable] = None,
                       timeout: int = 120) -> BatchRunResult:
    """Top-level convenience: run all (or selected) algorithms on X."""
    from clustering_registry import get_registry
    registry = get_registry()

    if algorithm_ids is None:
        algorithm_ids = registry.ids()

    runner = make_runner(
        n_clusters=n_clusters,
        parallel=True,
        timeout=timeout,
        progress_callback=progress_callback,
    )
    return runner.run_algorithms(algorithm_ids, X)


def labels_to_cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
    """Returns {cluster_id: size} for a label array."""
    from collections import Counter
    return dict(Counter(labels.tolist()))


def filter_degenerate_results(batch: BatchRunResult,
                               min_clusters: int = 2,
                               max_noise_ratio: float = 0.8) -> BatchRunResult:
    """Remove results where too few clusters or too much noise."""
    filtered = {}
    for k, r in batch.results.items():
        if not r.succeeded:
            filtered[k] = r
            continue
        if r.n_clusters_found < min_clusters:
            r.status = RunStatus.FAILED
            r.error_message = f"Only {r.n_clusters_found} cluster(s) found"
            filtered[k] = r
            continue
        if r.noise_ratio > max_noise_ratio:
            r.status = RunStatus.FAILED
            r.error_message = f"Noise ratio {r.noise_ratio:.1%} exceeds threshold"
            filtered[k] = r
            continue
        filtered[k] = r
    batch.results = filtered
    return batch


def compute_cluster_statistics(labels: np.ndarray) -> Dict[str, Any]:
    """Detailed per-cluster size statistics from label array."""
    valid = labels[labels != -1]
    if len(valid) == 0:
        return {"n_clusters": 0, "sizes": {}, "min_size": 0, "max_size": 0}
    from collections import Counter
    counts = Counter(valid.tolist())
    sizes = list(counts.values())
    return {
        "n_clusters": len(counts),
        "sizes": dict(counts),
        "min_size": int(min(sizes)),
        "max_size": int(max(sizes)),
        "mean_size": float(np.mean(sizes)),
        "std_size": float(np.std(sizes)),
        "balance_ratio": float(min(sizes) / max(sizes)) if max(sizes) > 0 else 0.0,
        "n_noise": int((labels == -1).sum()),
        "noise_ratio": float((labels == -1).mean()),
        "size_entropy": float(-sum(
            (c / len(valid)) * np.log2(c / len(valid) + 1e-12)
            for c in sizes
        )),
    }


# ══════════════════════════════════════════════════════════════════
# FINAL POLISH — ADVANCED RUNNER ADDITIONS
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# EXECUTION TIMELINE RECORDER
# ──────────────────────────────────────────────────────────────────

class ExecutionTimeline:
    """
    Records a detailed timeline of algorithm execution.
    Tracks: start time, end time, memory delta, status.
    Used for profiling and identifying bottlenecks.
    """
    def __init__(self):
        self._events: List[Dict[str, Any]] = []
        self._t_start = time.perf_counter()

    def record(self, algorithm_id: str, algorithm_name: str,
               status: str, runtime: float,
               n_clusters: int, extra: Optional[Dict] = None):
        elapsed = time.perf_counter() - self._t_start
        event = {
            "algorithm_id": algorithm_id,
            "algorithm_name": algorithm_name[:40],
            "status": status,
            "runtime_s": round(runtime, 4),
            "elapsed_s": round(elapsed, 4),
            "n_clusters": n_clusters,
        }
        if extra:
            event.update(extra)
        self._events.append(event)

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        if not self._events:
            return pd.DataFrame()
        df = pd.DataFrame(self._events)
        df = df.sort_values("elapsed_s").reset_index(drop=True)
        return df

    def summary(self) -> Dict[str, Any]:
        if not self._events:
            return {}
        runtimes = [e["runtime_s"] for e in self._events]
        statuses = [e["status"] for e in self._events]
        return {
            "n_events": len(self._events),
            "total_wall_time": round(time.perf_counter() - self._t_start, 3),
            "fastest": min(self._events, key=lambda e: e["runtime_s"])["algorithm_id"],
            "slowest": max(self._events, key=lambda e: e["runtime_s"])["algorithm_id"],
            "mean_runtime": round(float(np.mean(runtimes)), 3),
            "median_runtime": round(float(np.median(runtimes)), 3),
            "n_success": statuses.count("success") + statuses.count("cached"),
            "n_failed": statuses.count("failed"),
            "n_timeout": statuses.count("timeout"),
        }

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self._events


# ──────────────────────────────────────────────────────────────────
# MEMORY-AWARE SCHEDULER
# ──────────────────────────────────────────────────────────────────

class MemoryAwareScheduler:
    """
    Estimates memory usage per algorithm and reorders execution
    to avoid running multiple high-memory algorithms simultaneously.
    Also gates algorithms that exceed available system memory.
    """
    # Rough bytes-per-element multipliers
    MEMORY_CLASS = {
        "O(n)":       1,
        "O(n log n)": 2,
        "O(n²)":      4,
        "O(n²·k)":    6,
        "O(n³)":      10,
        "varies":     2,
    }

    def __init__(self, safety_factor: float = 0.6):
        self.safety_factor = safety_factor

    def estimate_memory_mb(self, algorithm_id: str, n: int, d: int) -> float:
        """Rough upper bound on peak memory in MB."""
        try:
            from clustering_registry import get_registry
            spec = get_registry().get(algorithm_id)
            mult = self.MEMORY_CLASS.get(spec.time_complexity.value, 2)
            bytes_est = mult * n * d * 8  # float64
            if "agglomerative" in algorithm_id or "spectral" in algorithm_id:
                bytes_est += n * n * 4  # distance matrix
            if "gmm" in algorithm_id:
                bytes_est += n * d * d * 8  # covariance matrices
            return bytes_est / 1e6
        except Exception:
            return float(n * d * 8 / 1e6)

    def available_memory_mb(self) -> float:
        """Available system RAM in MB."""
        try:
            import psutil
            return psutil.virtual_memory().available / 1e6
        except ImportError:
            return 4000.0  # Conservative fallback: 4 GB

    def filter_feasible(self, algorithm_ids: List[str],
                         n: int, d: int) -> Tuple[List[str], List[str]]:
        """Returns (feasible_ids, too_large_ids)."""
        avail = self.available_memory_mb() * self.safety_factor
        feasible, too_large = [], []
        for aid in algorithm_ids:
            est = self.estimate_memory_mb(aid, n, d)
            if est > avail:
                too_large.append(aid)
            else:
                feasible.append(aid)
        return feasible, too_large

    def schedule_by_memory(self, algorithm_ids: List[str],
                            n: int, d: int) -> List[str]:
        """Sort so light algorithms run first (better parallelism)."""
        estimates = {aid: self.estimate_memory_mb(aid, n, d) for aid in algorithm_ids}
        return sorted(algorithm_ids, key=lambda aid: estimates.get(aid, 0))


# ──────────────────────────────────────────────────────────────────
# RESULT FINGERPRINTER
# ──────────────────────────────────────────────────────────────────

class ResultFingerprinter:
    """
    Creates a compact fingerprint of a clustering result for
    deduplication, comparison, and reproducibility tracking.
    Two identical label arrays produce identical fingerprints.
    """
    @staticmethod
    def fingerprint(labels: np.ndarray) -> str:
        """Returns a 16-char hex fingerprint of a label array."""
        canonical = ResultFingerprinter._canonicalise(labels)
        return hashlib.md5(canonical.tobytes()).hexdigest()[:16]

    @staticmethod
    def _canonicalise(labels: np.ndarray) -> np.ndarray:
        """Relabel clusters in order of first appearance (canonical form)."""
        mapping = {}
        next_id = 0
        out = np.empty_like(labels)
        for i, lab in enumerate(labels):
            if lab == -1:
                out[i] = -1
                continue
            if lab not in mapping:
                mapping[lab] = next_id
                next_id += 1
            out[i] = mapping[lab]
        return out

    @staticmethod
    def are_equivalent(labels_a: np.ndarray,
                        labels_b: np.ndarray) -> bool:
        """True if two label arrays represent the same partition."""
        if len(labels_a) != len(labels_b):
            return False
        fp_a = ResultFingerprinter.fingerprint(labels_a)
        fp_b = ResultFingerprinter.fingerprint(labels_b)
        return fp_a == fp_b

    @staticmethod
    def deduplicate(results: Dict[str, "ClusteringResult"]
                    ) -> Tuple[Dict[str, "ClusteringResult"], Dict[str, str]]:
        """
        Remove duplicate clustering results.
        Returns (unique_results, duplicate_map: {dup_id → original_id}).
        """
        seen: Dict[str, str] = {}  # fingerprint → first algorithm_id
        unique: Dict[str, "ClusteringResult"] = {}
        duplicate_map: Dict[str, str] = {}
        for aid, cr in results.items():
            if not cr.succeeded or len(cr.labels) == 0:
                unique[aid] = cr
                continue
            fp = ResultFingerprinter.fingerprint(cr.labels)
            if fp not in seen:
                seen[fp] = aid
                unique[aid] = cr
            else:
                duplicate_map[aid] = seen[fp]
        return unique, duplicate_map


# ──────────────────────────────────────────────────────────────────
# INCREMENTAL / WARM-START RUNNER
# ──────────────────────────────────────────────────────────────────

class IncrementalRunner:
    """
    Supports incremental updates: when new data arrives, re-runs only
    the algorithms whose results might change, using warm-start where
    possible (Mini-Batch K-Means, BIRCH).
    """
    WARM_START_ALGOS = {"minibatch_kmeans", "birch", "online_gmm"}

    def __init__(self, base_runner: "ClusteringRunner"):
        self._runner = base_runner
        self._prev_results: Dict[str, "ClusteringResult"] = {}
        self._prev_X_hash: Optional[str] = None

    def update(self, algorithm_ids: List[str],
               X_new: np.ndarray,
               X_old_hash: Optional[str] = None) -> Dict[str, "ClusteringResult"]:
        """
        Update clustering with new data.
        Returns merged result dict (old results updated where needed).
        """
        new_hash = hashlib.md5(X_new.data.tobytes()).hexdigest()[:12]
        if new_hash == self._prev_X_hash and self._prev_results:
            logger.info("Data unchanged — returning cached results")
            return self._prev_results

        # For warm-start capable algorithms, attempt partial update
        warm_ids = [aid for aid in algorithm_ids if aid in self.WARM_START_ALGOS]
        cold_ids = [aid for aid in algorithm_ids if aid not in self.WARM_START_ALGOS]

        results = {}
        if cold_ids:
            batch = self._runner.run_algorithms(cold_ids, X_new)
            results.update(batch.results)

        # Warm-start: pass prev model to Mini-Batch KMeans
        from clustering_registry import get_registry
        for aid in warm_ids:
            try:
                prev_cr = self._prev_results.get(aid)
                if prev_cr and prev_cr.succeeded and prev_cr.model is not None:
                    model = prev_cr.model
                    if hasattr(model, "partial_fit"):
                        model.partial_fit(X_new)
                        labels = model.predict(X_new)
                    else:
                        cr = self._runner.run_single(aid, X_new)
                        labels = cr.labels
                        model = cr.model
                    from clustering_runner import ClusteringResult, RunStatus
                    valid = labels[labels != -1]
                    nk = len(np.unique(valid)) if len(valid) > 0 else 0
                    results[aid] = ClusteringResult(
                        algorithm_id=aid, algorithm_name=aid,
                        algorithm_family="warm_start",
                        labels=np.asarray(labels, dtype=np.int32),
                        status=RunStatus.SUCCESS,
                        runtime_seconds=0.0, n_clusters_found=nk,
                        n_noise_points=int((labels == -1).sum()),
                        noise_ratio=float((labels == -1).mean()),
                        params_used={}, model=model,
                    )
                else:
                    cr = self._runner.run_single(aid, X_new)
                    results[aid] = cr
            except Exception as e:
                logger.warning(f"Warm-start failed for {aid}: {e}")
                cr = self._runner.run_single(aid, X_new)
                results[aid] = cr

        self._prev_results = results
        self._prev_X_hash = new_hash
        return results


# ──────────────────────────────────────────────────────────────────
# PERFORMANCE LEADERBOARD
# ──────────────────────────────────────────────────────────────────

class PerformanceLeaderboard:
    """
    Tracks algorithm performance history across multiple datasets/runs.
    Useful for identifying consistently well-performing algorithms
    and building dataset-specific algorithm portfolios.
    """
    def __init__(self):
        self._history: List[Dict[str, Any]] = []

    def record(self, dataset_name: str,
               eval_results: List[Any],
               n_samples: int, n_features: int):
        """Record evaluation results for one run."""
        for er in eval_results:
            self._history.append({
                "dataset": dataset_name,
                "algorithm": er.algorithm_id,
                "algorithm_name": er.algorithm_name,
                "composite_score": er.composite_score,
                "silhouette": er.metric_value("silhouette"),
                "n_clusters": er.n_clusters,
                "rank": er.rank,
                "n_samples": n_samples,
                "n_features": n_features,
                "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            })

    def top_algorithms(self, metric: str = "composite_score",
                        top_n: int = 10) -> "pd.DataFrame":
        import pandas as pd
        if not self._history:
            return pd.DataFrame()
        df = pd.DataFrame(self._history)
        if metric not in df.columns:
            return df
        agg = df.groupby("algorithm_name")[metric].agg(
            mean="mean", std="std", count="count",
            best="max", worst="min"
        ).round(4).reset_index()
        return agg.sort_values("mean", ascending=False).head(top_n)

    def win_rates(self) -> "pd.DataFrame":
        """Fraction of runs where each algorithm ranked #1."""
        import pandas as pd
        if not self._history:
            return pd.DataFrame()
        df = pd.DataFrame(self._history)
        wins = df[df["rank"] == 1].groupby("algorithm_name").size()
        total = df.groupby("algorithm_name").size()
        win_rate = (wins / total).fillna(0).round(4)
        return win_rate.reset_index().rename(columns={0: "win_rate"}).sort_values(
            "win_rate", ascending=False)

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(self._history)

    def clear(self):
        self._history.clear()


# ──────────────────────────────────────────────────────────────────
# ALGORITHM COMPLEXITY ESTIMATOR
# ──────────────────────────────────────────────────────────────────

class ComplexityEstimator:
    """
    Empirically estimates actual time complexity by running on subsets
    and fitting power-law curve: T = a * n^b.
    Predicts runtime for full dataset before committing.
    """
    def __init__(self, sample_sizes: Optional[List[int]] = None):
        self.sample_sizes = sample_sizes or [200, 500, 1000, 2000]

    def estimate(self, algorithm_id: str,
                 X: np.ndarray,
                 n_clusters: int = 8) -> Dict[str, Any]:
        """Returns predicted runtime for full n and complexity exponent b."""
        import scipy.optimize as opt
        from clustering_runner import SingleAlgorithmExecutor, RunnerConfig
        config = RunnerConfig(n_clusters=n_clusters, timeout_seconds=30)
        executor = SingleAlgorithmExecutor(config)
        rng = np.random.default_rng(42)
        n = len(X)

        sizes_tested = [s for s in self.sample_sizes if s < n]
        if not sizes_tested or len(sizes_tested) < 2:
            return {"predicted_runtime_s": None, "exponent": None,
                    "status": "insufficient_data"}

        runtimes = []
        for size in sizes_tested:
            idx = rng.choice(n, size, replace=False)
            cr = executor.run(algorithm_id, X[idx], {})
            runtimes.append(cr.runtime_seconds if cr.succeeded else None)

        valid = [(s, t) for s, t in zip(sizes_tested, runtimes) if t is not None and t > 1e-5]
        if len(valid) < 2:
            return {"predicted_runtime_s": None, "exponent": None,
                    "status": "all_runs_failed"}

        vs, vt = zip(*valid)
        try:
            log_s = np.log(list(vs))
            log_t = np.log(list(vt))
            coeffs = np.polyfit(log_s, log_t, 1)
            b = float(coeffs[0])  # complexity exponent
            a = float(np.exp(coeffs[1]))
            predicted = a * (n ** b)
        except Exception:
            predicted = None; b = None

        return {
            "sizes_tested": list(vs),
            "runtimes_s": [round(t, 4) for t in vt],
            "exponent_b": round(b, 3) if b is not None else None,
            "complexity_class": self._classify_exponent(b) if b is not None else "unknown",
            "predicted_runtime_s": round(float(predicted), 2) if predicted else None,
            "status": "ok",
        }

    @staticmethod
    def _classify_exponent(b: float) -> str:
        if b < 1.3:   return "O(n) — linear, very fast"
        if b < 1.6:   return "O(n log n) — quasi-linear"
        if b < 2.3:   return "O(n²) — quadratic, moderate"
        if b < 2.8:   return "O(n²·k) — quadratic-plus"
        return f"O(n^{b:.1f}) — super-quadratic, slow on large data"

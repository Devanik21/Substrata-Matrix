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

"""
clustering_runner.py — ClusterX Parallel Clustering Execution Engine
=====================================================================
Handles: timeout-safe execution, parallel batch runs, hyperparameter sweeps,
         elbow analysis, result caching, and orchestration layer.
Author: ClusterX Intelligence Lab
"""

from __future__ import annotations

import time
import hashlib
import pickle
import threading
import warnings
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from clustering_registry import (
    REGISTRY, AlgorithmRegistry, AlgorithmMeta, AlgorithmResult,
    AlgorithmFamily, ParameterType
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# ENUMERATIONS & DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────

class RunMode(str, Enum):
    SINGLE    = "single"
    BATCH     = "batch"
    SWEEP     = "sweep"
    AUTO      = "auto"


class RunStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"
    TIMEOUT   = "timeout"
    SKIPPED   = "skipped"


@dataclass
class RunConfig:
    """Configuration for a clustering run."""
    algorithms: List[str] = field(default_factory=lambda: ["kmeans"])
    params_override: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_clusters: int = 3
    k_range: Optional[Tuple[int, int]] = None
    timeout_seconds: int = 120
    n_jobs: int = -1
    random_state: int = 42
    enable_cache: bool = True
    cache_dir: str = ".clusterx_cache"
    sweep_mode: bool = False
    sweep_metric: str = "silhouette"
    progress_callback: Optional[Callable[[str, float], None]] = None
    max_retries: int = 1
    verbose: bool = True


@dataclass
class SingleRunResult:
    """Result from running one algorithm with one parameter set."""
    algorithm_name: str
    display_name: str
    labels: np.ndarray
    n_clusters_found: int
    n_noise: int
    params_used: Dict[str, Any]
    fit_time_seconds: float
    status: RunStatus
    error_message: Optional[str] = None
    model: Optional[Any] = None
    centers: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    inertia: Optional[float] = None


@dataclass
class SweepPoint:
    """One point in a hyperparameter sweep."""
    k: int
    algorithm: str
    params: Dict[str, Any]
    labels: np.ndarray
    n_clusters_found: int
    metric_value: float
    metric_name: str
    fit_time: float


@dataclass
class BatchResult:
    """Results from running multiple algorithms."""
    results: List[SingleRunResult]
    sweep_points: List[SweepPoint]
    best_algorithm: Optional[str]
    best_score: float
    total_time_seconds: float
    n_algorithms_run: int
    n_algorithms_failed: int
    config: RunConfig


# ──────────────────────────────────────────────────────────────────
# RESULT CACHE
# ──────────────────────────────────────────────────────────────────

class ResultCache:
    """MD5-keyed disk cache for clustering results."""

    def __init__(self, cache_dir: str = ".clusterx_cache"):
        self.cache_dir = Path(cache_dir)
        self._memory: Dict[str, Any] = {}

    def _make_key(self, algo_name: str, params: Dict, data_hash: str) -> str:
        payload = f"{algo_name}|{sorted(params.items())}|{data_hash}"
        return hashlib.md5(payload.encode()).hexdigest()[:20]

    def get(self, algo_name: str, params: Dict, data_hash: str) -> Optional[SingleRunResult]:
        key = self._make_key(algo_name, params, data_hash)
        if key in self._memory:
            logger.debug(f"Cache HIT (memory): {algo_name}")
            return self._memory[key]
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                self._memory[key] = result
                logger.debug(f"Cache HIT (disk): {algo_name}")
                return result
            except Exception:
                pass
        return None

    def put(self, algo_name: str, params: Dict, data_hash: str, result: SingleRunResult):
        key = self._make_key(algo_name, params, data_hash)
        self._memory[key] = result
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def clear(self):
        self._memory.clear()
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.pkl"):
                try:
                    f.unlink()
                except Exception:
                    pass

    def size(self) -> int:
        return len(self._memory)


# ──────────────────────────────────────────────────────────────────
# TIMEOUT RUNNER
# ──────────────────────────────────────────────────────────────────

class TimeoutRunner:
    """Runs a clustering algorithm with wall-clock timeout."""

    def __init__(self, timeout_seconds: int = 120):
        self.timeout = timeout_seconds

    def run(self, model: Any, X: np.ndarray, algo_name: str,
            params: Dict[str, Any]) -> SingleRunResult:
        result_holder: Dict[str, Any] = {"result": None, "error": None}
        display_name = algo_name

        meta = REGISTRY.get(algo_name)
        if meta:
            display_name = meta.display_name

        def _execute():
            try:
                t0 = time.perf_counter()
                if hasattr(model, "fit_predict"):
                    labels = model.fit_predict(X)
                elif hasattr(model, "fit"):
                    model.fit(X)
                    if hasattr(model, "labels_"):
                        labels = model.labels_
                    elif hasattr(model, "predict"):
                        labels = model.predict(X)
                    else:
                        labels = np.zeros(len(X), dtype=int)
                else:
                    raise ValueError(f"Model {algo_name} has no fit/fit_predict")
                elapsed = time.perf_counter() - t0

                labels = np.asarray(labels, dtype=int)
                n_clusters = len(set(labels) - {-1})
                n_noise = int((labels == -1).sum())
                centers = getattr(model, "cluster_centers_", None)
                if centers is None:
                    centers = getattr(model, "centers_", None)
                probabilities = None
                if hasattr(model, "predict_proba"):
                    try:
                        probabilities = model.predict_proba(X)
                    except Exception:
                        pass
                elif hasattr(model, "membership_"):
                    probabilities = model.membership_
                inertia = getattr(model, "inertia_", None)

                result_holder["result"] = SingleRunResult(
                    algorithm_name=algo_name, display_name=display_name,
                    labels=labels, n_clusters_found=n_clusters, n_noise=n_noise,
                    params_used=params, fit_time_seconds=round(elapsed, 4),
                    status=RunStatus.SUCCESS, model=model,
                    centers=centers, probabilities=probabilities, inertia=inertia,
                )
            except Exception as e:
                result_holder["error"] = str(e)

        thread = threading.Thread(target=_execute, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            return SingleRunResult(
                algorithm_name=algo_name, display_name=display_name,
                labels=np.full(len(X), -1, dtype=int),
                n_clusters_found=0, n_noise=len(X),
                params_used=params, fit_time_seconds=float(self.timeout),
                status=RunStatus.TIMEOUT,
                error_message=f"Timeout after {self.timeout}s",
            )
        if result_holder["error"]:
            return SingleRunResult(
                algorithm_name=algo_name, display_name=display_name,
                labels=np.full(len(X), -1, dtype=int),
                n_clusters_found=0, n_noise=len(X),
                params_used=params, fit_time_seconds=0.0,
                status=RunStatus.FAILED,
                error_message=result_holder["error"],
            )
        return result_holder["result"]


# ──────────────────────────────────────────────────────────────────
# ELBOW & KNEEDLE FINDER
# ──────────────────────────────────────────────────────────────────

class ElbowFinder:
    """Finds optimal k using the Kneedle algorithm on inertia/metric curves."""

    @staticmethod
    def find_elbow(k_values: List[int], scores: List[float],
                   direction: str = "decreasing",
                   sensitivity: float = 1.0) -> int:
        if len(k_values) < 3:
            return k_values[0] if k_values else 2
        k_arr = np.array(k_values, dtype=float)
        s_arr = np.array(scores, dtype=float)
        k_norm = (k_arr - k_arr.min()) / max(k_arr.max() - k_arr.min(), 1e-16)
        s_norm = (s_arr - s_arr.min()) / max(s_arr.max() - s_arr.min(), 1e-16)
        if direction == "decreasing":
            diff = k_norm - s_norm
        else:
            diff = s_norm - k_norm
        elbow_idx = int(np.argmax(diff))
        return int(k_values[elbow_idx])

    @staticmethod
    def compute_inertia_curve(X: np.ndarray, k_range: Tuple[int, int],
                              random_state: int = 42,
                              callback: Optional[Callable] = None) -> Tuple[List[int], List[float]]:
        k_values = list(range(k_range[0], k_range[1] + 1))
        inertias = []
        for i, k in enumerate(k_values):
            km = KMeans(n_clusters=k, n_init=5, max_iter=200,
                        random_state=random_state)
            km.fit(X)
            inertias.append(float(km.inertia_))
            if callback:
                callback(f"Elbow k={k}", (i + 1) / len(k_values))
        return k_values, inertias

    @staticmethod
    def compute_silhouette_curve(X: np.ndarray, k_range: Tuple[int, int],
                                 random_state: int = 42, sample_size: int = 5000,
                                 callback: Optional[Callable] = None) -> Tuple[List[int], List[float]]:
        k_values = list(range(k_range[0], k_range[1] + 1))
        scores = []
        n_sample = min(sample_size, len(X))
        for i, k in enumerate(k_values):
            km = KMeans(n_clusters=k, n_init=5, max_iter=200,
                        random_state=random_state)
            labels = km.fit_predict(X)
            try:
                sc = silhouette_score(X, labels, sample_size=n_sample)
            except Exception:
                sc = -1.0
            scores.append(float(sc))
            if callback:
                callback(f"Silhouette k={k}", (i + 1) / len(k_values))
        return k_values, scores


# ──────────────────────────────────────────────────────────────────
# HYPERPARAMETER SWEEP
# ──────────────────────────────────────────────────────────────────

class HyperparamSweep:
    """Grid sweep over k-range and/or algorithm-specific params."""

    def __init__(self, registry: AlgorithmRegistry, config: RunConfig):
        self.registry = registry
        self.config = config
        self.runner = TimeoutRunner(config.timeout_seconds)

    def sweep_k(self, X: np.ndarray, algo_name: str,
                k_range: Tuple[int, int], data_hash: str = "",
                callback: Optional[Callable] = None) -> List[SweepPoint]:
        points: List[SweepPoint] = []
        k_vals = list(range(k_range[0], k_range[1] + 1))
        base_params = self.registry.get_default_params(algo_name)
        base_params["random_state"] = self.config.random_state
        n_sample = min(5000, len(X))

        for i, k in enumerate(k_vals):
            params = {**base_params, "n_clusters": k}
            try:
                model = self.registry.build(algo_name, params)
                result = self.runner.run(model, X, algo_name, params)
                if result.status == RunStatus.SUCCESS and result.n_clusters_found >= 2:
                    try:
                        metric_val = silhouette_score(
                            X, result.labels, sample_size=n_sample
                        )
                    except Exception:
                        metric_val = -1.0
                    points.append(SweepPoint(
                        k=k, algorithm=algo_name, params=params,
                        labels=result.labels, n_clusters_found=result.n_clusters_found,
                        metric_value=float(metric_val),
                        metric_name=self.config.sweep_metric,
                        fit_time=result.fit_time_seconds,
                    ))
            except Exception as e:
                logger.warning(f"Sweep k={k} {algo_name} failed: {e}")
            if callback:
                callback(f"Sweep {algo_name} k={k}", (i + 1) / len(k_vals))

        return points

    def sweep_params(self, X: np.ndarray, algo_name: str,
                     param_grid: Dict[str, List[Any]],
                     callback: Optional[Callable] = None) -> List[SweepPoint]:
        import itertools
        keys = list(param_grid.keys())
        vals = list(param_grid.values())
        combos = list(itertools.product(*vals))
        if len(combos) > 100:
            rng = np.random.RandomState(self.config.random_state)
            idx = rng.choice(len(combos), 100, replace=False)
            combos = [combos[i] for i in idx]

        points: List[SweepPoint] = []
        n_sample = min(5000, len(X))
        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            params["random_state"] = self.config.random_state
            try:
                model = self.registry.build(algo_name, params)
                result = self.runner.run(model, X, algo_name, params)
                if result.status == RunStatus.SUCCESS and result.n_clusters_found >= 2:
                    try:
                        metric_val = silhouette_score(
                            X, result.labels, sample_size=n_sample
                        )
                    except Exception:
                        metric_val = -1.0
                    k_val = params.get("n_clusters", result.n_clusters_found)
                    points.append(SweepPoint(
                        k=k_val, algorithm=algo_name, params=params,
                        labels=result.labels, n_clusters_found=result.n_clusters_found,
                        metric_value=float(metric_val),
                        metric_name=self.config.sweep_metric,
                        fit_time=result.fit_time_seconds,
                    ))
            except Exception as e:
                logger.warning(f"Param sweep {algo_name} failed: {e}")
            if callback:
                callback(f"Param sweep {algo_name}", (i + 1) / len(combos))

        return points


# ──────────────────────────────────────────────────────────────────
# PARALLEL BATCH RUNNER
# ──────────────────────────────────────────────────────────────────

class ParallelRunner:
    """Runs multiple algorithms in parallel with progress tracking."""

    def __init__(self, registry: AlgorithmRegistry, config: RunConfig):
        self.registry = registry
        self.config = config
        self.cache = ResultCache(config.cache_dir) if config.enable_cache else None
        self.timeout_runner = TimeoutRunner(config.timeout_seconds)

    def _compute_data_hash(self, X: np.ndarray) -> str:
        sample = X[:min(500, len(X))].tobytes()
        return hashlib.md5(sample).hexdigest()[:16]

    def run_single(self, X: np.ndarray, algo_name: str,
                   params: Optional[Dict[str, Any]] = None) -> SingleRunResult:
        meta = self.registry.get(algo_name)
        if meta is None:
            return SingleRunResult(
                algorithm_name=algo_name, display_name=algo_name,
                labels=np.full(len(X), -1), n_clusters_found=0, n_noise=len(X),
                params_used=params or {}, fit_time_seconds=0.0,
                status=RunStatus.FAILED, error_message=f"Unknown algorithm: {algo_name}",
            )
        final_params = self.registry.get_default_params(algo_name)
        final_params["random_state"] = self.config.random_state
        if meta.requires_n_clusters:
            final_params["n_clusters"] = self.config.n_clusters
        if params:
            final_params.update(params)
        data_hash = self._compute_data_hash(X)
        if self.cache:
            cached = self.cache.get(algo_name, final_params, data_hash)
            if cached:
                return cached
        try:
            model = self.registry.build(algo_name, final_params)
        except Exception as e:
            return SingleRunResult(
                algorithm_name=algo_name, display_name=meta.display_name,
                labels=np.full(len(X), -1), n_clusters_found=0, n_noise=len(X),
                params_used=final_params, fit_time_seconds=0.0,
                status=RunStatus.FAILED, error_message=f"Build error: {e}",
            )
        result = self.timeout_runner.run(model, X, algo_name, final_params)
        if self.cache and result.status == RunStatus.SUCCESS:
            self.cache.put(algo_name, final_params, data_hash, result)
        return result

    def run_batch(self, X: np.ndarray,
                  algorithms: Optional[List[str]] = None,
                  params_override: Optional[Dict[str, Dict]] = None,
                  callback: Optional[Callable] = None) -> BatchResult:
        t0 = time.perf_counter()
        algo_list = algorithms or self.config.algorithms
        overrides = params_override or self.config.params_override
        results: List[SingleRunResult] = []
        sweep_points: List[SweepPoint] = []
        n_total = len(algo_list)

        for i, algo_name in enumerate(algo_list):
            override = overrides.get(algo_name, {})
            result = self.run_single(X, algo_name, override)
            results.append(result)
            if callback:
                pct = (i + 1) / n_total
                callback(f"Completed {algo_name}", pct)
            if self.config.verbose and result.status != RunStatus.SUCCESS:
                logger.warning(f"{algo_name}: {result.status.value} — {result.error_message}")

        if self.config.sweep_mode and self.config.k_range:
            sweeper = HyperparamSweep(self.registry, self.config)
            for algo_name in algo_list:
                meta = self.registry.get(algo_name)
                if meta and meta.requires_n_clusters:
                    pts = sweeper.sweep_k(X, algo_name, self.config.k_range,
                                          callback=callback)
                    sweep_points.extend(pts)

        best_algo = None
        best_score = -np.inf
        for r in results:
            if r.status == RunStatus.SUCCESS and r.n_clusters_found >= 2:
                try:
                    sc = silhouette_score(X, r.labels,
                                          sample_size=min(5000, len(X)))
                    if sc > best_score:
                        best_score = sc
                        best_algo = r.algorithm_name
                except Exception:
                    pass

        elapsed = time.perf_counter() - t0
        n_failed = sum(1 for r in results if r.status != RunStatus.SUCCESS)

        return BatchResult(
            results=results, sweep_points=sweep_points,
            best_algorithm=best_algo, best_score=float(best_score),
            total_time_seconds=round(elapsed, 3),
            n_algorithms_run=len(results), n_algorithms_failed=n_failed,
            config=self.config,
        )


# ──────────────────────────────────────────────────────────────────
# LABEL POST-PROCESSING
# ──────────────────────────────────────────────────────────────────

class LabelPostProcessor:
    """Post-process cluster labels for consistency and analysis."""

    @staticmethod
    def relabel_by_size(labels: np.ndarray) -> np.ndarray:
        """Relabel clusters so that cluster 0 is the largest, etc."""
        unique = [l for l in np.unique(labels) if l >= 0]
        if not unique:
            return labels
        counts = [(l, (labels == l).sum()) for l in unique]
        counts.sort(key=lambda x: -x[1])
        mapping = {old: new for new, (old, _) in enumerate(counts)}
        mapping[-1] = -1
        return np.array([mapping.get(l, l) for l in labels])

    @staticmethod
    def separate_noise(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (clean_mask, noise_mask)."""
        noise_mask = labels == -1
        return ~noise_mask, noise_mask

    @staticmethod
    def merge_small_clusters(labels: np.ndarray, min_size: int = 5) -> np.ndarray:
        """Merge clusters smaller than min_size into nearest large cluster centroids."""
        result = labels.copy()
        unique = [l for l in np.unique(labels) if l >= 0]
        small = [l for l in unique if (labels == l).sum() < min_size]
        if not small:
            return result
        large = [l for l in unique if l not in small]
        if not large:
            return result
        for s in small:
            result[result == s] = large[0]
        return LabelPostProcessor.relabel_by_size(result)

    @staticmethod
    def compute_cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
        unique, counts = np.unique(labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    @staticmethod
    def compute_label_entropy(labels: np.ndarray) -> float:
        """Shannon entropy of cluster assignment distribution."""
        counts = np.bincount(labels[labels >= 0])
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-16)))

    @staticmethod
    def compute_balance_score(labels: np.ndarray) -> float:
        """How balanced are cluster sizes? 1.0 = perfectly balanced."""
        sizes = [int((labels == l).sum()) for l in np.unique(labels) if l >= 0]
        if not sizes:
            return 0.0
        return float(min(sizes) / max(max(sizes), 1))


# ──────────────────────────────────────────────────────────────────
# CLUSTERING ORCHESTRATOR — TOP-LEVEL API
# ──────────────────────────────────────────────────────────────────

class ClusteringOrchestrator:
    """Top-level API for the frontend. Wires together all components."""

    def __init__(self, config: Optional[RunConfig] = None):
        self.config = config or RunConfig()
        self.registry = REGISTRY
        self.runner = ParallelRunner(self.registry, self.config)
        self.elbow_finder = ElbowFinder()
        self.post_processor = LabelPostProcessor()
        self._last_batch_result: Optional[BatchResult] = None

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.runner = ParallelRunner(self.registry, self.config)

    def run(self, X: np.ndarray,
            algorithms: Optional[List[str]] = None,
            callback: Optional[Callable] = None) -> BatchResult:
        result = self.runner.run_batch(X, algorithms=algorithms, callback=callback)
        for r in result.results:
            if r.status == RunStatus.SUCCESS:
                r.labels = self.post_processor.relabel_by_size(r.labels)
        self._last_batch_result = result
        return result

    def run_elbow(self, X: np.ndarray, k_range: Tuple[int, int] = (2, 15),
                  callback: Optional[Callable] = None) -> Dict[str, Any]:
        k_vals, inertias = self.elbow_finder.compute_inertia_curve(
            X, k_range, self.config.random_state, callback
        )
        k_sil, sil_scores = self.elbow_finder.compute_silhouette_curve(
            X, k_range, self.config.random_state, callback=callback
        )
        optimal_k_inertia = self.elbow_finder.find_elbow(k_vals, inertias, "decreasing")
        optimal_k_silhouette = k_sil[int(np.argmax(sil_scores))]
        return {
            "k_values": k_vals,
            "inertias": inertias,
            "silhouette_scores": sil_scores,
            "optimal_k_inertia": optimal_k_inertia,
            "optimal_k_silhouette": optimal_k_silhouette,
            "recommended_k": optimal_k_silhouette,
        }

    def run_sweep(self, X: np.ndarray, algo_name: str,
                  k_range: Tuple[int, int] = (2, 15),
                  callback: Optional[Callable] = None) -> List[SweepPoint]:
        sweeper = HyperparamSweep(self.registry, self.config)
        return sweeper.sweep_k(X, algo_name, k_range, callback=callback)

    def get_last_result(self) -> Optional[BatchResult]:
        return self._last_batch_result

    def list_algorithms(self) -> List[Dict]:
        return self.registry.get_summary_table()

    def recommend_algorithms(self, n_samples: int, n_features: int,
                             **kwargs) -> List[Tuple[AlgorithmMeta, float]]:
        return self.registry.recommend(n_samples, n_features, **kwargs)

    def clear_cache(self):
        if self.runner.cache:
            self.runner.cache.clear()

    def get_algorithm_info(self, name: str) -> Optional[AlgorithmMeta]:
        return self.registry.get(name)

    def get_default_params(self, name: str) -> Dict[str, Any]:
        return self.registry.get_default_params(name)

    @property
    def available_algorithms(self) -> List[str]:
        return self.registry.list_names()

    @property
    def algorithm_families(self) -> List[str]:
        return self.registry.list_families()

    def get_results_dataframe(self) -> Optional[pd.DataFrame]:
        if self._last_batch_result is None:
            return None
        rows = []
        for r in self._last_batch_result.results:
            rows.append({
                "Algorithm": r.display_name,
                "Status": r.status.value,
                "Clusters": r.n_clusters_found,
                "Noise": r.n_noise,
                "Time (s)": r.fit_time_seconds,
                "Error": r.error_message or "",
            })
        return pd.DataFrame(rows)

    def get_best_result(self) -> Optional[SingleRunResult]:
        """Return the best single result by silhouette from the last batch."""
        if self._last_batch_result is None:
            return None
        best = None
        best_name = self._last_batch_result.best_algorithm
        for r in self._last_batch_result.results:
            if r.algorithm_name == best_name:
                best = r
                break
        return best

    def export_labels(self, result: SingleRunResult, index: Optional[pd.Index] = None) -> pd.DataFrame:
        """Export cluster labels as a DataFrame."""
        df = pd.DataFrame({"cluster_label": result.labels})
        if index is not None and len(index) == len(result.labels):
            df.index = index
        df["is_noise"] = result.labels == -1
        return df


# ──────────────────────────────────────────────────────────────────
# DATA SUBSAMPLER — Intelligent sampling for large datasets
# ──────────────────────────────────────────────────────────────────

class DataSubsampler:
    """Provides intelligent subsampling strategies for large datasets."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def uniform_sample(self, X: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform random subsampling with index tracking."""
        rng = np.random.RandomState(self.random_state)
        if n_samples >= len(X):
            return X, np.arange(len(X))
        idx = rng.choice(len(X), n_samples, replace=False)
        return X[idx], idx

    def stratified_sample(self, X: np.ndarray, labels: np.ndarray,
                          n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Stratified subsampling preserving cluster proportions."""
        rng = np.random.RandomState(self.random_state)
        if n_samples >= len(X):
            return X, np.arange(len(X))
        unique = np.unique(labels)
        counts = {u: (labels == u).sum() for u in unique}
        total = sum(counts.values())
        selected_idx = []
        for u in unique:
            n_from_u = max(1, int(n_samples * counts[u] / total))
            u_idx = np.where(labels == u)[0]
            if len(u_idx) > n_from_u:
                chosen = rng.choice(u_idx, n_from_u, replace=False)
            else:
                chosen = u_idx
            selected_idx.extend(chosen)
        idx = np.array(selected_idx)
        return X[idx], idx

    def density_sample(self, X: np.ndarray, n_samples: int,
                       n_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Density-based subsampling: keeps more points from sparse regions."""
        from sklearn.neighbors import NearestNeighbors
        rng = np.random.RandomState(self.random_state)
        if n_samples >= len(X):
            return X, np.arange(len(X))
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(X) - 1))
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        mean_dists = dists.mean(axis=1)
        probs = mean_dists / mean_dists.sum()
        idx = rng.choice(len(X), n_samples, replace=False, p=probs)
        return X[idx], idx

    def geometric_sketch(self, X: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Geometric sketching: farthest-point subsampling for maximum coverage."""
        rng = np.random.RandomState(self.random_state)
        if n_samples >= len(X):
            return X, np.arange(len(X))
        selected = [rng.randint(0, len(X))]
        min_dists = np.full(len(X), np.inf)
        for _ in range(n_samples - 1):
            last = X[selected[-1]]
            dists = np.linalg.norm(X - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected] = -1
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)
        idx = np.array(selected)
        return X[idx], idx

    def coreset_sample(self, X: np.ndarray, n_samples: int,
                       n_init_clusters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Lightweight coreset construction using K-Means++ seeding."""
        rng = np.random.RandomState(self.random_state)
        if n_samples >= len(X):
            return X, np.arange(len(X))
        n_proto = min(n_init_clusters, n_samples // 2, len(X))
        km = KMeans(n_clusters=n_proto, n_init=1, max_iter=50,
                    random_state=self.random_state)
        km.fit(X)
        per_cluster = max(1, n_samples // n_proto)
        selected_idx = []
        for c in range(n_proto):
            c_idx = np.where(km.labels_ == c)[0]
            if len(c_idx) == 0:
                continue
            dists = np.linalg.norm(X[c_idx] - km.cluster_centers_[c], axis=1)
            n_pick = min(per_cluster, len(c_idx))
            sorted_idx = c_idx[np.argsort(dists)]
            selected_idx.extend(sorted_idx[:n_pick].tolist())
        idx = np.array(selected_idx[:n_samples])
        return X[idx], idx


# ──────────────────────────────────────────────────────────────────
# RUN HISTORY — Tracks all clustering runs
# ──────────────────────────────────────────────────────────────────

class RunHistory:
    """Tracks and compares all clustering runs in a session."""

    def __init__(self):
        self._history: List[Dict[str, Any]] = []
        self._run_counter = 0

    def record(self, batch_result: BatchResult, tag: str = ""):
        """Record a batch result into history."""
        self._run_counter += 1
        for r in batch_result.results:
            entry = {
                "run_id": self._run_counter,
                "tag": tag,
                "timestamp": time.time(),
                "algorithm": r.algorithm_name,
                "display_name": r.display_name,
                "status": r.status.value,
                "n_clusters": r.n_clusters_found,
                "n_noise": r.n_noise,
                "fit_time": r.fit_time_seconds,
                "params": dict(r.params_used),
                "error": r.error_message,
            }
            if r.status == RunStatus.SUCCESS and r.n_clusters_found >= 2:
                try:
                    from sklearn.metrics import silhouette_score as ss
                    entry["silhouette"] = None
                except Exception:
                    entry["silhouette"] = None
            self._history.append(entry)

    def to_dataframe(self) -> pd.DataFrame:
        """Export full history as DataFrame."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history)

    def get_best_by_metric(self, metric: str = "silhouette") -> Optional[Dict]:
        """Return the best run entry by a given metric."""
        valid = [h for h in self._history if h.get(metric) is not None]
        if not valid:
            return None
        return max(valid, key=lambda x: x[metric])

    def compare_runs(self, run_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Compare specific runs or all runs side-by-side."""
        df = self.to_dataframe()
        if df.empty:
            return df
        if run_ids:
            df = df[df["run_id"].isin(run_ids)]
        return df.sort_values("run_id")

    def clear(self):
        self._history.clear()
        self._run_counter = 0

    @property
    def n_runs(self) -> int:
        return self._run_counter

    @property
    def n_entries(self) -> int:
        return len(self._history)


# ──────────────────────────────────────────────────────────────────
# CONVERGENCE DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────

class ConvergenceDiagnostics:
    """Analyzes KMeans convergence: inertia trajectory, center drift, label stability."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def trace_inertia(self, X: np.ndarray, n_clusters: int,
                      max_iter: int = 100) -> Dict[str, Any]:
        """Run KMeans step-by-step recording inertia at each iteration."""
        rng = np.random.RandomState(self.random_state)
        n, d = X.shape
        idx = rng.choice(n, n_clusters, replace=False)
        centers = X[idx].copy()
        inertia_trace = []
        center_drift = []
        label_changes = []
        prev_labels = None

        for iteration in range(max_iter):
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            inertia = 0.0
            new_centers = np.zeros_like(centers)
            for c in range(n_clusters):
                mask = labels == c
                if mask.sum() > 0:
                    new_centers[c] = X[mask].mean(axis=0)
                    inertia += np.sum((X[mask] - new_centers[c]) ** 2)
                else:
                    new_centers[c] = centers[c]
            inertia_trace.append(float(inertia))
            drift = float(np.linalg.norm(new_centers - centers))
            center_drift.append(drift)
            if prev_labels is not None:
                changes = int((labels != prev_labels).sum())
                label_changes.append(changes)
            else:
                label_changes.append(n)
            prev_labels = labels.copy()
            centers = new_centers.copy()
            if drift < 1e-10:
                break

        return {
            "inertia_trace": inertia_trace,
            "center_drift": center_drift,
            "label_changes": label_changes,
            "n_iterations": len(inertia_trace),
            "final_inertia": inertia_trace[-1] if inertia_trace else 0.0,
            "converged": center_drift[-1] < 1e-10 if center_drift else False,
            "final_labels": labels,
            "final_centers": centers,
        }

    def multi_init_comparison(self, X: np.ndarray, n_clusters: int,
                              n_inits: int = 10) -> Dict[str, Any]:
        """Run KMeans with multiple initializations and compare outcomes."""
        results = []
        for init in range(n_inits):
            km = KMeans(n_clusters=n_clusters, n_init=1, max_iter=300,
                        random_state=self.random_state + init)
            km.fit(X)
            results.append({
                "init_seed": self.random_state + init,
                "inertia": float(km.inertia_),
                "n_iter": km.n_iter_,
                "labels": km.labels_.copy(),
                "centers": km.cluster_centers_.copy(),
            })
        inertias = [r["inertia"] for r in results]
        best_idx = int(np.argmin(inertias))
        return {
            "results": results,
            "best_init_index": best_idx,
            "best_inertia": inertias[best_idx],
            "worst_inertia": max(inertias),
            "inertia_std": float(np.std(inertias)),
            "inertia_range": max(inertias) - min(inertias),
            "all_inertias": inertias,
        }


# ──────────────────────────────────────────────────────────────────
# MULTI-OBJECTIVE SWEEP
# ──────────────────────────────────────────────────────────────────

class MultiObjectiveSweep:
    """Sweep k-range optimizing multiple metrics simultaneously (Pareto front)."""

    def __init__(self, registry: AlgorithmRegistry, config: RunConfig):
        self.registry = registry
        self.config = config
        self.runner = TimeoutRunner(config.timeout_seconds)

    def sweep(self, X: np.ndarray, algo_name: str,
              k_range: Tuple[int, int],
              metrics: Optional[List[str]] = None,
              callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run sweep and compute multiple metrics per k."""
        from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
        if metrics is None:
            metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]
        k_values = list(range(k_range[0], k_range[1] + 1))
        base_params = self.registry.get_default_params(algo_name)
        base_params["random_state"] = self.config.random_state
        n_sample = min(5000, len(X))
        results_per_k = []
        for i, k in enumerate(k_values):
            params = {**base_params, "n_clusters": k}
            try:
                model = self.registry.build(algo_name, params)
                result = self.runner.run(model, X, algo_name, params)
                row = {"k": k, "status": result.status.value}
                if result.status == RunStatus.SUCCESS and result.n_clusters_found >= 2:
                    labels = result.labels
                    if "silhouette" in metrics:
                        try:
                            row["silhouette"] = float(silhouette_score(X, labels, sample_size=n_sample))
                        except Exception:
                            row["silhouette"] = -1.0
                    if "davies_bouldin" in metrics:
                        try:
                            row["davies_bouldin"] = float(davies_bouldin_score(X, labels))
                        except Exception:
                            row["davies_bouldin"] = 999.0
                    if "calinski_harabasz" in metrics:
                        try:
                            row["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
                        except Exception:
                            row["calinski_harabasz"] = 0.0
                    row["fit_time"] = result.fit_time_seconds
                    row["labels"] = labels
                results_per_k.append(row)
            except Exception as e:
                results_per_k.append({"k": k, "status": "failed", "error": str(e)})
            if callback:
                callback(f"Multi-obj sweep k={k}", (i + 1) / len(k_values))

        pareto = self._pareto_front(results_per_k, metrics)
        return {
            "results": results_per_k,
            "pareto_front": pareto,
            "k_values": k_values,
            "metrics_used": metrics,
        }

    def _pareto_front(self, results: List[Dict], metrics: List[str]) -> List[Dict]:
        """Extract Pareto-optimal solutions from sweep results."""
        valid = [r for r in results if r.get("status") == "success"
                 and all(m in r for m in metrics)]
        if not valid:
            return []
        objectives = []
        for r in valid:
            obj = []
            for m in metrics:
                val = r[m]
                if m == "davies_bouldin":
                    obj.append(-val)
                else:
                    obj.append(val)
            objectives.append(obj)
        objs = np.array(objectives)
        pareto = []
        for i in range(len(valid)):
            dominated = False
            for j in range(len(valid)):
                if i == j:
                    continue
                if np.all(objs[j] >= objs[i]) and np.any(objs[j] > objs[i]):
                    dominated = True
                    break
            if not dominated:
                pareto.append(valid[i])
        return pareto


# ──────────────────────────────────────────────────────────────────
# ENSEMBLE RUNNER — Voting-based cluster ensemble
# ──────────────────────────────────────────────────────────────────

class EnsembleRunner:
    """Creates ensemble clustering via majority-vote co-association."""

    def __init__(self, registry: AlgorithmRegistry, config: RunConfig):
        self.registry = registry
        self.config = config
        self.runner = ParallelRunner(registry, config)

    def run_ensemble(self, X: np.ndarray, algorithms: List[str],
                     n_clusters_final: int = 3,
                     callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run multiple algorithms and combine via co-association matrix."""
        from scipy.cluster.hierarchy import linkage, fcluster
        n = len(X)
        co_matrix = np.zeros((n, n), dtype=np.float32)
        n_valid = 0
        individual_labels = {}
        for i, algo in enumerate(algorithms):
            result = self.runner.run_single(X, algo)
            if result.status == RunStatus.SUCCESS:
                labels = result.labels
                individual_labels[algo] = labels
                for ii in range(n):
                    for jj in range(ii + 1, min(n, ii + 500)):
                        if labels[ii] == labels[jj] and labels[ii] >= 0:
                            co_matrix[ii, jj] += 1
                            co_matrix[jj, ii] += 1
                n_valid += 1
            if callback:
                callback(f"Ensemble {algo}", (i + 1) / len(algorithms))
        if n_valid == 0:
            return {"ensemble_labels": np.zeros(n, dtype=int), "n_valid": 0}
        co_matrix /= max(n_valid, 1)
        np.fill_diagonal(co_matrix, 1.0)
        dist_matrix = 1.0 - co_matrix
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.maximum(dist_matrix, 0.0)
        try:
            from scipy.spatial.distance import squareform
            condensed = squareform(dist_matrix, checks=False)
            Z = linkage(condensed, method="average")
            ensemble_labels = fcluster(Z, t=n_clusters_final, criterion="maxclust") - 1
        except Exception:
            from sklearn.cluster import AgglomerativeClustering
            ensemble_labels = AgglomerativeClustering(
                n_clusters=n_clusters_final
            ).fit_predict(dist_matrix)

        return {
            "ensemble_labels": ensemble_labels,
            "co_association_matrix": co_matrix,
            "n_valid_algorithms": n_valid,
            "individual_labels": individual_labels,
            "algorithms_used": algorithms,
        }


# ──────────────────────────────────────────────────────────────────
# BENCHMARK SUITE — Compare algorithms across multiple datasets
# ──────────────────────────────────────────────────────────────────

class BenchmarkSuite:
    """Benchmarks algorithms across multiple synthetic datasets."""

    GENERATORS = {
        "blobs_3": lambda rs: __import__("sklearn.datasets", fromlist=["make_blobs"]).make_blobs(
            n_samples=500, centers=3, n_features=5, random_state=rs),
        "blobs_7": lambda rs: __import__("sklearn.datasets", fromlist=["make_blobs"]).make_blobs(
            n_samples=1000, centers=7, n_features=8, random_state=rs),
        "moons": lambda rs: __import__("sklearn.datasets", fromlist=["make_moons"]).make_moons(
            n_samples=500, noise=0.08, random_state=rs),
        "circles": lambda rs: __import__("sklearn.datasets", fromlist=["make_circles"]).make_circles(
            n_samples=500, noise=0.05, factor=0.5, random_state=rs),
    }

    def __init__(self, orchestrator: ClusteringOrchestrator):
        self.orchestrator = orchestrator

    def run(self, algorithms: List[str],
            datasets: Optional[List[str]] = None,
            callback: Optional[Callable] = None) -> pd.DataFrame:
        """Run all algorithm × dataset combinations and return metric table."""
        from sklearn.metrics import adjusted_rand_score
        ds_names = datasets or list(self.GENERATORS.keys())
        rows = []
        total = len(ds_names) * len(algorithms)
        step = 0
        for ds_name in ds_names:
            gen = self.GENERATORS.get(ds_name)
            if gen is None:
                continue
            X, y_true = gen(42)
            for algo in algorithms:
                step += 1
                try:
                    result = self.orchestrator.runner.run_single(X, algo)
                    row = {
                        "Dataset": ds_name,
                        "Algorithm": algo,
                        "Status": result.status.value,
                        "Clusters Found": result.n_clusters_found,
                        "Time (s)": result.fit_time_seconds,
                    }
                    if result.status == RunStatus.SUCCESS and result.n_clusters_found >= 2:
                        try:
                            row["Silhouette"] = float(silhouette_score(X, result.labels,
                                                                        sample_size=min(5000, len(X))))
                        except Exception:
                            row["Silhouette"] = None
                        mask = result.labels >= 0
                        if mask.sum() >= 2:
                            try:
                                row["ARI"] = float(adjusted_rand_score(y_true[mask], result.labels[mask]))
                            except Exception:
                                row["ARI"] = None
                    rows.append(row)
                except Exception as e:
                    rows.append({"Dataset": ds_name, "Algorithm": algo,
                                 "Status": "error", "Error": str(e)})
                if callback:
                    callback(f"Bench {ds_name}/{algo}", step / total)
        return pd.DataFrame(rows)

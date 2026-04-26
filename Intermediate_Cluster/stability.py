"""
stability.py — ClusterX Robustness & Stability Analysis Engine
===============================================================
Measures how stable a clustering solution is under perturbations:
  • Bootstrap resampling stability
  • Gaussian / Laplacian noise injection
  • Feature dropout stability
  • Subset sampling stability
  • Hyperparameter sensitivity analysis
  • Cluster-level persistence profiling
  • Label agreement surfaces
  • Stability scorecards & rankings

Author: ClusterX Intelligence Lab
"""

from __future__ import annotations

import time
import warnings
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import mode as scipy_mode
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ──────────────────────────────────────────────────────────────────

class PerturbationType(str, Enum):
    BOOTSTRAP       = "bootstrap"
    GAUSSIAN_NOISE  = "gaussian_noise"
    LAPLACIAN_NOISE = "laplacian_noise"
    FEATURE_DROPOUT = "feature_dropout"
    SUBSET_SAMPLING = "subset_sampling"
    LABEL_FLIPPING  = "label_flipping"


class StabilityGrade(str, Enum):
    HIGHLY_STABLE   = "Highly Stable"
    STABLE          = "Stable"
    MODERATELY_STABLE = "Moderately Stable"
    UNSTABLE        = "Unstable"
    HIGHLY_UNSTABLE = "Highly Unstable"


# ──────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────

@dataclass
class PerturbationRun:
    """Single perturbed run result."""
    perturbation_type: PerturbationType
    perturbation_level: float
    run_index: int
    labels: np.ndarray
    n_clusters: int
    ari_score: float               # ARI vs reference
    ami_score: float               # AMI vs reference
    nmi_score: float               # NMI vs reference
    jaccard_mean: float            # Mean Jaccard overlap with reference clusters
    runtime_seconds: float
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class ClusterPersistence:
    """How stable a single cluster is across perturbations."""
    cluster_id: int
    n_members: int
    mean_jaccard: float
    std_jaccard: float
    survival_rate: float          # fraction of runs where cluster exists
    centroid: Optional[np.ndarray]
    is_stable: bool               # mean_jaccard >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "n_members": self.n_members,
            "mean_jaccard": round(self.mean_jaccard, 4),
            "std_jaccard": round(self.std_jaccard, 4),
            "survival_rate": round(self.survival_rate, 4),
            "is_stable": self.is_stable,
        }


@dataclass
class StabilityReport:
    """Full stability report for one algorithm."""
    algorithm_id: str
    algorithm_name: str
    n_perturbations: int
    n_samples: int

    # Aggregate scores per perturbation type
    ari_by_type: Dict[str, float]
    ami_by_type: Dict[str, float]
    nmi_by_type: Dict[str, float]

    # Grand aggregate
    mean_ari: float
    std_ari: float
    mean_ami: float
    mean_nmi: float

    # Per-cluster persistence
    cluster_persistence: List[ClusterPersistence]
    n_stable_clusters: int
    stability_grade: StabilityGrade
    stability_score: float          # 0–100

    # Raw run data
    runs: List[PerturbationRun] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    runtime_total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "algorithm_name": self.algorithm_name,
            "n_perturbations": self.n_perturbations,
            "mean_ari": round(self.mean_ari, 4),
            "std_ari": round(self.std_ari, 4),
            "mean_ami": round(self.mean_ami, 4),
            "mean_nmi": round(self.mean_nmi, 4),
            "stability_score": round(self.stability_score, 2),
            "stability_grade": self.stability_grade.value,
            "n_stable_clusters": self.n_stable_clusters,
            "ari_by_type": {k: round(v, 4) for k, v in self.ari_by_type.items()},
        }

    def summary_row(self) -> Dict[str, Any]:
        return {
            "Algorithm": self.algorithm_name,
            "Stability Score": round(self.stability_score, 1),
            "Grade": self.stability_grade.value,
            "Mean ARI": round(self.mean_ari, 4),
            "ARI Std": round(self.std_ari, 4),
            "Mean AMI": round(self.mean_ami, 4),
            "Mean NMI": round(self.mean_nmi, 4),
            "Stable Clusters": self.n_stable_clusters,
            "Runs": self.n_perturbations,
            "Total Time(s)": round(self.runtime_total, 2),
        }


@dataclass
class StabilityConfig:
    n_bootstrap_runs: int = 10
    bootstrap_fraction: float = 0.80
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.20])
    n_noise_runs_per_level: int = 3
    feature_dropout_rates: List[float] = field(default_factory=lambda: [0.10, 0.20, 0.30])
    subset_fractions: List[float] = field(default_factory=lambda: [0.70, 0.85])
    n_subset_runs_per_fraction: int = 3
    jaccard_stability_threshold: float = 0.60
    ari_stability_threshold: float = 0.60
    max_workers: int = 4
    timeout_per_run: int = 60
    random_state: int = 42
    enable_feature_dropout: bool = True
    enable_subset_sampling: bool = True
    enable_noise_injection: bool = True
    enable_bootstrap: bool = True
    sample_cap: int = 15_000          # Cap dataset for stability runs


# ──────────────────────────────────────────────────────────────────
# PERTURBATION GENERATORS
# ──────────────────────────────────────────────────────────────────

class PerturbationGenerator:
    """Generates perturbed versions of a dataset."""

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    def bootstrap(self, X: np.ndarray,
                  fraction: float = 0.80) -> Tuple[np.ndarray, np.ndarray]:
        """Sample with replacement. Returns (X_boot, original_indices)."""
        n = len(X)
        n_sample = max(int(n * fraction), 2)
        idx = self.rng.choice(n, size=n_sample, replace=True)
        return X[idx], idx

    def gaussian_noise(self, X: np.ndarray,
                       sigma_relative: float = 0.05) -> np.ndarray:
        """Add Gaussian noise proportional to feature std."""
        stds = X.std(axis=0)
        stds = np.where(stds < 1e-10, 1.0, stds)
        noise = self.rng.normal(0, sigma_relative * stds, size=X.shape)
        return X + noise

    def laplacian_noise(self, X: np.ndarray,
                        scale_relative: float = 0.05) -> np.ndarray:
        """Add Laplacian noise proportional to feature std."""
        stds = X.std(axis=0)
        stds = np.where(stds < 1e-10, 1.0, stds)
        noise = self.rng.laplace(0, scale_relative * stds, size=X.shape)
        return X + noise

    def feature_dropout(self, X: np.ndarray,
                        dropout_rate: float = 0.20) -> np.ndarray:
        """Randomly zero out features."""
        if X.shape[1] <= 2:
            return X.copy()
        X_copy = X.copy()
        n_drop = max(1, int(X.shape[1] * dropout_rate))
        drop_idx = self.rng.choice(X.shape[1], size=n_drop, replace=False)
        X_copy[:, drop_idx] = 0.0
        return X_copy

    def subset_sample(self, X: np.ndarray,
                      fraction: float = 0.80) -> Tuple[np.ndarray, np.ndarray]:
        """Sample without replacement (no duplicates)."""
        n = len(X)
        n_sample = max(int(n * fraction), 2)
        idx = self.rng.choice(n, size=n_sample, replace=False)
        return X[idx], idx

    def generate(self, X: np.ndarray,
                 ptype: PerturbationType,
                 level: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Unified generator. Returns (X_perturbed, indices_if_subsample)."""
        if ptype == PerturbationType.BOOTSTRAP:
            Xp, idx = self.bootstrap(X, fraction=level)
            return Xp, idx
        elif ptype == PerturbationType.GAUSSIAN_NOISE:
            return self.gaussian_noise(X, sigma_relative=level), None
        elif ptype == PerturbationType.LAPLACIAN_NOISE:
            return self.laplacian_noise(X, scale_relative=level), None
        elif ptype == PerturbationType.FEATURE_DROPOUT:
            return self.feature_dropout(X, dropout_rate=level), None
        elif ptype == PerturbationType.SUBSET_SAMPLING:
            Xp, idx = self.subset_sample(X, fraction=level)
            return Xp, idx
        else:
            return X.copy(), None


# ──────────────────────────────────────────────────────────────────
# LABEL AGREEMENT METRICS
# ──────────────────────────────────────────────────────────────────

class LabelAgreementMetrics:
    """Computes ARI, AMI, NMI, and per-cluster Jaccard between two labellings."""

    @staticmethod
    def ari(labels_true: np.ndarray,
            labels_pred: np.ndarray) -> float:
        try:
            from sklearn.metrics import adjusted_rand_score
            # Only compare shared indices (noise points excluded)
            mask = (labels_true != -1) & (labels_pred != -1)
            if mask.sum() < 2:
                return 0.0
            return float(adjusted_rand_score(labels_true[mask], labels_pred[mask]))
        except Exception:
            return 0.0

    @staticmethod
    def ami(labels_true: np.ndarray,
            labels_pred: np.ndarray) -> float:
        try:
            from sklearn.metrics import adjusted_mutual_info_score
            mask = (labels_true != -1) & (labels_pred != -1)
            if mask.sum() < 2:
                return 0.0
            return float(adjusted_mutual_info_score(labels_true[mask], labels_pred[mask]))
        except Exception:
            return 0.0

    @staticmethod
    def nmi(labels_true: np.ndarray,
            labels_pred: np.ndarray) -> float:
        try:
            from sklearn.metrics import normalized_mutual_info_score
            mask = (labels_true != -1) & (labels_pred != -1)
            if mask.sum() < 2:
                return 0.0
            return float(normalized_mutual_info_score(labels_true[mask], labels_pred[mask],
                                                       average_method="arithmetic"))
        except Exception:
            return 0.0

    @staticmethod
    def cluster_jaccard(ref_labels: np.ndarray,
                        new_labels: np.ndarray,
                        ref_indices: Optional[np.ndarray] = None
                        ) -> Dict[int, float]:
        """
        Per-cluster Jaccard similarity.
        If ref_indices provided, new_labels are defined on a subset of ref_labels.
        Returns {cluster_id: jaccard_score}.
        """
        unique_ref = [c for c in np.unique(ref_labels) if c != -1]
        jaccard_map = {}

        for c in unique_ref:
            ref_mask = (ref_labels == c)
            if ref_indices is not None:
                # Map new labels back to original indexing
                new_on_ref = np.full(len(ref_labels), -1, dtype=int)
                new_on_ref[ref_indices] = new_labels
                new_mask = (new_on_ref == c)
            else:
                new_mask = (new_labels == c)

            intersection = int((ref_mask & new_mask).sum())
            union = int((ref_mask | new_mask).sum())
            jaccard_map[c] = float(intersection / max(union, 1))

        return jaccard_map

    @staticmethod
    def mean_jaccard(ref_labels: np.ndarray,
                     new_labels: np.ndarray,
                     ref_indices: Optional[np.ndarray] = None) -> float:
        """Best-match mean Jaccard across all cluster pairs."""
        unique_ref  = np.unique(ref_labels[ref_labels != -1])
        unique_new  = np.unique(new_labels[new_labels != -1])

        if len(unique_ref) == 0 or len(unique_new) == 0:
            return 0.0

        # Build membership sets
        if ref_indices is not None:
            ref_idx_set  = {c: set(np.where(ref_labels == c)[0]) for c in unique_ref}
            new_idx_set  = {c: set(ref_indices[new_labels == c])
                            for c in unique_new if (new_labels == c).any()}
        else:
            ref_idx_set  = {c: set(np.where(ref_labels == c)[0]) for c in unique_ref}
            new_idx_set  = {c: set(np.where(new_labels == c)[0]) for c in unique_new}

        # Hungarian-style greedy match
        matched = []
        used_new = set()
        for c_ref, ref_set in ref_idx_set.items():
            best_j, best_score = None, -1.0
            for c_new, new_set in new_idx_set.items():
                if c_new in used_new:
                    continue
                inter = len(ref_set & new_set)
                union = len(ref_set | new_set)
                j = inter / max(union, 1)
                if j > best_score:
                    best_j = c_new
                    best_score = j
            matched.append(best_score)
            if best_j is not None:
                used_new.add(best_j)

        return float(np.mean(matched)) if matched else 0.0


# ──────────────────────────────────────────────────────────────────
# SINGLE-ALGORITHM STABILITY TESTER
# ──────────────────────────────────────────────────────────────────

class AlgorithmStabilityTester:
    """
    Runs a comprehensive stability battery for a single algorithm.
    """

    def __init__(self, config: StabilityConfig):
        self.config = config
        self._gen = PerturbationGenerator(random_state=config.random_state)
        self._metrics = LabelAgreementMetrics()
        self._log: List[str] = []

    def test(self,
             algorithm_id: str,
             X: np.ndarray,
             reference_labels: np.ndarray,
             n_clusters: int) -> StabilityReport:
        """Full stability battery."""
        t_start = time.perf_counter()
        self._log = []

        # Subsample if dataset is too large
        X_work, ref_labels_work, was_sampled = self._maybe_sample(X, reference_labels)
        n, _ = X_work.shape

        self._log.append(
            f"Testing stability for '{algorithm_id}' "
            f"(n={n}, k={n_clusters}, was_sampled={was_sampled})"
        )

        # Build run schedule
        runs_schedule = self._build_schedule(X_work, n_clusters)
        self._log.append(f"Scheduled {len(runs_schedule)} perturbation runs")

        # Execute runs
        completed_runs = self._execute_runs(
            algorithm_id, X_work, ref_labels_work, n_clusters, runs_schedule
        )

        # Compute per-cluster persistence
        persistence = self._compute_persistence(ref_labels_work, completed_runs, X_work)

        # Aggregate statistics
        ari_vals = [r.ari_score for r in completed_runs if r.succeeded]
        ami_vals = [r.ami_score for r in completed_runs if r.succeeded]
        nmi_vals = [r.nmi_score for r in completed_runs if r.succeeded]

        mean_ari = float(np.mean(ari_vals)) if ari_vals else 0.0
        std_ari  = float(np.std(ari_vals))  if ari_vals else 0.0
        mean_ami = float(np.mean(ami_vals)) if ami_vals else 0.0
        mean_nmi = float(np.mean(nmi_vals)) if nmi_vals else 0.0

        # Per-type breakdown
        ari_by_type, ami_by_type, nmi_by_type = self._aggregate_by_type(completed_runs)

        # Stability score (0–100)
        stability_score = self._compute_stability_score(
            mean_ari, std_ari, mean_ami, persistence
        )
        grade = self._assign_grade(stability_score)

        n_stable = sum(1 for p in persistence if p.is_stable)

        total_time = time.perf_counter() - t_start
        self._log.append(
            f"Stability test complete: score={stability_score:.1f}, "
            f"grade={grade.value}, mean_ari={mean_ari:.4f}"
        )

        return StabilityReport(
            algorithm_id=algorithm_id,
            algorithm_name=algorithm_id,
            n_perturbations=len(completed_runs),
            n_samples=n,
            ari_by_type=ari_by_type,
            ami_by_type=ami_by_type,
            nmi_by_type=nmi_by_type,
            mean_ari=mean_ari,
            std_ari=std_ari,
            mean_ami=mean_ami,
            mean_nmi=mean_nmi,
            cluster_persistence=persistence,
            n_stable_clusters=n_stable,
            stability_grade=grade,
            stability_score=stability_score,
            runs=completed_runs,
            runtime_total=total_time,
            warnings=self._log,
        )

    # ── Schedule Builder ──────────────────────────────────────────

    def _build_schedule(self, X: np.ndarray,
                        n_clusters: int) -> List[Tuple[PerturbationType, float, int]]:
        """Returns list of (ptype, level, run_index)."""
        cfg = self.config
        schedule = []

        if cfg.enable_bootstrap:
            for i in range(cfg.n_bootstrap_runs):
                schedule.append((PerturbationType.BOOTSTRAP,
                                  cfg.bootstrap_fraction, i))

        if cfg.enable_noise_injection:
            for level in cfg.noise_levels:
                for i in range(cfg.n_noise_runs_per_level):
                    schedule.append((PerturbationType.GAUSSIAN_NOISE, level, i))
            for level in cfg.noise_levels[:2]:
                for i in range(cfg.n_noise_runs_per_level):
                    schedule.append((PerturbationType.LAPLACIAN_NOISE, level, i))

        if cfg.enable_feature_dropout and X.shape[1] > 3:
            for rate in cfg.feature_dropout_rates:
                for i in range(2):
                    schedule.append((PerturbationType.FEATURE_DROPOUT, rate, i))

        if cfg.enable_subset_sampling:
            for frac in cfg.subset_fractions:
                for i in range(cfg.n_subset_runs_per_fraction):
                    schedule.append((PerturbationType.SUBSET_SAMPLING, frac, i))

        return schedule

    # ── Run Executor ──────────────────────────────────────────────

    def _execute_runs(self,
                      algorithm_id: str,
                      X: np.ndarray,
                      ref_labels: np.ndarray,
                      n_clusters: int,
                      schedule: List[Tuple[PerturbationType, float, int]]
                      ) -> List[PerturbationRun]:
        completed = []

        def _run_one(ptype, level, run_idx) -> PerturbationRun:
            return self._execute_single(
                algorithm_id, X, ref_labels, n_clusters, ptype, level, run_idx
            )

        max_workers = min(self.config.max_workers, len(schedule))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_one, ptype, level, ri): (ptype, level, ri)
                for ptype, level, ri in schedule
            }
            for future in as_completed(futures, timeout=self.config.timeout_per_run * 2):
                try:
                    run = future.result(timeout=self.config.timeout_per_run)
                    completed.append(run)
                except Exception as e:
                    ptype, level, ri = futures[future]
                    completed.append(PerturbationRun(
                        perturbation_type=ptype, perturbation_level=level,
                        run_index=ri, labels=np.array([]), n_clusters=0,
                        ari_score=0.0, ami_score=0.0, nmi_score=0.0,
                        jaccard_mean=0.0, runtime_seconds=0.0, error=str(e)
                    ))

        return completed

    def _execute_single(self,
                         algorithm_id: str,
                         X: np.ndarray,
                         ref_labels: np.ndarray,
                         n_clusters: int,
                         ptype: PerturbationType,
                         level: float,
                         run_idx: int) -> PerturbationRun:
        t0 = time.perf_counter()
        try:
            from clustering_runner import SingleAlgorithmExecutor, RunnerConfig
            from clustering_registry import get_registry

            Xp, idx = self._gen.generate(X, ptype, level)
            config = RunnerConfig(n_clusters=n_clusters, timeout_seconds=self.config.timeout_per_run)
            executor = SingleAlgorithmExecutor(config)
            cr = executor.run(algorithm_id, Xp, {})

            if not cr.succeeded or len(cr.labels) == 0:
                raise RuntimeError(f"Algorithm failed: {cr.error_message}")

            new_labels = cr.labels

            # For subsample-type perturbations, map back to full-array indices
            if idx is not None:
                # ref_labels on full array; new_labels on subset indexed by idx
                ref_sub = ref_labels[idx]
                ari = self._metrics.ari(ref_sub, new_labels)
                ami = self._metrics.ami(ref_sub, new_labels)
                nmi = self._metrics.nmi(ref_sub, new_labels)
                jac = self._metrics.mean_jaccard(ref_sub, new_labels)
            else:
                ari = self._metrics.ari(ref_labels, new_labels)
                ami = self._metrics.ami(ref_labels, new_labels)
                nmi = self._metrics.nmi(ref_labels, new_labels)
                jac = self._metrics.mean_jaccard(ref_labels, new_labels)

            valid = new_labels[new_labels != -1]
            n_cls = len(np.unique(valid)) if len(valid) > 0 else 0
            return PerturbationRun(
                perturbation_type=ptype,
                perturbation_level=level,
                run_index=run_idx,
                labels=new_labels,
                n_clusters=n_cls,
                ari_score=float(ari),
                ami_score=float(ami),
                nmi_score=float(nmi),
                jaccard_mean=float(jac),
                runtime_seconds=time.perf_counter() - t0,
            )
        except Exception as e:
            return PerturbationRun(
                perturbation_type=ptype,
                perturbation_level=level,
                run_index=run_idx,
                labels=np.array([]),
                n_clusters=0,
                ari_score=0.0, ami_score=0.0, nmi_score=0.0, jaccard_mean=0.0,
                runtime_seconds=time.perf_counter() - t0,
                error=str(e),
            )

    # ── Cluster Persistence ───────────────────────────────────────

    def _compute_persistence(self,
                              ref_labels: np.ndarray,
                              runs: List[PerturbationRun],
                              X: np.ndarray) -> List[ClusterPersistence]:
        unique_clusters = [c for c in np.unique(ref_labels) if c != -1]
        valid_runs = [r for r in runs if r.succeeded and len(r.labels) > 0]

        if not valid_runs:
            return [
                ClusterPersistence(c, int((ref_labels == c).sum()), 0.0, 0.0, 0.0, None, False)
                for c in unique_clusters
            ]

        persistence_list = []
        for c in unique_clusters:
            ref_mask = ref_labels == c
            n_members = int(ref_mask.sum())
            centroid = X[ref_mask].mean(axis=0) if n_members > 0 else None

            jaccard_scores = []
            survived = 0
            for run in valid_runs:
                if len(run.labels) == len(ref_labels):
                    # Same-size labels: direct comparison
                    new_mask = run.labels == c
                    inter = int((ref_mask & new_mask).sum())
                    union = int((ref_mask | new_mask).sum())
                    jac = inter / max(union, 1)
                else:
                    # Subset: best-match Jaccard
                    best_jac = 0.0
                    for cc in np.unique(run.labels[run.labels != -1]):
                        new_mask_sub = run.labels == cc
                        # Map via index (approximate for bootstrap)
                        jac_approx = float(min(n_members, new_mask_sub.sum())) / max(
                            n_members + new_mask_sub.sum() - min(n_members, new_mask_sub.sum()), 1
                        )
                        best_jac = max(best_jac, jac_approx)
                    jac = best_jac

                jaccard_scores.append(jac)
                if jac >= self.config.jaccard_stability_threshold:
                    survived += 1

            mean_jac = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
            std_jac  = float(np.std(jaccard_scores))  if jaccard_scores else 0.0
            surv_rate = survived / max(len(valid_runs), 1)

            persistence_list.append(ClusterPersistence(
                cluster_id=c,
                n_members=n_members,
                mean_jaccard=mean_jac,
                std_jaccard=std_jac,
                survival_rate=surv_rate,
                centroid=centroid,
                is_stable=mean_jac >= self.config.jaccard_stability_threshold,
            ))

        return sorted(persistence_list, key=lambda p: -p.mean_jaccard)

    # ── Aggregators ───────────────────────────────────────────────

    @staticmethod
    def _aggregate_by_type(runs: List[PerturbationRun]
                           ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        from collections import defaultdict
        ari_buckets: Dict[str, List[float]] = defaultdict(list)
        ami_buckets: Dict[str, List[float]] = defaultdict(list)
        nmi_buckets: Dict[str, List[float]] = defaultdict(list)

        for run in runs:
            if not run.succeeded:
                continue
            key = run.perturbation_type.value
            ari_buckets[key].append(run.ari_score)
            ami_buckets[key].append(run.ami_score)
            nmi_buckets[key].append(run.nmi_score)

        ari_agg = {k: round(float(np.mean(v)), 4) for k, v in ari_buckets.items()}
        ami_agg = {k: round(float(np.mean(v)), 4) for k, v in ami_buckets.items()}
        nmi_agg = {k: round(float(np.mean(v)), 4) for k, v in nmi_buckets.items()}
        return ari_agg, ami_agg, nmi_agg

    # ── Scoring & Grading ─────────────────────────────────────────

    def _compute_stability_score(self,
                                  mean_ari: float,
                                  std_ari: float,
                                  mean_ami: float,
                                  persistence: List[ClusterPersistence]) -> float:
        """Weighted composite stability score 0–100."""
        score = 0.0
        # ARI contribution (40%)
        score += 40.0 * max(0.0, mean_ari)
        # ARI stability (low std = good, 15%)
        score += 15.0 * max(0.0, 1 - min(std_ari * 3, 1.0))
        # AMI contribution (25%)
        score += 25.0 * max(0.0, mean_ami)
        # Cluster persistence (20%)
        if persistence:
            mean_persist = float(np.mean([p.mean_jaccard for p in persistence]))
            score += 20.0 * mean_persist
        return round(min(100.0, max(0.0, score)), 2)

    @staticmethod
    def _assign_grade(score: float) -> StabilityGrade:
        if score >= 80:
            return StabilityGrade.HIGHLY_STABLE
        elif score >= 65:
            return StabilityGrade.STABLE
        elif score >= 45:
            return StabilityGrade.MODERATELY_STABLE
        elif score >= 25:
            return StabilityGrade.UNSTABLE
        else:
            return StabilityGrade.HIGHLY_UNSTABLE

    # ── Utility ───────────────────────────────────────────────────

    def _maybe_sample(self,
                       X: np.ndarray,
                       labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        cap = self.config.sample_cap
        if len(X) <= cap:
            return X, labels, False
        rng = np.random.default_rng(self.config.random_state)
        idx = rng.choice(len(X), cap, replace=False)
        return X[idx], labels[idx], True

    @property
    def log(self) -> List[str]:
        return self._log


# ──────────────────────────────────────────────────────────────────
# HYPERPARAMETER SENSITIVITY ANALYSER
# ──────────────────────────────────────────────────────────────────

class HyperparamSensitivityAnalyser:
    """
    Measures how sensitive a clustering result is to small
    changes in key hyper-parameters.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def analyse(self,
                algorithm_id: str,
                X: np.ndarray,
                reference_labels: np.ndarray,
                param_perturbations: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        param_perturbations: {param_name: [value1, value2, ...]}
        Returns sensitivity scores per parameter.
        """
        from clustering_runner import SingleAlgorithmExecutor, RunnerConfig
        from clustering_registry import get_registry

        metrics = LabelAgreementMetrics()
        registry = get_registry()
        spec = registry.get(algorithm_id)
        base_params = spec.default_params()
        config = RunnerConfig(n_clusters=base_params.get("n_clusters", 8))
        executor = SingleAlgorithmExecutor(config)

        sensitivity = {}
        for param_name, values in param_perturbations.items():
            aris = []
            for val in values:
                params = base_params.copy()
                params[param_name] = val
                try:
                    cr = executor.run(algorithm_id, X, params)
                    if cr.succeeded and len(cr.labels) == len(X):
                        ari = metrics.ari(reference_labels, cr.labels)
                        aris.append(float(ari))
                except Exception:
                    pass

            if aris:
                sensitivity[param_name] = {
                    "values": values,
                    "ari_scores": [round(a, 4) for a in aris],
                    "mean_ari": round(float(np.mean(aris)), 4),
                    "std_ari": round(float(np.std(aris)), 4),
                    "sensitivity": round(float(np.std(aris)), 4),
                    "is_sensitive": float(np.std(aris)) > 0.1,
                }

        return sensitivity


# ──────────────────────────────────────────────────────────────────
# NOISE-RESPONSE PROFILER
# ──────────────────────────────────────────────────────────────────

class NoiseResponseProfiler:
    """Profiles ARI vs noise level to build a degradation curve."""

    def __init__(self, noise_levels: Optional[List[float]] = None,
                 n_runs: int = 5, random_state: int = 42):
        self.noise_levels = noise_levels or [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
        self.n_runs = n_runs
        self.rng = np.random.default_rng(random_state)
        self._gen = PerturbationGenerator(random_state=random_state)
        self._metrics = LabelAgreementMetrics()

    def profile(self, algorithm_id: str, X: np.ndarray,
                reference_labels: np.ndarray,
                n_clusters: int) -> Dict[str, Any]:
        from clustering_runner import SingleAlgorithmExecutor, RunnerConfig
        config = RunnerConfig(n_clusters=n_clusters)
        executor = SingleAlgorithmExecutor(config)

        mean_aris = []
        std_aris  = []
        for level in self.noise_levels:
            run_aris = []
            for i in range(self.n_runs):
                self._gen.rng = np.random.default_rng(i + int(level * 1000))
                Xp, _ = self._gen.generate(X, PerturbationType.GAUSSIAN_NOISE, level)
                try:
                    cr = executor.run(algorithm_id, Xp, {})
                    if cr.succeeded and len(cr.labels) == len(X):
                        run_aris.append(self._metrics.ari(reference_labels, cr.labels))
                except Exception:
                    pass
            mean_aris.append(float(np.mean(run_aris)) if run_aris else 0.0)
            std_aris.append(float(np.std(run_aris)) if run_aris else 0.0)

        # Compute half-ARI point (where performance drops to 0.5×original)
        half_ari_level = None
        if mean_aris and mean_aris[0] > 0:
            threshold = mean_aris[0] * 0.5
            for i, (lv, ari) in enumerate(zip(self.noise_levels, mean_aris)):
                if ari <= threshold:
                    half_ari_level = lv
                    break

        return {
            "algorithm_id": algorithm_id,
            "noise_levels": self.noise_levels,
            "mean_ari": mean_aris,
            "std_ari": std_aris,
            "half_ari_noise_level": half_ari_level,
            "auc_ari": float(np.trapz(mean_aris, self.noise_levels))
            if len(self.noise_levels) > 1 else 0.0,
        }


# ──────────────────────────────────────────────────────────────────
# MULTI-ALGORITHM STABILITY COMPARATOR
# ──────────────────────────────────────────────────────────────────

class MultiAlgorithmStabilityComparator:
    """
    Runs stability tests for multiple algorithms and produces
    a comparative scorecard.
    """

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()

    def compare(self,
                algorithm_ids: List[str],
                X: np.ndarray,
                clustering_results: Dict[str, Any],
                n_clusters: int,
                progress_callback: Optional[Callable[[str, int, int], None]] = None
                ) -> Dict[str, StabilityReport]:
        """
        clustering_results: {algorithm_id: ClusteringResult}
        Returns {algorithm_id: StabilityReport}
        """
        tester = AlgorithmStabilityTester(self.config)
        reports = {}
        total = len(algorithm_ids)

        for i, aid in enumerate(algorithm_ids, 1):
            if progress_callback:
                progress_callback(aid, i, total)
            cr = clustering_results.get(aid)
            if cr is None or not cr.succeeded or len(cr.labels) == 0:
                continue
            try:
                report = tester.test(aid, X, cr.labels, n_clusters)
                report.algorithm_name = cr.algorithm_name
                reports[aid] = report
            except Exception as e:
                logger.warning(f"Stability test failed for {aid}: {e}")

        return reports

    def build_scorecard(self,
                        reports: Dict[str, StabilityReport]) -> pd.DataFrame:
        """Build a ranked stability scorecard DataFrame."""
        rows = [report.summary_row() for report in reports.values()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.sort_values("Stability Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
        return df

    def noise_degradation_summary(self,
                                   reports: Dict[str, StabilityReport]
                                   ) -> pd.DataFrame:
        """Summary of ARI per noise level per algorithm."""
        rows = []
        for aid, report in reports.items():
            noise_ari = report.ari_by_type.get("gaussian_noise", None)
            base_ari  = report.ari_by_type.get("bootstrap", None)
            row = {
                "Algorithm": report.algorithm_name,
                "Bootstrap ARI": round(base_ari, 4) if base_ari is not None else None,
                "Gaussian Noise ARI": round(noise_ari, 4) if noise_ari is not None else None,
                "Mean ARI": round(report.mean_ari, 4),
                "ARI Std": round(report.std_ari, 4),
            }
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("Mean ARI", ascending=False)


# ──────────────────────────────────────────────────────────────────
# STABILITY VISUALISATION DATA BUILDERS
# ──────────────────────────────────────────────────────────────────

class StabilityVisDataBuilder:
    """
    Builds data structures suitable for Plotly charts
    from stability reports.
    """

    def ari_box_data(self,
                     reports: Dict[str, StabilityReport]
                     ) -> Dict[str, Any]:
        """Data for box-plot of ARI distributions."""
        data = {}
        for aid, report in reports.items():
            aris = [r.ari_score for r in report.runs if r.succeeded]
            data[report.algorithm_name] = aris
        return data

    def persistence_heatmap_data(self,
                                  report: StabilityReport) -> pd.DataFrame:
        """Per-cluster persistence for a single algorithm — heatmap data."""
        rows = [p.to_dict() for p in report.cluster_persistence]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def run_timeline(self,
                     report: StabilityReport) -> pd.DataFrame:
        """Timeline of ARI across all runs, grouped by perturbation type."""
        rows = []
        for run in report.runs:
            rows.append({
                "run_index": run.run_index,
                "perturbation_type": run.perturbation_type.value,
                "perturbation_level": run.perturbation_level,
                "ari_score": run.ari_score if run.succeeded else None,
                "ami_score": run.ami_score if run.succeeded else None,
                "nmi_score": run.nmi_score if run.succeeded else None,
                "n_clusters": run.n_clusters,
                "runtime": run.runtime_seconds,
                "status": "ok" if run.succeeded else "failed",
            })
        return pd.DataFrame(rows)

    def noise_response_chart(self,
                              profiles: Dict[str, Dict]) -> pd.DataFrame:
        """Noise-ARI degradation curves for multiple algorithms."""
        rows = []
        for alg_name, prof in profiles.items():
            for level, mean_ari in zip(prof["noise_levels"], prof["mean_ari"]):
                rows.append({
                    "Algorithm": alg_name,
                    "Noise Level": level,
                    "Mean ARI": mean_ari,
                })
        return pd.DataFrame(rows)

    def stability_radar_data(self,
                              reports: Dict[str, StabilityReport]
                              ) -> Dict[str, Dict[str, float]]:
        """
        Radar chart: each algorithm has scores on:
        bootstrap_ari, noise_ari, subset_ari, feature_dropout_ari,
        cluster_balance_stability
        """
        radar = {}
        for aid, report in reports.items():
            radar[report.algorithm_name] = {
                "Bootstrap ARI":      report.ari_by_type.get("bootstrap", 0.0),
                "Gaussian Noise ARI": report.ari_by_type.get("gaussian_noise", 0.0),
                "Laplacian Noise ARI":report.ari_by_type.get("laplacian_noise", 0.0),
                "Subset ARI":         report.ari_by_type.get("subset_sampling", 0.0),
                "Feature Dropout ARI":report.ari_by_type.get("feature_dropout", 0.0),
                "Cluster Persistence":float(np.mean(
                    [p.mean_jaccard for p in report.cluster_persistence]
                ) if report.cluster_persistence else 0.0),
            }
        return radar


# ──────────────────────────────────────────────────────────────────
# LABEL STABILITY MATRIX
# ──────────────────────────────────────────────────────────────────

class LabelStabilityMatrix:
    """
    Builds a pairwise label agreement matrix (ARI / AMI) between
    all pairs of evaluated clustering solutions.
    """

    def __init__(self):
        self._metrics = LabelAgreementMetrics()

    def build_ari_matrix(self,
                          clustering_results: Dict[str, Any],
                          algorithm_names: Optional[Dict[str, str]] = None
                          ) -> pd.DataFrame:
        """Returns symmetric ARI matrix DataFrame."""
        ids = [k for k, v in clustering_results.items()
               if hasattr(v, 'succeeded') and v.succeeded and len(v.labels) > 0]
        n = len(ids)
        if n < 2:
            return pd.DataFrame()

        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    mat[i, j] = 1.0
                elif i < j:
                    li = clustering_results[ids[i]].labels
                    lj = clustering_results[ids[j]].labels
                    if len(li) == len(lj):
                        ari = self._metrics.ari(li, lj)
                        mat[i, j] = mat[j, i] = round(float(ari), 4)

        names = [(algorithm_names or {}).get(aid, aid)[:30] for aid in ids]
        return pd.DataFrame(mat, index=names, columns=names)

    def build_nmi_matrix(self,
                          clustering_results: Dict[str, Any],
                          algorithm_names: Optional[Dict[str, str]] = None
                          ) -> pd.DataFrame:
        """Returns symmetric NMI matrix DataFrame."""
        ids = [k for k, v in clustering_results.items()
               if hasattr(v, 'succeeded') and v.succeeded and len(v.labels) > 0]
        n = len(ids)
        if n < 2:
            return pd.DataFrame()

        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    mat[i, j] = 1.0
                elif i < j:
                    li = clustering_results[ids[i]].labels
                    lj = clustering_results[ids[j]].labels
                    if len(li) == len(lj):
                        nmi = self._metrics.nmi(li, lj)
                        mat[i, j] = mat[j, i] = round(float(nmi), 4)

        names = [(algorithm_names or {}).get(aid, aid)[:30] for aid in ids]
        return pd.DataFrame(mat, index=names, columns=names)


# ──────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def quick_stability_test(algorithm_id: str,
                          X: np.ndarray,
                          reference_labels: np.ndarray,
                          n_clusters: int,
                          n_bootstrap: int = 5) -> Dict[str, Any]:
    """Fast stability estimate via bootstrap — suitable for real-time display."""
    config = StabilityConfig(
        n_bootstrap_runs=n_bootstrap,
        enable_noise_injection=False,
        enable_feature_dropout=False,
        enable_subset_sampling=False,
    )
    tester = AlgorithmStabilityTester(config)
    report = tester.test(algorithm_id, X, reference_labels, n_clusters)
    return {
        "algorithm_id": algorithm_id,
        "stability_score": report.stability_score,
        "stability_grade": report.stability_grade.value,
        "mean_ari": round(report.mean_ari, 4),
        "std_ari": round(report.std_ari, 4),
        "n_stable_clusters": report.n_stable_clusters,
    }


def get_stability_color(grade: str) -> str:
    """Returns a hex color for the stability grade."""
    grade_colors = {
        StabilityGrade.HIGHLY_STABLE.value:    "#00ff88",
        StabilityGrade.STABLE.value:           "#88ff44",
        StabilityGrade.MODERATELY_STABLE.value: "#ffcc00",
        StabilityGrade.UNSTABLE.value:         "#ff8800",
        StabilityGrade.HIGHLY_UNSTABLE.value:  "#ff3333",
    }
    return grade_colors.get(grade, "#888888")


def make_stability_config(fast: bool = False) -> StabilityConfig:
    """Factory for pre-configured stability configs."""
    if fast:
        return StabilityConfig(
            n_bootstrap_runs=5,
            noise_levels=[0.05, 0.10],
            n_noise_runs_per_level=2,
            feature_dropout_rates=[0.20],
            subset_fractions=[0.80],
            n_subset_runs_per_fraction=2,
            enable_feature_dropout=False,
        )
    return StabilityConfig()


def build_stability_summary(reports: Dict[str, StabilityReport]) -> Dict[str, Any]:
    """High-level summary of stability analysis batch."""
    if not reports:
        return {}
    scores = [r.stability_score for r in reports.values()]
    grades = [r.stability_grade.value for r in reports.values()]
    return {
        "n_algorithms_tested": len(reports),
        "mean_stability_score": round(float(np.mean(scores)), 2),
        "best_algorithm": max(reports.values(), key=lambda r: r.stability_score).algorithm_name,
        "worst_algorithm": min(reports.values(), key=lambda r: r.stability_score).algorithm_name,
        "grade_distribution": {g: grades.count(g) for g in set(grades)},
        "highly_stable_count": sum(1 for g in grades if g == StabilityGrade.HIGHLY_STABLE.value),
    }


# ══════════════════════════════════════════════════════════════════
# FINAL POLISH — ADVANCED STABILITY ADDITIONS
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# CLUSTER-MAP TRACKER — which clusters survive perturbation
# ──────────────────────────────────────────────────────────────────

class ClusterMapTracker:
    """
    Tracks which original clusters are recovered across perturbed runs.
    Produces a cluster survival map: for each run, which reference
    clusters were re-discovered and which were merged/split/lost.
    """
    def __init__(self, jaccard_threshold: float = 0.5):
        self.jaccard_threshold = jaccard_threshold

    def track(self, ref_labels: np.ndarray,
              perturbed_label_list: List[np.ndarray]) -> Dict[str, Any]:
        ref_unique = [c for c in np.unique(ref_labels) if c != -1]
        n_runs = len(perturbed_label_list)
        # survival_map[cluster_id] = list of recovered/merged/split/lost per run
        survival = {c: [] for c in ref_unique}
        for run_labels in perturbed_label_list:
            for c in ref_unique:
                ref_mask = ref_labels == c
                best_j = 0.0
                status = "lost"
                for nc in np.unique(run_labels[run_labels != -1]):
                    new_mask = run_labels == nc
                    if len(new_mask) != len(ref_mask): continue
                    inter = int((ref_mask & new_mask).sum())
                    union = int((ref_mask | new_mask).sum())
                    j = inter / max(union, 1)
                    if j > best_j:
                        best_j = j
                if best_j >= self.jaccard_threshold:
                    status = "recovered"
                elif best_j >= 0.25:
                    status = "partial"
                survival[c].append({"status": status, "jaccard": round(best_j, 3)})

        summary = {}
        for c, runs in survival.items():
            n_recovered = sum(1 for r in runs if r["status"] == "recovered")
            n_partial   = sum(1 for r in runs if r["status"] == "partial")
            n_lost      = sum(1 for r in runs if r["status"] == "lost")
            summary[int(c)] = {
                "recovery_rate": round(n_recovered / max(n_runs, 1), 3),
                "partial_rate":  round(n_partial  / max(n_runs, 1), 3),
                "lost_rate":     round(n_lost     / max(n_runs, 1), 3),
                "mean_jaccard":  round(float(np.mean([r["jaccard"] for r in runs])), 3),
                "is_robust":     (n_recovered / max(n_runs, 1)) >= 0.7,
            }
        robust_count = sum(1 for v in summary.values() if v["is_robust"])
        return {"cluster_map": summary, "n_robust": robust_count,
                "n_total": len(ref_unique),
                "overall_robustness": round(robust_count / max(len(ref_unique), 1), 3)}


# ──────────────────────────────────────────────────────────────────
# PERTURBATION TRAJECTORY ANALYSER
# ──────────────────────────────────────────────────────────────────

class PerturbationTrajectoryAnalyser:
    """
    Analyses how clustering quality degrades along a perturbation trajectory.
    Fits a degradation curve and extracts key statistics:
    - Half-life noise level
    - Degradation rate (slope)
    - Plateau ARI (ARI at maximum noise)
    """
    def analyse_trajectory(self, noise_levels: List[float],
                            mean_aris: List[float]) -> Dict[str, Any]:
        if len(noise_levels) < 3 or len(mean_aris) < 3:
            return {}
        import scipy.optimize as opt
        levels = np.array(noise_levels)
        aris   = np.array(mean_aris)
        # Fit exponential decay: ARI(noise) = a * exp(-b * noise) + c
        try:
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            popt, _ = opt.curve_fit(exp_decay, levels, aris,
                                     p0=[0.8, 5.0, 0.1], maxfev=1000,
                                     bounds=([0, 0, -0.5], [1.5, 50, 1.0]))
            a, b, c = popt
            half_life = float(np.log(2) / max(b, 1e-6))
            plateau = float(c)
            fitted = [round(float(exp_decay(x, *popt)), 4) for x in levels]
        except Exception:
            half_life = None; plateau = None; fitted = list(aris)

        degradation_rate = float((aris[0] - aris[-1]) / max(levels[-1] - levels[0], 1e-6))
        return {
            "initial_ari":     round(float(aris[0]), 4),
            "final_ari":       round(float(aris[-1]), 4),
            "degradation_rate": round(degradation_rate, 4),
            "half_life_noise": round(half_life, 4) if half_life else None,
            "plateau_ari":     round(plateau, 4) if plateau else None,
            "fitted_curve":    fitted,
            "is_resilient":    degradation_rate < 1.0 and (aris[-1] > 0.3 if len(aris) else False),
        }


# ──────────────────────────────────────────────────────────────────
# CLUSTER CONFIDENCE SCORER
# ──────────────────────────────────────────────────────────────────

class ClusterConfidenceScorer:
    """
    Assigns per-point confidence scores for cluster assignments.
    Based on: (1) silhouette value, (2) k-NN label consistency,
              (3) distance to cluster boundary.
    High confidence → point is well inside its cluster.
    Low confidence  → point is near a cluster boundary (ambiguous).
    """
    def __init__(self, n_neighbors: int = 15):
        self.n_neighbors = n_neighbors

    def score(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Returns per-point confidence in [0, 1]."""
        n = len(X)
        confidence = np.zeros(n, dtype=np.float32)
        valid_mask = labels != -1
        if valid_mask.sum() < 4:
            return confidence

        Xv, lv = X[valid_mask], labels[valid_mask]
        valid_idx = np.where(valid_mask)[0]

        # k-NN label consistency
        try:
            from sklearn.neighbors import NearestNeighbors
            k = min(self.n_neighbors, len(Xv) - 1)
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(Xv)
            _, indices = nbrs.kneighbors(Xv)
            knn_conf = np.zeros(len(Xv))
            for i in range(len(Xv)):
                neighbor_labels = lv[indices[i, 1:]]
                knn_conf[i] = float((neighbor_labels == lv[i]).mean())
        except Exception:
            knn_conf = np.ones(len(Xv)) * 0.5

        # Silhouette-based confidence
        sil_conf = np.zeros(len(Xv))
        try:
            from sklearn.metrics import silhouette_samples
            sil = silhouette_samples(Xv, lv)
            sil_conf = (sil + 1) / 2  # Map to [0,1]
        except Exception:
            sil_conf = knn_conf.copy()

        # Combined
        combined = 0.6 * sil_conf + 0.4 * knn_conf
        for i, orig_idx in enumerate(valid_idx):
            confidence[orig_idx] = float(np.clip(combined[i], 0, 1))

        return confidence

    def uncertain_points(self, X: np.ndarray, labels: np.ndarray,
                          threshold: float = 0.4) -> np.ndarray:
        """Returns indices of low-confidence (uncertain) points."""
        conf = self.score(X, labels)
        return np.where(conf < threshold)[0]

    def border_points(self, X: np.ndarray, labels: np.ndarray,
                       threshold: float = 0.5) -> Dict[int, np.ndarray]:
        """Returns {cluster_id: [indices of border points]}."""
        conf = self.score(X, labels)
        border = {}
        for c in np.unique(labels[labels != -1]):
            cluster_mask = labels == c
            low_conf = conf < threshold
            border_idx = np.where(cluster_mask & low_conf)[0]
            border[int(c)] = border_idx
        return border


# ──────────────────────────────────────────────────────────────────
# STABILITY TREND ANALYSER
# ──────────────────────────────────────────────────────────────────

class StabilityTrendAnalyser:
    """
    Analyses if stability changes systematically across
    perturbation levels (increasing noise → decreasing ARI).
    Tests monotonicity and significance of the trend.
    """
    def analyse(self, report: "StabilityReport") -> Dict[str, Any]:
        from scipy.stats import spearmanr, kendalltau
        runs = [r for r in report.runs if r.succeeded]
        if len(runs) < 4:
            return {"monotonic": None, "trend_score": None}

        # Group by perturbation level
        from collections import defaultdict
        by_level = defaultdict(list)
        for r in runs:
            by_level[round(r.perturbation_level, 3)].append(r.ari_score)
        levels = sorted(by_level.keys())
        mean_aris = [float(np.mean(by_level[l])) for l in levels]

        if len(levels) < 3:
            return {"monotonic": None, "trend_score": None}

        try:
            rho, p_rho = spearmanr(levels, mean_aris)
            tau, p_tau = kendalltau(levels, mean_aris)
        except Exception:
            return {"monotonic": None, "trend_score": None}

        is_monotonic = rho < -0.6 and p_rho < 0.1
        trend_score = float(-rho)  # Higher = more stability degradation with noise

        return {
            "levels": [round(l, 3) for l in levels],
            "mean_aris": [round(a, 4) for a in mean_aris],
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(p_rho), 4),
            "kendall_tau": round(float(tau), 4),
            "is_monotonically_degrading": bool(is_monotonic),
            "degradation_strength": round(float(np.clip(trend_score, 0, 1)), 4),
            "interpretation": (
                "Stability degrades predictably with noise." if is_monotonic
                else "Stability pattern is non-monotonic — clusterer may be noise-resistant in some regimes."
            ),
        }


# ──────────────────────────────────────────────────────────────────
# MULTI-RUN CONSENSUS LABELS (Stability-based)
# ──────────────────────────────────────────────────────────────────

class StabilityBasedConsensusLabeler:
    """
    Uses stability runs to build a robust label assignment.
    For each point, takes the majority vote across all stable runs.
    Only marks a point as 'confident' if agreement > threshold.
    """
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    def label(self, n_samples: int,
               runs: List["PerturbationRun"],
               n_clusters: int) -> Dict[str, Any]:
        valid_runs = [r for r in runs
                      if r.succeeded and len(r.labels) == n_samples]
        if not valid_runs:
            return {"labels": np.full(n_samples, -1), "confidence": np.zeros(n_samples)}

        # Majority vote matrix [n_samples x n_clusters]
        vote_matrix = np.zeros((n_samples, n_clusters + 1), dtype=np.float32)
        for r in valid_runs:
            for i in range(n_samples):
                c = int(r.labels[i])
                if 0 <= c < n_clusters:
                    vote_matrix[i, c] += 1
                else:
                    vote_matrix[i, n_clusters] += 1

        vote_matrix /= len(valid_runs)
        final_labels = vote_matrix[:, :n_clusters].argmax(axis=1).astype(np.int32)
        confidence   = vote_matrix[:, :n_clusters].max(axis=1)

        # Points below threshold marked as uncertain (-2)
        uncertain = confidence < self.confidence_threshold
        final_labels[uncertain] = -2

        return {
            "labels": final_labels,
            "confidence": confidence,
            "n_confident": int((~uncertain).sum()),
            "n_uncertain": int(uncertain.sum()),
            "confidence_rate": round(float((~uncertain).mean()), 4),
        }


# ──────────────────────────────────────────────────────────────────
# CROSS-DATASET STABILITY TESTER
# ──────────────────────────────────────────────────────────────────

class CrossDatasetStabilityTester:
    """
    Tests whether a clustering algorithm produces consistent results
    on two related datasets (e.g. train / holdout, or augmented pairs).
    Key question: 'Does this algorithm generalise, or is it overfit
    to the exact training sample?'
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def test(self, algorithm_id: str,
             X_a: np.ndarray, X_b: np.ndarray,
             n_clusters: int) -> Dict[str, Any]:
        from clustering_runner import SingleAlgorithmExecutor, RunnerConfig
        from stability import LabelAgreementMetrics

        config = RunnerConfig(n_clusters=n_clusters, timeout_seconds=120)
        executor = SingleAlgorithmExecutor(config)
        metrics  = LabelAgreementMetrics()

        cr_a = executor.run(algorithm_id, X_a, {})
        cr_b = executor.run(algorithm_id, X_b, {})

        if not cr_a.succeeded or not cr_b.succeeded:
            return {"status": "failed", "error": f"{cr_a.error_message} | {cr_b.error_message}"}

        # Only compare on shared indices if sizes differ
        n_min = min(len(cr_a.labels), len(cr_b.labels))
        la, lb = cr_a.labels[:n_min], cr_b.labels[:n_min]

        ari = metrics.ari(la, lb)
        ami = metrics.ami(la, lb)
        nmi = metrics.nmi(la, lb)
        jac = metrics.mean_jaccard(la, lb)

        ka = int(len(np.unique(la[la != -1])))
        kb = int(len(np.unique(lb[lb != -1])))

        return {
            "algorithm_id": algorithm_id,
            "ari": round(float(ari), 4),
            "ami": round(float(ami), 4),
            "nmi": round(float(nmi), 4),
            "mean_jaccard": round(float(jac), 4),
            "k_a": ka, "k_b": kb,
            "k_agreement": ka == kb,
            "is_stable_across_datasets": float(ari) >= 0.5,
            "interpretation": (
                f"ARI={ari:.3f} between datasets A (n={len(X_a)}) and B (n={len(X_b)}). "
                + ("Stable generalisation." if ari >= 0.5
                   else "Unstable — results differ significantly between datasets.")
            ),
            "status": "ok",
        }


# ──────────────────────────────────────────────────────────────────
# LABEL ENTROPY TRACKER
# ──────────────────────────────────────────────────────────────────

class LabelEntropyTracker:
    """
    Tracks how the entropy of cluster size distributions changes
    across perturbation runs. High entropy variance = unstable cluster
    sizes; low = stable cluster sizes even under perturbation.
    """
    @staticmethod
    def entropy_series(runs: List[PerturbationRun]) -> Dict[str, Any]:
        from scipy.stats import entropy as sp_entropy
        from collections import Counter
        entropies = []
        k_vals    = []
        for run in runs:
            if not run.succeeded or len(run.labels) == 0:
                continue
            valid = run.labels[run.labels != -1]
            if len(valid) == 0:
                continue
            counts = np.array(list(Counter(valid.tolist()).values()), dtype=float)
            p = counts / counts.sum()
            e = float(sp_entropy(p + 1e-12))
            entropies.append(e)
            k_vals.append(len(counts))

        if not entropies:
            return {}
        return {
            "mean_entropy": round(float(np.mean(entropies)), 4),
            "std_entropy":  round(float(np.std(entropies)), 4),
            "mean_k":       round(float(np.mean(k_vals)), 2),
            "std_k":        round(float(np.std(k_vals)), 3),
            "entropy_cv":   round(float(np.std(entropies) / (np.mean(entropies) + 1e-10)), 4),
            "k_cv":         round(float(np.std(k_vals) / (np.mean(k_vals) + 1e-10)), 4),
            "stable_k": float(np.std(k_vals)) < 1.5,
            "interpretation": (
                "Cluster count is stable across perturbations." if float(np.std(k_vals)) < 1.5
                else f"Cluster count varies (std={np.std(k_vals):.1f}) — algorithm is k-sensitive."
            ),
        }


# ──────────────────────────────────────────────────────────────────
# INTER-RUN CONSISTENCY MATRIX
# ──────────────────────────────────────────────────────────────────

class InterRunConsistencyMatrix:
    """
    Computes ARI between every pair of perturbation runs for one algorithm.
    Reveals: are all runs mutually consistent, or does the algorithm
    find different solutions on different perturbations (multi-modal landscape)?
    """
    def __init__(self):
        self._metrics = LabelAgreementMetrics()

    def compute(self, runs: List[PerturbationRun],
                max_runs: int = 20) -> Dict[str, Any]:
        import pandas as pd
        valid = [r for r in runs if r.succeeded and len(r.labels) > 0]
        valid = valid[:max_runs]
        n = len(valid)
        if n < 2:
            return {"matrix": None, "mean_pairwise_ari": None}

        # Pad/trim all to same length
        min_len = min(len(r.labels) for r in valid)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    mat[i, j] = 1.0
                elif i < j:
                    ari = self._metrics.ari(
                        valid[i].labels[:min_len],
                        valid[j].labels[:min_len])
                    mat[i, j] = mat[j, i] = round(float(ari), 4)

        run_names = [f"R{r.run_index}_{r.perturbation_type.value[:4]}" for r in valid]
        df = pd.DataFrame(mat, index=run_names, columns=run_names)
        upper_tri = mat[np.triu_indices(n, k=1)]
        return {
            "matrix": df,
            "mean_pairwise_ari": round(float(upper_tri.mean()), 4),
            "min_pairwise_ari":  round(float(upper_tri.min()), 4),
            "std_pairwise_ari":  round(float(upper_tri.std()), 4),
            "is_multimodal": float(upper_tri.std()) > 0.2,
            "interpretation": (
                "Algorithm consistently finds the same solution. (Unimodal landscape.)"
                if float(upper_tri.std()) < 0.2
                else "Algorithm finds different solutions on different perturbations. "
                     "(Multi-modal landscape — use consensus to combine.)"
            ),
        }


# ──────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE UNDER PERTURBATION
# ──────────────────────────────────────────────────────────────────

class PerturbationFeatureImportance:
    """
    Identifies which features are most responsible for clustering stability.
    Method: run feature dropout for each feature individually; features
    whose removal causes the largest ARI drop are 'clustering-critical'.
    """
    def __init__(self, max_features: int = 20, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state

    def compute(self, algorithm_id: str,
                X: np.ndarray,
                reference_labels: np.ndarray,
                feature_names: Optional[List[str]] = None,
                n_clusters: int = 8) -> Dict[str, Any]:
        from clustering_runner import SingleAlgorithmExecutor, RunnerConfig
        config = RunnerConfig(n_clusters=n_clusters, timeout_seconds=60)
        executor = SingleAlgorithmExecutor(config)
        metrics  = LabelAgreementMetrics()

        d = X.shape[1]
        n_feats = min(d, self.max_features)
        importances = {}
        rng = np.random.default_rng(self.random_state)

        for f in range(n_feats):
            # Zero out feature f
            X_drop = X.copy()
            X_drop[:, f] = rng.normal(0, 1e-6, len(X))
            cr = executor.run(algorithm_id, X_drop, {})
            if cr.succeeded and len(cr.labels) == len(X):
                ari = self._metrics_safe_ari(metrics, reference_labels, cr.labels)
            else:
                ari = 0.0
            fname = feature_names[f] if feature_names and f < len(feature_names) else f"f{f}"
            importances[fname] = round(1.0 - float(ari), 4)

        # Normalise
        vals = list(importances.values())
        max_v = max(vals) if vals else 1.0
        normalised = {k: round(v / max(max_v, 1e-10), 4) for k, v in importances.items()}
        ranked = sorted(normalised.items(), key=lambda x: -x[1])

        return {
            "feature_importance": dict(ranked),
            "top_features": [k for k, _ in ranked[:10]],
            "raw_ari_drop": importances,
            "most_critical": ranked[0][0] if ranked else None,
            "least_critical": ranked[-1][0] if ranked else None,
        }

    @staticmethod
    def _metrics_safe_ari(metrics, la, lb):
        try:
            return float(metrics.ari(la, lb))
        except Exception:
            return 0.0

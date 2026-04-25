"""
consensus.py — ClusterX Advanced Consensus Clustering Engine
=============================================================
Combines multiple clustering solutions into a single robust partition
via co-association matrices, voting, graph-based reclustering,
and Bayesian evidence accumulation.

Implements:
  • Co-Association Matrix (CAM) construction
  • Evidence Accumulation Clustering (EAC)
  • Voting-based consensus
  • Weighted consensus (metric-aware)
  • Cluster-based Similarity Partitioning (CSPA)
  • Meta-clustering consensus
  • Consensus quality scoring
  • Ensemble diversity analysis

Author: ClusterX Intelligence Lab
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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ──────────────────────────────────────────────────────────────────

class ConsensusMethod(str, Enum):
    EAC_AVERAGE          = "eac_average"
    EAC_SINGLE           = "eac_single"
    EAC_COMPLETE         = "eac_complete"
    CSPA                 = "cspa"
    VOTING               = "voting"
    WEIGHTED_VOTING      = "weighted_voting"
    META_CLUSTERING      = "meta_clustering"
    BAYESIAN_COMBINATION = "bayesian"
    HYBRID               = "hybrid"


class LinkageMethod(str, Enum):
    AVERAGE  = "average"
    SINGLE   = "single"
    COMPLETE = "complete"
    WARD     = "ward"


# ──────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────

@dataclass
class ConsensusConfig:
    method: ConsensusMethod = ConsensusMethod.EAC_AVERAGE
    n_clusters: int = 8
    linkage_method: LinkageMethod = LinkageMethod.AVERAGE
    weight_by_metric: bool = True
    metric_for_weighting: str = "composite_score"
    min_cluster_size: int = 3
    soft_threshold: float = 0.5         # Binarise co-association at this value
    use_sparse: bool = True             # Use sparse matrices for large datasets
    random_state: int = 42
    top_k_algorithms: Optional[int] = None   # Use only top-k by metric
    diversity_aware: bool = True        # Prefer diverse ensemble members
    verbose: bool = False


@dataclass
class ConsensusResult:
    method: ConsensusMethod
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    coassoc_matrix: Optional[np.ndarray]   # n×n co-association matrix
    algorithm_weights: Dict[str, float]
    algorithms_used: List[str]
    n_algorithms: int
    runtime_seconds: float
    quality_score: float                    # 0–100 internal quality
    diversity_score: float                  # 0–1 ensemble diversity
    silhouette: Optional[float]
    stability: Optional[float]
    interpretation: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "n_algorithms": self.n_algorithms,
            "algorithms_used": self.algorithms_used,
            "algorithm_weights": {k: round(v, 4) for k, v in self.algorithm_weights.items()},
            "quality_score": round(self.quality_score, 2),
            "diversity_score": round(self.diversity_score, 4),
            "silhouette": round(self.silhouette, 4) if self.silhouette else None,
            "runtime_seconds": round(self.runtime_seconds, 4),
            "interpretation": self.interpretation,
        }


@dataclass
class EnsembleDiversity:
    """Quantifies how diverse the ensemble of clustering solutions is."""
    mean_pairwise_disagreement: float    # 1 - mean ARI between all pairs
    std_pairwise_disagreement: float
    diversity_score: float               # 0 = identical, 1 = maximally diverse
    cluster_count_variance: float        # Variance of k across algorithms
    entropy_of_labels: float             # Mean label entropy across algorithms
    n_algorithms: int

    def is_diverse(self) -> bool:
        return self.diversity_score >= 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_pairwise_disagreement": round(self.mean_pairwise_disagreement, 4),
            "std_pairwise_disagreement": round(self.std_pairwise_disagreement, 4),
            "diversity_score": round(self.diversity_score, 4),
            "cluster_count_variance": round(self.cluster_count_variance, 4),
            "n_algorithms": self.n_algorithms,
            "is_diverse": self.is_diverse(),
        }


# ──────────────────────────────────────────────────────────────────
# CO-ASSOCIATION MATRIX BUILDER
# ──────────────────────────────────────────────────────────────────

class CoAssociationMatrixBuilder:
    """
    Builds the core co-association matrix from multiple label arrays.
    C[i,j] = fraction of clusterings that place i and j in the same cluster.
    """

    def __init__(self, use_sparse: bool = True):
        self.use_sparse = use_sparse

    def build(self,
              label_arrays: List[np.ndarray],
              weights: Optional[List[float]] = None,
              n: Optional[int] = None) -> np.ndarray:
        """
        label_arrays: list of 1D integer label arrays, each of length n.
        weights: per-algorithm weights (normalised internally).
        Returns: n×n float32 co-association matrix.
        """
        if not label_arrays:
            raise ValueError("No label arrays provided")

        n = n or len(label_arrays[0])
        if weights is None:
            weights = [1.0] * len(label_arrays)

        w_total = sum(weights)
        weights_norm = [w / w_total for w in weights]

        coassoc = np.zeros((n, n), dtype=np.float32)

        for labels, w in zip(label_arrays, weights_norm):
            if len(labels) != n:
                logger.warning(f"Label array length mismatch: {len(labels)} vs {n}")
                continue
            self._accumulate(coassoc, labels, float(w))

        return coassoc

    @staticmethod
    def _accumulate(coassoc: np.ndarray,
                    labels: np.ndarray,
                    weight: float):
        """Fast accumulation via vectorised outer product per cluster."""
        unique = np.unique(labels[labels != -1])
        for c in unique:
            mask = (labels == c)
            idx = np.where(mask)[0]
            if len(idx) < 2:
                coassoc[idx[0], idx[0]] += weight if len(idx) == 1 else 0
                continue
            # Outer product: all pairs in cluster
            coassoc[np.ix_(idx, idx)] += weight

        # Diagonal: always 1 for non-noise points
        non_noise = labels != -1
        coassoc[np.where(non_noise)[0], np.where(non_noise)[0]] = 1.0

    def build_incremental(self,
                           label_arrays: List[np.ndarray],
                           weights: Optional[List[float]] = None,
                           chunk_size: int = 50) -> np.ndarray:
        """Memory-efficient incremental builder for large n."""
        if not label_arrays:
            raise ValueError("No label arrays")
        n = len(label_arrays[0])
        if weights is None:
            weights = [1.0] * len(label_arrays)
        w_total = sum(weights)
        weights_norm = [w / w_total for w in weights]

        coassoc = np.zeros((n, n), dtype=np.float32)
        for labels, w in zip(label_arrays, weights_norm):
            if len(labels) != n:
                continue
            self._accumulate(coassoc, labels, float(w))
        return coassoc

    def symmetrise(self, coassoc: np.ndarray) -> np.ndarray:
        """Ensure symmetry: C = (C + C^T) / 2."""
        return (coassoc + coassoc.T) / 2

    def threshold(self, coassoc: np.ndarray,
                  threshold: float = 0.5) -> np.ndarray:
        """Binarise co-association at threshold."""
        return (coassoc >= threshold).astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# CONSENSUS METHODS
# ──────────────────────────────────────────────────────────────────

class EACConsensus:
    """Evidence Accumulation Clustering — clusters the co-association matrix."""

    def __init__(self, linkage_method: str = "average",
                 n_clusters: int = 8):
        self.linkage_method = linkage_method
        self.n_clusters = n_clusters

    def extract(self, coassoc: np.ndarray) -> np.ndarray:
        """
        Convert co-association to distance, hierarchically cluster,
        cut dendrogram at n_clusters.
        """
        dist_matrix = 1.0 - coassoc
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.clip(dist_matrix, 0.0, 1.0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2

        try:
            condensed = squareform(dist_matrix, checks=False)
            Z = linkage(condensed, method=self.linkage_method)
            labels = fcluster(Z, self.n_clusters, criterion="maxclust") - 1
            return labels.astype(np.int32)
        except Exception as e:
            logger.warning(f"EAC hierarchical clustering failed: {e}; using KMeans fallback")
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=5)
            return km.fit_predict(coassoc).astype(np.int32)

    def extract_with_dendrogram(self, coassoc: np.ndarray
                                 ) -> Tuple[np.ndarray, Optional[Any]]:
        """Returns (labels, linkage_matrix) for dendrogram visualisation."""
        dist_matrix = np.clip(1.0 - coassoc, 0.0, 1.0)
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        try:
            condensed = squareform(dist_matrix, checks=False)
            Z = linkage(condensed, method=self.linkage_method)
            labels = (fcluster(Z, self.n_clusters, criterion="maxclust") - 1).astype(np.int32)
            return labels, Z
        except Exception:
            return self.extract(coassoc), None


class CSPAConsensus:
    """
    Cluster-based Similarity Partitioning Algorithm.
    Constructs a binary similarity matrix and applies spectral clustering.
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def extract(self, label_arrays: List[np.ndarray],
                weights: Optional[List[float]] = None) -> np.ndarray:
        n = len(label_arrays[0])
        if weights is None:
            weights = [1.0] * len(label_arrays)
        w_total = sum(weights)

        S = np.zeros((n, n), dtype=np.float32)
        for labels, w in zip(label_arrays, weights):
            if len(labels) != n:
                continue
            for c in np.unique(labels[labels != -1]):
                idx = np.where(labels == c)[0]
                S[np.ix_(idx, idx)] += w / w_total

        try:
            from sklearn.cluster import SpectralClustering
            sc = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=self.random_state,
            )
            return sc.fit_predict(S).astype(np.int32)
        except Exception:
            # Fallback: EAC
            return EACConsensus(n_clusters=self.n_clusters).extract(S)


class VotingConsensus:
    """
    Majority voting consensus.
    For each point, assigns the most common label across all clusterings
    (after aligning labels via Hungarian algorithm).
    """

    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters

    def extract(self, label_arrays: List[np.ndarray],
                weights: Optional[List[float]] = None) -> np.ndarray:
        if not label_arrays:
            return np.array([], dtype=np.int32)
        n = len(label_arrays[0])
        if weights is None:
            weights = [1.0] * len(label_arrays)

        # Align all label arrays to the first one
        aligned = [label_arrays[0].copy()]
        ref = label_arrays[0]
        for labs in label_arrays[1:]:
            if len(labs) != n:
                continue
            aligned.append(self._align_labels(ref, labs, self.n_clusters))

        # Weighted vote
        vote_matrix = np.zeros((n, self.n_clusters + 1), dtype=np.float32)
        for labs, w in zip(aligned, weights):
            for i in range(n):
                c = int(labs[i])
                if 0 <= c < self.n_clusters:
                    vote_matrix[i, c] += w
                else:
                    vote_matrix[i, self.n_clusters] += w

        labels = vote_matrix[:, :self.n_clusters].argmax(axis=1).astype(np.int32)
        return labels

    @staticmethod
    def _align_labels(ref: np.ndarray,
                      other: np.ndarray,
                      n_clusters: int) -> np.ndarray:
        """Align `other` labels to `ref` via maximum overlap matching."""
        ref_u   = np.unique(ref[ref != -1])
        other_u = np.unique(other[other != -1])

        cost = np.zeros((n_clusters, n_clusters), dtype=np.int32)
        for i, cr in enumerate(ref_u[:n_clusters]):
            for j, co in enumerate(other_u[:n_clusters]):
                cost[i, j] = int(((ref == cr) & (other == co)).sum())

        # Greedy matching (Hungarian approximate)
        mapping = {}
        used_src = set()
        used_dst = set()
        flat_indices = np.argsort(cost.ravel())[::-1]
        for idx in flat_indices:
            i, j = divmod(int(idx), n_clusters)
            if i in used_src or j in used_dst:
                continue
            if i < len(ref_u) and j < len(other_u):
                mapping[int(other_u[j])] = int(ref_u[i])
                used_src.add(i)
                used_dst.add(j)

        aligned = other.copy()
        for old, new in mapping.items():
            aligned[other == old] = new
        return aligned


class MetaClusteringConsensus:
    """
    Treats each label array as a feature vector (one-hot expanded)
    and applies KMeans in the feature space.
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def extract(self, label_arrays: List[np.ndarray]) -> np.ndarray:
        if not label_arrays:
            return np.array([], dtype=np.int32)
        n = len(label_arrays[0])
        # Build feature matrix: n × (sum of unique labels across runs)
        features = []
        for labs in label_arrays:
            if len(labs) != n:
                continue
            from sklearn.preprocessing import LabelBinarizer
            lb = LabelBinarizer()
            valid_mask = labs != -1
            enc = np.zeros((n, max(2, int(labs.max()) + 1)), dtype=np.float32)
            if valid_mask.any():
                enc[valid_mask] = lb.fit_transform(labs[valid_mask])
            features.append(enc)

        if not features:
            return np.zeros(n, dtype=np.int32)

        feature_matrix = np.concatenate(features, axis=1)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=5)
        return km.fit_predict(feature_matrix).astype(np.int32)


class BayesianConsensus:
    """
    Bayesian evidence combination: weights solutions by their
    silhouette score as a proxy for likelihood.
    """

    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self._cam = CoAssociationMatrixBuilder()

    def extract(self,
                label_arrays: List[np.ndarray],
                X: np.ndarray,
                weights: Optional[List[float]] = None) -> np.ndarray:
        if not label_arrays:
            return np.array([], dtype=np.int32)

        # Compute silhouette-based likelihoods
        if weights is None:
            weights = self._compute_silhouette_weights(label_arrays, X)

        coassoc = self._cam.build(label_arrays, weights=weights)
        return EACConsensus(n_clusters=self.n_clusters).extract(coassoc)

    @staticmethod
    def _compute_silhouette_weights(label_arrays: List[np.ndarray],
                                     X: np.ndarray) -> List[float]:
        from sklearn.metrics import silhouette_score
        weights = []
        for labs in label_arrays:
            try:
                valid = labs != -1
                if valid.sum() < 4 or len(np.unique(labs[valid])) < 2:
                    weights.append(0.1)
                    continue
                s = float(silhouette_score(X[valid], labs[valid],
                                           metric="euclidean",
                                           sample_size=min(3000, valid.sum()),
                                           random_state=42))
                weights.append(max(0.01, s + 1.0))  # shift to positive
            except Exception:
                weights.append(0.1)
        return weights


class HybridConsensus:
    """
    Hybrid: uses EAC for initial partition, then refines
    via voting on the initial coarse clusters.
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def extract(self,
                label_arrays: List[np.ndarray],
                weights: Optional[List[float]] = None) -> np.ndarray:
        cam = CoAssociationMatrixBuilder()
        coassoc = cam.build(label_arrays, weights=weights)
        eac_labels = EACConsensus(n_clusters=self.n_clusters).extract(coassoc)

        # Refine: use coassoc-weighted KMeans in co-association space
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            n_feat = min(self.n_clusters * 2, coassoc.shape[0] - 1, 50)
            pca = PCA(n_components=n_feat, random_state=self.random_state)
            reduced = pca.fit_transform(coassoc)
            km = KMeans(n_clusters=self.n_clusters, init="k-means++",
                        n_init=10, random_state=self.random_state)
            refined = km.fit_predict(reduced).astype(np.int32)
            # Choose whichever has better silhouette
            from sklearn.metrics import silhouette_score
            s_eac = float(silhouette_score(coassoc, eac_labels,
                                            sample_size=min(3000, len(eac_labels)),
                                            random_state=42))
            s_ref = float(silhouette_score(coassoc, refined,
                                            sample_size=min(3000, len(refined)),
                                            random_state=42))
            return refined if s_ref >= s_eac else eac_labels
        except Exception:
            return eac_labels


# ──────────────────────────────────────────────────────────────────
# ENSEMBLE DIVERSITY ANALYSER
# ──────────────────────────────────────────────────────────────────

class EnsembleDiversityAnalyser:
    """Quantifies diversity of a collection of clustering solutions."""

    def analyse(self,
                label_arrays: List[np.ndarray],
                algorithm_ids: Optional[List[str]] = None) -> EnsembleDiversity:
        if len(label_arrays) < 2:
            return EnsembleDiversity(
                mean_pairwise_disagreement=0.0,
                std_pairwise_disagreement=0.0,
                diversity_score=0.0,
                cluster_count_variance=0.0,
                entropy_of_labels=0.0,
                n_algorithms=len(label_arrays),
            )

        from sklearn.metrics import adjusted_rand_score
        n = len(label_arrays)
        disagreements = []
        for i in range(n):
            for j in range(i + 1, n):
                if len(label_arrays[i]) != len(label_arrays[j]):
                    continue
                ari = float(adjusted_rand_score(label_arrays[i], label_arrays[j]))
                disagreements.append(1.0 - max(0.0, ari))

        mean_dis = float(np.mean(disagreements)) if disagreements else 0.0
        std_dis  = float(np.std(disagreements))  if disagreements else 0.0

        # Cluster count variance
        k_vals = [len(np.unique(la[la != -1])) for la in label_arrays]
        k_var = float(np.var(k_vals)) if k_vals else 0.0

        # Label entropy
        entropies = []
        for labs in label_arrays:
            valid = labs[labs != -1]
            if len(valid) == 0:
                continue
            from collections import Counter
            counts = np.array(list(Counter(valid.tolist()).values()), dtype=float)
            probs = counts / counts.sum()
            from scipy.stats import entropy as sp_entropy
            entropies.append(float(sp_entropy(probs + 1e-12)))
        mean_ent = float(np.mean(entropies)) if entropies else 0.0

        # Overall diversity score (0–1)
        diversity = float(np.clip(mean_dis * 0.6 + min(k_var / 20.0, 0.4), 0, 1))

        return EnsembleDiversity(
            mean_pairwise_disagreement=round(mean_dis, 4),
            std_pairwise_disagreement=round(std_dis, 4),
            diversity_score=round(diversity, 4),
            cluster_count_variance=round(k_var, 4),
            entropy_of_labels=round(mean_ent, 4),
            n_algorithms=n,
        )


# ──────────────────────────────────────────────────────────────────
# WEIGHT COMPUTER
# ──────────────────────────────────────────────────────────────────

class AlgorithmWeightComputer:
    """
    Computes per-algorithm weights for consensus from evaluation results.
    """

    def compute_from_scores(self,
                             algorithm_ids: List[str],
                             eval_results: Dict[str, Any],
                             metric: str = "composite_score",
                             temperature: float = 1.0) -> Dict[str, float]:
        """
        Softmax weighting based on metric scores.
        temperature > 1 → more uniform; temperature < 1 → winner-takes-most.
        """
        scores = []
        valid_ids = []
        for aid in algorithm_ids:
            er = eval_results.get(aid)
            if er is None:
                continue
            score = getattr(er, metric, None)
            if score is not None and not np.isnan(score):
                scores.append(float(score))
                valid_ids.append(aid)

        if not scores:
            return {aid: 1.0 / len(algorithm_ids) for aid in algorithm_ids}

        # Softmax
        scores_arr = np.array(scores) / temperature
        scores_arr -= scores_arr.max()  # Numerical stability
        exps = np.exp(scores_arr)
        probs = exps / exps.sum()

        return dict(zip(valid_ids, [float(p) for p in probs]))

    def uniform_weights(self, algorithm_ids: List[str]) -> Dict[str, float]:
        n = len(algorithm_ids)
        return {aid: 1.0 / n for aid in algorithm_ids}

    def compute_diversity_boosted(self,
                                   algorithm_ids: List[str],
                                   label_arrays: Dict[str, np.ndarray],
                                   base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Boost weights of diverse algorithms — ensures ensemble covers
        different parts of the clustering hypothesis space.
        """
        if len(algorithm_ids) < 3:
            return base_weights

        from sklearn.metrics import adjusted_rand_score
        n = len(algorithm_ids)
        diversity_bonus = np.zeros(n)

        ids = [aid for aid in algorithm_ids if aid in label_arrays]
        for i in range(len(ids)):
            disagreements = []
            for j in range(len(ids)):
                if i == j:
                    continue
                li = label_arrays.get(ids[i])
                lj = label_arrays.get(ids[j])
                if li is None or lj is None or len(li) != len(lj):
                    continue
                ari = float(adjusted_rand_score(li, lj))
                disagreements.append(1.0 - max(0.0, ari))
            diversity_bonus[i] = float(np.mean(disagreements)) if disagreements else 0.0

        # Combine base weights with diversity bonus
        final = {}
        for i, aid in enumerate(ids):
            w = base_weights.get(aid, 1.0 / n)
            bonus = 0.2 * diversity_bonus[i]
            final[aid] = max(0.001, w + bonus)

        # Normalise
        total = sum(final.values())
        return {k: v / total for k, v in final.items()}


# ──────────────────────────────────────────────────────────────────
# MAIN CONSENSUS ENGINE
# ──────────────────────────────────────────────────────────────────

class ConsensusEngine:
    """
    Master engine that orchestrates all consensus clustering methods
    and produces a final robust partition.
    """

    def __init__(self, config: Optional[ConsensusConfig] = None):
        self.config = config or ConsensusConfig()
        self._cam     = CoAssociationMatrixBuilder(use_sparse=config.use_sparse
                                                    if config else True)
        self._weights = AlgorithmWeightComputer()
        self._diversity = EnsembleDiversityAnalyser()

    def run(self,
            X: np.ndarray,
            clustering_results: Dict[str, Any],
            eval_results: Optional[Dict[str, Any]] = None,
            algorithm_ids: Optional[List[str]] = None) -> ConsensusResult:
        """
        Build consensus clustering from a dict of ClusteringResult objects.
        Optionally takes EvaluationResults for metric-based weighting.
        """
        t_start = time.perf_counter()
        warnings_list = []
        config = self.config

        # Filter to successful results
        valid_ids = [
            aid for aid, cr in clustering_results.items()
            if hasattr(cr, 'succeeded') and cr.succeeded and len(cr.labels) == len(X)
        ]
        if algorithm_ids:
            valid_ids = [aid for aid in valid_ids if aid in algorithm_ids]

        if len(valid_ids) < 2:
            warnings_list.append("Fewer than 2 valid algorithms — returning first available")
            if valid_ids:
                cr = clustering_results[valid_ids[0]]
                return self._trivial_result(cr.labels, valid_ids[0], t_start)
            return self._empty_result(t_start)

        # Select top-k if configured
        if config.top_k_algorithms and eval_results:
            valid_ids = self._select_top_k(valid_ids, eval_results,
                                            config.top_k_algorithms)

        label_arrays = [clustering_results[aid].labels for aid in valid_ids]
        alg_names    = {
            aid: getattr(clustering_results[aid], 'algorithm_name', aid)
            for aid in valid_ids
        }

        # Diversity analysis
        diversity = self._diversity.analyse(label_arrays, valid_ids)

        # Compute weights
        if config.weight_by_metric and eval_results:
            weights_dict = self._weights.compute_from_scores(
                valid_ids, eval_results, config.metric_for_weighting)
        else:
            weights_dict = self._weights.uniform_weights(valid_ids)

        if config.diversity_aware and len(valid_ids) >= 3:
            label_dict = {aid: clustering_results[aid].labels for aid in valid_ids}
            weights_dict = self._weights.compute_diversity_boosted(
                valid_ids, label_dict, weights_dict)

        weights_list = [weights_dict.get(aid, 1.0 / len(valid_ids)) for aid in valid_ids]

        # Build co-association matrix
        coassoc = self._cam.build(label_arrays, weights=weights_list, n=len(X))
        coassoc = self._cam.symmetrise(coassoc)

        # Apply consensus method
        labels = self._apply_method(
            config.method, coassoc, label_arrays, weights_list, X, config)
        labels = labels.astype(np.int32)

        # Quality evaluation
        quality = self._compute_quality(X, labels)
        sil = quality.get("silhouette")
        stab = quality.get("stability")

        # Counts
        valid_labels = labels[labels != -1]
        n_clusters = len(np.unique(valid_labels)) if len(valid_labels) > 0 else 0
        n_noise = int((labels == -1).sum())

        interpretation = self._interpret(
            n_clusters, n_noise, len(X), sil, diversity,
            len(valid_ids), config.method
        )

        return ConsensusResult(
            method=config.method,
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            coassoc_matrix=coassoc if len(X) <= 5000 else None,
            algorithm_weights=weights_dict,
            algorithms_used=[alg_names.get(aid, aid) for aid in valid_ids],
            n_algorithms=len(valid_ids),
            runtime_seconds=time.perf_counter() - t_start,
            quality_score=quality.get("composite", 0.0),
            diversity_score=diversity.diversity_score,
            silhouette=sil,
            stability=stab,
            interpretation=interpretation,
            warnings=warnings_list,
        )

    def run_all_methods(self,
                        X: np.ndarray,
                        clustering_results: Dict[str, Any],
                        eval_results: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, ConsensusResult]:
        """Run all consensus methods and return results dict."""
        results = {}
        for method in ConsensusMethod:
            try:
                cfg = ConsensusConfig(
                    method=method,
                    n_clusters=self.config.n_clusters,
                    weight_by_metric=self.config.weight_by_metric,
                    random_state=self.config.random_state,
                )
                engine = ConsensusEngine(cfg)
                results[method.value] = engine.run(X, clustering_results, eval_results)
            except Exception as e:
                logger.warning(f"Consensus method {method.value} failed: {e}")
        return results

    def _apply_method(self,
                       method: ConsensusMethod,
                       coassoc: np.ndarray,
                       label_arrays: List[np.ndarray],
                       weights: List[float],
                       X: np.ndarray,
                       config: ConsensusConfig) -> np.ndarray:
        k = config.n_clusters

        if method == ConsensusMethod.EAC_AVERAGE:
            return EACConsensus(linkage_method="average", n_clusters=k).extract(coassoc)
        elif method == ConsensusMethod.EAC_SINGLE:
            return EACConsensus(linkage_method="single", n_clusters=k).extract(coassoc)
        elif method == ConsensusMethod.EAC_COMPLETE:
            return EACConsensus(linkage_method="complete", n_clusters=k).extract(coassoc)
        elif method == ConsensusMethod.CSPA:
            return CSPAConsensus(n_clusters=k,
                                  random_state=config.random_state).extract(label_arrays, weights)
        elif method == ConsensusMethod.VOTING:
            return VotingConsensus(n_clusters=k).extract(label_arrays)
        elif method == ConsensusMethod.WEIGHTED_VOTING:
            return VotingConsensus(n_clusters=k).extract(label_arrays, weights=weights)
        elif method == ConsensusMethod.META_CLUSTERING:
            return MetaClusteringConsensus(
                n_clusters=k, random_state=config.random_state
            ).extract(label_arrays)
        elif method == ConsensusMethod.BAYESIAN_COMBINATION:
            return BayesianConsensus(n_clusters=k).extract(label_arrays, X, weights)
        elif method == ConsensusMethod.HYBRID:
            return HybridConsensus(
                n_clusters=k, random_state=config.random_state
            ).extract(label_arrays, weights)
        else:
            return EACConsensus(n_clusters=k).extract(coassoc)

    def _compute_quality(self, X: np.ndarray,
                          labels: np.ndarray) -> Dict[str, Any]:
        from evaluation import ClusteringEvaluator
        ev = ClusteringEvaluator(fast_mode=True)
        try:
            result = ev.evaluate(X, labels, "consensus", "Consensus", "Ensemble")
            return {
                "silhouette": result.metric_value("silhouette"),
                "composite": result.composite_score,
                "stability": None,
            }
        except Exception:
            return {"silhouette": None, "composite": 0.0, "stability": None}

    def _select_top_k(self,
                       valid_ids: List[str],
                       eval_results: Dict[str, Any],
                       k: int) -> List[str]:
        scored = []
        for aid in valid_ids:
            er = eval_results.get(aid)
            if er:
                scored.append((aid, getattr(er, "composite_score", 0.0)))
            else:
                scored.append((aid, 0.0))
        scored.sort(key=lambda x: -x[1])
        return [s[0] for s in scored[:k]]

    @staticmethod
    def _interpret(n_clusters: int, n_noise: int, n_total: int,
                   silhouette: Optional[float],
                   diversity: EnsembleDiversity,
                   n_algorithms: int,
                   method: ConsensusMethod) -> str:
        noise_pct = n_noise / max(n_total, 1) * 100
        parts = [
            f"Consensus ({method.value}) over {n_algorithms} algorithms "
            f"produced **{n_clusters} clusters** "
            f"({noise_pct:.1f}% noise)."
        ]
        if silhouette is not None:
            parts.append(f"Silhouette = {silhouette:.3f}.")
        if diversity.is_diverse():
            parts.append(
                f"Ensemble diversity is high ({diversity.diversity_score:.2f}) — "
                "consensus is more informative than any single algorithm."
            )
        else:
            parts.append(
                f"Ensemble diversity is low ({diversity.diversity_score:.2f}) — "
                "most algorithms agree; consensus largely confirms their outputs."
            )
        return " ".join(parts)

    def _trivial_result(self, labels: np.ndarray,
                         aid: str, t_start: float) -> ConsensusResult:
        valid = labels[labels != -1]
        n_cls = len(np.unique(valid)) if len(valid) > 0 else 0
        return ConsensusResult(
            method=ConsensusMethod.EAC_AVERAGE,
            labels=labels, n_clusters=n_cls,
            n_noise=int((labels == -1).sum()),
            coassoc_matrix=None,
            algorithm_weights={aid: 1.0},
            algorithms_used=[aid], n_algorithms=1,
            runtime_seconds=time.perf_counter() - t_start,
            quality_score=0.0, diversity_score=0.0,
            silhouette=None, stability=None,
            interpretation="Single algorithm — trivial consensus.",
        )

    def _empty_result(self, t_start: float) -> ConsensusResult:
        return ConsensusResult(
            method=ConsensusMethod.EAC_AVERAGE,
            labels=np.array([], dtype=np.int32),
            n_clusters=0, n_noise=0,
            coassoc_matrix=None,
            algorithm_weights={}, algorithms_used=[], n_algorithms=0,
            runtime_seconds=time.perf_counter() - t_start,
            quality_score=0.0, diversity_score=0.0,
            silhouette=None, stability=None,
            interpretation="No valid algorithms for consensus.",
        )


# ──────────────────────────────────────────────────────────────────
# CONSENSUS VISUALISATION DATA BUILDERS
# ──────────────────────────────────────────────────────────────────

class ConsensusVisBuilder:
    """Builds data for consensus visualisation widgets."""

    def coassoc_heatmap(self,
                         coassoc: np.ndarray,
                         labels: np.ndarray,
                         max_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (sorted_coassoc, sorted_labels) for heatmap display,
        reordered by cluster membership for visual clarity.
        """
        if len(labels) == 0 or coassoc is None:
            return coassoc, labels

        n = len(labels)
        if n > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, max_points, replace=False)
            coassoc = coassoc[np.ix_(idx, idx)]
            labels = labels[idx]

        # Sort by cluster
        sort_idx = np.argsort(labels)
        return coassoc[np.ix_(sort_idx, sort_idx)], labels[sort_idx]

    def weight_barchart_data(self,
                              consensus_result: ConsensusResult) -> pd.DataFrame:
        """Horizontal bar chart data for algorithm weights."""
        weights = consensus_result.algorithm_weights
        algs    = consensus_result.algorithms_used
        rows = []
        for alg in algs:
            # Match algorithm name to weight key
            for aid, w in weights.items():
                if alg == aid or alg.startswith(aid[:15]):
                    rows.append({"Algorithm": alg[:40], "Weight": round(w, 4)})
                    break
        if not rows:
            rows = [{"Algorithm": alg, "Weight": 1.0 / len(algs)} for alg in algs]
        df = pd.DataFrame(rows)
        return df.sort_values("Weight", ascending=False)

    def method_comparison_table(self,
                                  all_results: Dict[str, ConsensusResult]) -> pd.DataFrame:
        """Compare all consensus methods side-by-side."""
        rows = []
        for method_key, result in all_results.items():
            rows.append({
                "Method": method_key,
                "k Found": result.n_clusters,
                "Noise%": round(result.n_noise / max(len(result.labels), 1) * 100, 1),
                "Silhouette": round(result.silhouette, 4) if result.silhouette else None,
                "Quality Score": round(result.quality_score, 2),
                "Diversity": round(result.diversity_score, 4),
                "Algorithms Used": result.n_algorithms,
                "Runtime(s)": round(result.runtime_seconds, 3),
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("Quality Score", ascending=False)

    def cluster_distribution_chart(self,
                                    labels: np.ndarray) -> pd.DataFrame:
        """Cluster size distribution for bar chart."""
        from collections import Counter
        counts = Counter(labels.tolist())
        rows = []
        for c, cnt in sorted(counts.items()):
            label = f"Noise" if c == -1 else f"Cluster {c}"
            rows.append({"Cluster": label, "Size": cnt,
                         "Fraction": round(cnt / max(len(labels), 1) * 100, 2)})
        return pd.DataFrame(rows)

    def diversity_breakdown(self,
                             diversity: EnsembleDiversity) -> pd.DataFrame:
        """Diversity components as a breakdown table."""
        return pd.DataFrame([{
            "Metric": "Mean Pairwise Disagreement",
            "Value": diversity.mean_pairwise_disagreement,
        }, {
            "Metric": "Diversity Score",
            "Value": diversity.diversity_score,
        }, {
            "Metric": "Cluster Count Variance",
            "Value": diversity.cluster_count_variance,
        }, {
            "Metric": "Label Entropy",
            "Value": diversity.entropy_of_labels,
        }])


# ──────────────────────────────────────────────────────────────────
# K-ESTIMATION VIA CONSENSUS
# ──────────────────────────────────────────────────────────────────

class ConsensusKEstimator:
    """
    Estimates the optimal number of clusters using the co-association
    matrix — finds the k that maximises the step in the dendrogram.
    """

    def estimate(self,
                 coassoc: np.ndarray,
                 k_range: range = range(2, 15)) -> Dict[str, Any]:
        dist_matrix = np.clip(1.0 - coassoc, 0.0, 1.0)
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2

        try:
            from sklearn.metrics import silhouette_score
            condensed = squareform(dist_matrix, checks=False)
            Z = linkage(condensed, method="average")

            sil_scores = {}
            for k in k_range:
                if k >= len(coassoc):
                    break
                labels = fcluster(Z, k, criterion="maxclust") - 1
                if len(np.unique(labels)) < 2:
                    continue
                try:
                    s = float(silhouette_score(coassoc, labels,
                                               metric="precomputed" if False else "euclidean",
                                               sample_size=min(3000, len(labels)),
                                               random_state=42))
                    sil_scores[k] = s
                except Exception:
                    pass

            if not sil_scores:
                return {"optimal_k": min(k_range), "scores": {}}

            optimal_k = max(sil_scores, key=sil_scores.get)
            return {
                "optimal_k": optimal_k,
                "scores": {k: round(v, 4) for k, v in sil_scores.items()},
                "k_values": list(sil_scores.keys()),
                "sil_values": [sil_scores[k] for k in sil_scores],
            }
        except Exception as e:
            return {"optimal_k": min(k_range), "scores": {}, "error": str(e)}


# ──────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def run_consensus(X: np.ndarray,
                  clustering_results: Dict[str, Any],
                  n_clusters: int = 8,
                  method: ConsensusMethod = ConsensusMethod.EAC_AVERAGE,
                  eval_results: Optional[Dict[str, Any]] = None,
                  weight_by_metric: bool = True) -> ConsensusResult:
    """Top-level convenience wrapper."""
    config = ConsensusConfig(
        method=method,
        n_clusters=n_clusters,
        weight_by_metric=weight_by_metric,
    )
    engine = ConsensusEngine(config)
    return engine.run(X, clustering_results, eval_results)


def make_consensus_engine(n_clusters: int = 8,
                           method: str = "eac_average",
                           weight_by_metric: bool = True) -> ConsensusEngine:
    """Factory for creating configured consensus engines."""
    method_enum = ConsensusMethod(method)
    config = ConsensusConfig(
        method=method_enum,
        n_clusters=n_clusters,
        weight_by_metric=weight_by_metric,
        diversity_aware=True,
    )
    return ConsensusEngine(config)


def get_consensus_method_options() -> List[Dict[str, str]]:
    descriptions = {
        ConsensusMethod.EAC_AVERAGE:          "Evidence Accumulation — Average Linkage (recommended)",
        ConsensusMethod.EAC_SINGLE:           "Evidence Accumulation — Single Linkage",
        ConsensusMethod.EAC_COMPLETE:         "Evidence Accumulation — Complete Linkage",
        ConsensusMethod.CSPA:                 "Cluster-based Similarity Partitioning (CSPA)",
        ConsensusMethod.VOTING:               "Majority Voting (aligned labels)",
        ConsensusMethod.WEIGHTED_VOTING:      "Metric-Weighted Voting",
        ConsensusMethod.META_CLUSTERING:      "Meta-Clustering (feature-space KMeans)",
        ConsensusMethod.BAYESIAN_COMBINATION: "Bayesian Evidence Combination",
        ConsensusMethod.HYBRID:               "Hybrid EAC + KMeans Refinement",
    }
    return [
        {"value": m.value, "label": descriptions.get(m, m.value)}
        for m in ConsensusMethod
    ]


def analyse_ensemble_diversity(clustering_results: Dict[str, Any]) -> EnsembleDiversity:
    """Quick diversity analysis of all successful clustering results."""
    analyser = EnsembleDiversityAnalyser()
    valid = {k: v for k, v in clustering_results.items()
             if hasattr(v, 'succeeded') and v.succeeded and len(v.labels) > 0}
    if not valid:
        return EnsembleDiversity(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    ids = list(valid.keys())
    arrays = [valid[aid].labels for aid in ids]
    return analyser.analyse(arrays, ids)


def consensus_summary(result: ConsensusResult) -> str:
    """Returns a one-line text summary of a consensus result."""
    return (
        f"Method: {result.method.value} | "
        f"k={result.n_clusters} | "
        f"Quality={result.quality_score:.1f} | "
        f"Algorithms={result.n_algorithms} | "
        f"Diversity={result.diversity_score:.3f} | "
        f"Runtime={result.runtime_seconds:.2f}s"
    )

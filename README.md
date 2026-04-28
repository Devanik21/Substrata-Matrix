
# 🧬 UnSuPERvIsED-II — Advanced Clustering Intelligence System

> *"Sixty algorithms. Twelve families. One orchestration engine. Zero compromises."*

---

## About This Repository

**Author:** Devanik (GitHub: [Devanik21](https://github.com/Devanik21))
**Repository:** [Substrata-Matrix](https://github.com/Devanik21/Substrata-Matrix) · `Intermediate_Cluster/` · May 2026
**Affiliation:** Electronics & Communication Engineering, NIT Agartala · Samsung ISWDP Fellow (IISc, 98.58th percentile)

UnSuPERvIsED-II is the **second-generation, architecturally advanced** iteration of the *UnSuPERvIsED* series — a production-grade unsupervised clustering intelligence platform that orchestrates **60+ algorithms across 12 algorithmic families** through a unified, cache-aware, adaptive execution engine. This version substantially extends the foundation of UnSuPERvIsED-I with: neural and deep-learning-based clustering, fuzzy and possibilistic approaches, manifold-embedding pipelines, ensemble and subspace methods, grid-based density estimation, message-passing frameworks, adaptive parameter tuning, Pareto-front multi-objective ranking, and a fully refactored stability engine supporting Gaussian and Laplacian perturbation, persistence analysis, and stability grade certification. The entire system spans **~14,700 lines** of rigorously structured Python.

---

## What Is UnSuPERvIsED-II?

UnSuPERvIsED-II (`Intermediate_Cluster/`) delivers a complete clustering research and deployment environment. Every algorithmic family known in the literature is represented; the execution engine automatically classifies dataset size, blacklists incompatible algorithms, and schedules runs adaptively. A thread-safe LRU result cache avoids redundant computation. The evaluation layer provides internal indices, external label-agreement measures, Pareto-front selection, and k-sweep trend analysis. The stability engine operates under five distinct perturbation regimes and issues graded robustness certificates.

### Module Architecture

| Module | Role | Lines |
|--------|------|-------|
| `UnSuPERvIsED.py` | Streamlit orchestrator, lazy loading, dark-theme UI | ~3,558 |
| `clustering_registry.py` | 60+ algorithm specs, tags, factory system, 12 families | ~1,967 |
| `clustering_runner.py` | Cache-aware parallel runner, adaptive scheduling, timeout | ~1,367 |
| `evaluation.py` | Internal/external metrics, composite scorer, Pareto front | ~1,341 |
| `preprocessing.py` | Multi-stage preprocessing, outlier detection, encoding | ~1,506 |
| `stability.py` | 5-regime stability testing, persistence analysis, grading | ~1,635 |
| `consensus.py` | Co-association matrices, PAC, cophenetic correlation | ~1,561 |
| `visualization.py` | 30+ Plotly charts, dark-theme CSS, interactive widgets | ~1,763 |

---

## Algorithm Library — 60+ Algorithms Across 12 Families

### Family Taxonomy

```
AlgorithmFamily (12 families)
├── CENTROID         — Centroid-Based         (6 algorithms)
├── HIERARCHICAL     — Hierarchical           (7 algorithms)
├── DENSITY          — Density-Based          (5 algorithms)
├── DISTRIBUTION     — Distribution-Based     (5 algorithms)
├── GRAPH            — Graph-Based / Spectral (5 algorithms)
├── MESSAGE_PASSING  — Message-Passing        (1 algorithm)
├── NEURAL           — Neural / Deep          (2 algorithms)
├── SUBSPACE         — Subspace               (2 algorithms)
├── ENSEMBLE         — Ensemble               (2 algorithms)
├── FUZZY            — Fuzzy                  (2 algorithms)
├── GRID             — Grid-Based             (1 algorithm)
└── MANIFOLD         — Manifold               (3 algorithms)
    + EXTRA          — Additional             (6 algorithms)
```

Each `AlgorithmSpec` carries a rich metadata schema:

```python
@dataclass
class AlgorithmSpec:
    id: str                         # Unique snake_case identifier
    name: str                       # Human-readable name
    family: AlgorithmFamily         # One of 12 families
    tags: List[AlgorithmTag]        # fast, scalable, no_k, noise_robust, ...
    description: str
    paper_ref: str                  # Primary academic citation
    time_complexity: ComplexityClass
    space_complexity: ComplexityClass
    hyper_params: List[HyperParam]  # Full hyperparameter descriptors
    min_samples: int
    max_recommended_samples: int
    max_recommended_features: int
    produces_noise_label: bool      # Whether algorithm outputs -1 (noise)
    requires_n_clusters: bool
    factory: Callable               # factory(**params) → sklearn-compatible estimator
```

Algorithm tags: `fast`, `scalable`, `no_k_needed`, `noise_robust`, `probabilistic`, `hierarchical`, `online`, `gpu_ready`, `deterministic`, `shape_agnostic`, `high_dimensional`, `small_data`.

---

### Family I — Centroid-Based

**K-Means** (`kmeans`)

Standard Lloyd–Forgy algorithm minimising within-cluster sum of squares:

```math
$$\min_{\{C_j\}} \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - \mu_j\|_2^2, \quad \mu_j = \frac{1}{|C_j|}\sum_{x_i \in C_j} x_i$$

```

E-step: assign $c(i) = \arg\min_j \|x_i - \mu_j\|^2$. M-step: update centroids. Each iteration is a coordinate descent step guaranteeing non-increasing WCSS. Tags: `fast`, `deterministic`, `scalable`.

---

**K-Means++** (`kmeans_pp`)

Initialisation-enhanced K-Means. Seeding draws $c_1$ uniformly; each subsequent centroid is drawn with probability:

```math
P(x_i) \propto \min_{j < l} \|x_i - c_j\|^2
```

This $D^2$ weighting provides the approximation guarantee:

```math
\mathbb{E}[\mathrm{WCSS}_{k\text{-means}++}] \leq 8(\ln k + 2)\cdot \mathrm{WCSS}_{\mathrm{OPT}}
```

dramatically reducing convergence to poor local minima.

---

**MiniBatch K-Means** (`minibatch_kmeans`)

Stochastic gradient descent on WCSS. Per-centre learning rate $\eta_{c_j} = 1/n_{c_j}$. Each iteration processes batch $\mathcal{B}$ of size $b$:

```math
\mu_j \leftarrow (1 - \eta_{c_j})\,\mu_j + \eta_{c_j}\cdot \overline{x}_{\mathcal{B},j}
```

The harmonic decay satisfies Robbins-Monro conditions, guaranteeing almost-sure convergence. Tags: `fast`, `scalable`, `online`.

---

**Bisecting K-Means** (`bisecting_kmeans`)

Divisive strategy. At each step: (1) select $C^* = \arg\max_j \mathrm{WCSS}(C_j)$; (2) apply 2-means to $C^*$; (3) repeat until $k$ clusters. Total WCSS decreases monotonically at every bisection, avoiding global-initialisation fragility.

---

**K-Medoids** (`kmedoids`)

Medoids $m_j \in X$ minimise:

```math
\sum_{i=1}^n d\!\left(x_i,\, m_{c(i)}\right)
```

for arbitrary metric $d$. PAM: iterative swap of medoid $m_j$ with non-medoid $x_h$, accepted if total cost decreases. Tags: `noise_robust`, `shape_agnostic`.

---

**K-Means Auto** (`kmeans_auto`)

Bandwidth-adaptive K-Means that estimates the optimal $k$ via BIC on a Gaussian mixture model approximation, then runs K-Means++ at the selected $k$. Internal BIC criterion:

```math
\mathrm{BIC}(k) = -2\mathcal{L}(X;\, \hat{\theta}_k) + p_k \ln n
```

where $p_k = k(d + 1)$ is the number of free parameters (means + spherical variance per cluster). The selected $k^* = \arg\min_k \mathrm{BIC}(k)$.

---

### Family II — Hierarchical

**Agglomerative Clustering** (4 linkages: ward, complete, average, single)

Bottom-up merging with four linkage criteria. Ward linkage cost:

```math
\Delta(A, B) = \frac{n_A\, n_B}{n_A + n_B}\|\mu_A - \mu_B\|^2
```

Unified **Lance–Williams recurrence** for distance updates to merged cluster $A \cup B$ from $C$:

```math
D(A \cup B,\, C) = \alpha_A\, D(A,C) + \alpha_B\, D(B,C) + \beta\, D(A,B) + \gamma\,|D(A,C) - D(B,C)|
```

with family-specific coefficients $(\alpha_A, \alpha_B, \beta, \gamma)$. This enables $O(n^2 \log n)$ priority-queue implementations.

---

**BIRCH** (`birch`)

CF-tree with $O(n)$ complexity. Clustering Feature: $CF = (n, LS, SS)$ supports $O(1)$ merging. Maximum subcluster radius constraint:

```math
R = \sqrt{\frac{SS}{n} - \left\|\frac{LS}{n}\right\|^2} \leq \theta
```

Phase 1: build CF tree. Phase 2: agglomerate leaf subclusters. Phase 3: reassign.

---

**DIANA — Divisive Analysis** (`diana_divisive`)

Top-down divisive hierarchical clustering. Starting from the full dataset, identifies the most dissimilar point and builds the "splinter group" by iteratively moving points:

```math
$$d_{\mathrm{avg}}(x, S) = \frac{1}{|S|}\sum_{y \in S} d(x,y), \quad x^* = \underset{x \in C}{\arg\max}\ \bigl[d_{\mathrm{avg}}(x, C) - d_{\mathrm{avg}}(x, S)\bigr]$$

```

DIANA naturally handles elongated clusters and produces a balanced dendrogram in cases where AGNES is biased by chaining. The **diameter** of the resulting clusters satisfies a monotonicity property: $\mathrm{diam}(C_{\mathrm{parent}}) \geq \max(\mathrm{diam}(C_1), \mathrm{diam}(C_2))$.

---

### Family III — Density-Based

**DBSCAN** (`dbscan`)

Core-reachability graph: $p \xrightarrow{\text{reach}} q$ if $p$ is a core point and $d(p,q) \leq \varepsilon$. Clusters are maximal density-connected sets. Noise label $-1$. Tags: `no_k_needed`, `noise_robust`.

---

**OPTICS** (`optics`)

Reachability distance $\mathrm{rd}(p,q) = \max(\mathrm{cd}(q), d(p,q))$. Orders points by traversal; extracts clusters via $\xi$-steep descent ($\xi \in [0,1]$ controls minimum relative drop in reachability). The reachability plot encodes the complete density hierarchy without committing to a single $\varepsilon$.

---

**HDBSCAN** (`hdbscan`)

Mutual reachability distance and hierarchical cluster tree. Stability-based cluster selection:

```math
\mathrm{stab}(C) = \sum_{x \in C} \bigl(\lambda_{\mathrm{max}}(x, C) - \lambda_{\mathrm{birth}}(C)\bigr)
```

where $\lambda = 1/d_{\mathrm{mreach}}$. EOM (Excess of Mass) selects the subset of non-overlapping clusters maximising total stability. Soft membership probability: $\Pr(x \in C_j) = \lambda_{\mathrm{death}}(x,C_j)/\max_{x'}\lambda_{\mathrm{death}}(x',C_j)$. Tags: `no_k_needed`, `noise_robust`, `probabilistic`.

---

**Mean Shift** (`mean_shift`)

KDE gradient ascent. Bandwidth $h$ estimated via Silverman's rule: $h = 1.06\,\hat{\sigma}\,n^{-1/(d+4)}$. Mean shift vector:

```math
m_h(x) = \frac{\displaystyle\sum_i K_h(x - x_i)\, x_i}{\displaystyle\sum_i K_h(x - x_i)} - x \;\;\propto\;\; \nabla \hat{f}_h(x)
```

Tags: `no_k_needed`, `shape_agnostic`.

---

**DENCLUE** (`denclue`)

Density-based via Gaussian kernel density estimation:

```math
\hat{f}(x) = \frac{1}{n\,h^d} \sum_{i=1}^n K\!\left(\frac{x - x_i}{h}\right)
```

Clusters are maximal connected regions where $\hat{f}(x) \geq \xi$ (density threshold). Points are attracted to density attractors (local maxima of $\hat{f}$) via gradient ascent. Outperforms DBSCAN for continuously varying density distributions. The gradient of $\hat{f}$ in the Gaussian kernel case is analytically tractable:

```math
\nabla \hat{f}(x) = \frac{1}{n\,h^{d+2}}\sum_{i=1}^n K\!\left(\frac{x-x_i}{h}\right)(x_i - x)
```

---

### Family IV — Distribution-Based

**Gaussian Mixture Models** (4 covariance types)

```math
p(x) = \sum_{j=1}^k \pi_j\,\mathcal{N}(x \mid \mu_j, \Sigma_j)
```

EM with four covariance regimes:

| Type | Constraint | Parameters per cluster |
|------|-----------|------------------------|
| `full` | Unconstrained $\Sigma_j$ | $d(d+1)/2$ |
| `tied` | Shared $\Sigma$ | $d(d+1)/2$ total |
| `diag` | Diagonal $\Sigma_j$ | $d$ |
| `spherical` | $\sigma_j^2 I$ | $1$ |

BIC/AIC for model selection:

```math
\mathrm{BIC} = -2\hat{\mathcal{L}} + p\,\ln n, \quad \mathrm{AIC} = -2\hat{\mathcal{L}} + 2p
```

---

**Bayesian GMM** (`bgmm`)

Variational inference with Dirichlet process prior. ELBO:

```math
\mathcal{L}_{\mathrm{ELBO}} = \mathbb{E}_q\bigl[\log p(X,Z,\theta)\bigr] + \mathcal{H}[q(Z,\theta)]
```

where $\mathcal{H}$ is the variational entropy. This equals $\log p(X) - \mathrm{KL}[q \| p(\cdot|X)]$. Concentration $\alpha \to 0$: automatic component pruning (inactive components get $\pi_j \approx 0$).

---

### Family V — Graph-Based / Spectral

**Spectral Clustering** (2 assign modes: kmeans, discretize)

Graph Laplacian eigenvectors embed the data. Normalised Laplacian:

```math
\mathcal{L}_{\mathrm{sym}} = I - D^{-1/2}WD^{-1/2}
```

**Spectral gap heuristic** for $k$: the number of eigenvalues near 0 reveals the natural cluster count. Under ideal $k$ perfectly separated components: $\lambda_1 = \cdots = \lambda_k = 0$, $\lambda_{k+1} > 0$. Assignment mode `discretize` rounds the embedding matrix via iterative rotation:

```math
\min_{R \in \mathcal{O}(k)} \|U - VR\|_F, \quad V_{ij} = \frac{u_{ij}}{\|u_i\|_2}
```

---

**Spectral Biclustering** (`spectral_biclustering`)

Simultaneously clusters rows and columns of the data matrix $X \in \mathbb{R}^{n \times d}$ using the Laplacian normalisation of the biadjacency structure. Finds biclusters $B = (R, C)$ maximising the average value of $X_{RC}$ relative to $X$. SVD of the normalised matrix $D_r^{-1/2}X D_c^{-1/2}$ yields the bicluster structure.

---

**Spectral Coclustering** (`spectral_coclustering`)

Normalised cut on the bipartite graph of samples and features. SVD of the normalised matrix $D_r^{-1/2} X D_c^{-1/2}$ yields a low-rank structure; K-Means on the stacked singular vectors produces co-clusters simultaneously partitioning rows and columns.

---

**Affinity Propagation** (`affinity_propagation`)

Message-passing over the fully-connected similarity graph with damping:

```math
r_{t+1} = (1-\lambda)\,r_t + \lambda\,r_{\mathrm{new}}, \quad a_{t+1} = (1-\lambda)\,a_t + \lambda\,a_{\mathrm{new}}
```

Responsibility update:

```math
r(i,k) \leftarrow s(i,k) - \max_{k' \neq k}\bigl\{a(i,k') + s(i,k')\bigr\}
```

Availability update:

```math
a(i,k) \leftarrow \min\!\left(0,\; r(k,k) + \sum_{i' \notin \{i,k\}} \max(0, r(i',k))\right)
```

Exemplars emerge without specifying $k$. Tags: `no_k_needed`.

---

### Family VI — Neural / Deep

**Autoencoder + K-Means** (`autoencoder_kmeans`)

A deep autoencoder first learns a non-linear compression $z = f_\phi(x) \in \mathbb{R}^q$ ($q \ll d$), trained to minimise reconstruction loss:

```math
\mathcal{L}_{\mathrm{recon}} = \|x - g_\theta(f_\phi(x))\|_2^2
```

K-Means is then applied in the latent space $\{z_i\}$. Encoder architecture: Input($d$) → Dense(128, ReLU) → Dense(64, ReLU) → Dense($q$, Linear). Decoder mirrors. Joint fine-tuning minimises:

```math
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{recon}} + \lambda_c\,\mathcal{L}_{\mathrm{cluster}}, \quad \mathcal{L}_{\mathrm{cluster}} = \sum_{i=1}^n \|z_i - \mu_{c(i)}\|^2
```

The latent embedding $f_\phi$ learns a topology-aware compression superior to linear PCA for manifold-structured data. Tags: `shape_agnostic`, `high_dimensional`.

---

**Self-Organising Map Clustering** (`som_clustering`)

SOM learns a discrete low-dimensional manifold (typically 2D grid) that maps the data topology. For each input $x$, the best-matching unit (BMU) is:

```math
$$i^*(x) = \underset{i}{\arg\min}\ \|x - w_i\|$$

```

Weights update with decaying learning rate $\alpha(t)$ and neighbourhood function $h(i, i^*, t)$:

```math
w_i \leftarrow w_i + \alpha(t)\cdot h(i, i^*, t)\cdot (x - w_i)
```

where $h(i, i^*, t) = \exp(-\|r_i - r_{i^*}\|^2 / 2\sigma(t)^2)$ is the neighbourhood kernel on the grid and $\sigma(t)$ decays with training time. After training, the **U-Matrix** $U_{ij} = \|w_i - w_j\|$ between adjacent map units reveals cluster boundaries. K-Means on SOM prototypes yields final assignments. Tags: `shape_agnostic`, `high_dimensional`, `online`.

---

### Family VII — Subspace

**Projected K-Means** (`projected_kmeans`)

Random projection $\Phi \in \mathbb{R}^{q \times d}$ with entries $\phi_{ij} \sim \mathcal{N}(0, 1/q)$ reduces dimensionality before K-Means application. By the **Johnson-Lindenstrauss lemma**: for $q = O(\varepsilon^{-2}\log n)$, pairwise distances are preserved within factor $1 \pm \varepsilon$ with high probability:

```math
\Pr\!\left[\,(1-\varepsilon)\|x-y\|^2 \leq \|\Phi x - \Phi y\|^2 \leq (1+\varepsilon)\|x-y\|^2\,\right] \geq 1 - 2\exp\!\left(-q\varepsilon^2/4\right)
```

Effective for very high-dimensional sparse data.

---

**CLIQUE Subspace Clustering** (`clique_subspace`)

Grid-based subspace discovery. Divides each dimension into $\xi$ equal-width bins and identifies dense units (cells where point density $\geq \tau$). Connected dense units in lower-dimensional projections form clusters. CLIQUE automatically finds clusters in all subspace projections, addressing the curse of dimensionality by focusing on locally relevant subsets of dimensions.

---

### Family VIII — Ensemble

**Ensemble Voting (Multi-Init K-Means)** (`ensemble_voting`)

Runs K-Means with $M$ different random seeds, producing label sets $\{L_1, \ldots, L_M\}$. Builds a co-occurrence matrix:

```math
A_{ij} = \frac{1}{M}\sum_{m=1}^M \mathbf{1}\bigl[L_m(i) = L_m(j)\bigr]
```

Spectral or agglomerative clustering on $1 - A$ produces the consensus partition. The ensemble variance $\mathrm{Var}[A_{ij}]$ quantifies pointwise assignment uncertainty.

---

**Random Subspace Ensemble** (`random_subspace_ensemble`)

Trains $T$ clusterers, each on a random feature subset of size $\lfloor\sqrt{d}\rfloor$. Consensus via co-association matrix averaging. The feature subset variance reduction follows from:

```math
\mathrm{Var}\!\left[\frac{1}{T}\sum_{t=1}^T A_{ij}^{(t)}\right] = \frac{\mathrm{Var}[A_{ij}^{(t)}]}{T} + \frac{T-1}{T}\,\mathrm{Cov}[A_{ij}^{(t)}, A_{ij}^{(t')}]
```

Diversity between trees (low covariance term) reduces ensemble variance.

---

### Family IX — Fuzzy

**Fuzzy C-Means (FCM)** (`fuzzy_cmeans`)

FCM relaxes hard assignment to membership degrees $u_{ij} \in [0,1]$ with $\sum_j u_{ij} = 1$. Objective:

```math
J_m = \sum_{i=1}^n \sum_{j=1}^k u_{ij}^m \|x_i - \mu_j\|^2
```

where $m > 1$ is the fuzzifier (typically $m = 2$). Update rules from optimality conditions:

```math
u_{ij} = \left(\sum_{l=1}^k \left(\frac{\|x_i - \mu_j\|}{\|x_i - \mu_l\|}\right)^{2/(m-1)}\right)^{-1}, \qquad \mu_j = \frac{\sum_i u_{ij}^m x_i}{\sum_i u_{ij}^m}
```

EM-like iterations converge to a local minimum of $J_m$. The **Picard iteration** theorem guarantees convergence: $\|u^{(t+1)} - u^*\| = O(\|u^{(t)} - u^*\|^2)$ (quadratic local convergence under mild conditions). Tags: `probabilistic`, `shape_agnostic`.

---

**Possibilistic C-Means (PCM)** (`possibilistic_cmeans`)

PCM relaxes the probabilistic constraint $\sum_j u_{ij} = 1$. Membership degrees become independent **typicalities** $t_{ij} \in [0,1]$ measuring how typical a point is of a cluster. Objective:

```math
J_m = \sum_{i=1}^n \sum_{j=1}^k t_{ij}^m \|x_i - \mu_j\|^2 + \sum_{j=1}^k \eta_j \sum_{i=1}^n (1 - t_{ij})^m
```

where the per-cluster bandwidth:

```math
\eta_j = K\,\frac{\sum_i u_{ij}^m \|x_i - \mu_j\|^2}{\sum_i u_{ij}^m}
```

is estimated from FCM memberships. PCM is robust to outliers (outliers have low typicality to all clusters) and avoids the coincident clusters problem inherent in FCM.

---

### Family X — Grid-Based

**WaveCluster (Grid)** (`sting_grid`)

Quantises the feature space into a hierarchical grid. Each cell stores the statistical summary $(n, \mu, \sigma^2)$ of contained points. Density is estimated cell-wise; connected high-density cells form clusters. Multi-resolution analysis: coarse grids give global structure, fine grids capture local detail. Runs in $O(n)$ (linear in data, independent of grid resolution). Cell density threshold $\tau$ controls the minimum density for a cell to be considered part of a cluster.

---

### Family XI — Manifold

**Isomap + K-Means** (`isomap_kmeans`)

Isomap estimates geodesic distances by constructing a $k$-NN graph and computing shortest paths (Dijkstra or Floyd-Warshall); the geodesic distance matrix $D_G$ is embedded via **classical MDS** minimising strain:

```math
\mathrm{Strain}(Y) = \left\|\tau(D_G) - Y^\top Y\right\|_F
```

where $\tau(D_G) = -\frac{1}{2}H D_G^{(2)} H$ (double-centred squared distance, $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$). K-Means is applied to the geodesic embedding $Y \in \mathbb{R}^{n \times q}$. For data lying on a smooth $q$-dimensional manifold isometrically embedded in $\mathbb{R}^d$, Isomap recovers the true intrinsic geometry as $n \to \infty$.

---

**LLE + K-Means** (`lle_kmeans`)

Locally Linear Embedding represents each point as a linear combination of its $k$-NN:

```math
\min_W \sum_i \left\|x_i - \sum_j W_{ij} x_j\right\|^2, \quad \text{s.t.}\ \sum_j W_{ij} = 1,\; W_{ij} = 0\ \text{if}\ j \notin \mathcal{N}(i)
```

The low-dimensional embedding $Y$ minimises the same reconstruction cost with fixed $W$:

```math
\min_Y \sum_i \left\|y_i - \sum_j W_{ij} y_j\right\|^2 = \min_Y \mathrm{tr}\!\left(Y^\top M Y\right)
```

where $M = (I-W)^\top(I-W)$. The solution is the bottom $q$ eigenvectors of $M$ (excluding the trivial $\mathbf{1}$ eigenvector). LLE faithfully unrolls manifolds (e.g. Swiss roll), yielding Euclidean structure amenable to K-Means.

---

**UMAP + HDBSCAN** (`umap_hdbscan`)

UMAP learns a fuzzy topological representation via Riemannian geometry. For each point $x_i$ and its $k$-NN $\{x_{i,j}\}$, local fuzzy membership strength:

```math
\mu(x_i, x_{i,j}) = \exp\!\left(-\frac{d(x_i, x_{i,j}) - \rho_i}{\sigma_i}\right)
```

where $\rho_i = d(x_i, x_{i,1})$ is the nearest-neighbour distance and $\sigma_i$ is chosen to achieve a target fuzzy set cardinality (the perplexity analogue). The global embedding minimises the **cross-entropy** between fuzzy topological representations:

```math
\mathcal{L}_{\mathrm{UMAP}} = \sum_{e \in E} \left[w_e \log\frac{w_e}{q_e} + (1-w_e)\log\frac{1-w_e}{1-q_e}\right]
```

where $w_e$ are high-dimensional fuzzy edge weights and $q_e = (1 + a\|y_i - y_j\|^{2b})^{-1}$ are low-dimensional weights with learnable parameters $a, b$. HDBSCAN on the UMAP coordinates produces density-faithful cluster boundaries. Tags: `shape_agnostic`, `no_k_needed`, `noise_robust`.

---

### Extra Algorithms

| ID | Name | Description |
|----|------|-------------|
| `clara` | CLARA | K-Medoids on random subsamples; scalable to large $n$ |
| `threshold_clustering` | Threshold Clustering | Distance-based single-pass: merge if $d < \theta$ |
| `robust_cc` | Robust CC | Correlation clustering with noise tolerance |
| `sparse_spectral` | Sparse Spectral | Spectral on sparse affinity graph (k-NN only) |
| `online_gmm` | Online GMM | Mini-batch EM for streaming GMM estimation |
| `repeated_bisecting` | Repeated Bisecting | Bisecting K-Means with multiple restart averaging |

---

## Evaluation Framework

### Internal Validity Indices

**Silhouette Score**

```math
s(i) = \frac{b(i) - a(i)}{\max\!\left(a(i),\, b(i)\right)}, \qquad S = \frac{1}{n}\sum_{i=1}^n s(i) \in [-1, 1]
```

Graded interpretation: $[0.70, 1]$ strong, $[0.50, 0.70)$ reasonable, $[0.25, 0.50)$ weak, $[0, 0.25)$ no structure, $< 0$ incorrect assignments.

---

**Davies-Bouldin Index**

```math
\mathrm{DBI} = \frac{1}{k}\sum_{j=1}^k \max_{l \neq j} \frac{\sigma_j + \sigma_l}{d(\mu_j, \mu_l)}
```

---

**Calinski-Harabász Score**

```math
\mathrm{CH} = \frac{\mathrm{tr}(B_k)\,(n-k)}{\mathrm{tr}(W_k)\,(k-1)}, \quad B_k = \sum_j n_j(\mu_j - \bar{\mu})(\mu_j - \bar{\mu})^\top
```

The total scatter decomposition $T = B_k + W_k$ is invariant; maximising CH maximises the ratio of between- to within-cluster variance per degree of freedom.

---

**Dunn Index**

```math
\mathrm{DI} = \frac{\displaystyle\min_{i \neq j} \delta(C_i, C_j)}{\displaystyle\max_k \Delta(C_k)}
```

---

**Xie-Beni Index**

```math
\mathrm{XB} = \frac{\frac{1}{n}\displaystyle\sum_j \sum_{x \in C_j}\|x-\mu_j\|^2}{\displaystyle\min_{i \neq j}\|\mu_i - \mu_j\|^2}
```

---

**Inertia** (WCSS)

```math
\mathcal{I} = \sum_{j=1}^k \sum_{x \in C_j}\|x - \mu_j\|^2
```

The elbow in $\mathcal{I}(k)$ vs $k$ is detected by the kneedle algorithm (maximum discrete second derivative $|\mathcal{I}(k+1) - 2\mathcal{I}(k) + \mathcal{I}(k-1)|$).

---

**S\_DBw Index**

```math
\mathrm{S\_DBw} = \frac{1}{k}\sum_j \frac{\|\sigma_j\|}{\|\sigma\|} + \frac{1}{k(k-1)}\sum_{i \neq j} \frac{\rho(u_{ij})}{\max\!\left(\rho(\mu_i),\,\rho(\mu_j)\right)}
```

---

### External Validity Indices (when ground truth is available)

**Adjusted Rand Index (ARI)**

```math
\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}{\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]}
```

Chance-corrected pair-counting; ranges $[-1, 1]$, expected 0 for random assignments.

**Adjusted Mutual Information (AMI)**

```math
\mathrm{AMI}(U,V) = \frac{\mathrm{MI}(U,V) - \mathbb{E}[\mathrm{MI}(U,V)]}{\mathrm{avg}(H(U), H(V)) - \mathbb{E}[\mathrm{MI}(U,V)]}
```

where $\mathrm{MI}(U,V) = \sum_{u,v} P(u,v) \log \frac{P(u,v)}{P(u)P(v)}$ is mutual information and $H$ is Shannon entropy.

**Normalised Mutual Information (NMI)**

```math
\mathrm{NMI}(U,V) = \frac{2\cdot \mathrm{MI}(U,V)}{H(U) + H(V)}
```

NMI = 0 for independent labellings; NMI = 1 for identical labellings. The normalisation by $H(U) + H(V)$ corrects for the trivial dependence on cluster count.

**Cluster Purity**

```math
\mathrm{Purity} = \frac{1}{n}\sum_j \max_l |C_j \cap G_l|
```

where $G_l$ are ground-truth groups. Purity is always between $1/k$ (worst case) and 1 (perfect).

---

### Composite Scoring & Pareto Front

The `CompositeScorer` computes a normalised weighted combination:

```math
\mathrm{Score}(R) = \frac{\sum_m w_m \cdot \tilde{v}_m(R)}{\sum_m w_m}
```

with $\tilde{v}_m$ normalising metric $m$ to $[0,1]$ (respecting direction). Weights: Silhouette 3.0, DBI 2.0, CH 1.5, Dunn 1.0, XB 1.0.

The **Pareto Front** provides multi-objective selection — an algorithm $R_i$ is Pareto-optimal if no other algorithm is strictly better on all selected metrics simultaneously:

```math
\mathrm{PF} = \bigl\{R_i : \nexists\, R_j \text{ s.t. } v_m(R_j) \succeq v_m(R_i)\;\forall m,\; v_{m'}(R_j) \succ v_{m'}(R_i)\;\exists m'\bigr\}
```

The **k-Sweep Analyser** (`KSweepAnalyser`) runs any algorithm over a range of $k$ values, computes all metrics, and uses `find_optimal_k` combining elbow (maximum second derivative of $\mathcal{I}(k)$) and metric peak detection to recommend $k$.

---

## Preprocessing Pipeline

The `PreprocessingPipeline` executes 8 stages:

1. **Data Loading** — CSV/Excel/JSON/Parquet/TSV, max 500K × 2000
2. **Deep Profiling** — per-column: dtype, unique count, missing%, mean, std, skewness ($g_1 = \mu_3/\mu_2^{3/2}$), excess kurtosis ($g_2 = \mu_4/\mu_2^2 - 3$), outlier%, high-correlation pairs
3. **Missing Value Imputation** — mean, median, mode, KNN, MICE (iterative), constant, drop
4. **Outlier Detection** — Z-score, IQR, Isolation Forest, LOF, Elliptic Envelope; actions: remove, clip, winsorise, flag
5. **Categorical Encoding** — one-hot, ordinal, target encoding
6. **Feature Scaling** — Standard, MinMax, Robust (median/IQR), MaxAbs, L1/L2 Normalizer, Box-Cox/Yeo-Johnson, QuantileTransformer
7. **Feature Selection** — variance threshold, correlation filter, PCA (target variance), ANOVA F-test
8. **Column Dropping** — explicit user-selected columns

### Outlier Detection Mathematics

**Z-Score:**

```math
z_i = \frac{x_i - \mu}{\sigma}; \quad \text{flag}\ |z_i| > 3
```

**IQR:**

```math
\text{flag}\ x_i < Q_1 - 1.5\,\mathrm{IQR}\ \text{or}\ x_i > Q_3 + 1.5\,\mathrm{IQR}
```

**Isolation Forest:** anomaly score

```math
s(x,n) = 2^{-\,E[h(x)]\,/\,c(n)}, \quad c(n) = 2H(n-1) - \frac{2(n-1)}{n}
```

where $h(x)$ is the path length in a random isolation tree and $H$ is the harmonic number.

**LOF:**

```math
\mathrm{LOF}_k(p) = \frac{\displaystyle\sum_{o \in N_k(p)} \frac{\mathrm{lrd}_k(o)}{\mathrm{lrd}_k(p)}}{|N_k(p)|}, \quad \mathrm{lrd}_k(p) = \left(\frac{\sum_{o \in N_k(p)} \mathrm{reach\text{-}dist}_k(p,o)}{|N_k(p)|}\right)^{-1}
```

LOF $\approx 1$: regular point; LOF $\gg 1$: outlier (locally sparser than neighbours).

---

## Stability & Robustness Engine

### Five Perturbation Regimes

| Regime | Perturbation | Formal Definition |
|--------|-------------|-------------------|
| **Bootstrap** | Subsample 80% w/o replacement | $X^{(b)} \sim \mathrm{Subsample}(X, 0.8)$ |
| **Gaussian Noise** | Additive i.i.d. Gaussian | $\tilde{x}_i = x_i + \epsilon,\ \epsilon \sim \mathcal{N}(0, \sigma^2 I)$ |
| **Laplacian Noise** | Heavy-tailed additive | $\tilde{x}_i = x_i + \epsilon,\ \epsilon \sim \mathrm{Lap}(0, b)$ |
| **Feature Dropout** | Random column zeroing | $\tilde{x}_i^{(j)} = 0$ for $j \in \mathcal{D} \sim \mathrm{Binom}(d, p_{\mathrm{drop}})$ |
| **Subset Sampling** | Progressive data fraction | $X^{(f)} = \mathrm{Sample}(X, f\cdot n)$, $f \in [0.5, 1.0]$ |

For each regime at level $\sigma$, ARI is measured against the full-data reference labelling. The Laplacian distribution $\mathrm{Lap}(0, b)$ has heavier tails than Gaussian ($\mathrm{kurtosis} = 6$ vs 0), providing a more challenging corruption regime for testing outlier robustness.

The **persistence score** measures how consistently the same pairwise structure is preserved across noise levels:

```math
\mathrm{persistence}(C_j) = \int_{\sigma_{\min}}^{\sigma_{\max}} \mathrm{ARI}(L(\sigma), L_{\mathrm{ref}})\,d\sigma \approx \sum_l \mathrm{ARI}(L(\sigma_l), L_{\mathrm{ref}})\cdot\Delta\sigma
```

### Stability Grades

```math
\mathrm{Grade} = \begin{cases} \mathbf{S}^+ & \bar{\mathrm{ARI}} \geq 0.90\ \text{(Superior)} \\ \mathbf{A} & \bar{\mathrm{ARI}} \in [0.80, 0.90) \\ \mathbf{B} & \bar{\mathrm{ARI}} \in [0.65, 0.80) \\ \mathbf{C} & \bar{\mathrm{ARI}} \in [0.45, 0.65) \\ \mathbf{D} & \bar{\mathrm{ARI}} < 0.45\ \text{(Unstable)} \end{cases}
```

### Stability Report Schema

Each `StabilityReport` contains:
- Per-regime ARI distribution (mean, std, min, max, histogram)
- Per-cluster persistence scores
- Composite stability score (weighted average across regimes)
- Stability grade with certified assessment
- Noise degradation curve (ARI vs $\sigma$)
- Feature importance by dropout (features whose removal most degrades stability)
- `MultiAlgorithmStabilityComparator`: side-by-side comparison across all evaluated algorithms

### Consensus Analysis

Co-association matrix:

```math
A_{ij} = \frac{1}{M}\sum_{m=1}^M \mathbf{1}\bigl[L_m(i) = L_m(j)\bigr]
```

built over $M$ runs. Optimal $k$ minimises:

```math
\mathrm{PAC}(k) = F(0.9) - F(0.1)
```

from the empirical CDF $F$ of $A$ entries. Values of PAC near 0 indicate decisive co-association (all $A_{ij}$ near 0 or 1).

**Cophenetic correlation**: $r = \mathrm{corr}(1 - A_{ij},\ \mathrm{cophenetic\,distance}_{ij})$; measures how faithfully the dendrogram represents the co-association.

**LabelStabilityMatrix**: tracks individual sample label agreement across runs, identifying structurally ambiguous points — those with high entropy over their cluster assignments.

---

## Adaptive Execution Engine

### Dataset Size Classification

```python
class DatasetSizeClass(str, Enum):
    TINY    # n < 100
    SMALL   # n < 1,000
    MEDIUM  # n < 10,000
    LARGE   # n < 100,000
    XLARGE  # n ≥ 100,000
```

The engine consults `_SIZE_BLACKLIST` — a mapping from size class to algorithm IDs that are incompatible (e.g. Affinity Propagation for XLARGE datasets, with $O(n^2)$ space requirement), automatically filtering the run queue.

### Adaptive Parameter Tuning

`DBSCANParamTuner`: estimates optimal $\varepsilon$ from the $k$-NN distance elbow and optimal `min_samples` from $\lfloor\log n\rfloor$ heuristic (motivated by the expected number of points in a $d$-ball of radius $\varepsilon$ under a Poisson process). Applied automatically before each DBSCAN run.

### Thread-Safe LRU Result Cache

`ResultCache` (max 200 entries): keys results by `(algorithm_id, data_hash, params_hash)`. On cache hit, skips computation entirely. Reports hit rate as a session metric.

### Runtime Tracker

`RuntimeTracker` maintains an **exponential moving average** of per-algorithm runtimes:

```math
\hat{T}_j^{(t+1)} = (1-\alpha)\,\hat{T}_j^{(t)} + \alpha\,T_j^{(t)}, \quad \alpha = 0.3
```

enabling predictive scheduling and user-facing ETA estimates.

---

## AI Oracle — Gemini Intelligence

UnSuPERvIsED-II features a dedicated **AI Oracle** page — a deeply integrated conversational AI system powered by **Google Gemini 2.0 Flash Lite**, representing a more advanced AI integration than the basic AI tab in UnSuPERvIsED-I.

### Architecture

The AI Oracle is implemented as a full standalone page (Page 9 in the navigation) with a stateful conversation history preserved across the session. The core query function:

```python
def _gemini_query(prompt: str, context: str = "", model: str = "gemini-2.0-flash-lite") -> str:
    """Send a structured query to Gemini with full clustering context."""
```

Unlike UnSuPERvIsED-I's template-based AI tab, the Oracle supports **four integrated analysis modes**:

### Analysis Modes

**1. Contextual Result Interpretation**
Passes a comprehensive structured context including: all algorithm names, metric values (Silhouette, DBI, CH, Dunn, XB, ARI, AMI, NMI), stability grades (S+/A/B/C/D), perturbation regime scores, preprocessing diagnostics, and dataset shape to Gemini. The model synthesises these into natural-language interpretations of *why* one algorithm outperforms another on a given dataset.

**2. Algorithm Selection Advisor**
Given dataset characteristics (size class, dimensionality, estimated density, Hopkins statistic, skewness/kurtosis profile, missing rate), the Oracle reasons over the 60-algorithm registry and recommends the most suitable family and specific algorithms with mathematical justification.

**3. Hyperparameter Recommendation Engine**
Analyses the k-NN distance profile, elbow curve, gap statistic, and PAC curve together to recommend optimal $\varepsilon$, `min_samples`, $k$, and fuzzifier $m$ values with confidence intervals derived from the stability engine.

**4. Free-Form Conversational Q&A**
Full multi-turn conversation with session history preserved (up to 5 recent turns shown). Users can ask arbitrary questions about their clustering results, specific algorithms, metric interpretations, or deployment recommendations.

### Setup

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-key-here"
```

The Oracle gracefully degrades: if no API key is configured, it displays a clear setup prompt without breaking other dashboard functionality.

### Key Distinction from UnSuPERvIsED-I

| Feature | UnSuPERvIsED-I AI Tab | UnSuPERvIsED-II AI Oracle |
|---------|----------------------|--------------------------|
| Model | Gemini Flash Lite (legacy) | Gemini 2.0 Flash Lite |
| Placement | Tab 7 (basic tab) | Dedicated Page 9 |
| Analysis modes | 4 fixed templates | 4 modes + free-form chat |
| Context depth | Metric values + grades | Full pipeline state + perturbation data |
| Conversation history | Session-scoped | Session-scoped (last 5 shown) |
| Integration level | Post-hoc analysis | Context-aware at every stage |

---

## Visualisation Suite

UnSuPERvIsED-II renders 30+ interactive Plotly charts:

| Component | Description |
|-----------|-------------|
| 2D/3D scatter | PCA/t-SNE/UMAP projections with cluster colouring |
| Silhouette bar | Per-sample silhouette sorted by cluster and score |
| Elbow curve | WCSS vs k with kneedle detection |
| Silhouette curve | $S(k)$ with optimal k marker |
| Gap statistic | $\mathrm{Gap}(k) \pm 1\,\mathrm{SE}$ |
| PCA scree | Cumulative explained variance |
| PCA biplot | PC1 vs PC2 with loading vectors |
| Correlation heatmap | Feature correlation matrix |
| Missing heatmap | Missing value pattern |
| Consensus heatmap | Co-association matrix $A$ |
| Radar chart | Normalised multi-metric comparison |
| Violin plot | Per-cluster feature distribution |
| Pair plot | Scatter matrix (≤6 features) |
| Parallel coordinates | High-dim feature structure |
| Sunburst | Cluster membership hierarchy |
| Feature histograms | Per-cluster overlaid histograms |
| Box plots | Per-cluster feature box-and-whisker |
| Cluster size bar | Cluster membership counts |
| Dendrogram | Hierarchical tree (≤500 samples) |
| Stability bars | ARI per algorithm with grade colour codes |
| Perturbation curve | ARI vs noise level ± 1σ bands |
| Consensus CDF | Empirical CDF of co-association values |
| Noise response chart | ARI vs perturbation regime/level heatmap |
| ARI box plots | Bootstrap ARI distribution per algorithm |
| Persistence heatmap | Per-cluster persistence across perturbation levels |
| Metrics table | Colour-coded normalised comparison table |
| Hopkins gauge | Clusterability $H$ speedometer |
| k-NN distance | Sorted $k$-NN distances with ε suggestion |
| Overlap heatmap | Pairwise cluster overlap matrix |
| Score timeline | Metric scores vs run index |

**Dimensionality Reduction:**

**PCA**: $Z = XW_k$ (top-$k$ eigenvectors of $X^\top X$). Explained variance ratio $\mathrm{EVR}_l = \lambda_l / \sum_i \lambda_i$.

**t-SNE**: KL divergence minimisation with perplexity-adaptive bandwidth. High-dimensional symmetrised similarities $p_{ij}$; low-dimensional Student-t kernel $q_{ij} \propto (1 + \|y_i - y_j\|^2)^{-1}$. Cost: $\mathcal{L} = \mathrm{KL}(P\|Q)$.

**UMAP**: Fuzzy simplicial complex with Riemannian metric and stochastic gradient descent on cross-entropy $\mathcal{L}_{\mathrm{UMAP}}$.

---

## Dashboard Architecture

```
Navigation Tabs
├── 📤 Upload / 🎲 Samples / 🔍 Deep Profile   — Data ingestion and exploration
├── ⚙️  Preprocessing                           — Pipeline configuration and Hopkins test
├── 🧬 Algorithm Selection                      — Family browser, tag filters, recommendations
├── ⚡ Run & Evaluate                           — Batch execution, elbow, gap, Pareto front
├── 📊 Visualizations                           — Full visualisation suite
├── 🧪 Stability Lab                            — 5-regime stability, grades, comparator
├── 🤖 AI Oracle                                — Gemini 2.0 intelligence, multi-mode analysis
└── 💾 Export Suite                             — Labels, metrics, consensus, stability JSON
```

The lazy-loading architecture imports heavy modules (stability, consensus, visualisation) only when the user navigates to the relevant tab, keeping initial startup time minimal. Session state stores all intermediate results with downstream-reset logic ensuring consistency.

---

## Installation & Usage

```bash
git clone https://github.com/Devanik21/Substrata-Matrix.git
cd Substrata-Matrix/Intermediate_Cluster
pip install -r requirements.txt
streamlit run UnSuPERvIsED.py
```

**Supported formats:** CSV, Excel, JSON, Parquet, TSV — up to 500,000 rows × 2,000 columns.

---

## Dependencies

```
streamlit
numpy
pandas
scikit-learn
scipy
plotly
hdbscan
sklearn-extra
umap-learn
tensorflow          # Autoencoder-KMeans
minisom             # SOM Clustering
skfuzzy             # Fuzzy/Possibilistic C-Means
google-generativeai # AI Oracle
```

---

## Difference from UnSuPERvIsED-I

| Feature | UnSuPERvIsED-I | UnSuPERvIsED-II |
|---------|----------------|-----------------|
| Total lines | ~12,000 | ~14,700 |
| Algorithm count | 19 | 60+ |
| Algorithm families | 5 | 12 |
| Neural clustering | ✗ | ✓ (Autoencoder, SOM) |
| Fuzzy clustering | ✗ | ✓ (FCM, PCM) |
| Manifold pipelines | ✗ | ✓ (Isomap, LLE, UMAP+HDBSCAN) |
| Ensemble methods | ✗ | ✓ (Voting, Random Subspace) |
| Grid-based | ✗ | ✓ (WaveCluster) |
| Message-passing | Affinity Prop. | Full AP spec + enhanced |
| Perturbation regimes | 3 | 5 (+ Laplacian, Subset) |
| Pareto front selection | ✗ | ✓ |
| External metrics | ✗ | ✓ (ARI, AMI, NMI, Purity) |
| Adaptive param tuning | ✗ | ✓ (DBSCANParamTuner) |
| LRU result cache | Disk-based | In-memory, thread-safe |
| Dataset size blacklist | ✗ | ✓ |
| Runtime tracker | ✗ | ✓ (EMA scheduling) |
| Gemini AI | ✓ Flash Lite (template tab) | ✓ **2.0 Flash Lite (AI Oracle — dedicated page, multi-mode)** |
| Stability grades | A/B/C/D | S+/A/B/C/D |
| Persistence analysis | ✗ | ✓ |
| k-sweep trend analysis | Basic | ✓ (KSweepAnalyser) |

---

## Citation

If you use UnSuPERvIsED-II in research or build upon it, please cite:

```bibtex
@software{devanik2026unsupervised2,
  author       = {Devanik},
  title        = {UnSuPERvIsED-II: A 60-Algorithm Clustering Intelligence System
                  with Adaptive Execution, Multi-Family Coverage,
                  and Five-Regime Stability Certification},
  year         = {2026},
  month        = {May},
  version      = {2.0},
  url          = {https://github.com/Devanik21/Substrata-Matrix},
  note         = {Intermediate\_Cluster module of the Substrata-Matrix repository}
}
```

**Core algorithmic references:**

- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Trans. Inf. Theory*, 28(2), 129–137.
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA*, 1027–1035.
- Ester, M., et al. (1996). A density-based algorithm for discovering clusters. *KDD*, 226–231.
- Campello, R. J. G. B., et al. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD*, 160–172.
- Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms*. Plenum Press.
- Krishnapuram, R., & Keller, J. (1993). A possibilistic approach to clustering. *IEEE Trans. Fuzzy Syst.*, 1(2), 98–110.
- Xie, X. L., & Beni, G. (1991). A validity measure for fuzzy clustering. *IEEE TPAMI*, 13(8), 841–847.
- Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. *Science*, 290(5500), 2319–2323.
- Roweis, S. T., & Saul, L. K. (2000). Nonlinear dimensionality reduction by locally linear embedding. *Science*, 290(5500), 2323–2326.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*.
- Kohonen, T. (1990). The self-organizing map. *Proc. IEEE*, 78(9), 1464–1480.
- Xing, E., Jordan, M., Russell, S., & Ng, A. (2002). Distance metric learning with application to clustering. *NeurIPS*.
- Frey, B. J., & Dueck, D. (2007). Clustering by passing messages between data points. *Science*, 315(5814), 972–976.
- von Luxburg, U. (2007). A tutorial on spectral clustering. *Stat. Comput.*, 17(4), 395–416.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *J. R. Stat. Soc. B*, 39(1), 1–38.
- Monti, S., et al. (2003). Consensus Clustering. *Machine Learning*, 52(1–2), 91–118.
- Zhang, T., Ramakrishnan, R., & Livny, M. (1996). BIRCH. *SIGMOD Record*, 25(2), 103–114.
- Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE TPAMI*, 1(2), 224–227.
- Rousseeuw, P. J. (1987). Silhouettes. *J. Comput. Appl. Math.*, 20, 53–65.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *ICDM*, 413–422.
- Tibshirani, R., Walther, G., & Hastie, T. (2001). Gap statistic. *J. R. Stat. Soc. B*, 63(2), 411–423.
- Caliński, T., & Harabász, J. (1974). A dendrite method for cluster analysis. *Commun. Stat.*, 3(1), 1–27.
- Hinneburg, A., & Keim, D. (1998). An efficient approach to clustering in large multimedia databases with noise. *KDD*, 58–65. [DENCLUE]
- Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space. *Contemp. Math.*, 26, 189–206. [JL Lemma for Projected K-Means]
- Wang, W., Yang, J., & Muntz, R. (1997). STING. *VLDB*, 186–195. [Grid-based]
- Agrawal, R., et al. (1998). Automatic subspace clustering of high dimensional data. *KDD*, 94–105. [CLIQUE]

---

## Repository Structure

```
Intermediate_Cluster/
├── UnSuPERvIsED.py          # Main Streamlit application (~3,558 lines)
├── clustering_registry.py   # 60+ algorithm specs, 12 families (~1,967 lines)
├── clustering_runner.py     # Adaptive cache-aware execution engine (~1,367 lines)
├── evaluation.py            # Metrics, composite scorer, Pareto front (~1,341 lines)
├── preprocessing.py         # Multi-stage preprocessing pipeline (~1,506 lines)
├── stability.py             # 5-regime stability + persistence analysis (~1,635 lines)
├── consensus.py             # Co-association, PAC, cophenetic (~1,561 lines)
├── visualization.py         # 30+ Plotly chart components (~1,763 lines)
└── requirements.txt
```

---

*UnSuPERvIsED-II · May 2026 · Devanik · NIT Agartala · Samsung ISWDP Fellow*
*Part of the Substrata-Matrix project — [github.com/Devanik21/Substrata-Matrix](https://github.com/Devanik21/Substrata-Matrix)*

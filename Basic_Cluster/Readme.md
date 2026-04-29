
#  UnSuPERvIsED-I : The Fundamental Clustering Intelligence System

<img width="1672" height="941" alt="ChatGPT Image Apr 28, 2026, 07_21_20 PM" src="https://github.com/user-attachments/assets/c6a8e662-0b32-40af-a693-b61480690f68" />

> *"Structure is not imposed on data — it is discovered within it."*

---

## About This Repository

**Author:** Devanik (GitHub: [Devanik21](https://github.com/Devanik21))
**Repository:** [Substrata-Matrix](https://github.com/Devanik21/Substrata-Matrix) · `Basic_Cluster/` · May 2025
**Affiliation:** Electronics & Communication Engineering, NIT Agartala · Samsung ISWDP Fellow (IISc, 98.58th percentile)

This module is the **first generation** of the *UnSuPERvIsED* series — a production-grade unsupervised learning workbench that orchestrates 19+ clustering algorithms through a unified interactive platform. UnSuPERvIsED-I lays the mathematical and architectural foundation that all subsequent versions extend: rigorous multi-metric evaluation, clusterability testing, bootstrap stability, consensus clustering, and a full-featured dark-themed Streamlit dashboard with Gemini AI insights. Every design decision is grounded in the statistical theory of unsupervised learning.

---

## What Is UnSuPERvIsED-I?

UnSuPERvIsED-I (`Basic_Cluster/`) is a self-contained Streamlit application spanning **~12,000 lines** of modular Python across eight files. It ingests arbitrary tabular data, runs a configurable preprocessing pipeline, executes multiple clustering algorithms in parallel with timeout protection, evaluates results on a battery of internal and external validity indices, and provides deep stability analysis — all rendered in a responsive dark-themed UI. A Gemini-powered AI assistant synthesises results into natural-language insights and recommendations.

The architecture deliberately separates concerns into seven independent modules:

| Module | Role | Lines |
|--------|------|-------|
| `UnSuPERvIsED.py` | Streamlit orchestrator, session state, UI rendering | ~2,144 |
| `clustering_registry.py` | Algorithm registry, metadata, factory functions | ~1,131 |
| `clustering_runner.py` | Parallel execution, timeout, elbow/gap analysis | ~1,174 |
| `evaluation.py` | Metric engine, ranking, profiling, interpretability | ~1,073 |
| `preprocessing.py` | Multi-stage preprocessing pipeline | ~1,059 |
| `stability.py` | Bootstrap stability, perturbation robustness | ~1,634 |
| `stability_consensus.py` | Consensus matrices, PAC, cophenetic correlation | ~1,061 |
| `visualization.py` | 25+ Plotly visualisations, dark theme | ~1,116 |

---

## Algorithm Library

### Family Taxonomy

```
AlgorithmFamily
├── PARTITIONAL    — centroid-optimisation methods
├── HIERARCHICAL   — agglomerative and divisive linkage trees
├── DENSITY        — reachability and density-kernel approaches
├── SPECTRAL       — Laplacian eigendecomposition
├── MODEL_BASED    — probabilistic mixture models
├── FUZZY          — soft-membership frameworks
└── ENSEMBLE       — aggregated clustering
```

### Full Algorithm Registry

#### Partitional Family

**K-Means** (`kmeans`)

The canonical centroid algorithm. Given 
```math
n
```
 points 
```math
\{x_i\}_{i=1}^n \subset \mathbb{R}^d
```
 and 
```math
k
```
 clusters, K-Means minimises the within-cluster sum of squares (WCSS):

```math

```math
\underset{C}{\arg\min} \sum_{j=1}^{k} \sum_{x_i \in C_j} \|x_i - \mu_j\|_2^2
```

```

where 
```math
\mu_j = \frac{1}{|C_j|}\sum_{x_i \in C_j} x_i
```
 is the centroid of cluster 
```math
j
```
. The Lloyd–Forgy algorithm alternates the **E-step** (nearest-centroid assignment):

```math

```math
c(i) = \underset{j \in \{1,\ldots,k\}}{\arg\min}\ \|x_i - \mu_j\|_2^2
```

```

and the **M-step** (centroid recomputation) until convergence. Each step is individually optimal given the other, constituting coordinate descent on the joint WCSS objective. Convergence to a local minimum is guaranteed in finite steps since the number of distinct partitions is bounded by 
```math
k^n
```
; the WCSS is non-increasing at every iteration.

Initialisation uses **K-Means++**: the first centroid is sampled uniformly; subsequent centroids 
```math
c_l
```
 are sampled with probability 
```math
P(x_i) \propto \min_{j < l} \|x_i - c_j\|^2
```
. This 
```math
D^2
```
 weighting yields the approximation guarantee:

```math
\mathbb{E}[\mathrm{WCSS}_{k\text{-means}++}] \leq 8(\ln k + 2)\cdot \mathrm{WCSS}_{\mathrm{OPT}}
```

- **Complexity:** Time 
```math
O(nkd \cdot T)
```
 per iteration 
```math
T
```
; Space 
```math
O(kd)
```

- **Parameters:** `n_clusters` (k), `init` {k-means++, random}, `n_init`, `max_iter`, `tol`, `algorithm` {lloyd, elkan}
- **Scalability:** Medium (≤100K samples)

---

**MiniBatch K-Means** (`minibatch_kmeans`)

Stochastic approximation that processes random subsets (mini-batches) of size 
```math
b \ll n
```
 per iteration. The per-centre learning rate is 
```math
\eta_{c_j} = 1/(1 + n_{c_j})
```
 where 
```math
n_{c_j}
```
 counts prior assignments, and the centre update is:

```math
\mu_j^{(t+1)} = \left(1 - \eta_{c_j}\right)\mu_j^{(t)} + \eta_{c_j}\cdot \overline{x}_{\mathcal{B},j}
```

This harmonic learning rate schedule satisfies the Robbins-Monro conditions 
```math
\sum_t \eta_t = \infty
```
, 
```math
\sum_t \eta_t^2 < \infty
```
, guaranteeing almost-sure convergence. The algorithm achieves near-equivalent quality to full K-Means at 
```math
O(b)
```
 per step.

- **Complexity:** Time 
```math
O(bkd)
```
 per step; Space 
```math
O(kd)
```

- **Parameters:** `n_clusters`, `batch_size`, `max_iter`, `n_init`
- **Scalability:** Large (>1M samples viable)

---

**K-Medoids** (`kmedoids`)

A robust generalisation: medoids 
```math
m_j
```
 must be actual data points, minimising total dissimilarity:

```math

```math
\underset{m \subset X,\, |m|=k}{\arg\min} \sum_{j=1}^k \sum_{x_i \in C_j} d(x_i, m_j)
```

```

for arbitrary dissimilarity 
```math
d
```
. The PAM (Partitioning Around Medoids) algorithm evaluates swap cost 
```math
\Delta_{jh}
```
 for every medoid 
```math
m_j
```
 and non-medoid 
```math
x_h
```
, accepting swaps with 
```math
\Delta_{jh} < 0
```
. Unlike K-Means, K-Medoids is robust to outliers and applicable to non-Euclidean spaces (Jaccard, cosine, DTW). **K-Medoids++** provides an 
```math
O(\log k)
```
 approximation guarantee analogous to K-Means++.

- **Complexity:** Time 
```math
O(k(n-k)^2)
```
 per iteration; Space 
```math
O(n^2)
```
 for distance matrix
- **Parameters:** `n_clusters`, `metric`, `init` {k-medoids++, random}

---

**Bisecting K-Means** (`bisecting_kmeans`)

A divisive hierarchical strategy that iteratively bisects the cluster with maximum intra-cluster variance:

```math

```math
C^* = \underset{C_j}{\arg\max} \sum_{x_i \in C_j} \|x_i - \mu_j\|^2
```

```

At each step, K-Means with 
```math
k=2
```
 is applied to 
```math
C^*
```
, producing a binary tree. After 
```math
k-1
```
 bisections, 
```math
k
```
 clusters are produced. The total WCSS strictly decreases at every bisection since the two children always have lower combined WCSS than the parent. The strategy {`biggest_intra_cluster_variance`, `largest_cluster`} determines the selection policy.

- **Complexity:** Time 
```math
O(nk \cdot T \cdot \log k)
```
; Space 
```math
O(n)
```

---

#### Density Family

**DBSCAN** — Density-Based Spatial Clustering of Applications with Noise (`dbscan`)

DBSCAN constructs clusters from density-connected regions defined by radius 
```math
\varepsilon
```
 and minimum point count 
```math
\mathrm{MinPts}
```
. A point 
```math
p
```
 is a **core point** if 
```math
|N_\varepsilon(p)| \geq \mathrm{MinPts}
```
. Two core points are **density-connected** if there exists a chain 
```math
p = p_0, p_1, \ldots, p_m = q
```
 with 
```math
p_{i+1} \in N_\varepsilon(p_i)
```
. A cluster is the maximal density-connected set; border points are reachable from a core but not core themselves; noise points receive label 
```math
-1
```
.

```math
C_j = \bigl\{x \mid \exists\;\text{density-reachability path from some}\; c \in \mathrm{CorePoints}\;\text{to}\; x\bigr\}
```

The system estimates 
```math
\varepsilon
```
 automatically via the **k-NN Distance Profile**: sorting 
```math
k
```
-th nearest-neighbour distances and locating the elbow via maximum curvature (kneedle algorithm).

- **Complexity:** Time 
```math
O(n \log n)
```
 with spatial indexing; Space 
```math
O(n)
```

- **Parameters:** `eps` 
```math
\varepsilon
```
, `min_samples` MinPts, `metric`, `algorithm` {auto, ball_tree, kd_tree}

---

**HDBSCAN** — Hierarchical DBSCAN (`hdbscan`)

HDBSCAN extends DBSCAN by defining **mutual reachability distance**:

```math
d_{\mathrm{mreach}\text{-}k}(a,b) = \max\!\left(\mathrm{core}_k(a),\;\mathrm{core}_k(b),\;d(a,b)\right)
```

where 
```math
\mathrm{core}_k(a) = d(a, \mathrm{kNN}_k(a))
```
 is the core distance. The minimum spanning tree of the mutual-reachability graph is extracted; the condensed hierarchy selects persistent clusters by maximising **cluster stability**:

```math
\lambda_{\mathrm{birth}}(C) = \frac{1}{\varepsilon_{\mathrm{split}}(C)}, \qquad \mathrm{stability}(C) = \sum_{x \in C} \bigl(\lambda_{\mathrm{death}}(x) - \lambda_{\mathrm{birth}}(C)\bigr)
```

The Excess of Mass (EOM) criterion selects the subset of clusters maximising total stability. HDBSCAN requires no 
```math
\varepsilon
```
 parameter and produces soft cluster probabilities.

- **Complexity:** Time 
```math
O(n \log n)
```
; Space 
```math
O(n)
```

- **Parameters:** `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, `alpha`, `cluster_selection_method` {eom, leaf}

---

**OPTICS** — Ordering Points To Identify the Clustering Structure (`optics`)

OPTICS computes a reachability ordering encoding the full density hierarchy. For each point, the **reachability distance** is:

```math
\mathrm{reach\text{-}dist}_k(p, q) = \max\!\left(\mathrm{core\text{-}dist}_k(q),\;d(p,q)\right)
```

The reachability plot reveals cluster structure as valleys; cluster extraction applies 
```math
\xi
```
-steep descent detection or a global 
```math
\varepsilon
```
 threshold. The 
```math
\xi
```
-method requires no global density threshold, revealing multi-scale structure.

- **Complexity:** Time 
```math
O(n^2)
```
 worst case, 
```math
O(n \log n)
```
 with indexing; Space 
```math
O(n)
```

- **Parameters:** `min_samples`, `max_eps`, `xi`, `cluster_method` {xi, dbscan}

---

**Mean Shift** (`meanshift`)

Non-parametric mode-seeking algorithm. Each point iteratively migrates toward the mean of points within bandwidth 
```math
h
```
:

```math
m(x) = \frac{\displaystyle\sum_{x_i \in N_h(x)} K\!\left(\frac{x - x_i}{h}\right) x_i}{\displaystyle\sum_{x_i \in N_h(x)} K\!\left(\frac{x - x_i}{h}\right)}
```

with Gaussian kernel 
```math
K(u) = \exp(-\|u\|^2 / 2)
```
. The mean shift vector 
```math
m(x) - x
```
 is proportional to 
```math
\nabla \hat{f}_h(x)
```
 (gradient of the KDE), so updates climb the density surface. With bandwidth 
```math
h
```
 estimated via Silverman's rule, the algorithm requires no 
```math
k
```
.

- **Complexity:** Time 
```math
O(Tn^2)
```
; Space 
```math
O(n)
```

- **Parameters:** `bandwidth`, `bin_seeding`, `min_bin_freq`, `cluster_all`

---

#### Hierarchical Family

**Agglomerative Clustering** (`agglomerative`)

Bottom-up linkage clustering merges the closest pair at each step. For clusters 
```math
A, B
```
:

| Linkage | Distance 
```math
D(A,B)
```
 | Behaviour |
|---------|-------------------|-----------|
| **Ward** | 
```math
\frac{n_A n_B}{n_A + n_B}\|\mu_A - \mu_B\|^2
```
 | Minimises intra-cluster variance increase |
| **Complete** | 
```math
\max_{a \in A, b \in B} d(a,b)
```
 | Penalises outliers; compact clusters |
| **Average (UPGMA)** | 
```math
\frac{1}{|A||B|}\sum_{a \in A}\sum_{b \in B} d(a,b)
```
 | Compromise; less outlier-sensitive |
| **Single** | 
```math
\min_{a \in A, b \in B} d(a,b)
```
 | Chaining effect; detects elongated shapes |

Ward merge cost equals the increase in WCSS:

```math
\Delta(A, B) = \frac{n_A\, n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2
```

All linkages are unified by the **Lance–Williams recurrence** for updating distances to the merged cluster 
```math
A \cup B
```
 from existing cluster 
```math
C
```
:

```math
D(A \cup B,\, C) = \alpha_A\, D(A,C) + \alpha_B\, D(B,C) + \beta\, D(A,B) + \gamma\, |D(A,C) - D(B,C)|
```

with family-specific coefficients. This avoids recomputing all pairwise distances, enabling 
```math
O(n^2 \log n)
```
 priority-queue implementations.

- **Complexity:** Naïve 
```math
O(n^3)
```
; with Lance–Williams and priority queue 
```math
O(n^2 \log n)
```

- **Parameters:** `n_clusters`, `linkage` {ward, complete, average, single}, `metric`

---

**BIRCH** — Balanced Iterative Reducing and Clustering using Hierarchies (`birch`)

BIRCH uses a **Clustering Feature (CF)** tree to summarise data incrementally. Each CF node stores 
```math
(n, LS, SS)
```
: count, linear sum, and squared sum. Merging two CFs is 
```math
O(1)
```
:

```math
CF_1 + CF_2 = (n_1 + n_2,\; LS_1 + LS_2,\; SS_1 + SS_2)
```

The `threshold` controls maximum subcluster radius:

```math
R = \sqrt{\frac{SS}{n} - \left\|\frac{LS}{n}\right\|^2} \leq \mathrm{threshold}
```

Phase 1 builds the CF tree in a single pass; Phase 2 clusters leaf subclusters; Phase 3 reassigns original points.

- **Complexity:** Time 
```math
O(n)
```
; Space 
```math
O(\text{tree size})
```
; single-pass
- **Parameters:** `n_clusters`, `threshold`, `branching_factor`

---

**Feature Agglomeration** (`feature_agglomeration`)

Transposed agglomerative approach: clusters *features* (columns) rather than samples using Ward linkage on the feature correlation structure. Grouped features are replaced by their mean, enabling clustering in semantically coherent feature spaces. The feature affinity matrix 
```math
(\Sigma_F)_{ij} = \rho(f_i, f_j)
```
 defines the feature space.

- **Complexity:** 
```math
O(d^2 \log d)
```
 on 
```math
d
```
 features

---

#### Spectral Family

**Spectral Clustering** (`spectral`)

Spectral clustering embeds data into low-dimensional Euclidean space via graph Laplacian eigenvectors, then applies K-Means:

1. Affinity matrix: 
```math
W_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)
```

2. Normalised Laplacian:

```math
\mathcal{L}_{\mathrm{sym}} = D^{-1/2}(D - W)D^{-1/2}, \quad D_{ii} = \sum_j W_{ij}
```

3. Solve 
```math
\mathcal{L}_{\mathrm{sym}}\, u = \lambda\, u
```
; collect 
```math
k
```
 smallest eigenvectors into 
```math
U \in \mathbb{R}^{n \times k}
```

4. Row-normalise 
```math
U
```
 and apply K-Means

The **spectral gap** 
```math
\delta_k = \lambda_{k+1} - \lambda_k
```
 reveals natural cluster count: under ideal separation, 
```math
\lambda_1 = \cdots = \lambda_k = 0
```
 and 
```math
\lambda_{k+1} > 0
```
. The multiplicity of eigenvalue 0 of the unnormalised Laplacian equals the number of connected components. The `discretize` assignment variant rounds the embedding matrix via iterative rotation:

```math
\min_{R \in \mathcal{O}(k)} \|U - VR\|_F, \quad V_{ij} = u_{ij}/\|u_i\|_2
```

- **Complexity:** Time 
```math
O(n^3)
```
 for dense 
```math
W
```
; Space 
```math
O(n^2)
```

- **Parameters:** `n_clusters`, `affinity` {rbf, nearest_neighbors}, `gamma`, `eigen_solver`, `assign_labels`

---

#### Model-Based Family

**Gaussian Mixture Model** (`gmm`)

GMM models data as a convex combination of 
```math
k
```
 Gaussians:

```math
p(x) = \sum_{j=1}^{k} \pi_j \cdot \mathcal{N}(x \mid \mu_j, \Sigma_j)
```

Parameters estimated via **EM**:

**E-step** — posterior responsibilities:

```math
r_{ij} = \frac{\pi_j\,\mathcal{N}(x_i \mid \mu_j, \Sigma_j)}{\displaystyle\sum_{l=1}^k \pi_l\,\mathcal{N}(x_i \mid \mu_l, \Sigma_l)}
```

**M-step** — weighted sufficient statistics:

```math
\pi_j = \frac{1}{n}\sum_i r_{ij},\quad \mu_j = \frac{\sum_i r_{ij}\, x_i}{\sum_i r_{ij}},\quad \Sigma_j = \frac{\sum_i r_{ij}(x_i - \mu_j)(x_i - \mu_j)^\top}{\sum_i r_{ij}}
```

EM monotonically increases the log-likelihood 
```math
\mathcal{L} = \sum_i \log p(x_i)
```
 at every iteration (by Jensen's inequality applied to the Q-function lower bound). BIC for model selection:

```math
\mathrm{BIC}(k) = -2\hat{\mathcal{L}} + p_k \ln n, \quad p_k = k\bigl[d + d(d+1)/2 + 1\bigr] - 1 \;\text{(full covariance)}
```

Covariance types: `full`, `tied`, `diag`, `spherical`.

- **Complexity:** Time 
```math
O(nkd^2 \cdot T)
```
; Space 
```math
O(kd^2)
```

- **Parameters:** `n_clusters`, `covariance_type`, `max_iter`, `n_init`, `tol`, `init_params`

---

**Bayesian Gaussian Mixture** (`bgmm`)

BGMM places conjugate priors over all GMM parameters:

```math
\pi \sim \mathrm{Dir}(\alpha/k, \ldots, \alpha/k), \quad \mu_j \sim \mathcal{N}(0, \beta^{-1}I), \quad \Lambda_j \sim \mathcal{W}(\nu, W)
```

Inference via **variational EM** maximises the Evidence Lower BOund (ELBO):

```math
\mathcal{L}_{\mathrm{ELBO}} = \mathbb{E}_q\bigl[\log p(X, Z, \theta)\bigr] - \mathbb{E}_q\bigl[\log q(Z, \theta)\bigr]
```

The Dirichlet concentration 
```math
\alpha
```
 controls sparsity: as 
```math
\alpha \to 0
```
, most components receive 
```math
\pi_j \approx 0
```
 (automatic model order selection). The `dirichlet_process` prior encourages parsimonious solutions.

- **Parameters:** `n_clusters`, `covariance_type`, `weight_concentration_prior_type`, `weight_concentration_prior`

---

**Affinity Propagation** (`affinity_propagation`)

Message-passing algorithm. Similarity 
```math
s(i,k) = -\|x_i - x_k\|^2
```
 (negative squared Euclidean by default).

**Responsibility** 
```math
r(i,k)
```
 — evidence that 
```math
k
```
 should serve as exemplar for 
```math
i
```
:

```math
r(i,k) \leftarrow s(i,k) - \max_{k' \neq k}\bigl\{a(i,k') + s(i,k')\bigr\}
```

**Availability** 
```math
a(i,k)
```
 — evidence that 
```math
i
```
 should choose 
```math
k
```
:

```math
a(i,k) \leftarrow \min\!\left(0,\; r(k,k) + \sum_{i' \notin \{i,k\}} \max(0, r(i',k))\right)
```

Self-availability: 
```math
a(k,k) \leftarrow \sum_{i' \neq k} \max(0, r(i',k))
```
.

Exemplars identified where 
```math
a(k,k) + r(k,k) > 0
```
. Damping factor 
```math
\lambda \in [0.5, 1)
```
 stabilises oscillations: 
```math
r \leftarrow (1-\lambda)r_{\mathrm{new}} + \lambda r_{\mathrm{old}}
```
.

- **Complexity:** Time 
```math
O(n^2 T)
```
; Space 
```math
O(n^2)
```

- **Parameters:** `damping`, `preference`, `max_iter`, `convergence_iter`

---

## Evaluation Framework

UnSuPERvIsED-I computes a comprehensive battery of validity indices, combining them via weighted composite ranking.

### Internal Validity Indices

**Silhouette Score**

For each point 
```math
i
```
 in cluster 
```math
C_j
```
: 
```math
a(i)
```
 = mean intra-cluster distance; 
```math
b(i) = \min_{l \neq j} \overline{d}(i, C_l)
```
 = mean distance to nearest other cluster.

```math
s(i) = \frac{b(i) - a(i)}{\max\!\left(a(i),\, b(i)\right)}, \qquad S = \frac{1}{n}\sum_{i=1}^n s(i) \in [-1, 1]
```

```math
s(i) = 1
```
 indicates perfect cohesion (
```math
a(i) \to 0
```
); 
```math
s(i) = -1
```
 indicates misassignment (
```math
b(i) \to 0
```
). Interpretation thresholds: 
```math
[0.70, 1]
```
 strong; 
```math
[0.50, 0.70)
```
 reasonable; 
```math
[0.25, 0.50)
```
 weak; 
```math
< 0.25
```
 no structure.

---

**Davies-Bouldin Index**

```math
\mathrm{DBI} = \frac{1}{k}\sum_{j=1}^k \max_{l \neq j} \left\{ \frac{\sigma_j + \sigma_l}{d(\mu_j, \mu_l)} \right\}
```

where 
```math
\sigma_j = \frac{1}{|C_j|}\sum_{x \in C_j} d(x, \mu_j)
```
. DBI rewards compact clusters (small 
```math
\sigma_j
```
) that are well-separated (large 
```math
d(\mu_j, \mu_l)
```
). Lower is better; optimal clusters score 0.

---

**Calinski-Harabász Score (Variance Ratio Criterion)**

```math
\mathrm{CH} = \frac{\mathrm{tr}(B_k)\,/\,(k-1)}{\mathrm{tr}(W_k)\,/\,(n-k)}
```

where the between-cluster scatter matrix is 
```math
B_k = \sum_{j=1}^k n_j (\mu_j - \bar{\mu})(\mu_j - \bar{\mu})^\top
```
 and the within-cluster scatter matrix is 
```math
W_k = \sum_j \sum_{x \in C_j} (x - \mu_j)(x - \mu_j)^\top
```
. The total scatter 
```math
T = B_k + W_k
```
 is constant; maximising CH is equivalent to maximising the ratio of explained-to-unexplained variance. Higher is better.

---

**Dunn Index**

```math
\mathrm{DI} = \frac{\displaystyle\min_{i \neq j} \delta(C_i, C_j)}{\displaystyle\max_k \Delta(C_k)}
```

where 
```math
\delta(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x,y)
```
 is the inter-cluster minimum distance and 
```math
\Delta(C_k) = \max_{x,y \in C_k} d(x,y)
```
 is the cluster diameter. Higher values indicate compact, well-separated clusters. DI is sensitive to outliers (a single outlier inflates 
```math
\Delta
```
).

---

**Xie-Beni Index**

```math
\mathrm{XB} = \frac{\displaystyle\sum_{j=1}^k \sum_{x \in C_j} \|x - \mu_j\|^2}{n \cdot \displaystyle\min_{i \neq j} \|\mu_i - \mu_j\|^2}
```

Compactness-to-separation ratio; lower is better. The numerator is WCSS; the denominator is 
```math
n
```
 times the minimum squared centroid separation.

---

**S\_DBw Index**

```math
\mathrm{S\_DBw} = \mathrm{Scat}(k) + \mathrm{Dens\_bw}(k)
```

where 
```math
\mathrm{Scat}(k) = \frac{1}{k}\sum_j \frac{\|\sigma_j\|}{\|\sigma\|}
```
 measures average cluster scatter relative to dataset scatter, and

```math
\mathrm{Dens\_bw}(k) = \frac{1}{k(k-1)}\sum_{i \neq j} \frac{\rho(u_{ij})}{\max\!\left(\rho(\mu_i),\,\rho(\mu_j)\right)}
```

measures density at mid-points 
```math
u_{ij} = (\mu_i + \mu_j)/2
```
 relative to centroid densities. Both terms should be minimised.

---

**Gap Statistic**

Compares observed WCSS against a null reference distribution (uniform over data bounding box):

```math
\mathrm{Gap}(k) = \mathbb{E}_n^*\!\left[\log W_k\right] - \log W_k
```

where 
```math
W_k = \sum_j \frac{1}{2n_j} D_j
```
 is pooled within-cluster dispersion. The expectation is over 
```math
B
```
 uniformly drawn reference datasets. Optimal 
```math
k
```
 satisfies:

```math
\mathrm{Gap}(k) \geq \mathrm{Gap}(k+1) - s_{k+1}, \quad s_{k+1} = \mathrm{std}_b\!\left[\log W_{k+1}^{(b)}\right]\sqrt{1 + \tfrac{1}{B}}
```

---

**Hopkins Statistic** (Clusterability Test)

```math
H = \frac{\displaystyle\sum_{i=1}^m u_i^d}{\displaystyle\sum_{i=1}^m u_i^d + \sum_{i=1}^m w_i^d}
```

where 
```math
u_i
```
 = nearest-neighbour distance from a uniformly random point to data, and 
```math
w_i
```
 = nearest-neighbour distance from a data point to another data point. Under a uniform distribution, 
```math
H \sim \mathrm{Beta}(m,m)
```
 with mean 
```math
0.5
```
. 
```math
H \to 1
```
 indicates strong clustering tendency. The null hypothesis 
```math
H_0
```
: uniform distribution is rejected when 
```math
H > 0.75
```
.

---

**Partition Entropy**

The information content of a clustering assignment:

```math
\mathcal{H}(C) = -\sum_{j=1}^k \frac{n_j}{n}\log_2 \frac{n_j}{n}
```

A balanced clustering (
```math
n_j = n/k\ \forall j
```
) achieves maximum entropy 
```math
\log_2 k
```
; a degenerate single-cluster result gives 
```math
\mathcal{H} = 0
```
. Used internally to detect imbalanced assignments.

---

### Composite Ranking

```math
\mathrm{Score}(R) = \sum_{m} w_m \cdot \tilde{v}_m(R)
```

where 
```math
\tilde{v}_m
```
 normalises metric 
```math
m
```
 to 
```math
[0,1]
```
 (respecting direction) and 
```math
w_m
```
 are predefined weights (Silhouette: 3, DBI: 2, CH: 1.5, Dunn: 1). A pairwise win matrix counts how many metrics each algorithm "wins" against every other.

---

## Preprocessing Pipeline

The `PreprocessingPipeline` applies the following stages in sequence:

1. **Data Loading** — CSV, Excel (xlsx/xls), JSON, Parquet, TSV via `DataLoader` (max 500K × 2000)
2. **Data Profiling** — per-column statistics (dtype, unique count, missing%, mean, std, skewness 
```math
g_1 = \mu_3/\mu_2^{3/2}
```
, excess kurtosis 
```math
g_2 = \mu_4/\mu_2^2 - 3
```
, outlier%), high-correlation pair detection (
```math
r > 0.9
```
)
3. **Missing Value Imputation** — `mean`, `median`, `most_frequent`, `knn` (k-NN), `iterative` (MICE), `constant`, `drop_rows`, `drop_cols`
4. **Outlier Detection & Handling**
   - Z-score: flag 
```math
|z_i| > 3
```
 where 
```math
z_i = (x_i - \mu)/\sigma
```

   - IQR: flag 
```math
x_i < Q_1 - 1.5\,\mathrm{IQR}
```
 or 
```math
x_i > Q_3 + 1.5\,\mathrm{IQR}
```

   - Isolation Forest, Local Outlier Factor, Elliptic Envelope
   - Action: remove, clip to fence, winsorise, flag, ignore
5. **Categorical Encoding** — one-hot, ordinal, target encoding
6. **Feature Scaling** — StandardScaler, MinMaxScaler, RobustScaler (median/IQR), MaxAbsScaler, Normalizer (L1/L2/max), PowerTransformer (Box-Cox/Yeo-Johnson), QuantileTransformer
7. **Feature Selection**
   - Variance threshold: 
```math
\mathrm{Var}(f) < \tau
```
 → drop
   - Correlation filter: 
```math
|\rho(f_i, f_j)| > \theta
```
 → drop one
   - PCA: retain 
```math
v\%
```
 explained variance
   - ANOVA F-test feature ranking
8. **Column Dropping** — user-specified columns (e.g. ground-truth labels)

### k-NN Distance Profile for DBSCAN ε Estimation

Sorted 
```math
k
```
-th nearest-neighbour distances with **kneedle algorithm** (maximum discrete second derivative):

```math

```math
\hat{\varepsilon} = d_k\!\left[\underset{i}{\arg\max}\ \bigl|d_k[i+1] - 2\,d_k[i] + d_k[i-1]\bigr|\right]
```

```

This discrete approximation to curvature 
```math
\kappa \approx |y''|
```
 identifies the point of maximum rate-of-change in the sorted distance curve — the optimal density threshold for DBSCAN.

---

## Stability & Consensus Analysis

### Bootstrap Stability

The `StabilityPipeline` resamples the dataset 
```math
B
```
 times (subsample ratio 
```math
\rho
```
), re-runs the algorithm, and measures consistency via **Adjusted Rand Index (ARI)**:

```math
\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}{\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]}
```

where the Rand Index 
```math
\mathrm{RI} = \frac{a + d}{\binom{n}{2}}
```
 counts concordant pairs (
```math
a
```
: same cluster in both; 
```math
d
```
: different clusters in both). ARI ranges in 
```math
[-1, 1]
```
 with expected value 0 for random labelling and 1 for perfect agreement. **Jaccard Index**:

```math
J = \frac{|\mathrm{TP}|}{|\mathrm{TP}| + |\mathrm{FP}| + |\mathrm{FN}|}
```

Stability grades: **A** (ARI ≥ 0.85), **B** (≥0.65), **C** (≥0.40), **D** (<0.40).

### Consensus Clustering

The `ConsensusPipeline` builds the **co-association matrix** over 
```math
M
```
 runs:

```math
A_{ij} = \frac{\text{number of runs where } x_i,\, x_j \text{ co-cluster}}{\text{number of runs where both appear}}
```

```math
A \in [0,1]^{n \times n}
```
 symmetric; final clusters are obtained by agglomerative clustering (average linkage) on 
```math
1 - A
```
.

**PAC Score** (Proportion of Ambiguous Clustering):

```math
\mathrm{PAC}(k) = F(\theta_2) - F(\theta_1), \quad [\theta_1, \theta_2] = [0.1, 0.9]
```

where 
```math
F(\theta)
```
 is the empirical CDF of 
```math
A
```
 entries. Optimal 
```math
k
```
 minimises PAC — values near 0 indicate decisive (non-ambiguous) co-association.

**Cophenetic Correlation**: Pearson correlation between co-association distances 
```math
(1 - A_{ij})
```
 and dendrogram cophenetic distances; high values (>0.9) indicate the hierarchy faithfully represents the consensus matrix.

### Perturbation Analysis

Gaussian noise 
```math
\tilde{x}_i = x_i + \epsilon
```
, 
```math
\epsilon \sim \mathcal{N}(0, \sigma^2 I)
```
 at levels 
```math
\sigma_{\mathrm{noise}} \in \{0.05, 0.10, 0.20, 0.30\}
```
; ARI is measured against the clean-data result. Robustness score = mean ARI across noise levels.

### Cross-Validation Stability

```math
\mathrm{CV\text{-}ARI} = \frac{1}{k}\sum_{i=1}^k \mathrm{ARI}\!\left(\hat{y}^{(i)},\, \hat{y}_{\mathrm{full}}\right)
```

### Temporal Stability

Incremental data checkpoints 
```math
(0.1, 0.2, \ldots, 1.0)
```
 are clustered; ARI between consecutive checkpoints measures stability as data accumulates — a key property for streaming deployments.

---

## Visualisation Suite

UnSuPERvIsED-I renders 25+ interactive Plotly charts within the dark-theme CSS framework:

| Plot | Description |
|------|-------------|
| `ScatterPlotter.scatter_2d` | 2D cluster scatter after PCA/t-SNE/UMAP projection |
| `ScatterPlotter.scatter_3d` | 3D cluster scatter (interactive rotation) |
| `SilhouettePlotter.plot` | Per-sample silhouette bar chart sorted by cluster and score |
| `ElbowPlotter.plot_elbow` | WCSS vs k with marked elbow kneepoint |
| `ElbowPlotter.plot_silhouette_curve` | Silhouette vs k with marked optimal |
| `ElbowPlotter.plot_gap_statistic` | Gap(k) ± 1 s.e. with optimal k marker |
| `PCAPlotter.variance_explained` | Scree plot — cumulative explained variance |
| `PCAPlotter.biplot` | PC1 vs PC2 with feature loading vectors |
| `RadarPlotter.plot_comparison` | Normalised metric radar for ≤5 algorithms |
| `HeatmapPlotter.correlation_heatmap` | Feature correlation matrix |
| `HeatmapPlotter.missing_value_heatmap` | Missing value pattern heatmap |
| `HeatmapPlotter.consensus_heatmap` | Co-association matrix heatmap |
| `ViolinPlotter.plot` | Per-cluster feature value violin |
| `PairPlotter.plot` | Scatter matrix (up to 6 features) |
| `SunburstPlotter.plot` | Cluster membership sunburst |
| `DistributionPlotter.feature_histogram` | Per-cluster feature histogram overlay |
| `DistributionPlotter.parallel_coordinates` | Parallel coordinate plot |
| `DistributionPlotter.feature_boxplots` | Per-cluster feature box plot |
| `DistributionPlotter.cluster_sizes` | Cluster size bar chart |
| `DendrogramPlotter.plot` | Hierarchical dendrogram (≤500 samples) |
| `StabilityPlotter.stability_bars` | ARI per algorithm bar chart with grade colour-coding |
| `StabilityPlotter.perturbation_curve` | ARI vs noise level with error bands |
| `StabilityPlotter.consensus_cdf` | Consensus matrix empirical CDF |
| `GaugePlotter.metric_gauge` | Speedometer gauge for stability score |
| `OverlapHeatmapPlotter.plot` | Pairwise cluster overlap matrix |
| `NNDistancePlotter.plot` | k-NN distance curve with ε suggestion |
| `HopkinsPlotter.plot` | Hopkins H gauge |
| `MetricsTablePlotter.plot` | Colour-coded normalised comparison table |

**Dimensionality Reduction Methods:**

**PCA** — Linear projection: 
```math
Z = XW_k
```
 where 
```math
W_k \in \mathbb{R}^{d \times k}
```
 contains the top-
```math
k
```
 eigenvectors of 
```math
X^\top X
```
. Explained variance ratio: 
```math
\mathrm{EVR}_l = \lambda_l / \sum_i \lambda_i
```
.

**t-SNE** — KL divergence minimisation between pairwise similarity distributions. High-dimensional similarities (perplexity controls 
```math
\sigma_i
```
):

```math
p_{j|i} = \frac{\exp\!\left(-\|x_i - x_j\|^2 / 2\sigma_i^2\right)}{\displaystyle\sum_{k \neq i} \exp\!\left(-\|x_i - x_k\|^2 / 2\sigma_i^2\right)}, \quad p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}
```

Low-dimensional Student-t kernel (heavy-tailed, combats crowding):

```math
q_{ij} = \frac{\left(1 + \|y_i - y_j\|^2\right)^{-1}}{\displaystyle\sum_{k \neq l}\left(1 + \|y_k - y_l\|^2\right)^{-1}}
```

Minimised cost: 
```math
\mathcal{L}_{\mathrm{KL}} = \mathrm{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
```
.

**UMAP** — Topological approach preserving local manifold structure via Riemannian geometry and fuzzy simplicial complex representations.

---

## AI Insights — Gemini Integration

The AI tab passes a structured summary of clustering results (algorithm names, all metric values, stability grades, preprocessing diagnostics) to **Google Gemini Flash Lite** via the `google.generativeai` SDK. Four quick-analysis templates cover: data quality assessment, k-selection rationale, stability interpretation, and recommended next steps. Conversation history is preserved within the session; AI reports are exportable as Markdown.

---

## Dashboard Architecture

Seven navigation tabs provide the complete workflow:

```
1. 📊 Data Profile      — Column statistics, correlation matrix, missing heatmap
2. ⚙️  Preprocessing    — Pipeline config, Hopkins test, k-NN ε profile, diagnostics
3. 🧬 Algorithms        — Registry browser, family filter, parameter config, recommendations
4. 🚀 Run Clustering    — Batch execution, elbow/gap analysis, results table, radar chart
5. 📈 Visualizations    — 2D/3D scatter, silhouette, PCA biplot, violin, pair, sunburst
6. 🔒 Stability         — Bootstrap ARI, consensus heatmap, perturbation curves, CV
7. 🤖 AI Insights       — Gemini analysis, quick templates, export labels/metrics/report
```

Session state is managed centrally; all results are cached with MD5-keyed persistence across UI interactions.

---

## Installation & Usage

```bash
git clone https://github.com/Devanik21/Substrata-Matrix.git
cd Substrata-Matrix/Basic_Cluster
pip install -r requirements.txt
streamlit run UnSuPERvIsED.py
```

**Gemini AI Setup** (optional):

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-key-here"
```

**Supported input formats:** CSV, Excel (xlsx/xls), JSON, Parquet, TSV — up to 500,000 rows × 2,000 columns.

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
google-generativeai
umap-learn
```

---

## Citation

If you use UnSuPERvIsED-I in research or build upon it, please cite:

```bibtex
@software{devanik2025unsupervised1,
  author       = {Devanik},
  title        = {UnSuPERvIsED-I: A Clustering Intelligence Lab
                  with 19+ Algorithms, Multi-Metric Evaluation,
                  Consensus Clustering, and AI-Powered Insights},
  year         = {2025},
  month        = {May},
  version      = {1.0},
  url          = {https://github.com/Devanik21/Substrata-Matrix},
  note         = {Basic\_Cluster module of the Substrata-Matrix repository}
}
```

**Foundational algorithmic references:**

- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Trans. Inf. Theory*, 28(2), 129–137.
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA*, 1027–1035.
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters. *KDD*, 226–231.
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD*, 160–172.
- Ankerst, M., Breunig, M. M., Kriegel, H.-P., & Sander, J. (1999). OPTICS. *SIGMOD Record*, 28(2), 49–60.
- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters via the gap statistic. *J. R. Stat. Soc. B*, 63(2), 411–423.
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *J. Comput. Appl. Math.*, 20, 53–65.
- Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE TPAMI*, 1(2), 224–227.
- Frey, B. J., & Dueck, D. (2007). Clustering by passing messages between data points. *Science*, 315(5814), 972–976.
- Zhang, T., Ramakrishnan, R., & Livny, M. (1996). BIRCH: an efficient data clustering method. *SIGMOD Record*, 25(2), 103–114.
- Comaniciu, D., & Meer, P. (2002). Mean shift: a robust approach toward feature space analysis. *IEEE TPAMI*, 24(5), 603–619.
- Monti, S., et al. (2003). Consensus Clustering. *Machine Learning*, 52(1–2), 91–118.
- von Luxburg, U. (2007). A tutorial on spectral clustering. *Stat. Comput.*, 17(4), 395–416.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *J. R. Stat. Soc. B*, 39(1), 1–38.
- Hopkins, B. (1954). A new method for determining the type of distribution of plant individuals. *Ann. Bot.*, 18(2), 213–227.

---

## Repository Structure

```
Basic_Cluster/
├── UnSuPERvIsED.py          # Main Streamlit application
├── clustering_registry.py   # Algorithm metadata and factory functions
├── clustering_runner.py     # Parallel execution engine
├── evaluation.py            # Metric computation and ranking
├── preprocessing.py         # Multi-stage preprocessing pipeline
├── stability.py             # Bootstrap and perturbation stability
├── stability_consensus.py   # Consensus matrix and PAC analysis
├── visualization.py         # 25+ Plotly visualisation components
└── requirements.txt
```

---

*UnSuPERvIsED-I · May 2025 · Devanik · NIT Agartala · Samsung ISWDP Fellow*
*Part of the Substrata-Matrix project — [github.com/Devanik21/Substrata-Matrix](https://github.com/Devanik21/Substrata-Matrix)*

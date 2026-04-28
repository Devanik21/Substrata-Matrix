#  UnSuPERvIsED — Master Clustering Intelligence Reference

<img width="1672" height="941" alt="ChatGPT Image Apr 28, 2026, 07_41_12 PM" src="https://github.com/user-attachments/assets/b4e86343-154a-41d6-b37a-757e2acfc214" />


> *"Structure is not imposed on data — it is discovered within it."*

**Author:** Devanik · [GitHub: Devanik21](https://github.com/Devanik21)
**Repository:** [Substrata-Matrix](https://github.com/Devanik21/Substrata-Matrix)
**Affiliation:** Electronics & Communication Engineering, NIT Agartala · Samsung ISWDP Fellow (IISc, 98.58th percentile)

---

## Table of Contents

1. [Series Overview](#series-overview)
2. [Architecture Comparison](#architecture-comparison)
3. [Mathematical Foundations of Clustering](#mathematical-foundations-of-clustering)
4. [Algorithm Library — All Families](#algorithm-library)
   - [Centroid / Partitional Family](#centroid--partitional-family)
   - [Hierarchical Family](#hierarchical-family)
   - [Density Family](#density-family)
   - [Distribution / Model-Based Family](#distribution--model-based-family)
   - [Graph & Spectral Family](#graph--spectral-family)
   - [Neural / Deep Family](#neural--deep-family)
   - [Fuzzy Family](#fuzzy-family)
   - [Manifold Family](#manifold-family)
   - [Subspace & Ensemble Families](#subspace--ensemble-families)
   - [Grid-Based Family](#grid-based-family)
5. [Evaluation Framework](#evaluation-framework)
   - [Internal Validity Indices](#internal-validity-indices)
   - [External Validity Indices](#external-validity-indices)
   - [Composite Ranking & Pareto Front](#composite-ranking--pareto-front)
6. [Preprocessing Pipeline](#preprocessing-pipeline)
7. [Stability & Consensus Analysis](#stability--consensus-analysis)
8. [AI Integration — Gemini Oracle](#ai-integration--gemini-oracle)
9. [Visualisation Suite](#visualisation-suite)
10. [Dashboard Architecture](#dashboard-architecture)
11. [Installation & Usage](#installation--usage)
12. [Dependencies](#dependencies)
13. [Citation](#citation)
14. [References](#references)

---

## Series Overview

The **UnSuPERvIsED** series is a production-grade unsupervised machine learning platform built for rigorous scientific exploration of clustering algorithms. It provides a unified dark-themed Streamlit dashboard covering the complete unsupervised learning pipeline: data ingestion, preprocessing, algorithm execution, multi-metric evaluation, stability certification, consensus clustering, dimensionality reduction, interactive visualisation, and AI-powered analysis.

| Property | UnSuPERvIsED-I (`Basic_Cluster/`) | UnSuPERvIsED-II (`Intermediate_Cluster/`) |
|----------|-----------------------------------|-------------------------------------------|
| Version | 1.0 · May 2025 | 2.0 · May 2026 |
| Codebase | ~12,000 lines | ~14,700 lines |
| Algorithms | 19 | 60+ |
| Families | 7 | 12 |
| Stability regimes | 3 | 5 |
| External metrics | — | ARI, AMI, NMI, Purity |
| Pareto front | — | ✓ |
| Neural clustering | — | Autoencoder + K-Means, SOM |
| Fuzzy clustering | — | FCM, PCM |
| Manifold pipelines | — | Isomap, LLE, UMAP+HDBSCAN |
| AI assistant | Gemini Flash Lite (tab) | Gemini 2.0 Flash Lite (Oracle page) |
| LRU result cache | Disk-based | In-memory, thread-safe (200 entries) |
| Adaptive param tuning | — | DBSCANParamTuner |
| Stability grades | A/B/C/D | S+/A/B/C/D |
| Persistence analysis | — | ✓ |

---

## Architecture Comparison

### UnSuPERvIsED-I Module Map

| Module | Role | Lines |
|--------|------|-------|
| `UnSuPERvIsED.py` | Streamlit orchestrator, session state | ~2,144 |
| `clustering_registry.py` | Algorithm registry, factory functions | ~1,131 |
| `clustering_runner.py` | Parallel execution, timeout, elbow/gap | ~1,174 |
| `evaluation.py` | Metric engine, ranking, profiling | ~1,073 |
| `preprocessing.py` | Multi-stage preprocessing pipeline | ~1,059 |
| `stability.py` | Bootstrap stability, perturbation | ~1,634 |
| `stability_consensus.py` | Consensus matrices, PAC, cophenetic | ~1,061 |
| `visualization.py` | 25+ Plotly visualisations, dark theme | ~1,116 |

### UnSuPERvIsED-II Module Map

| Module | Role | Lines |
|--------|------|-------|
| `UnSuPERvIsED.py` | Streamlit orchestrator, lazy loading | ~3,558 |
| `clustering_registry.py` | 60+ specs, 12 families, tags, factory | ~1,967 |
| `clustering_runner.py` | Cache-aware adaptive scheduler | ~1,367 |
| `evaluation.py` | Internal + external metrics, Pareto | ~1,341 |
| `preprocessing.py` | Multi-stage pipeline + deep profiling | ~1,506 |
| `stability.py` | 5-regime stability, persistence | ~1,635 |
| `consensus.py` | Co-association, PAC, cophenetic | ~1,561 |
| `visualization.py` | 30+ Plotly charts, dark-theme CSS | ~1,763 |

---

## Mathematical Foundations of Clustering

### The Clustering Problem

Given a dataset  ```math
X = \{x_1, x_2, \ldots, x_n\} \subset \mathbb{R}^d
``` , clustering seeks a partition  ```math
\mathcal{C} = \{C_1, C_2, \ldots, C_k\}
```  such that  ```math
\bigcup_j C_j = X
``` ,  ```math
C_j \cap C_l = \emptyset
```  for  ```math
j \neq l
``` , and some objective function  ```math
\mathcal{F}(\mathcal{C})
```  is extremised.

The **fundamental trade-off** in all clustering is between *intra-cluster cohesion* and *inter-cluster separation*. Formally, let the within-cluster scatter and between-cluster scatter matrices be:


```math
W_k = \sum_{j=1}^k \sum_{x \in C_j} (x - \mu_j)(x - \mu_j)^\top
```



```math
B_k = \sum_{j=1}^k n_j (\mu_j - \bar{x})(\mu_j - \bar{x})^\top
```


where  ```math
\mu_j = \frac{1}{|C_j|} \sum_{x \in C_j} x
```  is the centroid of cluster ```math j ``` and  ```math
\bar{x} = \frac{1}{n}\sum_i x_i
```  is the grand mean. The total scatter matrix satisfies the identity  ```math
T = W_k + B_k
```  regardless of the partition. Good clustering maximises  ```math
\text{tr}(B_k)
```  and minimises  ```math
\text{tr}(W_k)
```  simultaneously.

### Clusterability — Hopkins Statistic

Before running any algorithm, clusterability of the data is assessed via the **Hopkins statistic**. Let  ```math
m \ll n
```  points  ```math
\{u_i\}
```  be sampled uniformly from the data bounding box, and let  ```math
\{w_i\}
```  be ```math m ``` randomly chosen data points. Define:


```math
H = \frac{\sum_{i=1}^m u_i^d}{\sum_{i=1}^m u_i^d + \sum_{i=1}^m w_i^d}
```


where  ```math
u_i = \min_{x \in X} \|u_i - x\|
```  is the nearest-data-point distance for each uniform sample, and ```math w_i ``` is the nearest-other-data-point distance. Under a completely random (Poisson) spatial distribution in  ```math
\mathbb{R}^d
``` :


```math
H \sim \text{Beta}(m, m), \quad \mathbb{E}[H] = 0.5
```


Strong clustering pushes  ```math
H \to 1
```  because data points are close to each other (small ```math w_i ```) while uniformly random points are far from data (large ```math u_i ```). The null hypothesis of spatial randomness is rejected when  ```math
H > 0.75
```  at the 0.05 significance level.

### Partition Entropy

The **information content** of a clustering assignment is:


```math
\mathcal{H}(\mathcal{C}) = -\sum_{j=1}^k \frac{n_j}{n} \log_2 \frac{n_j}{n}
```


A perfectly balanced clustering achieves  ```math
\mathcal{H} = \log_2 k
```  (maximum); a degenerate single-cluster solution yields  ```math
\mathcal{H} = 0
``` .

---

## Algorithm Library

### Full Family Taxonomy

```
UnSuPERvIsED Algorithm Registry
├── I.  CENTROID / PARTITIONAL  (6 algorithms in v2, 4 in v1)
├── II. HIERARCHICAL            (7 algorithms in v2, 3 in v1)
├── III.DENSITY                 (5 algorithms in v2, 4 in v1)
├── IV. DISTRIBUTION            (5 algorithms in v2, 2 in v1)
├── V.  GRAPH / SPECTRAL        (5 algorithms in v2, 1 in v1)
├── VI. NEURAL / DEEP           (2 algorithms, v2 only)
├── VII.FUZZY                   (2 algorithms, v2 only)
├── VIII.MANIFOLD               (3 algorithms, v2 only)
├── IX. SUBSPACE                (2 algorithms, v2 only)
├── X.  ENSEMBLE                (2 algorithms, v2 only)
├── XI. GRID                    (1 algorithm,  v2 only)
└── XII.EXTRA                   (6 algorithms, v2 only)
```

---

### Centroid / Partitional Family

#### K-Means

Given ```math n ``` points and ```math k ``` clusters, K-Means minimises the within-cluster sum of squares (WCSS):


```math
\min_{\{C_j\}} \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - \mu_j\|_2^2, \quad \mu_j = \frac{1}{|C_j|}\sum_{x_i \in C_j} x_i
```


The **Lloyd–Forgy** algorithm performs coordinate descent on this objective via alternating steps. The **E-step** assigns each point to its nearest centroid:


```math
c(i) = \arg\min_{j \in \{1,\ldots,k\}} \|x_i - \mu_j\|_2^2
```


The **M-step** recomputes centroids as cluster means. Each step is individually globally optimal given the other, so the WCSS is non-increasing at every iteration. Since the number of distinct partitions is at most ```math k^n ```, convergence to a local minimum in finite steps is guaranteed. The WCSS decrease per iteration satisfies:


```math
\mathcal{L}^{(t+1)} \leq \mathcal{L}^{(t)}, \quad \mathcal{L}^{(t)} = \sum_{j=1}^k \sum_{x_i \in C_j^{(t)}} \|x_i - \mu_j^{(t)}\|^2
```


**K-Means++ initialisation** provides an approximation guarantee. The first centroid is drawn uniformly; each subsequent centroid ```math c_l ``` is drawn with probability:


```math
P(x_i) \propto \min_{j < l} \|x_i - c_j\|^2
```


This ```math D^2 ```-weighting ensures:


```math
\mathbb{E}[\text{WCSS}_{k\text{-means}++}] \leq 8(\ln k + 2) \cdot \text{WCSS}_\text{OPT}
```


The **Elkan variant** avoids redundant distance calculations using the triangle inequality:  ```math
\|x - c_j\| \geq \max(0,\ \|x - c_i\| - \|c_i - c_j\|)
``` .

- **Complexity:**  ```math
O(nkdT)
```  time,  ```math
O(kd)
```  space
- **Parameters:** `n_clusters`, `init`, `n_init`, `max_iter`, `tol`, `algorithm`

---

#### K-Means Auto (v2)

Estimates the optimal ```math k ``` via BIC on a spherical GMM approximation, then runs K-Means++:


```math
\text{BIC}(k) = -2 \hat{\mathcal{L}}(X;\, \hat{\theta}_k) + p_k \ln n, \quad p_k = k(d+1)
```


The selected cluster count is  ```math
k^* = \arg\min_k \text{BIC}(k)
``` .

---

#### MiniBatch K-Means

Stochastic gradient descent on WCSS. Each iteration processes a batch  ```math
\mathcal{B}
```  of size  ```math
b \ll n
``` . The per-centre update uses a decreasing learning rate  ```math
\eta_{c_j} = 1/(1+n_{c_j})
```  where  ```math
n_{c_j}
```  counts prior assignments:


```math
\mu_j \leftarrow (1 - \eta_{c_j})\,\mu_j + \eta_{c_j}\,\bar{x}_{\mathcal{B},j}
```


where  ```math
\bar{x}_{\mathcal{B},j}
```  is the mean of batch points assigned to centre ```math j ```. The harmonic decay satisfies Robbins-Monro convergence conditions:  ```math
\sum_t \eta_t = \infty
```  and  ```math
\sum_t \eta_t^2 < \infty
``` .

- **Complexity:**  ```math
O(bkd)
```  per step, viable for  ```math
n > 10^6
``` 

---

#### K-Medoids (PAM)

Medoids ```math m_j ``` must be actual data points, minimising total dissimilarity for arbitrary metric ```math d ```:


```math
\min_{m \subset X,\, |m|=k} \sum_{j=1}^k \sum_{x_i \in C_j} d(x_i,\, m_j)
```


The **PAM** (Partitioning Around Medoids) algorithm evaluates the swap cost for every (medoid ```math m_j ```, non-medoid ```math x_h ```) pair:


```math
\Delta_{jh} = \sum_{i=1}^n \left[ d(x_i,\, m_{c(i)}^{\text{new}}) - d(x_i,\, m_{c(i)}) \right]
```


and accepts the swap minimising  ```math
\Delta_{jh} < 0
``` . Unlike K-Means, K-Medoids is robust to outliers and works with non-Euclidean distances (Jaccard, cosine, DTW, Earth Mover's).

- **Complexity:**  ```math
O(k(n-k)^2)
```  per iteration

---

#### Bisecting K-Means

Divisive strategy producing a binary tree over ```math k-1 ``` bisections. At each step, the cluster with maximum intra-cluster variance is selected and bisected:


```math
C^* = \arg\max_{C_j} \sum_{x_i \in C_j} \|x_i - \mu_j\|^2
```


Two-means is applied to ```math C^* ```. The WCSS decreases monotonically at every bisection because children always have lower combined WCSS than the parent. After ```math k-1 ``` bisections, ```math k ``` clusters are obtained.

- **Complexity:**  ```math
O(nkT \log k)
``` 

---

### Hierarchical Family

#### Agglomerative Clustering (AGNES)

Bottom-up linkage clustering merges the closest pair of clusters at each step. Four linkage criteria for clusters ```math A ```, ```math B ```:

| Linkage | Distance  ```math
D(A,B)
```  |
|---------|-------------------|
| Ward |  ```math
\frac{n_A n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2
```  |
| Complete |  ```math
\max_{a \in A,\, b \in B} d(a,b)
```  |
| Average (UPGMA) |  ```math
\frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} d(a,b)
```  |
| Single |  ```math
\min_{a \in A,\, b \in B} d(a,b)
```  |

All four are unified by the **Lance–Williams recurrence**, which updates the distance from the newly merged cluster  ```math
A \cup B
```  to any existing cluster ```math C ``` without recomputing pairwise distances:


```math
D(A \cup B,\, C) = \alpha_A D(A,C) + \alpha_B D(B,C) + \beta\, D(A,B) + \gamma |D(A,C) - D(B,C)|
```


with family-specific coefficients  ```math
(\alpha_A, \alpha_B, \beta, \gamma)
``` . This enables  ```math
O(n^2 \log n)
```  priority-queue implementations. Ward's linkage minimises the total increase in within-cluster variance at each merge:


```math
\Delta(A,B) = \frac{n_A n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2
```


The dendrogram cophenetic distance  ```math
c_{ij}
```  is the linkage level at which ```math x_i ``` and ```math x_j ``` first merge. **Cophenetic correlation**  ```math
\rho_c = \text{corr}(d_{ij}, c_{ij})
```  measures the faithfulness of the dendrogram representation.

- **Complexity:**  ```math
O(n^3)
```  naïve;  ```math
O(n^2 \log n)
```  with Lance–Williams and priority queue

---

#### BIRCH

BIRCH uses a **Clustering Feature (CF)** tree to summarise data in  ```math
O(n)
``` . Each CF node stores the triplet:


```math
CF = (n,\, LS,\, SS), \quad LS = \sum_{i=1}^n x_i, \quad SS = \sum_{i=1}^n \|x_i\|^2
```


All cluster statistics are computable from CF alone:

- Centroid:  ```math
\mu = LS / n
``` 
- Radius:  ```math
R = \sqrt{SS/n - \|LS/n\|^2}
``` 
- Diameter:  ```math
D = \sqrt{2 \cdot (SS/n - \|LS/n\|^2)}
``` 

Merging two CFs is  ```math
O(1)
``` :  ```math
CF_1 + CF_2 = (n_1+n_2,\, LS_1+LS_2,\, SS_1+SS_2)
``` .

The `threshold` parameter  ```math
\theta
```  controls the maximum subcluster radius:


```math
R = \sqrt{\frac{SS}{n} - \left\|\frac{LS}{n}\right\|^2} \leq \theta
```


Phase 1 builds the CF tree in a single data pass; Phase 2 applies agglomerative clustering to leaf-level subcluster centroids; Phase 3 reassigns original points to nearest leaf centroids.

- **Complexity:**  ```math
O(n)
```  Phase 1, single-pass, sub-linear space

---

#### DIANA — Divisive Analysis (v2)

Top-down divisive hierarchical clustering. Starting from the full dataset, the algorithm identifies the most dissimilar point and iteratively builds the *splinter group*:


```math
x^* = \arg\max_{x \in C} \left[ \bar{d}(x, C\setminus S) - \bar{d}(x, S) \right]
```


where  ```math
\bar{d}(x, A) = \frac{1}{|A|}\sum_{y \in A} d(x, y)
```  is the average distance from ```math x ``` to group ```math A ```, and ```math S ``` is the current splinter group. Points with positive *splinter preference*  ```math
\bar{d}(x, C\setminus S) - \bar{d}(x, S) > 0
```  are moved to the splinter group. This continues until no more points prefer the splinter group, then the split is finalised. DIANA avoids the chaining artefact of single-linkage AGNES for elongated clusters.

---

#### Feature Agglomeration (v1)

Transposed agglomerative clustering: clusters *features* (columns) using Ward linkage on the feature-correlation matrix  ```math
(\Sigma_F)_{ij} = \rho(f_i, f_j)
``` . Grouped features are replaced by their mean, giving a semantically coherent reduced feature space.

- **Complexity:**  ```math
O(d^2 \log d)
```  on ```math d ``` features

---

### Density Family

#### DBSCAN

DBSCAN constructs clusters from density-connected regions. Definitions:

- ** ```math
\varepsilon
``` -neighbourhood**:  ```math
N_\varepsilon(p) = \{q \in X : d(p,q) \leq \varepsilon\}
``` 
- **Core point**: ```math p ``` is core if  ```math
|N_\varepsilon(p)| \geq \text{MinPts}
``` 
- **Directly density-reachable**: ```math q ``` from ```math p ``` if ```math p ``` is core and  ```math
q \in N_\varepsilon(p)
``` 
- **Density-reachable**: chain of direct reachability
- **Density-connected**: both ```math p ```, ```math q ``` are density-reachable from some ```math o ```

A cluster is the maximal density-connected set containing at least one core point. Points not in any cluster receive label ```math -1 ``` (noise). Formally:


```math
C_j = \{ x \in X : \exists\text{ density-reachability path from some core point to }x \}
```


Automatic  ```math
\varepsilon
```  estimation uses the **k-NN distance profile elbow**. Sorting the ```math k ```-th nearest-neighbour distances  ```math
d_k(1) \leq d_k(2) \leq \cdots \leq d_k(n)
``` , the elbow is located via maximum discrete second derivative:


```math
\hat{\varepsilon} = d_k\!\left[\arg\max_i |d_k[i+1] - 2d_k[i] + d_k[i-1]|\right]
```


This approximates the curvature  ```math
\kappa \approx |y''| / (1 + y'^2)^{3/2}
```  in the sorted distance plot.

- **Complexity:**  ```math
O(n \log n)
```  with ball-tree or KD-tree spatial indexing

---

#### HDBSCAN

HDBSCAN extends DBSCAN to a full density hierarchy via the **mutual reachability distance**:


```math
d_{\text{mreach},k}(a, b) = \max\bigl(\text{core}_k(a),\; \text{core}_k(b),\; d(a,b)\bigr)
```


where 

```math
\text{core}_k(a) = d(a, \text{kNN}_k(a))
```
 is the core distance of ```math a ``` to its ```math k ```-th nearest neighbour. Mutual reachability smooths density fluctuations: 
 ```math
 d_{\text{mreach}} \geq d
```
always holds.

The algorithm:
1. Build the complete mutual-reachability graph  ```math
G_\text{mr}
``` 
2. Extract its Minimum Spanning Tree  ```math
\text{MST}(G_\text{mr})
```  via Borůvka's algorithm in  ```math
O(n \log n)
``` 
3. Convert to the condensed cluster hierarchy by removing clusters below `min_cluster_size`
4. Select persistent clusters by maximising cluster **stability**:


```math
\text{stability}(C) = \sum_{x \in C} \left( \lambda_{\text{death}}(x, C) - \lambda_{\text{birth}}(C) \right), \quad \lambda = 1/d_{\text{mreach}}
```


The **Excess of Mass (EOM)** criterion selects the subset of non-overlapping clusters maximising total stability. Soft membership probability for point ```math x ``` in cluster ```math C ```:


```math
P(x \in C) = \frac{\lambda_{\text{death}}(x, C)}{\max_{x'} \lambda_{\text{death}}(x', C)}
```


- **Complexity:**  ```math
O(n \log n)
```  time and  ```math
O(n)
```  space

---

#### OPTICS

OPTICS computes a reachability ordering encoding the complete density hierarchy without committing to a single  ```math
\varepsilon
``` . The reachability distance from ```math p ``` to ```math q ``` is:


```math
\text{rd}_k(p, q) = \max\!\left(\text{core-dist}_k(q),\; d(p,q)\right)
```


The sorted reachability plot reveals cluster structure as valleys. The **```math \xi ```-steep descent** extraction identifies valleys with relative depth ```math \xi ```:


```math
\text{steep descent at } i: \quad \frac{r[i] - r[i+1]}{r[i]} \geq \xi
```


This requires no global  ```math
\varepsilon
```  parameter and simultaneously reveals multi-scale cluster structure.

---

#### Mean Shift

Non-parametric mode-seeking algorithm. Each point ```math x ``` iterates toward the **kernel-weighted mean**:


```math
m_h(x) = \frac{\sum_{i=1}^n K_h(x - x_i)\, x_i}{\sum_{i=1}^n K_h(x - x_i)}
```


with Gaussian kernel  ```math
K_h(u) = \exp(-\|u\|^2 / 2h^2)
``` . The shift vector  ```math
m_h(x) - x
```  is proportional to  ```math
\nabla \hat{f}_h(x)
``` , the gradient of the kernel density estimate:


```math
\hat{f}_h(x) = \frac{1}{n h^d} \sum_{i=1}^n K\!\left(\frac{x - x_i}{h}\right)
```


Each shift step climbs the KDE gradient surface, converging to a mode (local maximum) of  ```math
\hat{f}_h
``` . Bandwidth ```math h ``` is estimated via Silverman's rule:  ```math
h = 1.06\, \hat{\sigma}\, n^{-1/(d+4)}
``` .

- **Complexity:**  ```math
O(Tn^2)
``` 

---

#### DENCLUE (v2)

Density-based clustering via Gaussian KDE. The KDE gradient in closed form (Gaussian kernel):


```math
\nabla \hat{f}(x) = \frac{1}{n h^{d+2}} \sum_{i=1}^n K\!\left(\frac{x-x_i}{h}\right)(x_i - x)
```


Points are attracted to **density attractors** (local maxima of  ```math
\hat{f}
``` ) by following the gradient. Clusters are maximal connected regions where  ```math
\hat{f}(x) \geq \xi
```  for a density threshold ```math \xi ```. DENCLUE handles continuously varying density distributions where DBSCAN's global  ```math
\varepsilon
```  is insufficient.

---

### Distribution / Model-Based Family

#### Gaussian Mixture Model (GMM)

GMM models data as a convex combination of ```math k ``` Gaussians:


```math
p(x) = \sum_{j=1}^k \pi_j\, \mathcal{N}(x \mid \mu_j, \Sigma_j), \quad \pi_j \geq 0,\; \sum_j \pi_j = 1
```


where  ```math
\mathcal{N}(x \mid \mu, \Sigma) = (2\pi)^{-d/2} |\Sigma|^{-1/2} \exp\!\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\right)
``` .

Parameters are estimated by **Expectation-Maximisation (EM)**. The complete-data log-likelihood is:


```math
\mathcal{L}_c(\theta) = \sum_{i=1}^n \sum_{j=1}^k z_{ij} \left[ \log \pi_j + \log \mathcal{N}(x_i \mid \mu_j, \Sigma_j) \right]
```


**E-step** — compute posterior responsibilities (soft cluster assignments):


```math
r_{ij} = \frac{\pi_j\, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}{\sum_{l=1}^k \pi_l\, \mathcal{N}(x_i \mid \mu_l, \Sigma_l)}
```


**M-step** — update parameters using weighted sufficient statistics:


```math
\pi_j = \frac{1}{n}\sum_i r_{ij}, \quad \mu_j = \frac{\sum_i r_{ij}\, x_i}{\sum_i r_{ij}}, \quad \Sigma_j = \frac{\sum_i r_{ij}(x_i-\mu_j)(x_i-\mu_j)^\top}{\sum_i r_{ij}}
```


EM monotonically increases the marginal log-likelihood  ```math
\mathcal{L}(\theta) = \sum_i \log p(x_i)
```  because:


```math
\mathcal{L}(\theta^{(t+1)}) - \mathcal{L}(\theta^{(t)}) = \text{KL}\!\left[q_t(Z) \;\|\; p(Z|X,\theta^{(t)})\right] + \Delta Q \geq 0
```


where  ```math
Q(\theta|\theta^{(t)}) = \mathbb{E}_{q_t}[\mathcal{L}_c(\theta)]
```  is the Q-function lower bound.

**Model selection** via BIC:  ```math
\text{BIC}(k) = -2\hat{\mathcal{L}} + p_k \ln n
```  where  ```math
p_k = k[d + d(d+1)/2 + 1] - 1
```  for full covariance.

Four covariance structures:

| Type | Constraint | ```math d ```-dim parameters |
|------|-----------|---------------------|
| `full` |  ```math
\Sigma_j
```  unconstrained |  ```math
d(d+1)/2
```  per cluster |
| `tied` | Shared  ```math
\Sigma
```  |  ```math
d(d+1)/2
```  total |
| `diag` |  ```math
\Sigma_j = \text{diag}(\sigma_{j1}^2,\ldots)
```  | ```math d ``` per cluster |
| `spherical` |  ```math
\Sigma_j = \sigma_j^2 I
```  | ```math 1 ``` per cluster |

- **Complexity:**  ```math
O(nkd^2 T)
```  time,  ```math
O(kd^2)
```  space

---

#### Bayesian GMM (BGMM)

BGMM places conjugate priors over all GMM parameters:


```math
\pi \sim \text{Dir}(\alpha/k, \ldots, \alpha/k), \quad \mu_j \sim \mathcal{N}(0, \beta^{-1}I), \quad \Lambda_j \sim \mathcal{W}(\nu, W)
```


Inference maximises the **Evidence Lower BOund (ELBO)**:


```math
\mathcal{L}_\text{ELBO} = \mathbb{E}_q[\log p(X, Z, \theta)] - \mathbb{E}_q[\log q(Z, \theta)]
```


This equals  ```math
\log p(X) - \text{KL}[q \| p(\cdot|X)] \leq \log p(X)
``` . As the Dirichlet concentration  ```math
\alpha \to 0
``` , inactive components receive  ```math
\pi_j \approx 0
``` , providing automatic model order selection equivalent to the Dirichlet Process (infinite mixture) limit.

---

#### Affinity Propagation

Message-passing over the similarity graph  ```math
s(i,k) = -\|x_i - x_k\|^2
``` . Two messages are exchanged at each iteration:

**Responsibility**  ```math
r(i,k)
```  — how much evidence that ```math k ``` should be ```math i ```'s exemplar:


```math
r(i,k) \leftarrow s(i,k) - \max_{k' \neq k} \{a(i,k') + s(i,k')\}
```


**Availability**  ```math
a(i,k)
```  — how appropriate it is for ```math i ``` to choose ```math k ``` as exemplar:


```math
a(i,k) \leftarrow \min\!\left(0,\; r(k,k) + \sum_{i' \notin \{i,k\}} \max(0, r(i',k))\right)
```


Self-availability:  ```math
a(k,k) \leftarrow \sum_{i' \neq k} \max(0, r(i',k))
``` .

Exemplars are identified where  ```math
a(k,k) + r(k,k) > 0
``` . Damped updates with factor  ```math
\lambda \in [0.5, 1)
```  prevent oscillations:


```math
r^{(t+1)} = (1-\lambda)\, r_\text{new} + \lambda\, r^{(t)}, \quad a^{(t+1)} = (1-\lambda)\, a_\text{new} + \lambda\, a^{(t)}
```


- **Complexity:**  ```math
O(n^2 T)
```  time and  ```math
O(n^2)
```  space

---

### Graph & Spectral Family

#### Spectral Clustering

Spectral clustering performs graph-cut optimisation in the eigenspace of the graph Laplacian.

**Step 1** — Affinity matrix via RBF kernel:  ```math
W_{ij} = \exp(-\|x_i-x_j\|^2 / 2\sigma^2)
``` .

**Step 2** — Normalised Laplacian:  ```math
\mathcal{L}_\text{sym} = I - D^{-1/2} W D^{-1/2}
```  where  ```math
D_{ii} = \sum_j W_{ij}
``` .

**Step 3** — Spectral decomposition: solve  ```math
\mathcal{L}_\text{sym} u = \lambda u
``` , collecting the ```math k ``` eigenvectors with smallest eigenvalues into  ```math
U \in \mathbb{R}^{n \times k}
``` .

**Step 4** — Row-normalise ```math U ```:  ```math
U_{i\cdot} \leftarrow U_{i\cdot} / \|U_{i\cdot}\|_2
``` .

**Step 5** — Apply K-Means to the rows of ```math U ```.

The **spectral gap heuristic** for ```math k ```: the multiplicity of eigenvalue 0 of the unnormalised Laplacian  ```math
L = D - W
```  equals the number of connected components. The optimal ```math k ``` is identified by the largest gap  ```math
\delta_k = \lambda_{k+1} - \lambda_k
``` .

The **Cheeger inequality** relates the spectral gap to the minimum graph cut:


```math
\frac{h^2}{2} \leq \lambda_2 \leq 2h, \quad h = \min_{S} \frac{|\partial S|}{\min(\text{vol}(S), \text{vol}(\bar{S}))}
```


where ```math h ``` is the Cheeger constant (normalised minimum cut) and  ```math
|\partial S|
```  counts edges crossing the cut.

The `discretize` assignment rounds the embedding via iterative rotation:


```math
\min_{R \in \mathcal{O}(k)} \|U - VR\|_F, \quad V_{ij} = u_{ij} / \|u_{i\cdot}\|_2
```


- **Complexity:**  ```math
O(n^3)
```  for dense ```math W ```;  ```math
O(n \cdot k \cdot d)
```  with sparse affinity

---

#### Spectral Biclustering (v2)

Simultaneously clusters rows and columns of  ```math
X \in \mathbb{R}^{n \times d}
``` . The normalised biadjacency form:


```math
\hat{X} = D_r^{-1/2} X D_c^{-1/2}
```


The SVD  ```math
\hat{X} = U \Sigma V^\top
```  yields row clusters from ```math U ``` and column clusters from ```math V ``` simultaneously. Biclusters  ```math
(R, C)
```  correspond to subsets of rows and columns where  ```math
X_{RC}
```  has elevated mean relative to the full matrix.

---

#### Spectral Coclustering (v2)

Minimum normalised cut on the bipartite graph  ```math
G = (I \cup J, E)
```  where nodes are samples and features. The adjacency matrix of ```math G ``` is the data matrix ```math X ``` itself. SVD of  ```math
D_r^{-1/2} X D_c^{-1/2}
```  yields singular vectors; K-Means on the stacked singular vectors simultaneously partitions rows and columns.

---

### Neural / Deep Family (v2 only)

#### Autoencoder + K-Means

A deep autoencoder learns a nonlinear latent embedding  ```math
z = f_\phi(x) \in \mathbb{R}^q
```  ( ```math
q \ll d
``` ), trained to minimise reconstruction loss:


```math
\mathcal{L}_\text{recon} = \frac{1}{n}\sum_{i=1}^n \|x_i - g_\theta(f_\phi(x_i))\|_2^2
```


K-Means is then applied in the latent space  ```math
\{z_i\}_{i=1}^n
``` . **Joint fine-tuning** minimises:


```math
\mathcal{L}_\text{total} = \mathcal{L}_\text{recon} + \lambda_c\, \mathcal{L}_\text{cluster}, \quad \mathcal{L}_\text{cluster} = \sum_{i=1}^n \|z_i - \mu_{c(i)}\|^2
```


Encoder architecture: Input(```math d ```) ```math \to ``` Dense(128, ReLU) ```math \to ``` Dense(64, ReLU) ```math \to ``` Dense(```math q ```, Linear). Decoder mirrors the encoder. The gradient of  ```math
\mathcal{L}_\text{total}
```  w.r.t. encoder parameters  ```math
\phi
``` :


```math
\frac{\partial \mathcal{L}_\text{total}}{\partial \phi} = \frac{\partial \mathcal{L}_\text{recon}}{\partial \phi} + \lambda_c \frac{\partial \mathcal{L}_\text{cluster}}{\partial \phi}
```


The cluster loss gradient 

```math
\frac{\partial \mathcal{L}_\text{cluster}}{\partial z_i} = 2(z_i - \mu_{c(i)})
```
encourages latent representations to form tight clusters.

---

#### Self-Organising Map (SOM)

SOM learns a discrete low-dimensional (typically 2D) grid manifold  ```math
\{w_i\}_{i \in G}
```  that maps the data topology. For each input ```math x ```, the **Best Matching Unit (BMU)** is:


```math
i^*(x) = \arg\min_{i \in G} \|x - w_i\|
```


Weights update with decaying neighbourhood function  ```math
h(i, i^*, t) = \exp(-\|r_i - r_{i^*}\|^2 / 2\sigma(t)^2)
```  and learning rate  ```math
\alpha(t)
``` :


```math
w_i \leftarrow w_i + \alpha(t)\cdot h(i, i^*, t) \cdot (x - w_i)
```


Both decay:  ```math
\alpha(t) = \alpha_0 \exp(-t/T)
```  and  ```math
\sigma(t) = \sigma_0 \exp(-t/T)
```  where ```math T ``` is the number of training epochs. The **U-Matrix** visualises cluster boundaries:  ```math
U_{ij} = \|w_i - w_j\|
```  between adjacent grid units — high values indicate inter-cluster boundaries. K-Means on the trained SOM prototypes yields final cluster assignments.

---

### Fuzzy Family (v2 only)

#### Fuzzy C-Means (FCM)

FCM relaxes hard assignment to membership degrees  ```math
u_{ij} \in [0,1]
```  with  ```math
\sum_j u_{ij} = 1
``` . The fuzzy WCSS objective:


```math
J_m = \sum_{i=1}^n \sum_{j=1}^k u_{ij}^m \|x_i - \mu_j\|^2, \quad m > 1
```


The fuzzifier ```math m ``` controls softness:  ```math
m \to 1
```  recovers hard K-Means;  ```math
m \to \infty
```  gives uniform memberships. Setting partial derivatives to zero yields the optimal update rules:


```math
u_{ij} = \left(\sum_{l=1}^k \left(\frac{\|x_i - \mu_j\|}{\|x_i - \mu_l\|}\right)^{2/(m-1)}\right)^{-1}, \quad \mu_j = \frac{\sum_i u_{ij}^m x_i}{\sum_i u_{ij}^m}
```


The **Picard iteration** converges quadratically in a neighbourhood of the solution:


```math
\|u^{(t+1)} - u^\ast\| = O(\|u^{(t)} - u^\ast\|^2)
```


under mild regularity conditions. Typically ```math m=2 ``` is used; the choice of ```math m ``` affects the transition sharpness. The boundary of cluster ```math j ``` is the locus  ```math
\{x : u_{j}(x) = 0.5\}
``` .

---

#### Possibilistic C-Means (PCM)

PCM relaxes the probabilistic constraint  ```math
\sum_j u_{ij} = 1
``` . Membership degrees become independent **typicalities**  ```math
t_{ij} \in [0,1]
``` . Objective:


```math
J_m = \sum_{i=1}^n \sum_{j=1}^k t_{ij}^m \|x_i - \mu_j\|^2 + \sum_{j=1}^k \eta_j \sum_{i=1}^n (1 - t_{ij})^m
```


where the per-cluster bandwidth (estimated from FCM memberships):


```math
\eta_j = \frac{\sum_i u_{ij}^m \|x_i - \mu_j\|^2}{\sum_i u_{ij}^m}
```


The second penalty term prevents the trivial solution  ```math
t_{ij} = 0
```  for all ```math i ```. The optimal typicality update:


```math
t_{ij} = \left(1 + \left(\frac{\|x_i - \mu_j\|^2}{\eta_j}\right)^{1/(m-1)}\right)^{-1}
```


Outliers achieve low typicality to all clusters (unlike FCM where they have equal membership to all), making PCM intrinsically outlier-robust.

---

### Manifold Family (v2 only)

#### Isomap + K-Means

Isomap estimates **geodesic distances** on the data manifold by constructing a ```math k ```-NN graph and computing all-pairs shortest paths via Dijkstra's algorithm. The geodesic distance matrix ```math D_G ``` is embedded in  ```math
\mathbb{R}^q
```  via **Classical Multidimensional Scaling (cMDS)**, minimising the strain:


```math
\text{Strain}(Y) = \left\|\tau(D_G) - Y^\top Y\right\|_F
```


where  ```math
\tau(D_G) = -\frac{1}{2} H D_G^{(2)} H
```  is the double-centred squared geodesic distance, and  ```math
H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top
```  is the centring matrix. The solution is the top-```math q ``` eigenvectors of  ```math
\tau(D_G)
```  scaled by their eigenvalues.

**Convergence guarantee:** For data uniformly sampled from a smooth compact Riemannian manifold  ```math
\mathcal{M}
```  of intrinsic dimension ```math q ``` isometrically embedded in  ```math
\mathbb{R}^d
``` :


```math
\left|D_G(x_i, x_j) - d_\mathcal{M}(x_i, x_j)\right| \to 0 \quad \text{as } n \to \infty
```


---

#### LLE + K-Means

LLE finds a locally linear representation of each point using its ```math k ```-NN. **Phase 1** — solve for optimal reconstruction weights:


```math
\min_W \sum_{i=1}^n \left\|x_i - \sum_{j \in \mathcal{N}(i)} W_{ij} x_j\right\|^2, \quad \sum_j W_{ij} = 1, \quad W_{ij}=0 \text{ if } j \notin \mathcal{N}(i)
```


Closed-form solution:  ```math
W_{ij} = \frac{\sum_l (G^i)^{-1}_{jl}}{\sum_{jl} (G^i)^{-1}_{jl}}
```  where  ```math
G^i_{jl} = (x_i - x_j)^\top(x_i - x_l)
``` .

**Phase 2** — find the low-dimensional embedding ```math Y ``` preserving the same weights:


```math
\min_Y \sum_i \left\|y_i - \sum_j W_{ij} y_j\right\|^2 = \text{tr}(Y^\top M Y), \quad M = (I-W)^\top(I-W)
```


subject to  ```math
\frac{1}{n} Y^\top Y = I
```  to avoid degenerate solutions. The embedding is the bottom ```math q ``` eigenvectors of ```math M ``` (excluding the constant eigenvector  ```math
\mathbf{1}
``` ).

---

#### UMAP + HDBSCAN

UMAP learns a fuzzy topological representation. For each point ```math x_i ``` and its ```math k ```-NN  ```math
\{x_{i,j}\}_{j=1}^k
``` , the local fuzzy membership strength is:


```math
\mu_{ij} = \exp\!\left(-\frac{d(x_i, x_{i,j}) - \rho_i}{\sigma_i}\right)
```


where  ```math
\rho_i = d(x_i, x_{i,1})
```  (nearest-neighbour distance) and  ```math
\sigma_i
```  is chosen so that  ```math
\sum_j \mu_{ij} = \log_2 k
```  (target fuzzy cardinality). The symmetrised fuzzy graph has edge weights:


```math
v_{ij} = \mu_{ij} + \mu_{ji} - \mu_{ij}\, \mu_{ji}
```


The low-dimensional embedding minimises cross-entropy between the high-dimensional fuzzy representation and the low-dimensional one:


```math
\mathcal{L}_\text{UMAP} = \sum_{(i,j)} \left[ v_{ij} \log \frac{v_{ij}}{q_{ij}} + (1-v_{ij}) \log \frac{1-v_{ij}}{1-q_{ij}} \right]
```


where  ```math
q_{ij} = (1 + a \|y_i - y_j\|^{2b})^{-1}
```  with learnable parameters  ```math
a, b
```  fit to a Student-t distribution. HDBSCAN is then applied to UMAP coordinates for density-faithful cluster boundaries.

---

### Subspace & Ensemble Families (v2 only)

#### Projected K-Means

Random projection  ```math
\Phi \in \mathbb{R}^{q \times d}
```  with  ```math
\Phi_{ij} \sim \mathcal{N}(0, 1/q)
```  reduces dimensionality before K-Means. The **Johnson-Lindenstrauss lemma** guarantees:


```math
\Pr\!\left[(1-\varepsilon)\|x-y\|^2 \leq \|\Phi x - \Phi y\|^2 \leq (1+\varepsilon)\|x-y\|^2\right] \geq 1 - 2\exp\!\left(-\frac{q\varepsilon^2}{4}\right)
```


for any  ```math
x, y \in \mathbb{R}^d
``` , provided  ```math
q = O(\varepsilon^{-2} \log n)
``` . Thus for  ```math
q \approx 8\varepsilon^{-2} \ln(1/\delta)
```  dimensions, all  ```math
\binom{n}{2}
```  pairwise distances are simultaneously preserved within factor  ```math
1 \pm \varepsilon
```  with probability  ```math
\geq 1-\delta
``` .

---

#### Ensemble Voting (Multi-Init K-Means)

Runs K-Means ```math M ``` times with different seeds, producing labellings  ```math
\{L_1,\ldots,L_M\}
``` . Builds the **co-occurrence matrix**:


```math
A_{ij} = \frac{1}{M}\sum_{m=1}^M \mathbf{1}[L_m(i) = L_m(j)]
```


Spectral or agglomerative clustering on ```math 1-A ``` produces the consensus partition. The ensemble variance satisfies:


```math
\text{Var}\!\left[\frac{1}{M}\sum_m A_{ij}^{(m)}\right] = \frac{\text{Var}[A_{ij}]}{M} + \frac{M-1}{M}\text{Cov}[A_{ij}^{(m)}, A_{ij}^{(m')}]
```


Diversity among initialisations (low covariance) reduces ensemble variance.

---

#### Random Subspace Ensemble

Trains ```math T ``` clusterers, each on a random feature subset of size  ```math
\lfloor\sqrt{d}\rfloor
``` . Co-association matrices are averaged:  ```math
A = \frac{1}{T}\sum_t A^{(t)}
``` . Effective for high-dimensional data where no single projection dominates.

---

### Grid-Based Family (v2 only)

#### WaveCluster / Grid

Quantises the feature space into a hierarchical grid at multiple resolutions. Each cell stores the statistical summary  ```math
(n, \mu, \sigma^2)
```  of contained points. Cell density:


```math
\hat{f}_\text{cell}(c) = \frac{n_c}{n \cdot \text{vol}(c)}
```


Connected high-density cells ( ```math
\hat{f} \geq \tau
``` ) form clusters. Multi-resolution: coarse grids give global structure, fine grids capture local detail. Runtime is  ```math
O(n)
```  — linear in data, independent of grid resolution.

---

## Evaluation Framework

### Internal Validity Indices

#### Silhouette Score

For each point ```math i ``` in cluster ```math C_j ```, define:
-  ```math
a(i) = \frac{1}{|C_j|-1}\sum_{i' \in C_j, i' \neq i} d(i, i')
```  — mean intra-cluster distance
-  ```math
b(i) = \min_{l \neq j} \frac{1}{|C_l|}\sum_{i' \in C_l} d(i, i')
```  — mean distance to nearest other cluster


```math
s(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))}, \quad S = \frac{1}{n}\sum_{i=1}^n s(i) \in [-1, 1]
```


Geometric interpretation:  ```math
s(i) = 1
```  means  ```math
a(i) \to 0
```  (perfect intra-cluster cohesion);  ```math
s(i) = -1
```  means  ```math
b(i) \to 0
```  (point is closer to the core of another cluster). The **per-cluster silhouette** is  ```math
S_j = \frac{1}{|C_j|}\sum_{i \in C_j} s(i)
``` .

Interpretation thresholds:  ```math
[0.70, 1.00]
```  strong structure;  ```math
[0.50, 0.70)
```  reasonable;  ```math
[0.25, 0.50)
```  weak;  ```math
< 0.25
```  no meaningful structure; ```math < 0 ``` misassignment.

---

#### Davies-Bouldin Index (DBI)


```math
\text{DBI} = \frac{1}{k}\sum_{j=1}^k \max_{l \neq j} \left\lbrace \frac{\sigma_j + \sigma_l}{d(\mu_j, \mu_l)} \right\rbrace
```


where  ```math
\sigma_j = \frac{1}{|C_j|}\sum_{x \in C_j} d(x, \mu_j)
```  is the average intra-cluster scatter radius. DBI rewards clusters that are simultaneously compact (small  ```math
\sigma_j
``` ) and well-separated (large  ```math
d(\mu_j, \mu_l)
``` ). Lower is better; DBI ```math = 0 ``` for non-overlapping clusters.

**Relation to cluster shapes:** DBI penalises elongated or overlapping clusters more than compact spherical ones.

---

#### Calinski-Harabász Score (CH)


```math
\text{CH} = \frac{\text{tr}(B_k)\, /\, (k-1)}{\text{tr}(W_k)\, /\, (n-k)}
```


Since  ```math
T = B_k + W_k
```  is constant for a fixed dataset, maximising CH is equivalent to maximising  ```math
\text{tr}(B_k)
```  while minimising  ```math
\text{tr}(W_k)
``` . Under a spherical Gaussian mixture with equal variances, CH is the ANOVA F-statistic for testing cluster mean differences. No absolute scale exists; CH is used comparatively across values of ```math k ```.

---

#### Dunn Index (DI)


```math
\text{DI} = \frac{\min_{i \neq j} \delta(C_i, C_j)}{\max_k \Delta(C_k)}
```


where  ```math
\delta(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x,y)
```  (single-linkage inter-cluster distance) and  ```math
\Delta(C_k) = \max_{x,y \in C_k} d(x,y)
```  (complete-linkage cluster diameter). Higher values indicate compact, well-separated clusters. DI is sensitive to outliers because a single outlier inflates  ```math
\Delta(C_k)
``` .

---

#### Xie-Beni Index (XB)


```math
\text{XB} = \frac{\sum_{j=1}^k \sum_{x \in C_j} \|x - \mu_j\|^2}{n \cdot \min_{i \neq j} \|\mu_i - \mu_j\|^2}
```


Compactness-to-separation ratio; lower is better. The numerator is the total WCSS; the denominator penalises configurations where centroids are too close.

---

#### S\_DBw Index


```math
S_{DBw} = \text{Scat}(k) + \text{Dens}_{bw}(k)
```



```math
\text{Scat}(k) = \frac{1}{k}\sum_{j=1}^k \frac{\|\sigma_j\|}{\|\sigma\|}
```



```math
\text{Dens}_{bw}(k) = \frac{1}{k(k-1)}\sum_{i \neq j} \frac{\rho(u_{ij})}{\max\left(\rho(\mu_i), \rho(\mu_j)\right)}, \quad u_{ij} = \frac{\mu_i + \mu_j}{2}
```


where  ```math
\rho(\mu)
```  is the density at point ```math \mu ``` (number of points within a radius ```math r ```). S\_DBw penalises dense inter-centroid mid-points relative to centroid densities — high mid-point density indicates cluster overlap.

---

#### Gap Statistic


```math
\text{Gap}(k) = \mathbb{E}_n^*[\log W_k] - \log W_k
```


where  ```math
W_k = \sum_{j=1}^k \frac{1}{2n_j} D_j
```  is the pooled within-cluster dispersion and the expectation is over ```math B ``` reference datasets drawn uniformly from the bounding box of ```math X ```. The **one-standard-error rule** selects the smallest ```math k ``` satisfying:


```math
\text{Gap}(k) \geq \text{Gap}(k+1) - s_{k+1}, \quad s_{k+1} = \text{std}_b[\log W_{k+1}^{(b)}]\sqrt{1 + 1/B}
```


The reference distribution choice (uniform bounding box vs PCA-aligned) affects the test: PCA-aligned is recommended when data is correlated.

---

#### Inertia (WCSS) and Elbow Detection


```math
\mathcal{I}(k) = \sum_{j=1}^k \sum_{x \in C_j} \|x - \mu_j\|^2
```


 ```math
\mathcal{I}(k)
```  is strictly decreasing in ```math k ```, asymptotically approaching zero. The **elbow** is the point of maximum curvature, detected by the kneedle algorithm:


```math
k^* = \arg\max_k |\mathcal{I}(k+1) - 2\mathcal{I}(k) + \mathcal{I}(k-1)|
```


This discrete second derivative approximates the curvature  ```math
\kappa
```  at each point.

---

#### Partition Entropy and Cluster Balance


```math
\mathcal{H}(\mathcal{C}) = -\sum_{j=1}^k \frac{n_j}{n} \log_2 \frac{n_j}{n} \in [0, \log_2 k]
```


Used internally to flag imbalanced assignments. The **Lorenz curve** of sorted cluster sizes  ```math
n_{(1)} \leq \cdots \leq n_{(k)}
```  visualises size inequality; the **Gini coefficient** measures it:


```math
G = \frac{\sum_{i,j} |n_i - n_j|}{2k \sum_j n_j}
```


---

### External Validity Indices

#### Adjusted Rand Index (ARI)

For labellings ```math U ``` and ```math V ```, let ```math a ``` = concordant same-cluster pairs, ```math d ``` = concordant different-cluster pairs. The Rand Index  ```math
\text{RI} = (a+d)/\binom{n}{2}
``` . ARI corrects for chance:


```math
\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}
```


where  ```math
\mathbb{E}[\text{RI}] = \frac{\sum_j \binom{n_j^U}{2} \cdot \sum_l \binom{n_l^V}{2}}{\binom{n}{2}^2}
``` .

ARI  ```math
\in [-1, 1]
``` ; expected value is 0 for random labellings, 1 for perfect agreement.

---

#### Adjusted Mutual Information (AMI)


```math
\text{AMI}(U,V) = \frac{\text{MI}(U,V) - \mathbb{E}[\text{MI}(U,V)]}{\text{avg}(H(U), H(V)) - \mathbb{E}[\text{MI}(U,V)]}
```


where the mutual information:


```math
\text{MI}(U,V) = \sum_{u,v} P(u,v) \log \frac{P(u,v)}{P(u)P(v)}
```


and Shannon entropy  ```math
H(U) = -\sum_u P(u) \log P(u)
``` . AMI = 1 for identical labellings; AMI  ```math
\approx 0
```  for independent ones.

---

#### Normalised Mutual Information (NMI)


```math
\text{NMI}(U,V) = \frac{2\cdot \text{MI}(U,V)}{H(U) + H(V)} \in [0, 1]
```


The denominator  ```math
H(U)+H(V)
```  corrects for cluster count: adding empty clusters inflates MI but not NMI.

---

#### Cluster Purity


```math
\text{Purity} = \frac{1}{n}\sum_{j=1}^k \max_l |C_j \cap G_l|
```


where ```math G_l ``` are ground-truth groups. Purity  ```math
\in [1/k, 1]
``` ; the lower bound ```math 1/k ``` is achieved when each cluster has exactly ```math n/k ``` members spread uniformly across all ground-truth classes.

---

#### V-Measure

The V-measure unifies homogeneity and completeness:


```math
\text{Homogeneity} = 1 - \frac{H(C|K)}{H(C)}, \quad \text{Completeness} = 1 - \frac{H(K|C)}{H(K)}
```



```math
V = \frac{(1+\beta)\cdot h \cdot c}{(\beta \cdot h) + c}
```


where  ```math
\beta = 1
```  gives equal weight to both. This is the conditional-entropy analogue of the F-measure.

---

### Composite Ranking & Pareto Front

#### Composite Score

The `CompositeScorer` computes a normalised weighted combination across all valid metrics:


```math
\text{Score}(R) = \frac{\sum_m w_m \cdot \tilde{v}_m(R)}{\sum_m w_m}
```


where  ```math
\tilde{v}_m(R) = \frac{v_m(R) - v_m^\text{min}}{v_m^\text{max} - v_m^\text{min}}
```  normalises metric ```math m ``` to  ```math
[0,1]
``` , respecting direction (some metrics are minimised, others maximised).

Predefined weights:

| Metric | Weight ```math w_m ``` |
|--------|-------------|
| Silhouette | 3.0 |
| DBI | 2.0 |
| CH | 1.5 |
| Dunn | 1.0 |
| XB | 1.0 |

A **pairwise win matrix**  ```math
W_{ij} = \sum_m \mathbf{1}[v_m(R_i) \succ v_m(R_j)]
```  counts on how many metrics algorithm ```math i ``` beats algorithm ```math j ```.

---

#### Pareto Front (v2)

An algorithm ```math R_i ``` is **Pareto-optimal** if no other algorithm ```math R_j ``` is simultaneously better on all selected metrics:


```math
\text{PF} = \{ R_i : \nexists\, R_j \text{ s.t. } v_m(R_j) \succeq v_m(R_i)\ \forall m \text{ and } v_{m'}(R_j) \succ v_{m'}(R_i) \text{ for some } m' \}
```


The Pareto front provides multi-objective selection without collapsing metrics into a single scalar, preserving the trade-off structure across validity indices.

---

#### k-Sweep Analysis

The `KSweepAnalyser` runs any algorithm over  ```math
k \in [k_{\min}, k_{\max}]
``` , computing all metrics at each ```math k ```. The recommended ```math k ``` combines:

1. **Elbow detection:**  ```math
k_\text{elbow} = \arg\max_k |\mathcal{I}(k+1) - 2\mathcal{I}(k) + \mathcal{I}(k-1)|
``` 
2. **Silhouette peak:**  ```math
k_S = \arg\max_k S(k)
``` 
3. **Gap statistic:**  ```math
k_\text{gap} = \min\{k : \text{Gap}(k) \geq \text{Gap}(k+1) - s_{k+1}\}
``` 

The consensus recommendation weights these three estimates.

---

## Preprocessing Pipeline

The `PreprocessingPipeline` applies 8 sequential stages:

### Stage 1 — Data Loading

CSV, Excel (xlsx/xls), JSON, Parquet, TSV via `DataLoader`. Maximum: 500,000 rows × 2,000 columns.

### Stage 2 — Deep Profiling

Per-column statistics: dtype, unique count, missing rate, mean, std, and distributional shape metrics.

**Skewness** (third standardised moment):  ```math
g_1 = \frac{\mu_3}{\mu_2^{3/2}}
```  where  ```math
\mu_r = \frac{1}{n}\sum_i (x_i - \bar{x})^r
``` .

**Excess kurtosis** (fourth standardised moment):  ```math
g_2 = \frac{\mu_4}{\mu_2^2} - 3
``` .

High-correlation pair detection flags feature pairs with  ```math
|\rho(f_i, f_j)| > 0.9
``` .

### Stage 3 — Missing Value Imputation

| Strategy | Description |
|----------|-------------|
| `mean` | Replace with feature mean |
| `median` | Replace with feature median |
| `most_frequent` | Replace with mode |
| `knn` | k-NN imputation:  ```math
\hat{x}_{ij} = \frac{\sum_{l \in \mathcal{N}_k(i)} x_{lj} \cdot w_{il}}{\sum_{l} w_{il}}
``` ,  ```math
w_{il} = 1/d(x_i, x_l)
```  |
| `iterative` (MICE) | Multiple Imputation by Chained Equations — iterative regression imputation |
| `constant` | Fill with user-specified value |
| `drop_rows` | Remove rows with any missing value |
| `drop_cols` | Remove columns exceeding missing threshold |

### Stage 4 — Outlier Detection & Handling

**Z-Score:**  ```math
z_i = (x_i - \mu)/\sigma
``` ; flag  ```math
|z_i| > 3
``` .

**IQR:** flag  ```math
x_i < Q_1 - 1.5\,\text{IQR}
```  or  ```math
x_i > Q_3 + 1.5\,\text{IQR}
``` .

**Isolation Forest:** anomaly score


```math
s(x, n) = 2^{-E[h(x)]/c(n)}, \quad c(n) = 2H(n-1) - \frac{2(n-1)}{n}
```


where  ```math
h(x)
```  is path length in a random isolation tree and ```math H ``` is the harmonic number. Expected path length for a normal point:  ```math
c(n)
``` ; anomalies have  ```math
E[h(x)] \ll c(n)
``` .

**Local Outlier Factor:**


```math
\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \text{lrd}_k(o)\, /\, \text{lrd}_k(p)}{|N_k(p)|}
```



```math
\text{lrd}_k(p) = \left(\frac{\sum_{o \in N_k(p)} \text{reach-dist}_k(p,o)}{|N_k(p)|}\right)^{-1}
```


LOF  ```math
\approx 1
```  for regular points; LOF  ```math
\gg 1
```  for outliers (locally sparser than their neighbours).

**Elliptic Envelope:** fits a robust Mahalanobis distance estimator  ```math
d_M(x) = \sqrt{(x-\hat{\mu})^\top \hat{\Sigma}^{-1}(x-\hat{\mu})}
```  using the Minimum Covariance Determinant (MCD) estimator.

**Outlier actions:** remove, clip to fence, winsorise, flag (add binary indicator), ignore.

### Stage 5 — Categorical Encoding

| Method | Description |
|--------|-------------|
| One-hot | Binary indicator columns, increases dimensionality by ```math c-1 ``` |
| Ordinal | Integer codes  ```math
0,1,\ldots,c-1
```  |
| Target encoding | Replace category ```math c_j ``` with  ```math
\mathbb{E}[y \mid c = c_j]
```  |

### Stage 6 — Feature Scaling

| Scaler | Formula |
|--------|---------|
| StandardScaler |  ```math
z = (x-\mu)/\sigma
```  |
| MinMaxScaler |  ```math
x' = (x - x_\min)/(x_\max - x_\min)
```  |
| RobustScaler |  ```math
x' = (x - Q_2)/(Q_3 - Q_1)
```  |
| MaxAbsScaler |  ```math
x' = x / \max|x|
```  |
| L1 Normalizer |  ```math
x' = x / \|x\|_1
```  |
| L2 Normalizer |  ```math
x' = x / \|x\|_2
```  |
| Box-Cox |  ```math
x'(\lambda) = (x^\lambda - 1)/\lambda
```  for  ```math
x > 0
```  |
| Yeo-Johnson | Extends Box-Cox to all  ```math
x \in \mathbb{R}
```  |
| QuantileTransformer | Maps to uniform or normal via rank transformation |

### Stage 7 — Feature Selection

**Variance threshold:** drop feature ```math f ``` if  ```math
\text{Var}(f) < \tau
``` .

**Correlation filter:** for correlated pair  ```math
|\rho(f_i, f_j)| > \theta
``` , drop ```math f_j ``` (retaining ```math f_i ``` with higher variance).

**PCA dimensionality reduction:** retain the top-```math q ``` principal components explaining ```math v\% ``` of total variance:


```math
q = \min\left\lbrace m : \frac{\sum_{l=1}^m \lambda_l}{\sum_{l=1}^d \lambda_l} \geq v\right\rbrace
```


**ANOVA F-test feature ranking:**


```math
F_j = \frac{(n-k)}{(k-1)} \cdot \frac{\sum_l n_l (\bar{x}_{lj} - \bar{x}_j)^2}{\sum_l \sum_{i \in C_l} (x_{ij} - \bar{x}_{lj})^2}
```


Higher ```math F_j ``` indicates feature ```math f_j ``` discriminates better between clusters.

### Stage 8 — Column Dropping

User-specified column removal (e.g., known label columns for purely unsupervised evaluation).

---

## Stability & Consensus Analysis

### Bootstrap Stability

The `StabilityPipeline` resamples ```math B ``` times with subsample ratio  ```math
\rho
``` . For each bootstrap run, ARI is computed against the full-data reference labelling. The **Adjusted Rand Index**:


```math
\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}
```


where the Rand Index  ```math
\text{RI} = (a+d)/\binom{n}{2}
``` , ```math a ``` = same-cluster concordant pairs, ```math d ``` = different-cluster concordant pairs. The chance correction term:


```math
\mathbb{E}[\text{RI}] = \frac{\left(\sum_j \binom{n_j^U}{2}\right)\!\left(\sum_l \binom{n_l^V}{2}\right)}{\binom{n}{2}^2}
```


The **Jaccard Index** provides an alternative stability measure:


```math
J = \frac{|\text{TP}|}{|\text{TP}| + |\text{FP}| + |\text{FN}|}
```


where TP = same-cluster concordant pairs, FP = same in prediction but different in reference, FN = different in prediction but same in reference.

### Five Perturbation Regimes (v2)

| Regime | Perturbation | Formal definition |
|--------|-------------|-------------------|
| Bootstrap | Subsample 80% |  ```math
X^{(b)} \sim \text{Subsample}(X, 0.8)
```  |
| Gaussian Noise | Additive i.i.d. |  ```math
\tilde{x}_i = x_i + \varepsilon,\ \varepsilon \sim \mathcal{N}(0, \sigma^2 I)
```  |
| Laplacian Noise | Heavy-tailed |  ```math
\tilde{x}_i = x_i + \varepsilon,\ \varepsilon \sim \text{Lap}(0, b)
```  |
| Feature Dropout | Column zeroing |  ```math
\tilde{x}_i^{(j)} = 0
```  for  ```math
j \in \mathcal{D} \sim \text{Binom}(d, p_\text{drop})
```  |
| Subset Sampling | Progressive fraction |  ```math
X^{(f)} = \text{Sample}(X, f\cdot n)
``` ,  ```math
f \in [0.5, 1.0]
```  |

The Laplacian distribution  ```math
\text{Lap}(0,b)
```  has kurtosis 6 (vs 0 for Gaussian), making it a harder corruption regime. For each regime at noise level  ```math
\sigma
``` , ARI is measured:


```math
\bar{\text{ARI}}(\text{regime}) = \frac{1}{L}\sum_{l=1}^L \text{ARI}(L(\sigma_l), L_\text{ref})
```


The **persistence score** integrates stability over the noise range:


```math
\text{persistence}(C_j) \approx \sum_l \text{ARI}(L(\sigma_l), L_\text{ref}) \cdot \Delta\sigma
```


### Stability Grades

UnSuPERvIsED-II introduces an extended grading scheme:


```math
\text{Grade} = \begin{cases} S^+ & \bar{\text{ARI}} \geq 0.90 \\ A & \bar{\text{ARI}} \in [0.80, 0.90) \\ B & \bar{\text{ARI}} \in [0.65, 0.80) \\ C & \bar{\text{ARI}} \in [0.45, 0.65) \\ D & \bar{\text{ARI}} < 0.45 \end{cases}
```


UnSuPERvIsED-I uses a four-grade scheme: **A** (ARI ≥ 0.85), **B** (≥0.65), **C** (≥0.40), **D** (<0.40).

### Consensus Clustering

The `ConsensusPipeline` runs the algorithm ```math M ``` times with different seeds on data subsamples and builds the **co-association matrix**:


```math
A_{ij} = \frac{\text{number of runs where } x_i, x_j \text{ co-cluster}}{\text{number of runs where both appear}} \in [0,1]
```


```math A ``` is symmetric;  ```math
A_{ij} = 1
```  means always co-clustered,  ```math
A_{ij} = 0
```  means never. Final consensus clusters are obtained by agglomerative clustering (average linkage) on the dissimilarity matrix ```math 1-A ```.

**PAC Score** (Proportion of Ambiguous Clustering):


```math
\text{PAC}(k) = F(\theta_2) - F(\theta_1), \quad [\theta_1, \theta_2] = [0.1, 0.9]
```


where  ```math
F(\theta) = \frac{1}{n^2}\sum_{i,j} \mathbf{1}[A_{ij} \leq \theta]
```  is the empirical CDF of co-association values. Optimal ```math k ``` minimises PAC — values near 0 indicate decisive co-association (all  ```math
A_{ij}
```  close to 0 or 1).

**Cophenetic correlation:**


```math
r_c = \text{corr}\!\left((1 - A_{ij}),\; c_{ij}\right)
```


where  ```math
c_{ij}
```  is the cophenetic distance from the consensus dendrogram. High  ```math
r_c > 0.9
```  indicates the hierarchy faithfully represents the consensus matrix.

**LabelStabilityMatrix** tracks individual sample label entropy across runs:


```math
H_i = -\sum_j p_{ij} \log p_{ij}, \quad p_{ij} = \frac{\text{number of runs assigning } x_i \text{ to cluster } j}{M}
```


High-entropy points are **structurally ambiguous** — lying near cluster boundaries.

### Cross-Validation Stability


```math
\text{CV-ARI} = \frac{1}{K}\sum_{i=1}^K \text{ARI}\!\left(\hat{y}^{(i)}, \hat{y}_\text{full}\right)
```


### Temporal Stability

Incremental data fractions  ```math
(0.1, 0.2, \ldots, 1.0)
```  are clustered; ARI between consecutive fractions:


```math
\text{TS}(f, f') = \text{ARI}\!\left(L(X_{0:f}),\; L(X_{0:f'})\right)
```


A monotonically high temporal stability curve indicates cluster structure is robust even with limited data — essential for online and streaming applications.

---

## AI Integration — Gemini Oracle

### UnSuPERvIsED-I: AI Insights Tab

The AI tab passes a structured summary (algorithm names, all metric values, stability grades) to **Google Gemini Flash Lite** via the `google.generativeai` SDK. Four quick-analysis templates cover: (1) data quality, (2) k-selection rationale, (3) stability interpretation, (4) recommended next steps. Conversation history is preserved within the session; reports are exportable as Markdown.

### UnSuPERvIsED-II: AI Oracle (Page 9)

A fully dedicated **AI Oracle** page powered by **Gemini 2.0 Flash Lite**, with deeper integration and four operational modes:

**Mode 1 — Contextual Result Interpretation:** Passes comprehensive pipeline state (all metrics, stability grades across all five perturbation regimes, preprocessing diagnostics, dataset shape, Hopkins statistic) to Gemini. The model synthesises these into explanations of *why* specific algorithms outperform others on the given data geometry.

**Mode 2 — Algorithm Selection Advisor:** Given dataset characteristics (size class, dimensionality, density estimates, Hopkins ```math H ```, skewness/kurtosis profile), the Oracle reasons over the 60-algorithm registry and recommends families and specific algorithms with mathematical justification.

**Mode 3 — Hyperparameter Recommendation Engine:** Analyses the k-NN distance profile, elbow curve, gap statistic, and PAC curve jointly to recommend optimal  ```math
\varepsilon
``` , `min_samples`, ```math k ```, and fuzzifier ```math m ``` with mathematical rationale.

**Mode 4 — Free-Form Conversational Q&A:** Full multi-turn conversation with session history. The Oracle can answer any question about the clustering results, specific algorithms, metric interpretations, or deployment recommendations.

| Property | v1 AI Tab | v2 AI Oracle |
|----------|-----------|--------------|
| Model | Gemini Flash Lite | Gemini 2.0 Flash Lite |
| Placement | Tab 7 | Dedicated Page 9 |
| Analysis modes | 4 templates | 4 modes + free-form chat |
| Context depth | Metric values + grades | Full pipeline state + perturbation data |
| Conversation history | Session-scoped | Session-scoped (last 5 turns shown) |

**Setup:**

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-key-here"
```

---

## Visualisation Suite

UnSuPERvIsED-I provides 25+ charts; UnSuPERvIsED-II extends this to 30+.

| Chart | Description | v1 | v2 |
|-------|-------------|----|----|
| 2D scatter | PCA/t-SNE/UMAP cluster colouring | ✓ | ✓ |
| 3D scatter | Interactive rotation | ✓ | ✓ |
| Silhouette bar | Per-sample silhouette sorted by cluster | ✓ | ✓ |
| Elbow curve | WCSS vs k with kneedle detection | ✓ | ✓ |
| Silhouette curve | S(k) with optimal k marker | ✓ | ✓ |
| Gap statistic | Gap(k) ± 1 SE with optimal k marker | ✓ | ✓ |
| PCA scree | Cumulative explained variance | ✓ | ✓ |
| PCA biplot | PC1 vs PC2 with loading vectors | ✓ | ✓ |
| Radar chart | Normalised multi-metric comparison | ✓ | ✓ |
| Correlation heatmap | Feature correlation matrix | ✓ | ✓ |
| Missing heatmap | Missing value pattern | ✓ | ✓ |
| Consensus heatmap | Co-association matrix ```math A ``` | ✓ | ✓ |
| Violin plot | Per-cluster feature distribution | ✓ | ✓ |
| Pair plot | Scatter matrix (≤6 features) | ✓ | ✓ |
| Sunburst | Cluster membership hierarchy | ✓ | ✓ |
| Feature histograms | Per-cluster overlaid histograms | ✓ | ✓ |
| Box plots | Per-cluster feature box-and-whisker | ✓ | ✓ |
| Cluster size bar | Membership counts | ✓ | ✓ |
| Dendrogram | Hierarchical tree (≤500 samples) | ✓ | ✓ |
| Stability bars | ARI per algorithm with grade colours | ✓ | ✓ |
| Perturbation curve | ARI vs noise level ± 1σ bands | ✓ | ✓ |
| Consensus CDF | Empirical CDF of ```math A ``` entries | ✓ | ✓ |
| Hopkins gauge | Clusterability ```math H ``` speedometer | ✓ | ✓ |
| k-NN distance | Sorted distances with ε suggestion | ✓ | ✓ |
| Overlap heatmap | Pairwise cluster overlap matrix | ✓ | ✓ |
| Metrics table | Colour-coded comparison table | ✓ | ✓ |
| Noise response chart | ARI vs regime/level heatmap | — | ✓ |
| ARI box plots | Bootstrap ARI distributions | — | ✓ |
| Persistence heatmap | Per-cluster persistence levels | — | ✓ |
| Score timeline | Metric scores vs run index | — | ✓ |

### Dimensionality Reduction Mathematics

**PCA — Principal Component Analysis**

Embedding  ```math
Z = XW_k
```  where  ```math
W_k \in \mathbb{R}^{d \times k}
```  holds the top-```math k ``` eigenvectors of the sample covariance  ```math
\hat{\Sigma} = \frac{1}{n-1}(X-\bar{X})^\top(X-\bar{X})
``` . Equivalently, the right singular vectors of  ```math
X - \bar{X}
```  from SVD  ```math
(X-\bar{X}) = U\Sigma V^\top
``` . The ```math l ```-th principal component has variance  ```math
\lambda_l
```  (the ```math l ```-th eigenvalue) and explained variance ratio:


```math
\text{EVR}_l = \frac{\lambda_l}{\sum_{r=1}^d \lambda_r}
```


**t-SNE — t-Distributed Stochastic Neighbour Embedding**

High-dimensional pairwise similarities with perplexity-controlled bandwidth  ```math
\sigma_i
``` :


```math
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}, \quad p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}
```


Low-dimensional Student-t similarities (heavy tails combat the crowding problem):


```math
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
```


Minimised objective:


```math
\mathcal{L}_\text{KL} = \text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
```


Gradient:  ```math
\frac{\partial \mathcal{L}}{\partial y_i} = 4\sum_j (p_{ij} - q_{ij})(y_i - y_j)(1+\|y_i-y_j\|^2)^{-1}
``` .

**UMAP — Uniform Manifold Approximation and Projection**

Fuzzy topological cross-entropy minimisation as described in the manifold family section. UMAP preserves both local and global structure better than t-SNE for large datasets.

---

## Dashboard Architecture

### UnSuPERvIsED-I Navigation

```
Tab 1: 📊 Data Profile      — Column stats, correlation matrix, missing heatmap
Tab 2: ⚙️  Preprocessing    — Pipeline config, Hopkins test, k-NN ε profile
Tab 3: 🧬 Algorithms        — Registry browser, family filter, parameter config
Tab 4: 🚀 Run Clustering    — Batch execution, elbow/gap, results table, radar
Tab 5: 📈 Visualizations    — 2D/3D scatter, silhouette, PCA, violin, pair, sunburst
Tab 6: 🔒 Stability         — Bootstrap ARI, consensus heatmap, perturbation curves
Tab 7: 🤖 AI Insights       — Gemini analysis, quick templates, export
```

### UnSuPERvIsED-II Navigation

```
Page 1: 📤 Upload / 🎲 Samples / 🔍 Deep Profile  — Ingestion and exploration
Page 2: ⚙️  Preprocessing                          — Pipeline config and Hopkins
Page 3: 🧬 Algorithm Selection                     — Family browser, tag filters
Page 4: ⚡ Run & Evaluate                          — Adaptive batch execution, Pareto
Page 5: 📊 Visualizations                          — Full 30+ chart suite
Page 6: 🧪 Stability Lab                           — 5-regime stability, grades
Page 7: 🤖 AI Oracle                               — Gemini 2.0 multi-mode analysis
Page 8: 💾 Export Suite                            — Labels, metrics, consensus JSON
```

**Adaptive execution engine features (v2):**

- Dataset size classification: TINY / SMALL / MEDIUM / LARGE / XLARGE
- `_SIZE_BLACKLIST`: automatically filters  ```math
O(n^2)
```  algorithms for XLARGE datasets
- `DBSCANParamTuner`: auto-estimates  ```math
\varepsilon
```  and `min_samples`
- **Thread-safe LRU cache** (200 entries): keyed by `(algorithm_id, data_hash, params_hash)`
- **Runtime tracker** via exponential moving average:  ```math
\hat{T}_j^{(t+1)} = (1-\alpha)\hat{T}_j^{(t)} + \alpha T_j^{(t)}
``` ,  ```math
\alpha = 0.3
``` 
- **Lazy loading**: heavy modules imported only when the user navigates to their tab

---

## Installation & Usage

### UnSuPERvIsED-I

```bash
git clone https://github.com/Devanik21/Substrata-Matrix.git
cd Substrata-Matrix/Basic_Cluster
pip install -r requirements.txt
streamlit run UnSuPERvIsED.py
```

### UnSuPERvIsED-II

```bash
git clone https://github.com/Devanik21/Substrata-Matrix.git
cd Substrata-Matrix/Intermediate_Cluster
pip install -r requirements.txt
streamlit run UnSuPERvIsED.py
```

### Gemini AI Setup (both versions)

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-key-here"
```

**Supported input formats:** CSV, Excel (xlsx/xls), JSON, Parquet, TSV — up to 500,000 rows × 2,000 columns.

---

## Dependencies

### UnSuPERvIsED-I

```
streamlit           # Dashboard framework
numpy               # Numerical computing
pandas              # Tabular data handling
scikit-learn        # Core ML algorithms
scipy               # Scientific computing, linkage, stats
plotly              # Interactive visualisations
hdbscan             # HDBSCAN implementation
sklearn-extra       # K-Medoids (sklearn_extra.cluster)
google-generativeai # Gemini AI integration
umap-learn          # UMAP dimensionality reduction
```

### UnSuPERvIsED-II (adds)

```
tensorflow          # Autoencoder-KMeans (Keras backend)
minisom             # Self-Organising Map implementation
skfuzzy             # Fuzzy C-Means and Possibilistic C-Means
```

---

## Repository Structure

```
Substrata-Matrix/
├── Basic_Cluster/                    ← UnSuPERvIsED-I
│   ├── UnSuPERvIsED.py               # Main Streamlit app
│   ├── clustering_registry.py        # Algorithm registry
│   ├── clustering_runner.py          # Parallel execution engine
│   ├── evaluation.py                 # Metric computation
│   ├── preprocessing.py              # Preprocessing pipeline
│   ├── stability.py                  # Bootstrap stability
│   ├── stability_consensus.py        # Consensus matrices
│   ├── visualization.py              # Plotly visualisations
│   └── requirements.txt
│
└── Intermediate_Cluster/             ← UnSuPERvIsED-II
    ├── UnSuPERvIsED.py               # Main Streamlit app (lazy-loaded)
    ├── clustering_registry.py        # 60+ algorithm specs, 12 families
    ├── clustering_runner.py          # Cache-aware adaptive runner
    ├── evaluation.py                 # Internal + external + Pareto
    ├── preprocessing.py              # Deep profiling + full pipeline
    ├── stability.py                  # 5-regime stability + persistence
    ├── consensus.py                  # Co-association + PAC
    ├── visualization.py              # 30+ Plotly charts
    └── requirements.txt
```

---

## Citation

If you use UnSuPERvIsED-I or UnSuPERvIsED-II in research or build upon them, please cite:

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

@software{devanik2026unsupervised2,
  author       = {Devanik},
  title        = {UnSuPERvIsED-II: A 60-Algorithm Clustering Intelligence System
                  with Adaptive Execution, 12-Family Coverage,
                  and Five-Regime Stability Certification},
  year         = {2026},
  month        = {May},
  version      = {2.0},
  url          = {https://github.com/Devanik21/Substrata-Matrix},
  note         = {Intermediate\_Cluster module of the Substrata-Matrix repository}
}
```

---

## References

**Centroid & Partitional:**
- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Trans. Inf. Theory*, 28(2), 129–137.
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA*, 1027–1035.
- Sculley, D. (2010). Web-scale k-means clustering. *WWW*, 1177–1178. [MiniBatch]

**Hierarchical:**
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *JASA*, 58, 236–244.
- Zhang, T., Ramakrishnan, R., & Livny, M. (1996). BIRCH: an efficient data clustering method. *SIGMOD*, 103–114.
- Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley. [DIANA]

**Density:**
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters. *KDD*, 226–231. [DBSCAN]
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD*, 160–172. [HDBSCAN]
- Ankerst, M., Breunig, M. M., Kriegel, H.-P., & Sander, J. (1999). OPTICS. *SIGMOD Record*, 28(2), 49–60.
- Comaniciu, D., & Meer, P. (2002). Mean shift: a robust approach toward feature space analysis. *IEEE TPAMI*, 24(5), 603–619.
- Hinneburg, A., & Keim, D. (1998). An efficient approach to clustering in large multimedia databases with noise. *KDD*, 58–65. [DENCLUE]

**Distribution & Probabilistic:**
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS-B*, 39(1), 1–38.
- Frey, B. J., & Dueck, D. (2007). Clustering by passing messages between data points. *Science*, 315(5814), 972–976. [AP]

**Spectral & Graph:**
- von Luxburg, U. (2007). A tutorial on spectral clustering. *Stat. Comput.*, 17(4), 395–416.
- Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation. *IEEE TPAMI*, 22(8), 888–905.

**Evaluation:**
- Rousseeuw, P. J. (1987). Silhouettes. *J. Comput. Appl. Math.*, 20, 53–65.
- Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE TPAMI*, 1(2), 224–227.
- Caliński, T., & Harabász, J. (1974). A dendrite method for cluster analysis. *Commun. Stat.*, 3(1), 1–27.
- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters via the gap statistic. *JRSS-B*, 63(2), 411–423.
- Xie, X. L., & Beni, G. (1991). A validity measure for fuzzy clustering. *IEEE TPAMI*, 13(8), 841–847.
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *J. Classif.*, 2(1), 193–218. [ARI]
- Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for clusterings comparison. *JMLR*, 11, 2837–2854. [AMI, NMI]
- Monti, S., et al. (2003). Consensus clustering. *Mach. Learn.*, 52, 91–118.

**Stability:**
- Hopkins, B. (1954). A new method for determining the type of distribution of plant individuals. *Ann. Bot.*, 18(2), 213–227.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *ICDM*, 413–422.
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF. *SIGMOD*, 93–104.

**Neural & Manifold:**
- Kohonen, T. (1990). The self-organizing map. *Proc. IEEE*, 78(9), 1464–1480.
- Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. *Science*, 290(5500), 2319–2323. [Isomap]
- Roweis, S. T., & Saul, L. K. (2000). Nonlinear dimensionality reduction by locally linear embedding. *Science*, 290(5500), 2323–2326. [LLE]
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*.

**Fuzzy:**
- Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms*. Plenum Press. [FCM]
- Krishnapuram, R., & Keller, J. (1993). A possibilistic approach to clustering. *IEEE Trans. Fuzzy Syst.*, 1(2), 98–110. [PCM]

**Subspace & Ensemble:**
- Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space. *Contemp. Math.*, 26, 189–206. [JL lemma]
- Agrawal, R., et al. (1998). Automatic subspace clustering of high dimensional data. *KDD*, 94–105. [CLIQUE]

---

*UnSuPERvIsED Series · 2026–2027 · Devanik · NIT Agartala · Samsung ISWDP Fellow*
*Substrata-Matrix — [github.com/Devanik21/Substrata-Matrix](https://github.com/Devanik21/Substrata-Matrix)*

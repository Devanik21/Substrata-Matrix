# Performance Tuning & Scalability

## Adaptive Execution Engine
For datasets > 100,000 rows, UnSuPERvIsED automatically activates the following optimizations:

### 1. Spatial Indexing
KD-Trees and Ball-Trees are forcefully enabled for DBSCAN, OPTICS, and HDBSCAN to reduce $O(N^2)$ to $O(N \log N)$.

### 2. O(N²) Algorithm Blacklisting
Algorithms with hard quadratic time/space bounds (e.g., standard Affinity Propagation, Agglomerative Clustering without Lance-Williams optimizations) are gracefully skipped unless strictly overridden.

### 3. LRU Response Cache
Repeated computations for Hyperparameter sweeps utilize a Thread-Safe In-Memory LRU Cache.

## Hardware Acceleration
- Ensure `scikit-learn-intelex` is installed for Intel architecture CPU optimizations.
- TensorFlow backends for `DeepClusteringNetwork` leverage CUDA natively if `nvidia-smi` detects active GPUs.

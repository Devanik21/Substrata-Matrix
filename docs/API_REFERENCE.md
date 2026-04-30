# API Reference - UnSuPERvIsED Framework

## Core Modules

### `clustering_registry.py`
Provides a central registry for all algorithms across 12 families.
- `get_algorithm(name: str)`: Retrieves the class implementation.
- `list_families()`: Returns available clustering families.

### `clustering_runner.py`
Cache-aware, adaptive asynchronous task scheduler for algorithms.
- `AdaptiveRunner(max_workers: int)`: Class orchestrating batch runs.

### `evaluation.py`
- `CompositeScorer`: Normalizes and aggregates multi-metric evaluation.
- `ParetoFrontCalculator`: Identifies non-dominated algorithms.

## Advanced Cluster (Extension)

- `DeepClusteringNetwork`: Deep autoencoder + KMeans.
- `TimeSeriesClustering`: DTW-based agglomerative merging.
- `FederatedKMeans`: Distributed centroid aggregation without raw data exchange.

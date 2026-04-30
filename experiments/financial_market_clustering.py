import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import logging

logging.basicConfig(level=logging.INFO)

def run_financial_experiment():
    """
    Simulates fetching financial time series data and applying spectral clustering
    to identify market regimes and correlated assets.
    """
    logging.info("Initializing Financial Market Clustering Experiment...")
    np.random.seed(42)
    n_assets = 50
    n_days = 252

    # Generate synthetic log returns
    returns = np.random.normal(0.0005, 0.015, (n_assets, n_days))

    # Compute correlation matrix
    correlation_matrix = np.corrcoef(returns)

    # Convert correlation to distance matrix for clustering
    distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
    affinity_matrix = np.exp(- distance_matrix ** 2 / (2 * np.std(distance_matrix)**2))

    logging.info("Running Spectral Clustering on asset affinity matrix...")
    sc = SpectralClustering(n_clusters=4, affinity='precomputed', random_state=42)
    labels = sc.fit_predict(affinity_matrix)

    logging.info("Experiment complete. Asset regime clusters:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        logging.info("Cluster %d: %d assets", u, c)

if __name__ == "__main__":
    run_financial_experiment()

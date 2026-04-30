import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import logging

logging.basicConfig(level=logging.INFO)

def run_segmentation_experiment():
    """
    Uses Gaussian Mixture Models (GMM) to soft-cluster synthetic customer data
    based on RFM (Recency, Frequency, Monetary) metrics.
    """
    logging.info("Initializing Customer Segmentation Analysis...")
    np.random.seed(101)

    # Synthetic RFM data
    # Cluster 1: High value, recent
    c1 = np.random.normal(loc=[10, 50, 1000], scale=[5, 10, 200], size=(300, 3))
    # Cluster 2: At-risk, low value
    c2 = np.random.normal(loc=[100, 5, 50], scale=[20, 2, 10], size=(500, 3))
    # Cluster 3: New customers
    c3 = np.random.normal(loc=[5, 1, 100], scale=[2, 0.5, 30], size=(200, 3))

    data = np.vstack([c1, c2, c3])

    logging.info("Fitting Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=101)
    labels = gmm.fit_predict(data)

    score = silhouette_score(data, labels)
    logging.info("GMM Clustering complete. Silhouette Score: %.4f", score)

    # Soft assignments
    probs = gmm.predict_proba(data)
    ambiguous_customers = np.sum((np.max(probs, axis=1) < 0.6))
    logging.info("Identified %d structurally ambiguous customers requiring targeted retention.", ambiguous_customers)

if __name__ == "__main__":
    run_segmentation_experiment()

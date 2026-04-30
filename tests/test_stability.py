import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

def test_bootstrap_stability(basic_dataset):
    X, _ = basic_dataset

    kmeans1 = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)

    # Perturb data slightly
    X_perturbed = X + np.random.normal(0, 0.1, X.shape)
    kmeans2 = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_perturbed)

    ari = adjusted_rand_score(kmeans1.labels_, kmeans2.labels_)
    assert ari > 0.8, "Clustering should be stable under minor perturbation."

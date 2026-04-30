import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def test_internal_validity_indices(basic_dataset):
    X, y = basic_dataset

    # Test Silhouette
    sil = silhouette_score(X, y)
    assert sil > 0.5, "Silhouette score should be high for well-separated blobs."

    # Test Davies-Bouldin
    db = davies_bouldin_score(X, y)
    assert db < 1.0, "DB Index should be low for compact clusters."

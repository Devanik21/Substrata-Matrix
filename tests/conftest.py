import pytest
import numpy as np
from sklearn.datasets import make_blobs

@pytest.fixture
def basic_dataset():
    X, y = make_blobs(n_samples=200, centers=4, n_features=5, random_state=42)
    return X, y

@pytest.fixture
def messy_dataset():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=2.5, random_state=99)
    # Inject outliers
    outliers = np.random.uniform(low=-15, high=15, size=(10, 2))
    X = np.vstack([X, outliers])
    y = np.concatenate([y, [-1]*10])
    return X, y

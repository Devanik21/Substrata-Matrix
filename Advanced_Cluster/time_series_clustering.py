import numpy as np
from sklearn.cluster import AgglomerativeClustering
import warnings

def dtw_distance(s1, s2, w=None):
    """
    Computes Dynamic Time Warping (DTW) distance between two time series.
    """
    n, m = len(s1), len(s2)
    if w is None:
        w = max(n, m)
    w = max(w, abs(n - m))

    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(max(1, i-w), min(m+1, i+w+1)):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                          dtw_matrix[i, j-1],    # deletion
                                          dtw_matrix[i-1, j-1])  # match

    return dtw_matrix[n, m]

class TimeSeriesClustering:
    """
    Agglomerative clustering optimized for Time Series data using DTW.
    """
    def __init__(self, n_clusters=3, window_size=10):
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.labels_ = None

    def fit(self, X):
        """
        Fit the time series clustering model.
        X: array-like of shape (n_samples, n_timestamps)
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = dtw_distance(X[i], X[j], self.window_size)
                distances[i, j] = dist
                distances[j, i] = dist

        # Hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters=self.n_clusters, metric='precomputed', linkage='average')
        self.labels_ = clustering.fit_predict(distances)
        return self

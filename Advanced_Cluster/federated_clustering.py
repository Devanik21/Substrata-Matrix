import numpy as np
from sklearn.cluster import KMeans

class FederatedKMeans:
    """
    Federated Learning architecture for K-Means clustering.
    Allows distributed clients to compute clusters without sharing raw data.
    """
    def __init__(self, n_clusters=5, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.global_centroids = None

    def fit(self, client_datasets):
        """
        client_datasets: list of ndarrays, each representing data on a remote client.
        """
        # Initialize centroids from the first client's data
        np.random.seed(42)
        all_data = np.vstack(client_datasets)
        initial_indices = np.random.choice(all_data.shape[0], self.n_clusters, replace=False)
        self.global_centroids = all_data[initial_indices]

        for iteration in range(self.max_iter):
            client_centroids = []
            client_counts = []

            # Client side: compute local updates
            for X in client_datasets:
                distances = np.linalg.norm(X[:, np.newaxis] - self.global_centroids, axis=2)
                labels = np.argmin(distances, axis=1)

                local_centroids = np.zeros_like(self.global_centroids)
                counts = np.zeros(self.n_clusters)

                for k in range(self.n_clusters):
                    cluster_points = X[labels == k]
                    if len(cluster_points) > 0:
                        local_centroids[k] = np.mean(cluster_points, axis=0)
                        counts[k] = len(cluster_points)

                client_centroids.append(local_centroids)
                client_counts.append(counts)

            # Server side: aggregate updates
            new_centroids = np.zeros_like(self.global_centroids)
            total_counts = np.zeros(self.n_clusters)

            for local_centroids, counts in zip(client_centroids, client_counts):
                for k in range(self.n_clusters):
                    new_centroids[k] += local_centroids[k] * counts[k]
                    total_counts[k] += counts[k]

            for k in range(self.n_clusters):
                if total_counts[k] > 0:
                    new_centroids[k] /= total_counts[k]
                else:
                    new_centroids[k] = self.global_centroids[k]

            shift = np.linalg.norm(new_centroids - self.global_centroids)
            self.global_centroids = new_centroids

            if shift < self.tol:
                break

        return self

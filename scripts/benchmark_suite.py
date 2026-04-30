import time
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
import logging

logging.basicConfig(level=logging.INFO)

def run_benchmarks():
    logging.info("Starting Benchmark Suite...")

    sample_sizes = [1000, 10000, 50000]

    for n in sample_sizes:
        logging.info("Benchmarking with N=%d", n)
        X, _ = make_blobs(n_samples=n, centers=5, n_features=10, random_state=42)

        # Benchmark KMeans
        start = time.time()
        KMeans(n_clusters=5, n_init=5).fit(X)
        km_time = time.time() - start

        # Benchmark DBSCAN
        start = time.time()
        DBSCAN(eps=1.5, min_samples=10).fit(X)
        db_time = time.time() - start

        logging.info("N=%d | KMeans: %.4fs | DBSCAN: %.4fs", n, km_time, db_time)

if __name__ == "__main__":
    run_benchmarks()

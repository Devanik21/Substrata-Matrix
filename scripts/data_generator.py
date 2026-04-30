import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_moons, make_circles
import argparse

def generate_synthetic_data(output_path, n_samples=10000, n_features=20, n_clusters=5):
    print(f"Generating synthetic dataset: {n_samples} samples, {n_features} features.")

    # Mix of isotropic and anisotropic blobs
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=15, n_redundant=5,
                               n_classes=n_clusters, random_state=42)

    # Introduce some noise
    noise = np.random.normal(0, 0.5, X.shape)
    X = X + noise

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df['target'] = y

    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="synthetic_benchmark_data.csv")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()

    generate_synthetic_data(args.output, n_samples=args.samples)

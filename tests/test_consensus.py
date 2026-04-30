import numpy as np

def test_co_association_matrix():
    # 5 samples, 3 runs
    labels_run1 = [0, 0, 1, 1, 2]
    labels_run2 = [0, 0, 1, 1, 2]
    labels_run3 = [1, 1, 0, 0, 2]  # Identical partition, different label names

    n_samples = 5
    co_assoc = np.zeros((n_samples, n_samples))
    runs = [labels_run1, labels_run2, labels_run3]

    for run in runs:
        for i in range(n_samples):
            for j in range(n_samples):
                if run[i] == run[j]:
                    co_assoc[i, j] += 1

    co_assoc /= len(runs)

    # Samples 0 and 1 always together
    assert co_assoc[0, 1] == 1.0
    # Samples 0 and 2 never together
    assert co_assoc[0, 2] == 0.0

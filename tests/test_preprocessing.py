import numpy as np
from sklearn.preprocessing import StandardScaler

def test_standard_scaler_behavior():
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    assert np.allclose(np.mean(scaled, axis=0), [0, 0])
    assert np.allclose(np.std(scaled, axis=0), [1, 1])

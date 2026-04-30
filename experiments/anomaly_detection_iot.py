import numpy as np
from sklearn.ensemble import IsolationForest
import logging

logging.basicConfig(level=logging.INFO)

def run_iot_anomaly_experiment():
    """
    Utilizes Isolation Forest to detect anomalies in high-dimensional IoT sensor arrays.
    """
    logging.info("Initializing IoT Anomaly Detection Experiment...")
    np.random.seed(404)

    n_sensors = 120
    n_samples = 10000

    # Normal operations
    X_normal = np.random.normal(0, 1, (n_samples, n_sensors))

    # Anomalous events (sensor drift/spikes)
    n_anomalies = 150
    X_anomalous = np.random.normal(3, 2, (n_anomalies, n_sensors))

    X = np.vstack([X_normal, X_anomalous])

    logging.info("Training Isolation Forest...")
    clf = IsolationForest(contamination=0.015, random_state=404)
    preds = clf.fit_predict(X)

    detected_anomalies = np.sum(preds == -1)
    logging.info("Experiment complete. Detected %d anomalous events out of %d true anomalies.", detected_anomalies, n_anomalies)

if __name__ == "__main__":
    run_iot_anomaly_experiment()

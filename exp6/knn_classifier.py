# knn_classifier.py
import numpy as np
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        logging.info(f"KNN Classifier initialized with k={self.k}")

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        logging.info(f"Model trained with {len(X_train)} samples")

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        logging.info(f"Predicting {len(X_test)} samples...")
        predictions = [self._predict(x) for x in X_test]
        logging.info("Prediction complete")
        return np.array(predictions)

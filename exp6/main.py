# main.py
import numpy as np
import matplotlib.pyplot as plt
import logging
from data import load_iris_data, load_wine_data
from utils import train_test_split
from knn_classifier import KNNClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------ IRIS DATASET ------------------
logging.info("=== IRIS DATASET ===")
X, y = load_iris_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)

accuracy = np.sum(preds == y_test) / len(y_test)
logging.info(f"Iris Dataset Accuracy (k=3): {accuracy * 100:.2f}%")

# Hyperparameter tuning for Iris
k_values = [1, 3, 5, 7, 9, 11, 15]
accuracies = []
for k in k_values:
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = np.sum(preds == y_test) / len(y_test)
    accuracies.append(acc)
    logging.info(f"k={k}, Accuracy={acc * 100:.2f}%")

plt.figure(figsize=(6, 4))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. K value (Iris Dataset)")
plt.grid(True)
plt.show()

best_k_iris = k_values[np.argmax(accuracies)]
logging.info(f"Best k for Iris Dataset: {best_k_iris}, Accuracy: {max(accuracies) * 100:.2f}%")

# ------------------ WINE DATASET ------------------
logging.info("=== WINE DATASET ===")
X_wine, y_wine = load_wine_data()
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Train & evaluate using best k from Iris
knn_wine = KNNClassifier(k=best_k_iris)
knn_wine.fit(X_train_w, y_train_w)
preds_wine = knn_wine.predict(X_test_w)

wine_acc = np.sum(preds_wine == y_test_w) / len(y_test_w)
logging.info(f"Wine Dataset Accuracy (using Iris best k={best_k_iris}): {wine_acc * 100:.2f}%")

# Hyperparameter tuning for Wine
k_values_wine = [1, 3, 5, 7, 9, 11, 15]
accuracies_wine = []
for k in k_values_wine:
    knn = KNNClassifier(k=k)
    knn.fit(X_train_w, y_train_w)
    preds = knn.predict(X_test_w)
    acc = np.sum(preds == y_test_w) / len(y_test_w)
    accuracies_wine.append(acc)
    logging.info(f"k={k}, Wine Accuracy={acc * 100:.2f}%")

plt.figure(figsize=(6, 4))
plt.plot(k_values_wine, accuracies_wine, marker='o', color='green')
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. K value (Wine Dataset)")
plt.grid(True)
plt.show()

best_k_wine = k_values_wine[np.argmax(accuracies_wine)]
logging.info(f"Best k for Wine Dataset: {best_k_wine}, Accuracy: {max(accuracies_wine) * 100:.2f}%")

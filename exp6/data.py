# data.py
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_iris_data():
    logging.info("Loading Iris dataset...")
    iris = fetch_ucirepo(id=53)
    X = pd.DataFrame(iris.data.features)
    y = pd.Series(iris.data.targets.iloc[:, 0])
    
    logging.info("Preprocessing Iris labels...")
    y = y.map(lambda label: label.replace("Iris-", ""))
    
    X = X.to_numpy()
    y = y.to_numpy()
    
    logging.info(f"Iris dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    return X, y


def load_wine_data():
    logging.info("Loading Wine dataset...")
    wine = fetch_ucirepo(id=109)
    X = pd.DataFrame(wine.data.features).to_numpy()
    y = pd.Series(wine.data.targets.iloc[:, 0]).to_numpy()
    
    logging.info(f"Wine dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    return X, y

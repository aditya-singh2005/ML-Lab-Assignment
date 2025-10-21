import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data import load_iris_data
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_iris_eda(save_path=None):
    logging.info("Starting EDA for Iris dataset...")
    X, y = load_iris_data()
    df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df['species'] = y

    logging.info("Generating pairplot...")
    sns.pairplot(df, hue='species', diag_kind='hist')
    plt.suptitle("Iris Dataset - Feature Relationships", y=1.02)

    if save_path:
        plt.savefig(save_path)
        logging.info(f"EDA plot saved at {save_path}")
    plt.show()
    logging.info("EDA plot displayed")

if __name__ == "__main__":
    plot_iris_eda("iris_eda_plot.png")

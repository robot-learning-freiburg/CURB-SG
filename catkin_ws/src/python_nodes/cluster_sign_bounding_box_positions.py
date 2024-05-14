"""This is a helper script to test out some clustering stuff."""
# pylint: skip-file
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

def read_csv(file_path):
    return pd.read_csv(file_path)

def plot_points(df):
    plt.scatter(df['X'], df['Y'], c='blue', label='Data Points')

def dbscan_clustering(df):
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(df)
    labels = clustering.labels_
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for noise
        class_member_mask = (labels == k)
        xy = df[class_member_mask]
        plt.scatter(xy['X'], xy['Y'], c=[col], label=f'Cluster {k}')

def main(file_path):
    df = read_csv(file_path)
    plot_points(df)
    # dbscan_clustering(df)
    
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('DBSCAN Clustering')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file_path>")
    else:
        file_path = sys.argv[1]
        main(file_path)

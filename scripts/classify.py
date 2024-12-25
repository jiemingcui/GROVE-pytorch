# Aim:     improve data-balance
# Method:  classify pose data into 500 clusters
# Modules: k-means++, Silhouette scores
# Args:    plot_counts(bool), data_dir(str), classify_dir(str), n_clusters(int)

import argparse
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PoseDataClustering:
    def __init__(self, data_dir="data", classify_dir="classify", n_clusters=500):
        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self.classify_dir = os.path.join(os.getcwd(), classify_dir)
        self.n_clusters = n_clusters
        self.pose_data = None
        self.labels = None
        self.cluster_indices = None

    def load_data(self):
        """Load pose data from specified directory."""
        pose_data_1 = joblib.load(os.path.join(self.data_dir, "pose_data_amass.joblib"))
        pose_data_2 = joblib.load(os.path.join(self.data_dir, "pose_data_idea400.joblib"))
        pose_data_3 = np.load(os.path.join(self.data_dir, "failure_cases.npy"))
        pose_data_4 = np.load(os.path.join(self.data_dir, "pose_data_other_motionx.npy"))
        self.pose_data = np.concatenate((pose_data_1, pose_data_2, pose_data_3, pose_data_4), axis=0)
        logging.info(f"Pose Data's Shape: {self.pose_data.shape}")

    def normalize_data(self):
        """Normalize the pose data."""
        scaler = StandardScaler()
        self.pose_data = scaler.fit_transform(self.pose_data)

    def classify_data(self):
        """Perform K-means clustering on the normalized data."""
        kmeans = KMeans(init='k-means++', n_clusters=self.n_clusters, n_init=10)
        
        logging.info("Now classifying...")
        start_time = time.time()
        kmeans.fit(self.pose_data)
        elapsed_time = time.time() - start_time
        logging.info(f"Classification complete. Time taken: {elapsed_time:.2f} seconds.")
        
        self.labels = kmeans.labels_
        self.cluster_indices = {cluster_id: np.where(self.labels == cluster_id)[0] for cluster_id in range(self.n_clusters)}

    def save_results(self, plot_counts=False):
        """Save cluster indices and generate optional plots."""
        os.makedirs(self.classify_dir, exist_ok=True)
        
        # Save cluster indices to file
        joblib.dump(self.cluster_indices, os.path.join(self.classify_dir, f"indices.joblib"))

        # Calculate and log silhouette score
        score = silhouette_score(self.pose_data, self.labels)
        logging.info(f"n_clusters: {self.n_clusters}, Silhouette Coefficient: {score:.4f}")

        # Optional: Plot cluster counts
        if plot_counts:
            self.plot_cluster_counts()

    def plot_cluster_counts(self):
        """Plot the number of clusters for each sample count."""
        cluster_counts = pd.Series(self.labels).value_counts()
        sample_size_counts = cluster_counts.value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        plt.plot(sample_size_counts.index, sample_size_counts.values, marker='o', linestyle='-', color='skyblue')
        plt.title('Number of Clusters for Each Sample Count')
        plt.xlabel('Number of Samples in Cluster')
        plt.ylabel('Number of Clusters')
        plt.xticks([sample_size_counts.index.min(), sample_size_counts.index.max()])  
        plt.grid(True)
        plt.savefig(os.path.join(self.classify_dir, f'cluster_counts.png'))
        plt.close()

    def run(self, plot_counts=False):
        """Execute the full pipeline with optional plotting."""
        self.load_data()
        self.normalize_data()
        self.classify_data()
        self.save_results(plot_counts=plot_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Data Clustering')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the pose data files')
    parser.add_argument('--classify_dir', type=str, default='classify', help='Directory to save classification results')
    parser.add_argument('--n_clusters', type=int, default=500, help='Number of clusters for K-means')
    parser.add_argument('--plot_counts', action='store_true', help='Flag to plot the cluster counts')

    args = parser.parse_args()

    clustering = PoseDataClustering(data_dir=args.data_dir, classify_dir=args.classify_dir, n_clusters=args.n_clusters)
    clustering.run(plot_counts=args.plot_counts)

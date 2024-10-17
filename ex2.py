import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.metrics import silhouette_score

# Function to apply clustering and visualization
def apply_clustering(X, dataset_name, num_centers):
    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c='black', marker='o', s=15, label='Unlabeled data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{dataset_name}: Unlabeled Data')
    plt.show()

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_centers, init='random')
    kmeans.fit(X)
    labels_kmeans = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Plot KMeans results
    plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', s=50, marker='o', label='KMeans clusters')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{dataset_name}: KMeans Clustering')
    plt.legend()
    plt.show()

    # Apply Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=num_centers, covariance_type='full')
    gmm.fit(X)
    labels_gmm = gmm.predict(X)

    # Plot GMM results
    plt.scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap='coolwarm', s=50, marker='o', label='GMM clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{dataset_name}: GMM Clustering')
    plt.legend()
    plt.show()

    # Silhouette score for GMM
    silhouette_avg = silhouette_score(X, labels_gmm)
    print(f"Silhouette Score ({dataset_name} - GMM): {silhouette_avg}")

# Load datasets
datasets = {
    "Iris": load_iris().data[:, :2],  # First two features
    "Wine": load_wine().data[:, :2],  # First two features
    "Digits": load_digits().data[:, :2]  # First two features
}

# Number of centers for clustering
num_centers = int(input("Number of centers: "))

# Apply clustering to each dataset
for dataset_name, X in datasets.items():
    apply_clustering(X, dataset_name, num_centers)

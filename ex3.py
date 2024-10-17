import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import scipy.io
import pandas as pd

# Load the .mat file
mat_contents = scipy.io.loadmat('wksp_XY.mat')

# Extract the X (features) and Y (labels) data
X = pd.DataFrame(mat_contents['X'])  # Assuming 'X' contains feature data
Y = pd.DataFrame(mat_contents['Y'], columns=['Label'])  # Assuming 'Y' contains labels

# Convert labels from 'M' and 'B' to numerical values (Malignant = 1, Benign = 0)
Y['Label'] = Y['Label'].str[0].map({'M': 1, 'B': 0})

# Visualize the first two features of the dataset with the true labels
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y['Label'], cmap='coolwarm', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('WBCD Data Visualization (Feature 1 vs Feature 2)')
plt.colorbar(label='Label')
plt.show()

# Apply Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict the cluster assignments
gmm_labels = gmm.predict(X)

# Determine the cluster indices corresponding to benign (0) and malignant (1)
benign_indices = np.where(Y['Label'].values == 0)[0]
malignant_indices = np.where(Y['Label'].values == 1)[0]

# Count the occurrences of each cluster for benign and malignant groups
benign_cluster_counts = np.bincount(gmm_labels[benign_indices])
malignant_cluster_counts = np.bincount(gmm_labels[malignant_indices])

# Get the cluster with the highest count for each label
benign_cluster = np.argmax(benign_cluster_counts) if benign_cluster_counts.size > 0 else None
malignant_cluster = np.argmax(malignant_cluster_counts) if malignant_cluster_counts.size > 0 else None

# Print the determined clusters
print(f'Benign cluster: {benign_cluster}, Malignant cluster: {malignant_cluster}')

# Map the GMM labels to benign (0) and malignant (1)
mapped_labels = np.where(gmm_labels == benign_cluster, 0, 1)

# Visualize the clustering result, now with red for benign and blue for malignant
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=mapped_labels, cmap=plt.cm.coolwarm, s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('GMM Clustering (Benign = Red, Malignant = Blue)')
plt.colorbar(label='Cluster')
plt.show()

# Compute and display the silhouette score for evaluating the quality of the clustering
silhouette_avg = silhouette_score(X, mapped_labels)
print(f'Silhouette Score (GMM): {silhouette_avg}')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D

# Generate first dataset
MU1 = [1, 0.5]
SIGMA1 = [[4, 0], [0, 0.5]]
MU2 = [-2, -2]
SIGMA2 = [[1, 0], [0, 1]]

dataSet1 = np.random.multivariate_normal(MU1, SIGMA1, 1000)
dataSet2 = np.random.multivariate_normal(MU2, SIGMA2, 1000)
dataSet = np.vstack((dataSet1, dataSet2))

# Generate second dataset
MU3 = [4, 4]
SIGMA3 = [[1, 0], [0, 2]]
MU4 = [-3, 3]
SIGMA4 = [[1, 0.2], [0.2, 0.8]]

dataSet3 = np.random.multivariate_normal(MU3, SIGMA3, 1000)
dataSet4 = np.random.multivariate_normal(MU4, SIGMA4, 1000)
dataSet_additional = np.vstack((dataSet3, dataSet4))


# Function to apply GMM and visualize both 2D and 3D plots
def visualize_gmm(data, n_components, title_suffix):
    # Apply GMM to the data
    gmm = GaussianMixture(n_components=n_components, covariance_type='full').fit(data)

    # Plot original data
    plt.scatter(data[:, 0], data[:, 1], s=10)
    plt.title(f"Generated Data - {title_suffix}")
    plt.show()

    # Generate grid for contour/surface plotting
    x = np.linspace(-8, 8, 100)
    y = np.linspace(-8, 8, 100)
    X, Y = np.meshgrid(x, y)
    XY = np.array([X.ravel(), Y.ravel()]).T
    Z = np.exp(gmm.score_samples(XY)).reshape(X.shape)

    # 2D Contour plot
    plt.contour(X, Y, Z)
    plt.scatter(data[:, 0], data[:, 1], s=10)
    plt.title(f"GMM - 2D Contours - {title_suffix}")
    plt.show()

    # 3D Surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')
    plt.title(f"GMM Surface Plot (3D) - {title_suffix}")
    plt.show()


# Visualize GMM for the first dataset
visualize_gmm(dataSet, n_components=2, title_suffix="Dataset 1")

# Visualize GMM for the second dataset
visualize_gmm(dataSet_additional, n_components=2, title_suffix="Dataset 2")

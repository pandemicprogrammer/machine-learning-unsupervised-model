import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming X is your data
X = -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = X1

# Initialize KMeans
kmeans = KMeans(n_clusters=2)

# Fitting with inputs
kmeans = kmeans.fit(X)

# Predicting clusters
labels = kmeans.predict(X)

# Getting the cluster centers
C = kmeans.cluster_centers_

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=1000)
plt.show()

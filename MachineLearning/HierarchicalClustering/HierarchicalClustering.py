import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Example data
X = np.array([
    [1],
    [2],
    [8],
    [9]
])

# Create the model
model = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

# Fit the model and predict cluster labels
labels = model.fit_predict(X)

print("Cluster labels:", labels)

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

X = [[1], [2], [8], [9]]

Z = linkage(X, method="ward")

dendrogram(Z)
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.title("Hierarchical Clustering Dendrogram")
plt.show()
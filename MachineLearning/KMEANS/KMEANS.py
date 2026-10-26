import numpy as np
from sklearn.cluster import KMeans

# Example data: study hours and sleep hours
X = np.array([
    [1, 8],
    [2, 7],
    [1.5, 8.5],
    [7, 5],
    [8, 4],
    [7.5, 4.5]
])

# Create the model
model = KMeans(n_clusters=2, random_state=42, n_init=10)

# Train the model and assign clusters
model.fit(X)

# Display the cluster assigned to each student
print("Cluster labels:", model.labels_)

# Display the final centroids
print("Centroids:")
print(model.cluster_centers_)
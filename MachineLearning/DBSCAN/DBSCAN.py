import numpy as np
from sklearn.cluster import DBSCAN

# Example data
X = np.array([
    [1],
    [2],
    [3],
    [10],
    [11],
    [25]
])

# Create the model
model = DBSCAN(
    eps=2,
    min_samples=2
)

# Find clusters
labels = model.fit_predict(X)

print("Cluster labels:", labels)
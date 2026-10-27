import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Dataset: study hours and preparation hours
X = np.array([
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
    [5, 10]
])

# Step 1: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Create PCA and keep one component
pca = PCA(n_components=1)

# Step 3: Transform the data
X_reduced = pca.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)

print("Reduced data:")
print(X_reduced)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
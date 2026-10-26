from sklearn.neighbors import KNeighborsClassifier

X_train = [
    [1, 1],
    [2, 2],
    [3, 3],
    [6, 6],
    [7, 7]
]

y_train = ["Red", "Red", "Blue", "Blue", "Blue"]

# Create the model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X_train, y_train)

# Predict
prediction = model.predict([[3, 2]])

print("Predicted class:", prediction[0])

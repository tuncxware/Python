from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Training data: study hours
X = [[1], [2], [2], [3], [4], [5]]

# Results: 0 = Fail, 1 = Pass
y = [0, 0, 0, 1, 1, 1]

# Create the model
model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

# Train the model
model.fit(X, y)

# Predict for a student who studies 4 hours
prediction = model.predict([[4]])

print("Prediction:", prediction[0])

# Predict probabilities
probability = model.predict_proba([[4]])

print("Probabilities:", probability)

# Draw the tree
plt.figure(figsize=(10, 6))

plot_tree(
    model,
    feature_names=["Study hours"],
    class_names=["Fail", "Pass"],
    filled=True
)

plt.show()
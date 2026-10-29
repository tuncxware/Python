from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Input: study hours
X = [[1], [2], [3], [4], [5], [6], [7], [8]]

# Target: 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1]

# Create a neural network
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        max_iter=3000,
        random_state=42
    )
)

# Train the model
model.fit(X, y)

# Predict for a student who studies 4.5 hours
prediction = model.predict([[4.5]])
probability = model.predict_proba([[4.5]])

print("Predicted class:", prediction[0])
print("Probability of passing:", probability[0][1])
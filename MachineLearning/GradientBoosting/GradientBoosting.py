from sklearn.ensemble import GradientBoostingRegressor

# Study hours and exam scores
X = [[1], [2], [3], [4], [5], [6]]
y = [50, 55, 65, 70, 80, 85]

# Create the model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=2,
    random_state=42
)

# Train the model
model.fit(X, y)

# Predict the score for a student who studies 3.5 hours
prediction = model.predict([[3.5]])

print("Predicted score:", prediction[0])
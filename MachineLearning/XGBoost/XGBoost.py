from xgboost import XGBRegressor

# Study hours
X = [[1], [2], [3], [4], [5], [6]]

# Actual exam scores
y = [50, 55, 65, 70, 80, 85]

# Create the model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=2,
    objective="reg:squarederror",
    random_state=42
)

# Train the model
model.fit(X, y)

# Predict the score for 3.5 study hours
prediction = model.predict([[3.5]])

print("Predicted score:", prediction[0])
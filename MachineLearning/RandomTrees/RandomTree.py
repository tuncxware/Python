from sklearn.ensemble import RandomForestClassifier

# Training data: study hours
X = [[1], [2], [2], [3], [4], [5]]

# Results: 0 = Fail, 1 = Pass
y = [0, 0, 0, 1, 1, 1]

# Create the Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)

# Predict for a student who studies 4 hours
prediction = model.predict([[4]])

if prediction[0] == 1:
    print("Pass")
else:
    print("Fail")
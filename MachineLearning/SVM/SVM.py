from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Features: [study hours, attendance percentage]
X = [
    [1, 50],
    [2, 55],
    [2, 60],
    [5, 85],
    [6, 90],
    [7, 95]
]

# 0 = Fail, 1 = Pass
y = [0, 0, 0, 1, 1, 1]

# Scale the features, then train a linear SVM
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", C=1.0)
)

model.fit(X, y)

# Predict for a new student
new_student = [[4, 75]]

prediction = model.predict(new_student)

print("Prediction:", "Pass" if prediction[0] == 1 else "Fail")
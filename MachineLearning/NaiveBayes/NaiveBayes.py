from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training emails
emails = [
    "free prize winner",
    "win free prize",
    "meeting schedule today",
    "project meeting schedule",
    "free winner prize",
    "schedule a meeting"
]

# 1 = Spam, 0 = Not Spam
labels = [1, 1, 0, 0, 1, 0]

# Convert text into numerical features
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(emails)

# Create and train the model
model = MultinomialNB()
model.fit(X_train, labels)

# Predict a new email
new_email = ["free prize winner"]

X_new = vectorizer.transform(new_email)

prediction = model.predict(X_new)
probabilities = model.predict_proba(X_new)

print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")
print("Class probabilities:", probabilities)
print("Class order:", model.classes_)

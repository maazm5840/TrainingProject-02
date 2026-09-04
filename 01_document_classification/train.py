import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
data = pd.read_csv("dataset/documents.csv")

# 2. Separate document text and category
X = data["document_text"]
y = data["category"]

# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create the classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 5. Train the model
model.fit(X_train, y_train)

# 6. Test the model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 7. Save the trained model
joblib.dump(model, "model/document_classifier.pkl")

print("\nModel saved successfully!")
import joblib

model = joblib.load("model/document_classifier.pkl")

print("Hospital Document Classifier")
print("Type your document text below.")
print("Type 'exit' to stop.\n")

while True:
    document = input("Enter document: ")

    if document.lower() == "exit":
        print("Program stopped.")
        break

    predicted_category = model.predict([document])[0]
    probabilities = model.predict_proba([document])[0]
    confidence = max(probabilities) * 100

    print("Predicted Category:", predicted_category)
    print("Confidence:", round(confidence, 2), "%")
    print()
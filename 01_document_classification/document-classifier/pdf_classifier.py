import joblib
from pypdf import PdfReader

# Load trained model
model = joblib.load("model/document_classifier.pkl")

# Ask for PDF path
pdf_path = input("Enter PDF file path: ")

# Read PDF
reader = PdfReader(pdf_path)

document = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        document += text + "\n"

# Check whether text was extracted
if not document.strip():
    print("No text found in the PDF.")
else:
    predicted_category = model.predict([document])[0]
    probabilities = model.predict_proba([document])[0]
    confidence = max(probabilities) * 100

    print("\nPredicted Category:", predicted_category)
    print("Confidence:", round(confidence, 2), "%")
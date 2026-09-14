# ============================================================
# HOSPITAL RAG SYSTEM - QUERY CLASSIFICATION
# TF-IDF + Logistic Regression
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. TRAINING DATA
# ============================================================

data = [

    # ---------------- HOSPITAL_INFO ----------------
    ("What are the hospital visiting hours?", "HOSPITAL_INFO"),
    ("What time does the hospital open?", "HOSPITAL_INFO"),
    ("What time does the hospital close?", "HOSPITAL_INFO"),
    ("What are the hospital working hours?", "HOSPITAL_INFO"),
    ("What are the hospital timings?", "HOSPITAL_INFO"),
    ("When can I visit a patient?", "HOSPITAL_INFO"),
    ("What are the visiting hours for patients?", "HOSPITAL_INFO"),
    ("Where is the hospital located?", "HOSPITAL_INFO"),
    ("What is the hospital address?", "HOSPITAL_INFO"),
    ("What is the hospital phone number?", "HOSPITAL_INFO"),
    ("How can I contact the hospital?", "HOSPITAL_INFO"),
    ("Give me the hospital contact details.", "HOSPITAL_INFO"),
    ("Is the hospital open on Sunday?", "HOSPITAL_INFO"),
    ("Is the hospital open on weekends?", "HOSPITAL_INFO"),
    ("What are the general hospital timings?", "HOSPITAL_INFO"),
    ("How do I contact the hospital reception?", "HOSPITAL_INFO"),

    # ---------------- ADMISSION ----------------
    ("How can I get admitted to the hospital?", "ADMISSION"),
    ("What documents are required for admission?", "ADMISSION"),
    ("What documents are needed for admission?", "ADMISSION"),
    ("What is the hospital admission process?", "ADMISSION"),
    ("How do I register a patient?", "ADMISSION"),
    ("How can I register for hospital admission?", "ADMISSION"),
    ("What is required for patient registration?", "ADMISSION"),
    ("How can a new patient register?", "ADMISSION"),
    ("What are the admission requirements?", "ADMISSION"),
    ("Where can I complete admission registration?", "ADMISSION"),
    ("How do I admit a patient?", "ADMISSION"),
    ("What information is needed during admission?", "ADMISSION"),
    ("What forms are required for admission?", "ADMISSION"),
    ("Can I register a patient online?", "ADMISSION"),
    ("Where is the admission counter?", "ADMISSION"),
    ("How long does hospital registration take?", "ADMISSION"),

    # ---------------- DEPARTMENT ----------------
    ("Where is the cardiology department?", "DEPARTMENT"),
    ("Where is the neurology department?", "DEPARTMENT"),
    ("Where is the orthopedic department?", "DEPARTMENT"),
    ("Where is the pediatric department?", "DEPARTMENT"),
    ("Where is the dermatology department?", "DEPARTMENT"),
    ("Which department treats heart problems?", "DEPARTMENT"),
    ("Which department treats brain problems?", "DEPARTMENT"),
    ("Which department treats bone problems?", "DEPARTMENT"),
    ("Which department treats skin problems?", "DEPARTMENT"),
    ("What departments are available in the hospital?", "DEPARTMENT"),
    ("What medical departments does the hospital have?", "DEPARTMENT"),
    ("Where can I find cardiology?", "DEPARTMENT"),
    ("Where can I find neurology?", "DEPARTMENT"),
    ("Which department should I visit for a heart problem?", "DEPARTMENT"),
    ("Which department treats children?", "DEPARTMENT"),
    ("Where is the emergency department?", "DEPARTMENT"),

    # ---------------- APPOINTMENT ----------------
    ("How can I book an appointment?", "APPOINTMENT"),
    ("How do I schedule a doctor appointment?", "APPOINTMENT"),
    ("Can I book an appointment with a doctor?", "APPOINTMENT"),
    ("How can I make a doctor appointment?", "APPOINTMENT"),
    ("I want to book a doctor visit.", "APPOINTMENT"),
    ("How do I schedule a hospital appointment?", "APPOINTMENT"),
    ("Can I schedule an appointment online?", "APPOINTMENT"),
    ("How can I cancel my appointment?", "APPOINTMENT"),
    ("How can I reschedule my appointment?", "APPOINTMENT"),
    ("How do I change my appointment time?", "APPOINTMENT"),
    ("How can I check my appointment?", "APPOINTMENT"),
    ("How do I book an appointment with a specialist?", "APPOINTMENT"),
    ("Can I get an appointment today?", "APPOINTMENT"),
    ("How do I make an appointment with a cardiologist?", "APPOINTMENT"),
    ("Where can I book a doctor appointment?", "APPOINTMENT"),
    ("How do I cancel a doctor visit?", "APPOINTMENT"),

    # ---------------- PROCEDURE ----------------
    ("How should I prepare for an MRI?", "PROCEDURE"),
    ("What is the procedure for an MRI scan?", "PROCEDURE"),
    ("How is an MRI performed?", "PROCEDURE"),
    ("How should I prepare for a CT scan?", "PROCEDURE"),
    ("What is the procedure for a CT scan?", "PROCEDURE"),
    ("How is an X-ray performed?", "PROCEDURE"),
    ("What should I do before an X-ray?", "PROCEDURE"),
    ("What preparation is required for surgery?", "PROCEDURE"),
    ("What is the surgery procedure?", "PROCEDURE"),
    ("How is the medical test performed?", "PROCEDURE"),
    ("What should I do before a medical test?", "PROCEDURE"),
    ("What are the steps involved in surgery?", "PROCEDURE"),
    ("How is the procedure carried out?", "PROCEDURE"),
    ("What preparation is needed before the scan?", "PROCEDURE"),
    ("What should I know before an MRI?", "PROCEDURE"),
    ("How long does the medical procedure take?", "PROCEDURE"),

    # ---------------- MEDICATION ----------------
    ("What is this medicine used for?", "MEDICATION"),
    ("What is this medication used for?", "MEDICATION"),
    ("What are the side effects of this medicine?", "MEDICATION"),
    ("What are the side effects of this tablet?", "MEDICATION"),
    ("What is the dosage of this medicine?", "MEDICATION"),
    ("How should I take this medication?", "MEDICATION"),
    ("How often should I take this tablet?", "MEDICATION"),
    ("What is this drug used to treat?", "MEDICATION"),
    ("What are the uses of this capsule?", "MEDICATION"),
    ("What are the medicine instructions?", "MEDICATION"),
    ("Can this medicine cause side effects?", "MEDICATION"),
    ("What is the correct dosage?", "MEDICATION"),
    ("When should I take this medicine?", "MEDICATION"),
    ("What medicine is prescribed for this condition?", "MEDICATION"),
    ("What are the uses of this tablet?", "MEDICATION"),
    ("How should this drug be taken?", "MEDICATION"),

    # ---------------- DISCHARGE ----------------
    ("What is the discharge process?", "DISCHARGE"),
    ("How can a patient get discharged?", "DISCHARGE"),
    ("What documents are required for discharge?", "DISCHARGE"),
    ("What documents are needed for discharge?", "DISCHARGE"),
    ("When can I leave the hospital?", "DISCHARGE"),
    ("What is a discharge summary?", "DISCHARGE"),
    ("How does hospital discharge work?", "DISCHARGE"),
    ("What is required before discharge?", "DISCHARGE"),
    ("How do I get discharge papers?", "DISCHARGE"),
    ("What are the discharge procedures?", "DISCHARGE"),
    ("Who approves patient discharge?", "DISCHARGE"),
    ("How long does discharge take?", "DISCHARGE"),
    ("What should I do before leaving the hospital?", "DISCHARGE"),
    ("Can I get my discharge summary?", "DISCHARGE"),
    ("What documents should I collect at discharge?", "DISCHARGE"),
    ("How can I complete the discharge formalities?", "DISCHARGE"),

    # ---------------- HOSPITAL_POLICY ----------------
    ("What is the visitor policy?", "HOSPITAL_POLICY"),
    ("What are the hospital rules?", "HOSPITAL_POLICY"),
    ("What are the visiting rules?", "HOSPITAL_POLICY"),
    ("Are there any hospital policies for visitors?", "HOSPITAL_POLICY"),
    ("What are the guidelines for visitors?", "HOSPITAL_POLICY"),
    ("What is the hospital policy?", "HOSPITAL_POLICY"),
    ("Are visitors allowed in the ICU?", "HOSPITAL_POLICY"),
    ("What are the rules for patients and visitors?", "HOSPITAL_POLICY"),
    ("How many visitors are allowed?", "HOSPITAL_POLICY"),
    ("What are the visitor restrictions?", "HOSPITAL_POLICY"),
    ("Are children allowed to visit patients?", "HOSPITAL_POLICY"),
    ("What are the hospital visitor guidelines?", "HOSPITAL_POLICY"),
    ("Can visitors stay overnight?", "HOSPITAL_POLICY"),
    ("What are the rules for hospital visitors?", "HOSPITAL_POLICY"),
    ("Is there a visiting restriction?", "HOSPITAL_POLICY"),
    ("What are the patient visitor rules?", "HOSPITAL_POLICY"),

    # ---------------- HR_POLICY ----------------
    ("What is the nurse leave policy?", "HR_POLICY"),
    ("What is the employee leave policy?", "HR_POLICY"),
    ("What are the staff attendance rules?", "HR_POLICY"),
    ("What is the hospital employee policy?", "HR_POLICY"),
    ("How does staff leave work?", "HR_POLICY"),
    ("What are the HR policies?", "HR_POLICY"),
    ("What is the employee attendance policy?", "HR_POLICY"),
    ("How many days of leave can staff take?", "HR_POLICY"),
    ("What is the nurse attendance policy?", "HR_POLICY"),
    ("What are the employee working hours?", "HR_POLICY"),
    ("What are the staff rules?", "HR_POLICY"),
    ("How can an employee apply for leave?", "HR_POLICY"),
    ("What is the staff leave procedure?", "HR_POLICY"),
    ("What are the hospital HR rules?", "HR_POLICY"),
    ("What is the employee holiday policy?", "HR_POLICY"),
    ("What are the staff benefits?", "HR_POLICY"),

    # ---------------- EMERGENCY ----------------
    ("What is the emergency number?", "EMERGENCY"),
    ("Where is the emergency room?", "EMERGENCY"),
    ("How can I call an ambulance?", "EMERGENCY"),
    ("Where is the ICU?", "EMERGENCY"),
    ("What should I do in a medical emergency?", "EMERGENCY"),
    ("Where is the blood bank?", "EMERGENCY"),
    ("How do I contact emergency services?", "EMERGENCY"),
    ("Where should I go during an emergency?", "EMERGENCY"),
    ("What is the emergency contact number?", "EMERGENCY"),
    ("How can I get an ambulance?", "EMERGENCY"),
    ("Where is the emergency department?", "EMERGENCY"),
    ("What should I do if a patient needs urgent care?", "EMERGENCY"),
    ("How can I contact the emergency department?", "EMERGENCY"),
    ("Where can I find emergency services?", "EMERGENCY"),
    ("What number should I call during an emergency?", "EMERGENCY"),
    ("Where is the hospital ICU located?", "EMERGENCY"),

    # ---------------- OUT_OF_SCOPE ----------------
    ("Tell me a joke", "OUT_OF_SCOPE"),
    ("What is the weather today?", "OUT_OF_SCOPE"),
    ("Who won the cricket match?", "OUT_OF_SCOPE"),
    ("Write a Python program", "OUT_OF_SCOPE"),
    ("What is the capital of India?", "OUT_OF_SCOPE"),
    ("Tell me a movie recommendation", "OUT_OF_SCOPE"),
    ("Who is the president of India?", "OUT_OF_SCOPE"),
    ("What is artificial intelligence?", "OUT_OF_SCOPE"),
    ("Tell me a story", "OUT_OF_SCOPE"),
    ("What is the latest news?", "OUT_OF_SCOPE"),
    ("Who is the best cricket player?", "OUT_OF_SCOPE"),
    ("Give me a recipe for pizza", "OUT_OF_SCOPE"),
    ("How do I learn Python?", "OUT_OF_SCOPE"),
    ("What is the temperature today?", "OUT_OF_SCOPE"),
    ("Tell me about football", "OUT_OF_SCOPE"),
    ("What is the stock market?", "OUT_OF_SCOPE")
]


# ============================================================
# 2. SEPARATE QUESTIONS AND LABELS
# ============================================================

queries = [item[0] for item in data]
labels = [item[1] for item in data]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    queries,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


# ============================================================
# 4. TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# 5. LOGISTIC REGRESSION MODEL
# ============================================================

classifier = LogisticRegression(
    max_iter=2000,
    C=5,
    random_state=42
)


classifier.fit(X_train_tfidf, y_train)


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

y_pred = classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)


print("\n" + "=" * 70)
print("        HOSPITAL RAG - QUERY CLASSIFICATION MODEL")
print("=" * 70)

print("\nModel: TF-IDF + Logistic Regression")

print("\nTotal Training Samples :", len(X_train))
print("Total Testing Samples  :", len(X_test))

print("\nModel Accuracy:", f"{accuracy * 100:.2f}%")


print("\nClassification Report:")
print("-" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 7. QUERY CLASSIFICATION FUNCTION
# ============================================================

def classify_query(query):

    query_vector = vectorizer.transform([query])

    probabilities = classifier.predict_proba(query_vector)[0]

    best_index = probabilities.argmax()

    category = classifier.classes_[best_index]

    confidence = probabilities[best_index]

    return category, confidence


# ============================================================
# 8. TOP 3 PREDICTIONS
# ============================================================

def get_top_predictions(query):

    query_vector = vectorizer.transform([query])

    probabilities = classifier.predict_proba(query_vector)[0]

    results = list(
        zip(
            classifier.classes_,
            probabilities
        )
    )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results[:3]


# ============================================================
# 9. DISPLAY RESULT
# ============================================================

def display_result(query):

    category, confidence = classify_query(query)

    print("\n" + "=" * 70)
    print("QUERY CLASSIFICATION RESULT")
    print("=" * 70)

    print("Query      :", query)
    print("Category   :", category)
    print("Confidence :", f"{confidence * 100:.2f}%")

    print("\nTop 3 Predictions:")

    predictions = get_top_predictions(query)

    for i, (label, probability) in enumerate(
        predictions,
        start=1
    ):

        print(
            f"{i}. {label:<25}"
            f"{probability * 100:.2f}%"
        )

    print("=" * 70)


# ============================================================
# 10. INTERACTIVE QUERY SYSTEM
# ============================================================

if __name__ == "__main__":

    print("\nAvailable Query Categories:")

    for category in sorted(classifier.classes_):
        print(" -", category)

    print("\nType 'exit' to stop.")

    while True:

        query = input("\nEnter your query: ")

        if query.lower().strip() == "exit":

            print("\nProgram stopped.")
            break

        if not query.strip():

            print("Please enter a valid query.")
            continue

        display_result(query)
from query_classifier import classify_query, get_top_predictions


# Test queries
test_queries = [
    "What are the visiting hours?",
    "What documents are required for admission?",
    "Where is the cardiology department?",
    "How can I book an appointment?",
    "How should I prepare for an MRI?",
    "What are the side effects of this medicine?",
    "What is the discharge process?",
    "What is the visitor policy?",
    "What is the nurse leave policy?",
    "What is the emergency contact number?",
    "Tell me a joke"
]


print("=" * 70)
print("       HOSPITAL RAG - QUERY CLASSIFICATION TEST")
print("=" * 70)


for query in test_queries:

    category, confidence = classify_query(query)

    print("\nQuery      :", query)
    print("Category   :", category)
    print("Confidence :", f"{confidence * 100:.2f}%")

    print("Top Predictions:")

    predictions = get_top_predictions(query)

    for i, (label, probability) in enumerate(predictions, start=1):

        print(
            f"  {i}. {label:<25} "
            f"{probability * 100:.2f}%"
        )


print("\n" + "=" * 70)
print("Testing completed successfully.")
print("=" * 70)
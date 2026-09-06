from vector_store import add_documents, search_documents
from reranker import Reranker


documents = [

    "The hospital provides treatment for diabetes using insulin therapy and oral medications.",

    "The hospital emergency department is open 24 hours a day.",

    "Patients with diabetes should monitor blood glucose regularly.",

    "The hospital cafeteria provides breakfast and lunch.",

    "Diabetes management includes healthy diet and regular physical activity.",

    "The hospital has parking facilities for patients and visitors.",

    "Insulin may be prescribed to patients who require blood glucose control.",

    "The hospital pharmacy provides prescribed medications.",

    "Regular exercise can help patients manage type 2 diabetes.",

    "The hospital visiting hours are from 9 AM to 6 PM."

]


# Add documents to ChromaDB
add_documents(documents)


# User question
query = "What treatments are available for diabetes?"


# Retrieve initial candidates
retrieved_documents = search_documents(
    query,
    top_k=10
)


print("\n========== CHROMADB RESULTS ==========\n")


for i, document in enumerate(
    retrieved_documents,
    start=1
):

    print(f"{i}. {document}")


# Create reranker
reranker = Reranker()


# Rerank retrieved documents
reranked_results = reranker.rerank(
    query=query,
    documents=retrieved_documents,
    top_k=5
)


print("\n========== RERANKED RESULTS ==========\n")


for rank, (document, score) in enumerate(
    reranked_results,
    start=1
):

    print(f"Rank {rank}")
    print(f"Score: {score}")
    print(f"Document: {document}")
    print("-" * 60)
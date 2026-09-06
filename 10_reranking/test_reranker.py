from reranker import Reranker


def main():

    query = "What are the treatments for diabetes?"

    documents = [

        "Hospital visiting hours are from 9 AM to 6 PM.",

        "Diabetes treatment includes insulin therapy and oral medications.",

        "The hospital emergency department is open 24 hours.",

        "Regular exercise and a healthy diet help manage diabetes.",

        "The hospital provides parking facilities for visitors."

    ]

    # Create reranker
    reranker = Reranker()

    # Rerank documents
    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=3
    )

    print("\nRERANKED RESULTS:\n")

    for rank, (document, score) in enumerate(results, start=1):

        print(f"Rank: {rank}")
        print(f"Document: {document}")
        print(f"Score: {score}")
        print("-" * 50)


if __name__ == "__main__":
    main()
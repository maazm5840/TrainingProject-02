from vector_store import search_documents
from reranker import Reranker


TOP_RETRIEVAL = 20

TOP_RERANK = 5


def main():

    print("=" * 70)

    print("RAG RETRIEVAL + RERANKING")

    print("=" * 70)


    query = input(
        "\nEnter your question: "
    ).strip()


    if not query:

        print(
            "Question cannot be empty."
        )

        return


    print(
        "\nSearching vector database..."
    )


    retrieved = search_documents(
        query,
        top_k=TOP_RETRIEVAL
    )


    if not retrieved:

        print(
            "\nNo documents found."
        )

        print(
            "Run ingest_documents.py first."
        )

        return


    print(
        f"Retrieved {len(retrieved)} candidates."
    )


    print(
        "\nLoading reranker..."
    )


    reranker = Reranker()


    print(
        "\nReranking candidates..."
    )


    reranked = reranker.rerank(
        query=query,
        documents=retrieved,
        top_k=TOP_RERANK
    )


    print("\n")

    print("=" * 70)

    print("FINAL RERANKED RESULTS")

    print("=" * 70)


    for rank, result in enumerate(
        reranked,
        start=1
    ):

        document = result[
            "document"
        ]

        score = result[
            "rerank_score"
        ]


        print(
            f"\nRANK {rank}"
        )

        print(
            f"RERANK SCORE: {score:.4f}"
        )


        print(
            f"ID: {document.get('id')}"
        )


        metadata = document.get(
            "metadata",
            {}
        )


        print(
            f"SOURCE: "
            f"{metadata.get('source', 'Unknown')}"
        )


        print(
            f"FILE TYPE: "
            f"{metadata.get('file_type', 'Unknown')}"
        )


        print(
            f"PAGE: "
            f"{metadata.get('page', 'Unknown')}"
        )


        print(
            f"CHUNK: "
            f"{metadata.get('chunk', 'Unknown')}"
        )


        print("\nTEXT:")

        print(
            document.get(
                "text",
                ""
            )
        )


        print(
            "-" * 70
        )


if __name__ == "__main__":

    main()
from vector_store import search_documents
from reranker import Reranker


# ============================================================
# EVALUATION DATASET
# ============================================================

TEST_CASES = [
    {
        "query": "What are the treatment options for cancer?",
        "keywords": [
            "cancer",
            "treatment",
            "therapy",
            "malignancy"
        ]
    },
    {
        "query": "How is cervical cancer diagnosed and treated?",
        "keywords": [
            "cervical",
            "cancer",
            "diagnosis",
            "treatment",
            "screening"
        ]
    },
    {
        "query": "What treatment is recommended for urinary tract cancer?",
        "keywords": [
            "urinary",
            "cancer",
            "treatment",
            "bladder",
            "kidney"
        ]
    },
    {
        "query": "How should diabetes be managed?",
        "keywords": [
            "diabetes",
            "blood glucose",
            "insulin",
            "diet",
            "exercise"
        ]
    },
    {
        "query": "What emergency services are available?",
        "keywords": [
            "emergency",
            "hospital",
            "services"
        ]
    },
    {
        "query": "What are the hospital visiting hours?",
        "keywords": [
            "visiting",
            "hours",
            "visitor"
        ]
    }
]


# ============================================================
# CHECK WHETHER A DOCUMENT IS RELEVANT
# ============================================================

def is_relevant(text, keywords):

    text = text.lower()

    matches = 0

    for keyword in keywords:

        if keyword.lower() in text:
            matches += 1

    # At least one important keyword must match
    return matches >= 1


# ============================================================
# PRECISION@K
# ============================================================

def calculate_precision(results, keywords):

    if not results:
        return 0.0

    relevant = 0

    for result in results:

        document = result.get("document", {})

        text = document.get("text", "")

        if is_relevant(text, keywords):
            relevant += 1

    return relevant / len(results)


# ============================================================
# RECIPROCAL RANK
# ============================================================

def calculate_reciprocal_rank(results, keywords):

    for rank, result in enumerate(results, start=1):

        document = result.get("document", {})

        text = document.get("text", "")

        if is_relevant(text, keywords):

            return 1 / rank

    return 0.0


# ============================================================
# EVALUATE ONE QUERY
# ============================================================

def evaluate_query(
    query,
    keywords,
    reranker,
    retrieval_k=20,
    rerank_k=5
):

    print("\n" + "=" * 70)

    print("QUERY:")
    print(query)

    print("=" * 70)


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    retrieved = search_documents(
        query,
        top_k=retrieval_k
    )

    if not retrieved:

        print("\nNo documents retrieved.")

        return {
            "retrieval_precision": 0.0,
            "rerank_precision": 0.0,
            "rerank_mrr": 0.0
        }


    # --------------------------------------------------------
    # CHECK RETRIEVED RESULTS
    # --------------------------------------------------------

    retrieved_relevant = 0

    for document in retrieved:

        text = document.get(
            "text",
            ""
        )

        if is_relevant(
            text,
            keywords
        ):

            retrieved_relevant += 1


    retrieval_precision = (
        retrieved_relevant /
        len(retrieved)
    )


    # --------------------------------------------------------
    # RERANK
    # --------------------------------------------------------

    reranked = reranker.rerank(
        query=query,
        documents=retrieved,
        top_k=rerank_k
    )


    # --------------------------------------------------------
    # RERANK PRECISION
    # --------------------------------------------------------

    rerank_precision = calculate_precision(
        reranked,
        keywords
    )


    # --------------------------------------------------------
    # MRR
    # --------------------------------------------------------

    reciprocal_rank = calculate_reciprocal_rank(
        reranked,
        keywords
    )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print(
        f"\nRetrieved candidates: "
        f"{len(retrieved)}"
    )

    print(
        f"Relevant before reranking: "
        f"{retrieved_relevant}"
    )

    print(
        f"Precision@{retrieval_k} before reranking: "
        f"{retrieval_precision:.2f}"
    )

    print(
        f"\nPrecision@{rerank_k} after reranking: "
        f"{rerank_precision:.2f}"
    )

    print(
        f"Reciprocal Rank: "
        f"{reciprocal_rank:.2f}"
    )


    print("\nTop reranked results:")

    for rank, result in enumerate(
        reranked,
        start=1
    ):

        document = result.get(
            "document",
            {}
        )

        score = result.get(
            "rerank_score",
            0
        )

        metadata = (
            document.get("metadata")
            or {}
        )

        print(
            f"\nRank {rank}"
        )

        print(
            f"Score: {score:.4f}"
        )

        print(
            f"Source: "
            f"{metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: "
            f"{metadata.get('page', 'Unknown')}"
        )

        text = document.get(
            "text",
            ""
        )

        print(
            f"Text: {text[:300]}..."
        )


    return {
        "retrieval_precision": retrieval_precision,
        "rerank_precision": rerank_precision,
        "rerank_mrr": reciprocal_rank
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("RERANKER EVALUATION")
    print("=" * 70)

    print(
        f"\nNumber of test queries: "
        f"{len(TEST_CASES)}"
    )


    # Load reranker only once

    print("\nLoading reranker...")

    reranker = Reranker()

    print("\nReranker ready.")


    results = []


    # --------------------------------------------------------
    # RUN ALL TEST CASES
    # --------------------------------------------------------

    for test_case in TEST_CASES:

        result = evaluate_query(
            query=test_case["query"],
            keywords=test_case["keywords"],
            reranker=reranker
        )

        results.append(result)


    # --------------------------------------------------------
    # CALCULATE AVERAGES
    # --------------------------------------------------------

    if results:

        avg_retrieval_precision = (
            sum(
                r["retrieval_precision"]
                for r in results
            )
            / len(results)
        )

        avg_rerank_precision = (
            sum(
                r["rerank_precision"]
                for r in results
            )
            / len(results)
        )

        avg_mrr = (
            sum(
                r["rerank_mrr"]
                for r in results
            )
            / len(results)
        )

    else:

        avg_retrieval_precision = 0
        avg_rerank_precision = 0
        avg_mrr = 0


    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n\n")

    print("=" * 70)
    print("FINAL EVALUATION REPORT")
    print("=" * 70)

    print(
        f"\nAverage Precision before reranking: "
        f"{avg_retrieval_precision:.2f}"
    )

    print(
        f"Average Precision after reranking:  "
        f"{avg_rerank_precision:.2f}"
    )

    print(
        f"Average MRR: "
        f"{avg_mrr:.2f}"
    )


    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    improvement = (
        avg_rerank_precision -
        avg_retrieval_precision
    )

    print(
        f"\nPrecision improvement: "
        f"{improvement:+.2f}"
    )


    if improvement > 0:

        print(
            "\nSUCCESS: Reranking improved "
            "retrieval relevance."
        )

    elif improvement == 0:

        print(
            "\nRESULT: No measurable improvement "
            "for this test dataset."
        )

    else:

        print(
            "\nWARNING: Reranking performed worse "
            "for this test dataset."
        )


    print("\n" + "=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
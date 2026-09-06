from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        print("\nLoading reranking model...")

        self.model = CrossEncoder(
            model_name
        )

        print(
            "Reranking model loaded successfully!"
        )


    def rerank(
        self,
        query,
        documents,
        top_k=5
    ):

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        if not documents:
            return []

        valid_documents = []

        texts = []

        for document in documents:

            if isinstance(
                document,
                dict
            ):

                text = document.get(
                    "text",
                    ""
                )

            else:

                text = str(document)

            if text and text.strip():

                valid_documents.append(
                    document
                )

                texts.append(text)

        if not texts:
            return []

        pairs = [
            (query, text)
            for text in texts
        ]

        scores = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False
        )

        results = []

        for document, score in zip(
            valid_documents,
            scores
        ):

            results.append({
                "document": document,
                "rerank_score": float(score)
            })

        results.sort(
            key=lambda x:
                x["rerank_score"],
            reverse=True
        )

        return results[:top_k]
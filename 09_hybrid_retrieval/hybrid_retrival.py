import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(
        self,
        documents,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        alpha=0.7
    ):
        """
        documents: list of dictionaries containing:
            {
                "id": "chunk_001",
                "text": "...",
                "metadata": {...}
            }

        alpha:
            Weight given to vector similarity.
            0.7 = 70% vector + 30% BM25
        """

        self.documents = documents
        self.alpha = alpha

        # Load Hugging Face embedding model
        print("Loading embedding model...")
        self.model = SentenceTransformer(embedding_model)

        # Extract document text
        self.texts = [doc["text"] for doc in documents]

        # -----------------------------
        # Create BM25 index
        # -----------------------------
        tokenized_documents = [
            text.lower().split()
            for text in self.texts
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

        # -----------------------------
        # Create vector embeddings
        # -----------------------------
        print("Creating document embeddings...")

        self.embeddings = self.model.encode(
            self.texts,
            normalize_embeddings=True
        )

        print("Hybrid Retriever Ready!")

    # =========================================================
    # VECTOR SEARCH
    # =========================================================

    def vector_search(self, query, top_k=5):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # Cosine similarity because embeddings are normalized
        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            results.append({
                "id": self.documents[index]["id"],
                "text": self.documents[index]["text"],
                "metadata": self.documents[index].get("metadata", {}),
                "score": float(scores[index])
            })

        return results

    # =========================================================
    # BM25 SEARCH
    # =========================================================

    def bm25_search(self, query, top_k=5):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            results.append({
                "id": self.documents[index]["id"],
                "text": self.documents[index]["text"],
                "metadata": self.documents[index].get("metadata", {}),
                "score": float(scores[index])
            })

        return results

    # =========================================================
    # RECIPROCAL RANK FUSION
    # =========================================================

    def reciprocal_rank_fusion(
        self,
        vector_results,
        bm25_results,
        k=60
    ):

        fused_scores = {}
        document_data = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):

            doc_id = result["id"]

            fused_scores[doc_id] = fused_scores.get(
                doc_id, 0
            ) + self.alpha * (
                1 / (k + rank)
            )

            document_data[doc_id] = result

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):

            doc_id = result["id"]

            fused_scores[doc_id] = fused_scores.get(
                doc_id, 0
            ) + (1 - self.alpha) * (
                1 / (k + rank)
            )

            document_data[doc_id] = result

        # Sort according to fused score
        sorted_documents = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        final_results = []

        for doc_id, score in sorted_documents:

            result = document_data[doc_id].copy()

            result["hybrid_score"] = score

            final_results.append(result)

        return final_results

    # =========================================================
    # COMPLETE HYBRID RETRIEVAL
    # =========================================================

    def retrieve(self, query, top_k=5):

        print("\nUser Query:")
        print(query)

        # Vector retrieval
        vector_results = self.vector_search(
            query,
            top_k=top_k
        )

        # BM25 retrieval
        bm25_results = self.bm25_search(
            query,
            top_k=top_k
        )

        # Combine results
        hybrid_results = self.reciprocal_rank_fusion(
            vector_results,
            bm25_results
        )

        return hybrid_results[:top_k]


# =============================================================
# SAMPLE HOSPITAL SOP DATA
# =============================================================

documents = [

    {
        "id": "SOP001_CHUNK001",
        "text": """
        Hand hygiene must be performed before and after
        patient contact. Healthcare workers should follow
        the hospital hand hygiene procedure.
        """,
        "metadata": {
            "department": "Infection Control",
            "sop": "Hand Hygiene SOP",
            "version": "2.0"
        }
    },

    {
        "id": "SOP002_CHUNK001",
        "text": """
        In case of accidental needle stick injury,
        immediately wash the affected area with soap
        and water and report the incident according
        to the occupational exposure procedure.
        """,
        "metadata": {
            "department": "Infection Control",
            "sop": "Needle Stick Injury SOP",
            "version": "1.5"
        }
    },

    {
        "id": "SOP003_CHUNK001",
        "text": """
        Emergency department staff must assess the patient,
        record vital signs and follow the emergency
        admission procedure.
        """,
        "metadata": {
            "department": "Emergency",
            "sop": "Emergency Admission SOP",
            "version": "3.0"
        }
    },

    {
        "id": "SOP004_CHUNK001",
        "text": """
        Personal protective equipment must be selected
        according to the infection control requirements

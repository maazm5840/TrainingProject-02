from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from reranker import Reranker


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RAG Reranking API",
    description="CrossEncoder-based document reranking service",
    version="1.0.0"
)


# ============================================================
# LOAD RERANKER ONCE
# ============================================================

print("\nLoading reranking model...")

reranker = Reranker()

print("Reranking API is ready!")


# ============================================================
# REQUEST MODEL
# ============================================================

class RerankRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        description="User's question"
    )

    documents: List[str] = Field(
        ...,
        min_length=1,
        description="Retrieved document chunks"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        description="Number of documents to return"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "message": "RAG Reranking API is running",
        "status": "healthy"
    }


# ============================================================
# RERANK ENDPOINT
# ============================================================

@app.post("/rerank")
def rerank_documents(request: RerankRequest):

    documents = request.documents

    top_k = min(
        request.top_k,
        len(documents)
    )

    results = reranker.rerank(
        query=request.query,
        documents=documents,
        top_k=top_k
    )

    response = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        response.append({
            "rank": rank,
            "score": result["rerank_score"],
            "text": result["document"]
        })

    return {
        "query": request.query,
        "total_documents_received": len(documents),
        "documents_returned": len(response),
        "results": response
    }
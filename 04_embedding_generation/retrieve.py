import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


INDEX_DIR = Path("index")

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 4


def load_database():

    index_file = INDEX_DIR / "sop.faiss"
    metadata_file = INDEX_DIR / "metadata.json"

    if not index_file.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run 'python ingest.py' first."
        )

    index = faiss.read_index(
        str(index_file)
    )

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as file:

        documents = json.load(file)

    model = SentenceTransformer(
        MODEL_NAME
    )

    return index, documents, model


def retrieve(question, top_k=TOP_K):

    index, documents, model = load_database()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(
        question_embedding,
        top_k
    )

    results = []

    for score, document_id in zip(
        scores[0],
        ids[0]
    ):

        if document_id >= 0:

            result = documents[document_id].copy()

            result["score"] = float(score)

            results.append(result)

    return results
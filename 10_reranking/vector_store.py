import hashlib

import chromadb
from chromadb.utils import embedding_functions


# ============================================================
# CHROMADB CONFIGURATION
# ============================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"


# ============================================================
# CREATE CHROMADB CLIENT
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_function = (
    embedding_functions
    .SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)


# ============================================================
# CREATE / LOAD COLLECTION
# ============================================================

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)


# ============================================================
# CREATE UNIQUE CHUNK ID
# ============================================================

def create_chunk_id(
    source,
    page,
    chunk_number,
    text
):

    raw_id = (
        f"{source}|"
        f"{page}|"
        f"{chunk_number}|"
        f"{text}"
    )

    hash_value = hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()

    return f"chunk_{hash_value}"


# ============================================================
# ADD CHUNKS TO CHROMADB
# ============================================================

def add_chunks(chunks):

    if not chunks:
        return 0

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:

        text = chunk.get("text", "").strip()

        if not text:
            continue

        metadata = chunk.get("metadata") or {}

        chunk_id = create_chunk_id(
            metadata.get(
                "source",
                "unknown"
            ),
            metadata.get(
                "page",
                None
            ),
            metadata.get(
                "chunk",
                0
            ),
            text
        )

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(metadata)

    if not documents:
        return 0

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    return len(documents)


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(
    query,
    top_k=20
):

    if not query or not query.strip():
        return []

    count = collection.count()

    if count == 0:
        return []

    n_results = min(
        top_k,
        count
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    ids = results.get(
        "ids",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    retrieved = []

    for index, document in enumerate(
        documents
    ):

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        metadata = metadata or {}

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        chunk_id = (
            ids[index]
            if index < len(ids)
            else None
        )

        retrieved.append({
            "id": chunk_id,
            "text": document,
            "metadata": metadata,
            "vector_distance": distance
        })

    return retrieved


# ============================================================
# DELETE DOCUMENT
# ============================================================

def delete_document(source):

    results = collection.get(
        where={
            "source": source
        }
    )

    ids = results.get(
        "ids",
        []
    )

    if ids:
        collection.delete(
            ids=ids
        )

    return len(ids)


# ============================================================
# LIST DOCUMENTS
# ============================================================

def list_documents():

    results = collection.get()

    metadatas = results.get(
        "metadatas",
        []
    )

    sources = set()

    for metadata in metadatas:

        metadata = metadata or {}

        source = metadata.get(
            "source"
        )

        if source:
            sources.add(source)

    return sorted(sources)


# ============================================================
# COLLECTION COUNT
# ============================================================

def collection_count():

    return collection.count()
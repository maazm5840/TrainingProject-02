from pathlib import Path

from document_loader import load_document
from chunker import chunk_text
from reranker import Reranker

import chromadb
from chromadb.utils import embedding_functions


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_FOLDER = "./documents"

TOP_RETRIEVAL = 20
TOP_RERANK = 5


# ============================================================
# CHROMADB SETUP
# ============================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

embedding_function = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

collection = client.get_or_create_collection(
    name="real_documents",
    embedding_function=embedding_function
)


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_all_documents():

    all_chunks = []

    document_folder = Path(DOCUMENT_FOLDER)

    files = list(document_folder.glob("*.pdf"))

    files += list(document_folder.glob("*.docx"))

    if not files:

        print("No PDF or DOCX files found.")

        return []

    print("\nDocuments found:")

    for file in files:

        print(f" - {file.name}")

        pages = load_document(str(file))

        for page in pages:

            text = page["text"]

            metadata = page["metadata"]

            chunks = chunk_text(
                text,
                chunk_size=500,
                overlap=100
            )

            for chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):

                all_chunks.append({

                    "text": chunk,

                    "metadata": {
                        **metadata,
                        "chunk": chunk_number
                    }

                })

    return all_chunks


# ============================================================
# STORE CHUNKS IN CHROMADB
# ============================================================

def store_chunks(chunks):

    if not chunks:

        return

    ids = []

    texts = []

    metadatas = []

    for index, chunk in enumerate(chunks):

        ids.append(
            f"chunk_{index}"
        )

        texts.append(
            chunk["text"]
        )

        metadatas.append(
            chunk["metadata"]
        )

    collection.add(

        ids=ids,

        documents=texts,

        metadatas=metadatas

    )

    print(
        f"\nStored {len(chunks)} chunks in ChromaDB."
    )


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(query):

    results = collection.query(

        query_texts=[query],

        n_results=TOP_RETRIEVAL

    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    retrieved = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        retrieved.append({

            "text": document,

            "metadata": metadata

        })

    return retrieved


# ============================================================
# RERANK DOCUMENTS
# ============================================================

def rerank_documents(
    query,
    retrieved_documents
):

    reranker = Reranker()

    results = reranker.rerank(

        query=query,

        documents=retrieved_documents,

        top_k=TOP_RERANK

    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print("REAL DOCUMENT RAG + RERANKING TEST")

    print("=" * 70)


    # --------------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------------

    chunks = load_all_documents()

    if not chunks:

        print(
            "\nPlease put a PDF or DOCX file inside:"
        )

        print(
            "documents/"
        )

        return


    print(
        f"\nTotal chunks created: {len(chunks)}"
    )


    # --------------------------------------------------------
    # 2. Store chunks
    # --------------------------------------------------------

    store_chunks(chunks)


    # --------------------------------------------------------
    # 3. Get user query
    # --------------------------------------------------------

    query = input(
        "\nEnter your question: "
    )


    if not query.strip():

        print(
            "Query cannot be empty."
        )

        return


    # --------------------------------------------------------
    # 4. Retrieve top 20
    # --------------------------------------------------------

    print(
        "\nRetrieving relevant chunks..."
    )

    retrieved = retrieve_documents(
        query
    )


    print(
        f"Retrieved {len(retrieved)} chunks."
    )


    # --------------------------------------------------------
    # 5. Rerank
    # --------------------------------------------------------

    print(
        "\nReranking retrieved chunks..."
    )

    reranked = rerank_documents(

        query,

        retrieved

    )


    # --------------------------------------------------------
    # 6. Display final results
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)

    print("FINAL RERANKED RESULTS")

    print("=" * 70)


    for rank, result in enumerate(

        reranked,

        start=1

    ):

        document = result["document"]

        score = result["rerank_score"]


        if isinstance(document, dict):

            text = document["text"]

            metadata = document["metadata"]

        else:

            text = str(document)

            metadata = {}


        print(
            f"\nRANK: {rank}"
        )

        print(
            f"RERANK SCORE: {score:.4f}"
        )

        print(
            f"SOURCE: {metadata.get('source', 'Unknown')}"
        )

        print(
            f"PAGE: {metadata.get('page', 'Unknown')}"
        )

        print(
            f"CHUNK: {metadata.get('chunk', 'Unknown')}"
        )

        print("\nTEXT:")

        print(text)

        print("-" * 70)


if __name__ == "__main__":

    main()
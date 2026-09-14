from pathlib import Path
import json
import re

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data")
INDEX_DIR = Path("index")

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 40


def extract_text(pdf_path):
    reader = PdfReader(str(pdf_path))

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(text):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):

        end = min(start + CHUNK_SIZE, len(words))

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():

    INDEX_DIR.mkdir(exist_ok=True)

    documents = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the data folder.")
        return

    print("Reading SOP documents...\n")

    for pdf_file in pdf_files:

        print(f"Processing: {pdf_file.name}")

        text = extract_text(pdf_file)

        text = clean_text(text)

        chunks = create_chunks(text)

        for number, chunk in enumerate(chunks):

            documents.append({
                "source": pdf_file.name,
                "chunk": number,
                "text": chunk
            })

    print("\nGenerating embeddings...")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        [document["text"] for document in documents],
        normalize_embeddings=True
    ).astype("float32")

    print("Creating FAISS vector database...")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    faiss.write_index(
        index,
        str(INDEX_DIR / "sop.faiss")
    )

    with open(
        INDEX_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n--------------------------------")
    print("RAG indexing completed successfully!")
    print("--------------------------------")

    print(f"Documents: {len(pdf_files)}")
    print(f"Chunks: {len(documents)}")

    print("\nCreated:")
    print("index/sop.faiss")
    print("index/metadata.json")


if __name__ == "__main__":
    main()
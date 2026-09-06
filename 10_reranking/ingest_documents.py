from pathlib import Path

from document_loader import load_document
from chunker import chunk_text
from vector_store import add_chunks


DOCUMENT_FOLDER = "./documents"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def ingest_file(file_path):

    print("\n" + "=" * 70)
    print(f"Processing: {file_path.name}")
    print("=" * 70)

    pages = load_document(str(file_path))

    if not pages:
        print("No readable text found.")
        return 0

    all_chunks = []

    for page in pages:

        text = page["text"]
        metadata = page["metadata"]

        chunks = chunk_text(
            text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            chunk_metadata = {
                **metadata,
                "chunk": chunk_number,
                "document_name": metadata.get(
                    "source",
                    "unknown"
                )
            }

            all_chunks.append({
                "text": chunk,
                "metadata": chunk_metadata
            })

    stored = add_chunks(all_chunks)

    print(f"Pages processed: {len(pages)}")
    print(f"Chunks generated: {len(all_chunks)}")
    print(f"Chunks stored/updated: {stored}")

    return stored


def ingest_all_documents():

    folder = Path(DOCUMENT_FOLDER)

    if not folder.exists():

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        print(f"Created folder: {folder}")
        return

    files = []

    files.extend(folder.glob("*.pdf"))
    files.extend(folder.glob("*.docx"))

    if not files:

        print("\nNo PDF or DOCX files found.")

        print("Put your files inside:")
        print("documents/")

        return

    print(f"\nFound {len(files)} document(s).")

    total_chunks = 0

    for file_path in files:

        total_chunks += ingest_file(
            file_path
        )

    print("\n" + "=" * 70)
    print(
        f"TOTAL CHUNKS PROCESSED: {total_chunks}"
    )
    print("=" * 70)


if __name__ == "__main__":
    ingest_all_documents()
import re


def clean_chunk(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def chunk_text(
    text,
    chunk_size=300,
    overlap=50
):

    text = clean_chunk(text)

    if not text:
        return []

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        )

        if chunk:

            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks
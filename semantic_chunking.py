import re
import nltk
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Download NLTK sentence tokenizer
nltk.download("punkt", quiet=True)


# --------------------------------------------------
# 1. LOAD HOSPITAL SOP DOCUMENT
# --------------------------------------------------

with open("hospital_sop.txt", "r", encoding="utf-8") as file:
    sop_text = file.read()


# --------------------------------------------------
# 2. PREPROCESS THE TEXT
# --------------------------------------------------

# Remove extra spaces
sop_text = re.sub(r"\s+", " ", sop_text).strip()


# --------------------------------------------------
# 3. SPLIT TEXT INTO SENTENCES
# --------------------------------------------------

sentences = nltk.sent_tokenize(sop_text)


print("\n========== SOP SENTENCES ==========\n")

for i, sentence in enumerate(sentences):
    print(f"{i + 1}. {sentence}")


# --------------------------------------------------
# 4. LOAD SEMANTIC EMBEDDING MODEL
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 5. GENERATE SEMANTIC EMBEDDINGS
# --------------------------------------------------

embeddings = model.encode(sentences)


# --------------------------------------------------
# 6. CALCULATE SEMANTIC SIMILARITY
# --------------------------------------------------

similarities = []

print("\n========== SEMANTIC SIMILARITY ==========\n")

for i in range(len(sentences) - 1):

    similarity = cosine_similarity(
        [embeddings[i]],
        [embeddings[i + 1]]
    )[0][0]

    similarities.append(similarity)

    print(
        f"Sentence {i + 1} <-> Sentence {i + 2} "
        f"= {similarity:.2f}"
    )


# --------------------------------------------------
# 7. IDENTIFY TOPIC CHANGES
# --------------------------------------------------

# Similarity threshold
threshold = 0.45


# --------------------------------------------------
# 8. CREATE MEANINGFUL SEMANTIC CHUNKS
# --------------------------------------------------

chunks = []

current_chunk = [sentences[0]]

for i in range(len(similarities)):

    similarity = similarities[i]

    if similarity >= threshold:

        # Similar meaning
        current_chunk.append(sentences[i + 1])

    else:

        # Topic change detected
        chunks.append(" ".join(current_chunk))

        current_chunk = [sentences[i + 1]]


# Add the last chunk
if current_chunk:
    chunks.append(" ".join(current_chunk))


# --------------------------------------------------
# 9. DISPLAY FINAL CHUNKS
# --------------------------------------------------

print("\n========== MEANINGFUL SEMANTIC CHUNKS ==========\n")

for i, chunk in enumerate(chunks):

    print(f"Chunk {i + 1}")
    print("-" * 50)
    print(chunk)
    print()
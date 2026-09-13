# Module 9: Hybrid Retrieval and Deployment

This module implements a **hybrid RAG pipeline** that combines semantic (vector-based) and lexical (BM25) retrieval, wrapped in a cross-platform Docker deployment.  We will integrate a Hugging Face model for answer generation, and containerize the system so it runs identically on Windows, macOS, and Linux. The objectives are:

- **Combine BM25 and vector search**: use exact-match keyword search (BM25) alongside embedding-based semantic search to get both precise and contextual results.
- **Choose Hugging Face models**: select open-source transformer models for embeddings and answer generation (e.g. Sentence-Transformer for embedding, LLM for answering).
- **Data preprocessing and indexing**: prepare the knowledge base by chunking, embedding, and indexing into both a BM25 index and a vector database.
- **Containerize the pipeline**: write a Dockerfile and use multi-architecture builds so the service runs uniformly across all OS environments.
- **Testing, evaluation, monitoring**: define metrics and tests for retrieval accuracy and model performance, and set up basic logging/monitoring.

## Hybrid Retrieval Architecture

A **hybrid retriever** queries both a BM25 (lexical) index and a vector store (semantic) and merges the results.  In a typical RAG pipeline, the system works as follows:

1. **Query processing**: The user’s question is processed (e.g. lowercased, tokenized) for BM25 and *also* embedded into a dense vector via a pre-trained model (e.g. a Sentence-Transformer or CLIP model).
2. **BM25 retrieval**: The query’s tokens are matched against the BM25 index (an inverted-index engine like Elasticsearch, OpenSearch, or a library like [rank-bm25] or the new [BM25S] parser).  BM25 ranks documents by term frequency/rarity, returning the top **exact-match** passages. This is fast and yields precise keyword hits, especially for well-structured queries.
3. **Vector search**: Concurrently, the query vector is used to find nearest neighbors in a vector database (Chroma, FAISS, Pinecone, etc.) that stores pre-computed embeddings of each document chunk.  These dense embeddings capture semantic meaning, so vector search finds contextually relevant passages even if they don’t contain the exact query words.
4. **Ensemble merging**: The top-K results from BM25 and vector search are combined (e.g. via a simple merge or LangChain’s `EnsembleRetriever`).  One can weight them equally or prioritize one over the other.  This hybrid ensures that exact keyword matches (BM25) and semantic matches (vectors) are both considered.
5. **Answer generation**: The merged passages are concatenated into a context block and fed into a Hugging Face language model (e.g. a GPT or LLAMA variant).  This RAG model generates the answer conditioned on the retrieved context.  By grounding on actual documents, the answers are more factual and updatable (you can change the index without retraining the model).

This hybrid approach “combines the sharp precision of keyword search and the deep contextual understanding of semantic search” to improve retrieval accuracy.  As noted in RAG literature, using both methods helps catch cases where one method alone would fail (e.g. a BM25 keyword misses semantic synonyms, or a vector search misses an exact key). 

## BM25 (Lexical Retrieval)

- **What is BM25?**  BM25 (Best-Match 25) is a classic term-frequency-based ranking algorithm used in search engines.  It scores documents by matching query words, normalizing for term frequency and document length.  For example, `rank-bm25` is a popular Python library implementing Okapi BM25.
- **Implementations:**  We can use a ready engine like Elasticsearch/OpenSearch (which uses Lucene under the hood) or an in-memory solution:
  - *Elasticsearch/OpenSearch:* indexes text and supports BM25 out of the box. LangChain and other frameworks can connect via `elasticsearch` client.
  - *Python libraries:* `rank_bm25` (pip-installable) is simple but single-threaded. Newer libraries like [BM25S] offer much faster search in Python while staying in the local process.
  - *LangChain Retriever:* The [LangChain BM25Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/bm25) uses either internal or Elasticsearch BM25 to run `.similarity_search(query)`.
- **Strengths:** BM25 excels when the user query uses the same terms as the document (exact keywords).  It’s deterministic, fast, and easy to debug (you see exactly which passages matched).
- **Weaknesses:** It does **not** handle paraphrases or semantic queries.  If the query words are different from the document wording, BM25 might retrieve poorly or nothing.  In such cases, the LM might hallucinate because the necessary evidence wasn’t retrieved.

## Semantic (Vector) Retrieval

- **Embeddings:** We use a pre-trained Transformer encoder (from Hugging Face) to turn text into dense vectors.  Popular choices include Sentence-Transformers (e.g. `all-MiniLM`), **E5**, **GTE**, **BGE** family models, or even CLIP-style models for images/text.  These models map semantically similar texts to nearby points in vector space.
- **Vector Database:** The vectors are indexed in a nearest-neighbor search engine. Examples:
  - *Chroma, FAISS:* open-source, easy to integrate (LanChain has built-ins). Good for prototyping.
  - *Weaviate, Milvus:* scalable open-source solutions with more features.
  - *Pinecone, Qdrant:* managed cloud services with high performance.  
  DataCamp notes that in 2026 the top vector DBs include Chroma, Pinecone, Weaviate, FAISS, Qdrant, Milvus, and pgvector.  For a module prototype, Chroma or FAISS are popular free choices.
- **Indexing:** Each document chunk (see next section) is fed through the encoder to get a fixed-size embedding. These embeddings (often 384–1024 dimensions) are stored in the vector store. The store also supports querying by returning the nearest neighbors (by cosine or Euclidean distance).
- **Multimodal Note:** For a *multimodal* system, we can index both text and images.  For example, we might embed images using CLIP or OpenCLIP and index those alongside text embeddings.  CLIP provides joint image-text embeddings useful for cross-modal retrieval.

## Data Preprocessing & Indexing

Before running retrieval, we must prepare the document corpus:

- **Collect Documents:** Gather all relevant text (and images, if any). This could be PDFs, markdown, database records, etc.
- **Chunking:** Split large documents into smaller passages (e.g. ~500 tokens with some overlap) to preserve context.  Tools like LangChain’s `RecursiveCharacterTextSplitter` or Haystack’s splitters can do this efficiently. Chunking ensures granular retrieval.
- **Embedding Generation:** For each chunk, generate an embedding vector:
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('all-MiniLM-L6-v2')
  vectors = [model.encode(chunk) for chunk in chunks]
  ```
  This uses a Hugging Face model to create a numerical representation.
- **Indexing in Vector Store:** Insert each chunk’s embedding into the chosen vector DB.  Also store metadata (e.g. source doc ID, chunk index) along with the vector.
- **Indexing in BM25 Store:** Index the same chunks in a BM25 engine. If using Elasticsearch, send the chunk texts via the API to index. If using `rank-bm25`, keep the list of chunk texts in memory and let the library preprocess it (it builds internal term-frequency tables).
- **Retrieval Pipeline:** Once indexed, set up the query pipeline.  Given a query:
  1. Use the BM25 retriever to get top-N chunks by keyword.
  2. Embed the query and run vector search for top-N semantic matches.
  3. Merge the results, possibly using LangChain’s `EnsembleRetriever(BM25Retriever, VectorRetriever)`.

**Example (BM25 with Elasticsearch):**  
LangChain’s `ElasticsearchStore` and `BM25Strategy` can be used:
```python
from langchain.retrievers import BM25Retriever
from langchain.vectorstores.elasticsearch_store import ElasticsearchStore
bm25 = BM25Strategy(k1=1.2, b=0.75)
store = ElasticsearchStore(es_connection=es_client, index_name="docs", strategy=bm25)
results = store.similarity_search(query, k=5)
```  
This runs BM25 in Elasticsearch and returns matching chunks.

## Implementation Steps

1. **Select Models and Libraries:**  
   - Choose an embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2`).  Install Transformers or Sentence-Transformers.  
   - Choose a retriever: e.g. LangChain with Chroma or Faiss for vectors, plus an Elasticsearch or rank-bm25 for BM25.
   - Choose an answer generation model: e.g. Hugging Face’s GPT-NeoX, Llama 2/3, or any suitable LLM.
2. **Prepare Environment:**  
   - Create a Python project. Use virtualenv or Conda. Install dependencies (`transformers`, `sentence-transformers`, `langchain`, `chromadb` or `elasticsearch`, `fastapi`/`flask`, etc.).
   - (Optional) Download or fine-tune any necessary model checkpoints.
3. **Build Index:**  
   - Split documents and generate embeddings as above.  
   - Load embeddings into Chroma/FAISS: e.g. using `chromadb.Client()` to create a collection and upsert documents + embeddings.  
   - Index text for BM25: e.g., run `es.index(...)` for each chunk into Elasticsearch.
4. **Implement Retriever:**  
   - Code a function that takes `query` and runs both retrievals. For instance, with LangChain:
     ```python
     bm25_ret = BM25Retriever(index=es_index)
     vector_ret = ChromaRetriever(collection=my_chroma)
     ensemble = EnsembleRetriever([bm25_ret, vector_ret], weights=[0.5, 0.5])
     contexts = ensemble.get_relevant_documents(query)
     ```
   - Tune weights or ranks as needed.
5. **Answer Generation:**  
   - Given the retrieved contexts, format a prompt for the HF model. For example:
     ```
     "Use the following context to answer the question:\nContext: {combined_text}\nQuestion: {query}\nAnswer:"
     ```  
   - Tokenize and feed this to the model (via `pipeline("text-generation")` or `AutoModelForCausalLM.generate`).
   - Return the model’s answer (decode the tokens).
6. **Wrap in an API:**  
   - Optionally create a FastAPI or Flask app that accepts queries and returns answers.  This app will call the retrieval + generation pipeline.  
7. **Testing Locally:**  
   - Test with example questions to ensure retrieval and answering works. Print out which documents were retrieved to verify correctness.

## Containerization and Deployment

To ensure OS independence, we Dockerize the pipeline:

- **Write a Dockerfile:** Use an official Python base image (e.g. `python:3.10-slim`) and install requirements. Example:
  ```dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  CMD ["python", "app.py"]
  ```
  This structure copies code and installs dependencies.
- **Include Models:** If using local model files, copy them into the image (or download them at build time).  Ensure `transformers` can find the model (set `TRANSFORMERS_CACHE` or use `HF_HOME`).
- **Expose ports:** If using an API server, `EXPOSE 8000` or relevant port.
- **Build Multi-arch Image:** Use Docker Buildx to target multiple architectures. For example:
  ```
  docker buildx build --platform linux/amd64,linux/arm64 -t myuser/hybrid-rag:latest .
  ```  
  This creates a multi-platform image that runs on x86 and ARM hosts.
- **Run the Container:** On any OS (Windows with Docker Desktop, macOS, Linux), simply:
  ```
  docker run -p 8000:8000 myuser/hybrid-rag:latest
  ```
  Docker will select the appropriate architecture variant at runtime.
- **Volumes and Config:** If large indices or models shouldn’t be baked into the image, mount volumes or use initialization scripts to download/index at start.

This containerization ensures the same code works across Windows, macOS, and Linux without modification. The Docker documentation emphasizes that multi-platform images let one image run on different architectures seamlessly.

## Testing, Evaluation, and Monitoring

- **Functional Testing:** Verify the system end-to-end with test queries:
  - Check that BM25 returns expected exact matches (e.g. see retrieved chunks).
  - Check that vector search handles synonyms (different phrasing questions).
  - Ensure the merged result context leads to correct answers from the model.
- **Evaluation Metrics:** For retrieval, use metrics like recall@K (does at least one relevant doc appear in top-K), or precision@K.  For generation, simple metrics include BLEU/ROUGE against known answers, but human or heuristic evaluation is common. Pinecone’s guide suggests measuring retrieval accuracy (precision/recall) before trusting the LLM.
- **Logging:** Log each query, the top BM25 results, top vector results, and final answer. This helps diagnose failures (e.g. if the answer hallucinates, check retrieved context).
- **Performance Testing:** Measure latency of retrieval (BM25 vs vector) and model inference. Optimize embedding dimension, index parameters, or model size as needed.
- **Monitoring:** In production, monitor the Docker container (CPU, memory). Tools like Prometheus or simple Docker stats can alert if the service is down or slow.

## References

- RAG definition and workflow.  
- BM25 lexical search and usage.  
- Hugging Face embedding models and uses.  
- Vector database options and RAG applications.  
- Hybrid (BM25 + vector) retrieval for RAG.  
- Dockerfile example for ML service.  
- Multi-platform Docker builds for cross-OS deployment.
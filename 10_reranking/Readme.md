# RAG Document Reranking Module

A production-oriented **document reranking module for Retrieval-Augmented Generation (RAG)** using **ChromaDB**, **Sentence Transformers**, and a **CrossEncoder** model.

The purpose of this module is to improve the relevance of retrieved documents before they are passed to the Large Language Model (LLM).

---

# 1. Project Overview

In a RAG system, a vector database retrieves documents that are semantically similar to a user's question.

However, vector similarity does not always place the most relevant document first.

For example, suppose the user asks:

> What are the treatment options for cancer?

The vector database may retrieve:

```text
1. Hospital visiting hours
2. Cancer treatment options
3. Hospital emergency services
4. Cancer referral guidelines
5. Hospital parking information
```

The relevant cancer information may not always be ranked first.

The **Reranking Module** solves this problem.

It takes the initially retrieved documents and performs a more detailed **query-document relevance comparison** using a CrossEncoder model.

```text
Initial Retrieval
       ↓
Top 20 Candidate Chunks
       ↓
CrossEncoder Reranker
       ↓
Relevance Scores
       ↓
Sort by Score
       ↓
Top 5 Relevant Chunks
       ↓
LLM
       ↓
Final Answer
```

---

# 2. Main Objective

The objective of this module is:

> To improve the quality of retrieved context by reranking candidate document chunks according to their relevance to the user's query.

The module is designed to work with:

* PDF documents
* DOCX documents
* Multiple documents
* Hospital guidelines
* Medical/hospital information documents
* Other text-based documents supported by the ingestion pipeline

---

# 3. Technology Stack

| Technology               | Purpose                          |
| ------------------------ | -------------------------------- |
| Python                   | Main programming language        |
| ChromaDB                 | Vector database                  |
| Sentence Transformers    | Embeddings and reranking         |
| CrossEncoder             | Query-document relevance scoring |
| `all-MiniLM-L6-v2`       | Embedding model                  |
| `ms-marco-MiniLM-L-6-v2` | Reranking model                  |
| pypdf                    | PDF text extraction              |
| python-docx              | DOCX text extraction             |
| FastAPI                  | Reranking API                    |
| Uvicorn                  | API server                       |
| Requests                 | API testing                      |

---

# 4. System Architecture

## 4.1 Complete RAG Architecture

The reranking module is one component inside the complete RAG architecture.

```text
                         USER
                          │
                          ▼
                    User Question
                          │
                          ▼
                  Document Retrieval
                          │
                          ▼
              ┌───────────────────────┐
              │      ChromaDB         │
              │                       │
              │ Vector Similarity     │
              │ Search                │
              └───────────┬───────────┘
                          │
                          ▼
                Top 10 / Top 20 Chunks
                          │
                          ▼
              ┌───────────────────────┐
              │    RERANKING MODULE   │
              │                       │
              │     CrossEncoder      │
              │                       │
              │ Query + Document      │
              │       ↓               │
              │ Relevance Score       │
              │       ↓               │
              │ Sort Documents        │
              └───────────┬───────────┘
                          │
                          ▼
                    Top 5 Chunks
                          │
                          ▼
                         LLM
                          │
                          ▼
                   Final Answer
```

---

# 5. Internal Architecture of This Module

The module itself contains several supporting components.

```text
                         DOCUMENTS
                            │
                            ▼
                  ┌──────────────────┐
                  │ Document Loader  │
                  └────────┬─────────┘
                           │
                    PDF / DOCX Text
                           │
                           ▼
                  ┌──────────────────┐
                  │     Chunker      │
                  └────────┬─────────┘
                           │
                     Text Chunks
                           │
                           ▼
                  ┌──────────────────┐
                  │    ChromaDB      │
                  │  Vector Storage  │
                  └────────┬─────────┘
                           │
                           │ Query
                           ▼
                    Candidate Chunks
                           │
                           ▼
                  ┌──────────────────┐
                  │    Reranker      │
                  │   CrossEncoder   │
                  └────────┬─────────┘
                           │
                    Relevance Scores
                           │
                           ▼
                  Top Ranked Chunks
                           │
                           ▼
                    Return to RAG
```

---

# 6. Project Folder Structure

The complete project structure is:

```text
reranking_module/
│
├── documents/
│   ├── hospital_guidelines.pdf
│   └── another_document.docx
│
├── chroma_db/
│   └── ChromaDB persistent storage
│
├── reranker.py
│
├── document_loader.py
│
├── chunker.py
│
├── vector_store.py
│
├── ingest_documents.py
│
├── search_and_rerank.py
│
├── evaluation.py
│
├── api.py
│
├── test_api.py
│
├── requirements.txt
│
└── README.md
```

---

# 7. Description of Each File

## `reranker.py`

This is the **core module**.

It loads the CrossEncoder model and reranks documents.

Main responsibility:

```text
Query + Documents
       ↓
CrossEncoder
       ↓
Relevance Scores
       ↓
Sorted Documents
       ↓
Top K
```

The model used is:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## `document_loader.py`

Responsible for extracting text from documents.

Supported formats:

```text
PDF
DOCX
```

Workflow:

```text
PDF/DOCX
   ↓
Document Loader
   ↓
Extracted Text
```

---

## `chunker.py`

Large documents are divided into smaller pieces called **chunks**.

Current configuration:

```text
Chunk size = 300 words
Overlap = 50 words
```

Example:

```text
Document
   ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
...
```

The overlap helps preserve context between neighboring chunks.

---

## `vector_store.py`

Responsible for storing and searching document chunks in ChromaDB.

It performs:

```text
Text Chunks
     ↓
Embeddings
     ↓
ChromaDB
```

During search:

```text
User Query
     ↓
Embedding
     ↓
Vector Similarity Search
     ↓
Top Candidate Chunks
```

---

## `ingest_documents.py`

This script processes documents placed inside the `documents/` directory.

It performs:

```text
Document
   ↓
Load
   ↓
Extract Text
   ↓
Chunk
   ↓
Generate Embeddings
   ↓
Store in ChromaDB
```

---

## `search_and_rerank.py`

This is the main local testing pipeline.

It performs:

```text
User Query
     ↓
ChromaDB Retrieval
     ↓
Top 20 Chunks
     ↓
CrossEncoder Reranking
     ↓
Top 5 Chunks
```

---

## `evaluation.py`

Used to evaluate whether reranking improves retrieval quality.

It compares:

```text
Vector Retrieval
       VS
Vector Retrieval + Reranking
```

Metrics include:

* Precision@20
* Precision@5
* Mean Reciprocal Rank (MRR)

---

## `api.py`

Creates the FastAPI service.

Main endpoint:

```text
POST /rerank
```

The API accepts:

```text
Query
+
Retrieved Documents
+
Top K
```

and returns:

```text
Rank
+
Score
+
Document
```

---

## `test_api.py`

Used to test the FastAPI endpoint automatically.

A successful request should return:

```text
HTTP 200
```

---

# 8. Installation

## Step 1 — Open the project

Open PowerShell or Git Bash.

Navigate to:

```powershell
cd C:\Users\poornima\Desktop\reranking_module
```

---

# 9. Create Virtual Environment

Recommended:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\activate
```

After activation, the terminal should show something similar to:

```text
(venv)
```

---

# 10. Install Dependencies

Create/use:

```text
requirements.txt
```

with:

```text
sentence-transformers
torch
numpy
chromadb
pypdf
python-docx
fastapi
uvicorn
requests
```

Install everything:

```powershell
pip install -r requirements.txt
```

---

# 11. Verify Installation

Check Python:

```powershell
python --version
```

Check Sentence Transformers:

```powershell
python -c "import sentence_transformers; print('sentence-transformers OK')"
```

Check ChromaDB:

```powershell
python -c "import chromadb; print('chromadb OK')"
```

Check FastAPI:

```powershell
python -c "import fastapi; print('fastapi OK')"
```

Check Uvicorn:

```powershell
python -c "import uvicorn; print('uvicorn OK')"
```

---

# 12. Add Documents

Put your documents inside:

```text
documents/
```

Example:

```text
documents/
├── hospital_guidelines.pdf
├── hospital_policy.pdf
└── medical_guidelines.docx
```

The current ingestion pipeline supports:

```text
.pdf
.docx
```

---

# 13. Ingest Documents

Run:

```powershell
python ingest_documents.py
```

The system performs:

```text
Find Documents
       ↓
Load PDF/DOCX
       ↓
Extract Text
       ↓
Create Chunks
       ↓
Generate Embeddings
       ↓
Store in ChromaDB
```

Example output:

```text
Found 2 document(s).

======================================================================
Processing: hospital_guidelines.pdf
======================================================================

Pages processed: 120
Chunks generated: 149
Chunks stored/updated: 149

======================================================================
TOTAL CHUNKS PROCESSED: 149
======================================================================
```

---

# 14. Test Retrieval + Reranking

Run:

```powershell
python search_and_rerank.py
```

Example query:

```text
What are the treatment options for cancer?
```

The system performs:

```text
Query
 ↓
ChromaDB
 ↓
Retrieve Top 20
 ↓
CrossEncoder
 ↓
Score 20 Documents
 ↓
Sort Descending
 ↓
Return Top 5
```

Example:

```text
Rank 1
Score: -4.9930
Page: 35
Cancer treatment/referral information

Rank 2
Score: -7.1487
Page: 28
Cancer diagnosis and treatment information

Rank 3
Score: -7.3417
Page: 33
Cancer-related treatment information
```

---

# 15. Understanding Reranking Scores

The CrossEncoder produces a relevance score for every query-document pair.

For example:

```text
Document A → 8.72
Document B → 3.15
Document C → -2.41
Document D → -7.82
```

The system sorts them:

```text
8.72
 ↓
3.15
 ↓
-2.41
 ↓
-7.82
```

Therefore:

```text
Higher score = More relevant
Lower score = Less relevant
```

Important:

The raw scores are model logits and are **not percentages**.

For example:

```text
-4.99
```

does not mean:

```text
-4.99% relevance
```

Only the relative ranking is important.

---

# 16. Run Evaluation

Run:

```powershell
python evaluation.py
```

The evaluation compares:

```text
Original Vector Retrieval
             VS
Reranked Retrieval
```

The test queries include topics such as:

```text
Cancer treatment
Cervical cancer
Urinary tract cancer
Diabetes management
Emergency services
Visiting hours
```

The evaluation calculates:

```text
Precision@20
Precision@5
MRR
```

This demonstrates whether reranking improves the quality of the retrieved context.

---

# 17. Start the Reranking API

Run:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Expected output:

```text
Uvicorn running on http://127.0.0.1:8000
```

Keep this terminal running.

---

# 18. Open FastAPI Documentation

Open your browser:

```text
http://127.0.0.1:8000/docs
```

FastAPI provides an interactive Swagger UI.

You will see:

```text
POST /rerank
```

---

# 19. Test the API Using Swagger

Click:

```text
POST /rerank
```

Click:

```text
Try it out
```

Enter:

```json
{
  "query": "What are the treatment options for cancer?",
  "documents": [
    "The hospital cafeteria provides breakfast.",
    "Cancer treatment may include surgery, chemotherapy and radiation therapy.",
    "The hospital has parking facilities.",
    "Patients with cancer should be referred for appropriate treatment.",
    "The emergency department is open 24 hours."
  ],
  "top_k": 3
}
```

Click:

```text
Execute
```

Expected response:

```text
200 OK
```

The API returns the documents in relevance order.

---

# 20. Test API Using Python

Keep the FastAPI server running.

Open a **second terminal**.

Navigate to the project:

```powershell
cd C:\Users\poornima\Desktop\reranking_module
```

Activate the environment if necessary:

```powershell
.\venv\Scripts\activate
```

Run:

```powershell
python test_api.py
```

Expected:

```text
200
```

followed by the JSON response containing ranked documents.

---

# 21. Why Two Terminals Are Required

### Terminal 1

Runs the API:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

This terminal must remain running.

### Terminal 2

Runs the API test:

```powershell
python test_api.py
```

Architecture:

```text
Terminal 1
     │
     ▼
FastAPI Server
     │
     │ HTTP
     ▼
Terminal 2
test_api.py
```

Do **not** press `Ctrl+C` in Terminal 1 while testing.

---

# 22. Complete Execution Order

Whenever you want to run the complete local pipeline from the beginning:

## Step 1

Open terminal.

```powershell
cd C:\Users\poornima\Desktop\reranking_module
```

## Step 2

Activate environment.

```powershell
.\venv\Scripts\activate
```

## Step 3

Add PDF/DOCX files.

```text
documents/
```

## Step 4

Ingest documents.

```powershell
python ingest_documents.py
```

## Step 5

Run retrieval and reranking test.

```powershell
python search_and_rerank.py
```

## Step 6

Run evaluation.

```powershell
python evaluation.py
```

## Step 7

Start API.

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

## Step 8

Open:

```text
http://127.0.0.1:8000/docs
```

## Step 9

Test:

```text
POST /rerank
```

## Step 10

Expected result:

```text
200 OK
```

---

# 23. Complete Data Flow

The entire process can be represented as:

```text
                ┌─────────────────┐
                │ PDF / DOCX      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Document Loader │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Text Chunking   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Embedding Model │
                │ MiniLM          │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    ChromaDB     │
                └────────┬────────┘
                         │
                    User Query
                         │
                         ▼
                ┌─────────────────┐
                │ Vector Search   │
                └────────┬────────┘
                         │
                         ▼
                   Top 20 Chunks
                         │
                         ▼
          ┌─────────────────────────────┐
          │      CROSSENCODER           │
          │                             │
          │ Query + Chunk               │
          │      ↓                      │
          │ Relevance Score             │
          │      ↓                      │
          │ Ranking                     │
          └──────────────┬──────────────┘
                         │
                         ▼
                    Top 5 Chunks
                         │
                         ▼
                  Return to RAG
                         │
                         ▼
                        LLM
                         │
                         ▼
                   Final Answer
```

---

# 24. How CrossEncoder Reranking Works

Unlike a normal embedding model that independently creates embeddings for the query and documents, the CrossEncoder receives both together.

For every candidate:

```text
Query
+
Document
```

Example:

```text
Query:
What are the treatment options for cancer?

Document:
Cancer treatment may include surgery,
chemotherapy and radiation therapy.
```

The CrossEncoder evaluates their relevance:

```text
Query + Document
       ↓
CrossEncoder
       ↓
Score
```

This process is repeated for every retrieved candidate.

If there are 20 candidates:

```text
Query + Document 1 → Score
Query + Document 2 → Score
Query + Document 3 → Score
...
Query + Document 20 → Score
```

Then:

```text
Sort scores
     ↓
Top 5
```

---

# 25. Why Reranking Is Needed

Vector retrieval is designed for **fast candidate retrieval**.

Reranking is designed for **more precise relevance ordering**.

Therefore:

```text
Vector Search
     ↓
Fast but approximate
     ↓
Top 20 candidates
```

Then:

```text
CrossEncoder
     ↓
More detailed query-document comparison
     ↓
Top 5 relevant documents
```

This creates a two-stage retrieval architecture:

```text
Stage 1
───────
Bi-Encoder / Vector Search
        ↓
Fast candidate retrieval


Stage 2
───────
CrossEncoder
        ↓
Precise reranking
```

---

# 26. Role of This Module in the Team Project

This module does **not** generate the final answer.

Your responsibility is:

```text
Retrieved Documents
        ↓
      Reranker
        ↓
Ranked Documents
        ↓
Return to RAG
```

The team's complete RAG system may look like:

```text
User
 ↓
Query
 ↓
Retriever
 ↓
Top 20 Documents
 ↓
YOUR RERANKER
 ↓
Top 5 Documents
 ↓
LLM
 ↓
Answer
```

---

# 27. API Contract for Team Integration

Your team can send a request to:

```text
POST /rerank
```

Request:

```json
{
  "query": "What are the treatment options for cancer?",
  "documents": [
    "Document chunk 1",
    "Document chunk 2",
    "Document chunk 3",
    "Document chunk 4"
  ],
  "top_k": 3
}
```

Response:

```json
{
  "query": "What are the treatment options for cancer?",
  "total_documents_received": 4,
  "documents_returned": 3,
  "results": [
    {
      "rank": 1,
      "score": 8.52,
      "text": "Cancer treatment may include surgery..."
    },
    {
      "rank": 2,
      "score": 5.21,
      "text": "Patients with cancer should..."
    },
    {
      "rank": 3,
      "score": 1.14,
      "text": "The hospital provides..."
    }
  ]
}
```

The team can then send these top-ranked chunks to the LLM.

---

# 28. Important Commands — Quick Reference

## Navigate to project

```powershell
cd C:\Users\poornima\Desktop\reranking_module
```

## Create environment

```powershell
python -m venv venv
```

## Activate environment

```powershell
.\venv\Scripts\activate
```

## Install dependencies

```powershell
pip install -r requirements.txt
```

## Ingest documents

```powershell
python ingest_documents.py
```

## Test retrieval + reranking

```powershell
python search_and_rerank.py
```

## Run evaluation

```powershell
python evaluation.py
```

## Start FastAPI

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

## API documentation

```text
http://127.0.0.1:8000/docs
```

## Test API

Open another terminal:

```powershell
python test_api.py
```

Expected:

```text
200
```

---

# 29. Troubleshooting

## Problem: `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'sentence_transformers'
```

Solution:

```powershell
pip install sentence-transformers
```

Or reinstall all dependencies:

```powershell
pip install -r requirements.txt
```

---

## Problem: FastAPI not found

```text
ModuleNotFoundError: No module named 'fastapi'
```

Run:

```powershell
pip install fastapi uvicorn
```

---

## Problem: Uvicorn not found

Run:

```powershell
pip install uvicorn
```

Then:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Using:

```text
python -m uvicorn
```

is recommended because it uses the Uvicorn installed in the active Python environment.

---

## Problem: No documents found

Make sure files are inside:

```text
reranking_module/
└── documents/
    ├── hospital_guidelines.pdf
    └── medical_guidelines.docx
```

Then run:

```powershell
python ingest_documents.py
```

---

## Problem: ChromaDB error

If the local ChromaDB becomes corrupted or contains incompatible old data, delete the local database and rebuild it.

PowerShell:

```powershell
Remove-Item -Recurse -Force .\chroma_db
```

Then:

```powershell
python ingest_documents.py
```

---

## Problem: API returns 200

This is **not a problem**.

```text
200 OK
```

means the request was successfully processed.

---

# 30. Important Notes About PDF Documents

Text-based PDFs can normally be processed using `pypdf`.

However, some PDFs may have:

* scanned pages
* images instead of text
* unusual fonts
* broken text encoding
* tables
* columns
* merged words

In these cases, extracted text may not be perfect.

For scanned PDFs, an OCR-based extraction pipeline may be required.

---

# 31. Current Module Limitations

The current implementation primarily supports:

```text
PDF
DOCX
```

The current DOCX loader primarily extracts paragraph text.

Complex documents containing:

* tables
* images
* scanned pages
* unusual PDF layouts

may require additional document-processing techniques.

These limitations belong to the document ingestion stage and are separate from the CrossEncoder reranking algorithm.

---

# 32. Performance Consideration

Reranking every document in a large database would be expensive.

Therefore, the recommended architecture is:

```text
Millions of Documents
        ↓
Vector Search
        ↓
Top 20 / Top 50
        ↓
CrossEncoder
        ↓
Top 5
```

The CrossEncoder is applied only to a relatively small candidate set.

This provides a balance between:

```text
Speed
+
Relevance
```

---

# 33. Security and Production Considerations

For a production deployment, the API should additionally consider:

* authentication
* input validation
* rate limiting
* logging
* monitoring
* HTTPS
* containerization
* environment-based configuration
* model caching
* GPU acceleration when appropriate

These are deployment concerns and are not required for the basic reranking implementation.

---

# 34. Final Module Workflow

The final workflow of this project is:

```text
1. User provides documents
          ↓
2. Document loader extracts text
          ↓
3. Text is divided into chunks
          ↓
4. Chunks are embedded
          ↓
5. Chunks are stored in ChromaDB
          ↓
6. User asks a question
          ↓
7. ChromaDB retrieves top 20 candidates
          ↓
8. Candidates are sent to CrossEncoder
          ↓
9. CrossEncoder calculates relevance scores
          ↓
10. Documents are sorted by score
          ↓
11. Top 5 documents are selected
          ↓
12. Top 5 are returned to the RAG pipeline
          ↓
13. LLM uses the reranked context
          ↓
14. Final answer is generated
```

---

# 35. Project Completion Checklist

```text
[✓] PDF document loading
[✓] DOCX document loading
[✓] Text chunking
[✓] ChromaDB integration
[✓] Vector retrieval
[✓] CrossEncoder reranking
[✓] Top-K ranking
[✓] Real hospital document testing
[✓] Evaluation module
[✓] FastAPI API
[✓] API testing
[✓] HTTP 200 response
[✓] Team integration interface
[ ] Final integration with team's RAG pipeline
```

The final team-integration checkbox depends on connecting this API/module to the team's retrieval component.

---

# 36. One-Line Project Description

> **A CrossEncoder-based RAG document reranking module that improves retrieval quality by reordering candidate document chunks according to query-document relevance before passing context to the LLM.**

---

# 37. Conclusion

This project implements the **reranking stage of a Retrieval-Augmented Generation system**.

The system uses:

```text
ChromaDB
   ↓
Candidate Retrieval
   ↓
CrossEncoder
   ↓
Relevance Scoring
   ↓
Reranking
   ↓
Top-K Relevant Context
```

The primary contribution of this module is the **CrossEncoder-based relevance reranking layer**, which improves the ordering of retrieved documents before they are provided to the generation component of the RAG system.

The module can be executed independently for testing and exposed through FastAPI for integration with the complete team RAG system.

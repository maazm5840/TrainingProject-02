Hospital SOP RAG

Overview

This project implements the retrieval part of a Retrieval-Augmented Generation (RAG) system for hospital Standard Operating Procedure (SOP) documents.

The system converts SOP documents into embeddings, stores them in a FAISS vector database, and retrieves the most relevant SOP sections when a user asks a question.

How It Works

SOP PDFs
↓
Text Extraction
↓
Chunking
↓
Embedding Generation
↓
FAISS Vector Database
↓
User Question
↓
Question Embedding
↓
Similarity Search
↓
Relevant SOP Chunks

Project Structure

hospital-sop-rag/
│
├── data/
│   └── README.md
│
├── index/
│   └── .gitkeep
│
├── ingest.py
├── retrieve.py
├── run_rag.py
├── requirements.txt
├── README.md
└── .gitignore

How to Run

Install the required packages:

pip install -r requirements.txt

Place approved SOP PDF files inside the "data" folder.

Then create the searchable database:

python ingest.py

Run the retrieval system:

python run_rag.py

Example question:

What is the procedure for patient admission?

File Description

- "ingest.py" — extracts, chunks and embeds SOP documents and creates the FAISS index.
- "retrieve.py" — searches the FAISS database for relevant SOP content.
- "run_rag.py" — provides a simple command-line interface for testing.
- "requirements.txt" — contains the required Python packages.
- "data/" — contains approved SOP documents.
- "index/" — stores the generated local vector database.

Team Integration

The retrieved SOP content can later be passed to an LLM/generation module by another team member.

retrieve.py
     ↓
Relevant SOP Context
     ↓
LLM / Generation Module
     ↓
Final Answer

Important

Use only approved and authorized SOP documents.

Do not upload confidential patient information or sensitive hospital documents to a public Git repository.

This project is intended for educational and demonstration purposes.
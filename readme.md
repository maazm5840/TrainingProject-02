# 🏥 Hospital RAG Feedback Continuous Improvement System

A Retrieval-Augmented Generation (RAG) project that answers hospital-related questions using AI and continuously improves its responses through user feedback.

---

## 📌 Project Overview

The **Hospital RAG Feedback Continuous Improvement System** is an intelligent question-answering application designed for hospitals. It retrieves relevant information from hospital documents using semantic search and generates accurate, context-aware responses. After each response, users can rate the answer as **Good** or **Bad**, allowing the system to collect feedback for future improvements.

This project demonstrates the practical implementation of **Retrieval-Augmented Generation (RAG)**, **vector databases**, and **continuous learning through feedback**.

---

## 🎯 Objectives

* Build a hospital AI assistant using RAG.
* Retrieve relevant information from hospital documents.
* Generate accurate responses using retrieved context.
* Collect user feedback after every interaction.
* Store feedback for continuous improvement.
* Improve retrieval quality over time.

---

## ✨ Features

* 🔍 Semantic document retrieval
* 🤖 AI-generated hospital answers
* 📚 FAISS vector database
* 📝 User feedback collection
* 📊 Feedback history stored in CSV
* 🔄 Continuous improvement workflow
* 💻 Simple Python implementation

---

## 🛠️ Technologies Used

| Technology            | Purpose                   |
| --------------------- | ------------------------- |
| Python                | Core programming language |
| Sentence Transformers | Text embeddings           |
| FAISS                 | Vector similarity search  |
| Transformers          | Response generation       |
| Pandas                | Feedback storage          |
| NumPy                 | Numerical processing      |
| Streamlit             | Optional web interface    |

---

## 📁 Project Structure

```text
rag_feedback/
│
├── data/
│   ├── hospital_docs.txt
│   └── feedback.csv
│
├── models/
│   └── vector_index.faiss
│
├── src/
│   ├── ingest.py
│   ├── retriever.py
│   ├── generator.py
│   ├── feedback.py
│   └── app.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

### Folder Description

* **data/** → Stores hospital documents and feedback records.
* **models/** → Contains the FAISS vector index.
* **src/** → All Python source files.
* **requirements.txt** → Required Python libraries.
* **README.md** → Project documentation.

---

## 🧠 System Architecture

```text
            User Question
                  │
                  ▼
      Sentence Embedding Model
                  │
                  ▼
      FAISS Vector Retrieval
                  │
                  ▼
     Relevant Hospital Context
                  │
                  ▼
      Language Model Generator
                  │
                  ▼
          Generated Answer
                  │
                  ▼
        User Feedback (👍/👎)
                  │
                  ▼
        feedback.csv Storage
                  │
                  ▼
      Continuous Improvement
```

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/hospital-rag-feedback.git
cd hospital-rag-feedback
```

### Step 2: Create Virtual Environment

**Windows**

```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

```text
sentence-transformers
faiss-cpu
transformers
torch
pandas
numpy
streamlit
```

Or install manually:

```bash
pip install sentence-transformers faiss-cpu transformers torch pandas numpy streamlit
```

---

## 📄 Hospital Dataset

Create a file named **hospital_docs.txt** inside the `data` folder.

Example:

```text
The emergency department is open 24 hours.

Patients should carry their hospital ID card during every visit.

The cardiology department is located on the second floor.

Visitors are allowed between 10 AM and 6 PM.

The pharmacy operates from 8 AM to 8 PM.
```

You can add hospital policies, departments, FAQs, and medical guidelines.

---

## 🚀 Build Vector Database

Generate embeddings and create the FAISS index:

```bash
python src/ingest.py
```

Output:

```text
models/vector_index.faiss
```

---

## ▶️ Run the Application

### Python Version

```bash
python src/app.py
```

### Streamlit Version

```bash
streamlit run src/app.py
```

---

## 💬 Example Interaction

**Question**

```text
Where is the cardiology department?
```

**Retrieved Context**

```text
The cardiology department is located on the second floor.
```

**Generated Answer**

```text
The Cardiology Department is located on the second floor of the hospital.
```

---

## ⭐ Feedback System

After every response, users provide feedback.

| Feedback | Description                    |
| -------- | ------------------------------ |
| 👍 Good  | Correct and useful answer      |
| 👎 Bad   | Incorrect or incomplete answer |

Example `feedback.csv`

| Question        | Answer      | Feedback |
| --------------- | ----------- | -------- |
| Visiting hours? | 10 AM–6 PM  | Good     |
| ICU location?   | First floor | Bad      |

This feedback is automatically stored for future analysis.

---

## 🔄 Continuous Improvement Workflow

1. User asks a hospital question.
2. Documents are converted into embeddings.
3. FAISS retrieves the most relevant information.
4. The language model generates an answer.
5. User rates the response.
6. Feedback is stored in `feedback.csv`.
7. Developers analyze poor responses.
8. Hospital documents are updated.
9. Vector database is rebuilt for improved accuracy.

This creates a **continuous learning cycle** without retraining the language model.

---

## 📂 Module Description

### `ingest.py`

* Reads hospital documents
* Splits text into chunks
* Generates embeddings
* Builds FAISS index

### `retriever.py`

* Loads vector database
* Converts query into embedding
* Retrieves relevant document chunks

### `generator.py`

* Uses retrieved context
* Generates natural language responses

### `feedback.py`

* Saves questions
* Saves generated answers
* Records user feedback

### `app.py`

* Main application
* Accepts user input
* Displays AI responses
* Collects feedback

---

## 📈 Future Enhancements

* 🎙️ Voice-based hospital assistant
* 🌐 Multi-language support
* 📄 PDF document ingestion
* 📊 Feedback analytics dashboard
* 📅 Appointment scheduling integration
* 🏥 Electronic Health Record connectivity

---

## 🎓 Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Vector embeddings
* Semantic similarity search
* FAISS vector databases
* AI-powered document retrieval
* Feedback-driven continuous improvement
* Practical healthcare AI application development

---

## 👩‍💻 Author

**Rakshitha H A**

**Project:** Hospital RAG Feedback Continuous Improvement System

Built using Python, FAISS, Sentence Transformers, Transformers, Pandas, NumPy, and Streamlit.

---

## 📜 License

This project is developed for **educational and research purposes only**. It is not intended for real-world medical diagnosis or clinical decision-making without professional validation.

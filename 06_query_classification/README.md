## Query Classification in Hospital RAG

**Query classification** is the process of identifying the **type, intent, and required knowledge source** of a user's question before sending it to the RAG retrieval system.

For example:

| User Query                                 | Classification         |
| ------------------------------------------ | ---------------------- |
| “What are the hospital visiting hours?”    | Hospital Information   |
| “What documents are needed for admission?” | Admission              |
| “What is the MRI preparation procedure?”   | Procedure              |
| “What is the leave policy for nurses?”     | HR / Staff Policy      |
| “Where is the cardiology department?”      | Department Information |
| “What is this medicine used for?”          | Medication Information |

### Role in the RAG Pipeline

```text
User Query
    ↓
Query Classification
    ↓
Identify Intent / Category
    ↓
Select Appropriate Knowledge Base
    ↓
Retrieve Relevant Documents
    ↓
Reranking
    ↓
LLM
    ↓
Answer + Source
```

### Main Objective

The objective is to **route each query to the correct document collection or retrieval strategy**, improving the relevance and accuracy of the final response.

For example:

```text
Query:
"What documents are required for hospital admission?"

        ↓

Classifier

        ↓

Category: Admission

        ↓

Search:
Admission Guidelines + Registration Documents

        ↓

Relevant Context

        ↓

LLM

        ↓

Answer with Sources
```

### Possible Classification Categories

For your project, you could define:

1. **Hospital Information**
2. **Admission & Registration**
3. **Departments & Services**
4. **Appointment Information**
5. **Medical Procedures**
6. **Medication Information**
7. **Discharge Information**
8. **Hospital Policies**
9. **HR & Employee Policies**
10. **Emergency/Operational Information**
11. **Out-of-Scope Query**

### Classification Methods

You can implement query classification using different approaches:

**1. Rule-based classification**

* Keyword matching
* Simple and easy to implement
* Less flexible

**2. ML-based classification**

* TF-IDF + Logistic Regression
* SVM
* Random Forest
* Requires a labeled dataset

**3. Transformer-based classification**

* BERT / DistilBERT
* Better understanding of natural-language queries
* Requires training/fine-tuning

**4. LLM-based classification**

* Give the LLM a predefined set of categories
* Useful when you have limited training data
* Can return structured JSON containing the category and confidence

### A good project implementation

For a college project, I'd recommend:

```text
                 User Query
                     ↓
             Query Classifier
                     ↓
          ┌──────────┼──────────┐
          ↓          ↓          ↓
       Admission   Medical     HR/Policy
          ↓          ↓          ↓
       Vector DB  Vector DB   Vector DB
          └──────────┼──────────┘
                     ↓
                Retrieved Data
                     ↓
                    LLM
                     ↓
              Final Response
```

**Key benefit:** Query classification prevents the RAG system from searching the entire hospital knowledge base for every question. It narrows the search to the **most relevant domain**, which can improve retrieval precision, response quality, speed, and access control.

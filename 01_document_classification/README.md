# Document Classification Module

## Overview

The Document Classification Module is a component of our Hospital RAG Assistant.

Its purpose is to automatically identify the category of an incoming hospital document before it is processed by the RAG pipeline.

This helps organize documents and supports category-based retrieval.

## Problem Statement

Hospitals contain different types of documents, such as:

- Emergency protocols
- Hospital SOPs
- Department guidelines
- Equipment manuals
- Hospital policies

Manually identifying the document type can be time-consuming.

This module automates the classification process using Machine Learning.

## Objectives

- Automatically classify hospital documents.
- Reduce manual document categorization.
- Support better organization of hospital knowledge.
- Prepare documents for integration with the RAG system.
- Accept both text input and PDF documents.
- Display the predicted category and confidence score.

## Document Categories

The model classifies documents into five categories:

| Category | Description |
|---|---|
| Emergency Protocol | Instructions for handling emergencies and critical situations |
| Hospital SOP | Standard operating procedures followed in the hospital |
| Department Guideline | Guidelines specific to a hospital department |
| Equipment Manual | Instructions for operating and maintaining medical equipment |
| Hospital Policy | Rules and policies followed by hospital staff |

## Technologies Used

- **Python** — Programming language
- **Pandas** — Dataset loading and processing
- **Scikit-learn** — Machine Learning
- **TF-IDF** — Text feature extraction
- **Logistic Regression** — Classification algorithm
- **Joblib** — Saving and loading the trained model
- **pypdf** — Extracting text from PDF documents

## Machine Learning Approach

### 1. Dataset Creation

A labeled dataset was created containing hospital-related document examples.

Each document contains:

- Document text
- Document category

Example:

| Document Text | Category |
|---|---|
| Instructions for operating an ECG machine | Equipment Manual |
| Steps to follow during cardiac arrest | Emergency Protocol |
| Rules for hospital visitors | Hospital Policy |

### 2. Text Preprocessing

The document text is provided to the TF-IDF vectorizer.

TF-IDF converts text into numerical features that the Machine Learning model can understand.

### 3. Feature Extraction

The module uses:

```python
TfidfVectorizer()

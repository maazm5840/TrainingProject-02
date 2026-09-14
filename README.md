# Query Classification

## Overview

Query Classification is a sub-module of the Hospital RAG System.

The purpose of this module is to identify the intent or category of a user's query before the query is passed to the Query Router and retrieval components.

This helps the Hospital RAG system identify the appropriate type of information required for a query.

## Workflow

User Query
    ↓
Text Feature Extraction using TF-IDF
    ↓
Logistic Regression Classifier
    ↓
Query Category
    ↓
Confidence Score
    ↓
Query Router

## Classification Categories

The system supports the following categories:

1. HOSPITAL_INFO
2. ADMISSION
3. DEPARTMENT
4. APPOINTMENT
5. PROCEDURE
6. MEDICATION
7. DISCHARGE
8. HOSPITAL_POLICY
9. HR_POLICY
10. EMERGENCY
11. OUT_OF_SCOPE

## Technologies Used

- Python
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

## How It Works

The user query is first converted into numerical features using TF-IDF.

The Logistic Regression model then analyzes these features and predicts the most suitable query category.

The system also provides a confidence score and the top three predicted categories.

## Example

### Input

What documents are required for admission?

### Output

Category: ADMISSION

### Another Example

Input:

Where is the cardiology department?

Output:

Category: DEPARTMENT

### Out-of-Scope Example

Input:

Tell me a joke

Output:

Category: OUT_OF_SCOPE

## Files

### query_classifier.py

Contains the training data, TF-IDF vectorizer, Logistic Regression model, classification function, and interactive query interface.

### test_classifier.py

Contains sample hospital queries used to test the classifier.

### requirements.txt

Contains the required Python dependency.

## Running the Project

Install the dependency:

```bash
pip install -r requirements.txt
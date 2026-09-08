# NER Model + Metadata

Hospital-specific Named Entity Recognition (NER) and metadata extraction module for hospital SOP documents.

## Overview

This module extracts important medical entities from hospital SOP documents and converts them into structured metadata.

The module uses a fine-tuned Hugging Face NER model, rule-based extraction, and entity normalization.

## Entity Types

```text
MEDICATION
PROCEDURE
DISEASE
SYMPTOM
EQUIPMENT
DEPARTMENT
PERSON_ROLE
LOCATION
LAB_TEST
DOSAGE
FREQUENCY
DURATION
POLICY
```

## Pipeline

```text
Hospital SOP
     ↓
Text / PDF Extraction
     ↓
NER Model
     ↓
Entity Extraction
     ↓
Rule-Based Extraction
     ↓
Normalization
     ↓
Metadata
```

## Project Structure

```text
02_ner_metadata_extration/
│
├── dataset/
│   ├── train.json
│   ├── val.json
│   └── test.json
│
├── model/
│   └── hospital_ner/
│       └── Fine-tuned NER model
│
├── src/
│   ├── __init__.py
│   ├── metadata.py
│   ├── normalizer.py
│   └── pipeline.py
│
├── train.py
├── evaluate.py
├── predict.py
├── predict_pdf.py
├── requirements.txt
└── README.md
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Requirements

```text
transformers
datasets
torch
seqeval
pypdf
accelerate
```

## Trained Model

The fine-tuned NER model is stored in:

```text
model/hospital_ner/
```

The model is based on DistilBERT and is fine-tuned for hospital-specific entity extraction.

## Train

Train the NER model using:

```bash
python train.py
```

The trained model is saved in:

```text
model/hospital_ner/
```

## Evaluate

Evaluate the model using:

```bash
python evaluate.py
```

The evaluation includes:

* Precision
* Recall
* F1 Score

## Predict from Text

Run:

```bash
python predict.py
```

This extracts medical entities from hospital-related text and generates metadata.

Example:

```text
The nurse should administer Paracetamol 500 mg twice daily for 5 days.
```

Example entities:

```text
nurse          → PERSON_ROLE
Paracetamol    → MEDICATION
500 mg         → DOSAGE
twice daily    → FREQUENCY
5 days         → DURATION
```

## Predict from PDF

To process a Hospital SOP PDF:

```bash
python predict_pdf.py hospital_sop.pdf
```

The PDF pipeline:

1. Extracts text from the PDF
2. Splits the text into sentences
3. Runs the NER model
4. Applies rule-based extraction
5. Normalizes entities
6. Generates metadata

## Metadata

The extracted entities are organized into:

```text
medications
procedures
diseases
symptoms
equipment
departments
person_roles
locations
lab_tests
dosages
frequencies
durations
policies
```

Example:

```json
{
  "medications": ["paracetamol"],
  "procedures": [],
  "diseases": ["diabetes"],
  "symptoms": ["fever"],
  "equipment": [],
  "departments": ["emergency department"],
  "person_roles": ["nurse"],
  "locations": [],
  "lab_tests": [],
  "dosages": ["500 mg"],
  "frequencies": ["twice daily"],
  "durations": ["5 days"],
  "policies": []
}
```

## Rule-Based Extraction

Rules are used to extract dosage, frequency, and duration.

### Dosage

```text
500 mg
10 mL
2 tablets
```

### Frequency

```text
once daily
twice daily
every 4 hours
```

### Duration

```text
5 days
2 weeks
3 months
```

## Normalization

Entity normalization converts different forms of the same entity into a consistent format.

Example:

```text
Staff Nurse
staff nurse
STAFF NURSE
```

becomes:

```text
staff nurse
```

## Dataset

The dataset contains training, validation, and test data.

```text
dataset/
├── train.json
├── val.json
└── test.json
```

## Purpose

This module extracts structured medical information and metadata from Hospital SOP documents.

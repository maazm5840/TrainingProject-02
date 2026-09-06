from pathlib import Path
from pypdf import PdfReader
from docx import Document
import re


def clean_text(text):
    if not text:
        return ""

    # Fix words broken by spaces around common OCR/PDF extraction issues
    text = re.sub(r"\b([A-Za-z])\s+([A-Za-z])\b", r"\1\2", text)

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize newlines
    text = re.sub(r"\n+", "\n", text)

    # Fix common extracted words
    replacements = {
        "tr eat": "treat",
        "tre at": "treat",
        "treatment/Referral": "treatment / referral",
        "Diagnoseanrefer": "Diagnose and refer",
        "Diagnosanrefer": "Diagnose and refer",
        "Diagnoseantreat": "Diagnose and treat",
        "Investigatean": "Investigate and ",
        "RecommendeServicM": "Recommended Services",
        "NamothIllness": "Name of Illness",
        "SlNo.": "Sl No.",
        "Cance": "Cancer",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


def load_pdf(file_path):

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        raw_text = page.extract_text() or ""

        text = clean_text(raw_text)

        if text:

            pages.append({
                "text": text,
                "metadata": {
                    "source": Path(file_path).name,
                    "file_path": str(file_path),
                    "file_type": "pdf",
                    "page": page_number
                }
            })

    return pages


def load_docx(file_path):

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = clean_text(
            paragraph.text
        )

        if text:
            paragraphs.append(text)

    text = "\n".join(paragraphs)

    if not text:
        return []

    return [{
        "text": text,
        "metadata": {
            "source": Path(file_path).name,
            "file_path": str(file_path),
            "file_type": "docx",
            "page": None
        }
    }]


def load_document(file_path):

    extension = Path(
        file_path
    ).suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )
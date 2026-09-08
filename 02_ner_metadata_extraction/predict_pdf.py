import json
import sys

import spacy
from pypdf import PdfReader

from src.pipeline import HospitalNERPipeline


def main():

    # Check PDF file argument
    if len(sys.argv) < 2:
        print("Usage: python predict_pdf.py <pdf_file>")
        return

    pdf_path = sys.argv[1]

    print("\n======================================")
    print("     HOSPITAL SOP PDF NER")
    print("======================================")

    print("\nLoading Hospital NER model...")
    pipeline = HospitalNERPipeline()

    # Sentence splitter
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    # Read PDF
    print(f"Reading PDF: {pdf_path}")

    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            print(f"\nPage {page_number}: No text found")
            continue

        print(f"\n========== PAGE {page_number} ==========")

        # Split PDF text into sentences
        doc = nlp(text)

        for sentence in doc.sents:

            sentence_text = sentence.text.strip()

            if not sentence_text:
                continue

            print("\nSentence:", sentence_text)

            # Run NER + metadata pipeline
            result = pipeline.process(
                text=sentence_text,
                document_id=pdf_path,
                document_type="SOP",
                title=pdf_path,
                source="PDF",
                page=page_number
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False
                )
            )


if __name__ == "__main__":
    main()


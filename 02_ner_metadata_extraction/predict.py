"""
Interactive prediction script for the hospital SOP NER pipeline.

Usage:
    python predict.py
    Enter text at the prompt; press Enter on an empty line (or Ctrl+D) to quit.
"""

import json

from src.pipeline import HospitalNERPipeline


def print_result(result: dict) -> None:
    entities = result["entities"]
    if not entities:
        print("(no entities found)")
    else:
        for ent in entities:
            print(f"{ent['label']}: {ent['text']}")

    print("\nNormalized metadata:")
    print(json.dumps(result["medical_metadata"], indent=2, ensure_ascii=False))


def main() -> None:
    try:
        pipeline = HospitalNERPipeline()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    print("Hospital SOP NER - interactive prediction")
    print("Enter text and press Enter. Enter a blank line or Ctrl+D to quit.\n")

    while True:
        try:
            text = input("Input: ").strip()
        except EOFError:
            print()
            break

        if not text:
            break

        result = pipeline.process(text)
        print("\nOutput:")
        print_result(result)
        print()


if __name__ == "__main__":
    main()

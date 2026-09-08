import re

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

from src.normalizer import EntityNormalizer
from src.metadata import create_medical_metadata


MODEL_DIR = "model/hospital_ner"


class HospitalNERPipeline:

    def __init__(self):

        print(
            "Loading Hospital NER model..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            MODEL_DIR
        )

        self.model.eval()

        self.normalizer = EntityNormalizer()

        self.id2label = self.model.config.id2label

    # ========================================================
    # MODEL EXTRACTION
    # ========================================================

    def extract_entities(self, text):

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512
        )

        offsets = encoded.pop(
            "offset_mapping"
        )[0].tolist()

        with torch.no_grad():

            outputs = self.model(
                **encoded
            )

        predictions = torch.argmax(
            outputs.logits,
            dim=-1
        )[0].tolist()

        entities = []

        current_entity = None

        for token_index, label_id in enumerate(
            predictions
        ):

            label = self.id2label[
                label_id
            ]

            start, end = offsets[
                token_index
            ]

            if start == end:
                continue

            if label == "O":

                if current_entity:

                    entities.append(
                        current_entity
                    )

                    current_entity = None

                continue

            if label.startswith("B-"):

                if current_entity:

                    entities.append(
                        current_entity
                    )

                entity_type = label[2:]

                current_entity = {

                    "start": start,

                    "end": end,

                    "label": entity_type
                }

            elif label.startswith("I-"):

                entity_type = label[2:]

                if (
                    current_entity
                    and current_entity["label"]
                    == entity_type
                ):

                    current_entity["end"] = end

                else:

                    current_entity = {

                        "start": start,

                        "end": end,

                        "label": entity_type
                    }

        if current_entity:

            entities.append(
                current_entity
            )

        # Convert offsets to actual text
        final_entities = []

        for entity in entities:

            start = entity["start"]
            end = entity["end"]

            entity_text = text[
                start:end
            ]

            if not entity_text.strip():
                continue

            normalized = self.normalizer.normalize(
                entity_text,
                entity["label"]
            )

            final_entities.append(
                {
                    "text": entity_text,
                    "label": entity["label"],
                    "start": start,
                    "end": end,
                    "source": "model",
                    "normalized": normalized
                }
            )

        return final_entities

    # ========================================================
    # RULE EXTRACTION
    # ========================================================

    def extract_rules(self, text):

        entities = []

        patterns = [

            (
                r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|kg|ml|mL|L)\b",
                "DOSAGE"
            ),

            (
                r"\b\d+\s+(?:tablet|tablets|capsule|capsules)\b",
                "DOSAGE"
            ),

            (
                r"\b(?:once|twice|three times|four times)\s+daily\b",
                "FREQUENCY"
            ),

            (
                r"\bevery\s+\d+\s+(?:hour|hours|day|days|week|weeks)\b",
                "FREQUENCY"
            ),

            (
                r"\bevery\s+hour\b",
                "FREQUENCY"
            ),

            (
                r"\b\d+\s+(?:day|days|week|weeks|month|months|year|years|minutes|hours)\b",
                "DURATION"
            )
        ]

        for pattern, label in patterns:

            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE
            ):

                entity_text = match.group()

                entities.append(
                    {
                        "text": entity_text,
                        "label": label,
                        "start": match.start(),
                        "end": match.end(),
                        "source": "rule",
                        "normalized": self.normalizer.normalize(
                            entity_text,
                            label
                        )
                    }
                )

        return entities

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    def merge_entities(
        self,
        model_entities,
        rule_entities
    ):

        combined = []

        for entity in model_entities:

            combined.append(
                entity
            )

        for rule_entity in rule_entities:

            duplicate = False

            for existing in combined:

                if (
                    existing["start"]
                    == rule_entity["start"]

                    and existing["end"]
                    == rule_entity["end"]

                    and existing["label"]
                    == rule_entity["label"]
                ):

                    duplicate = True

                    break

            if not duplicate:

                combined.append(
                    rule_entity
                )

        combined.sort(
            key=lambda x: x["start"]
        )

        return combined

    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    def process(
        self,
        text,
        document_id="interactive_query",
        document_type="SOP",
        title="Interactive Text",
        source="user_input",
        page=None
    ):

        model_entities = self.extract_entities(
            text
        )

        rule_entities = self.extract_rules(
            text
        )

        entities = self.merge_entities(
            model_entities,
            rule_entities
        )

        medical_metadata = create_medical_metadata(
            entities
        )

        result = {

            "document_id": document_id,

            "document_type": document_type,

            "title": title,

            "source": source,

            "page": page,

            "entities": entities,

            "medical_metadata": medical_metadata
        }

        return result
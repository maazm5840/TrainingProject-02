import json
import os
import shutil
import glob

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import precision_score, recall_score, f1_score


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

TRAIN_FILE = "dataset/train.json"
VAL_FILE = "dataset/val.json"

# Final trained model location
MODEL_DIR = "model/hospital_ner"

MAX_LENGTH = 512

EPOCHS = 10
LEARNING_RATE = 3e-5
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01


# ============================================================
# ENTITY LABELS
# ============================================================

ENTITY_TYPES = [
    "MEDICATION",
    "PROCEDURE",
    "DISEASE",
    "SYMPTOM",
    "EQUIPMENT",
    "DEPARTMENT",
    "PERSON_ROLE",
    "LOCATION",
    "LAB_TEST",
    "DOSAGE",
    "FREQUENCY",
    "DURATION",
    "POLICY"
]


# ============================================================
# BIO LABELS
# ============================================================

LABEL_LIST = ["O"]

for entity in ENTITY_TYPES:
    LABEL_LIST.append("B-" + entity)
    LABEL_LIST.append("I-" + entity)


LABEL2ID = {
    label: index
    for index, label in enumerate(LABEL_LIST)
}

ID2LABEL = {
    index: label
    for label, index in LABEL2ID.items()
}


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# ============================================================
# CONVERT DATA
# ============================================================

def prepare_dataset(data):

    records = []

    for item in data:

        text = item["text"]

        entities = []

        for annotation in item.get("annotations", []):

            entity_text = annotation[0]
            entity_label = annotation[1]

            start = text.find(entity_text)

            if start == -1:

                print(
                    "WARNING: Entity not found:",
                    repr(entity_text)
                )

                continue

            end = start + len(entity_text)

            entities.append(
                {
                    "start": start,
                    "end": end,
                    "label": entity_label
                }
            )

        records.append(
            {
                "text": text,
                "entities": entities
            }
        )

    return Dataset.from_list(records)


# ============================================================
# TOKEN LABELING
# ============================================================

def tokenize_and_align_labels(examples):

    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True
    )

    all_labels = []

    for batch_index in range(len(examples["text"])):

        entities = examples["entities"][batch_index]

        offsets = tokenized["offset_mapping"][batch_index]

        labels = []

        for token_start, token_end in offsets:

            # Special tokens
            if token_start == token_end:

                labels.append(-100)

                continue

            token_label = "O"

            for entity in entities:

                entity_start = entity["start"]
                entity_end = entity["end"]
                entity_type = entity["label"]

                # No overlap
                if token_end <= entity_start:
                    continue

                if token_start >= entity_end:
                    continue

                # Token overlaps entity
                if token_start == entity_start:

                    token_label = "B-" + entity_type

                else:

                    token_label = "I-" + entity_type

                break

            labels.append(
                LABEL2ID[token_label]
            )

        all_labels.append(labels)

    tokenized["labels"] = all_labels

    tokenized.pop("offset_mapping")

    return tokenized


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predictions = np.argmax(
        predictions,
        axis=2
    )

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        predictions,
        labels
    ):

        current_predictions = []
        current_labels = []

        for pred, lab in zip(
            prediction,
            label
        ):

            if lab == -100:
                continue

            current_predictions.append(
                ID2LABEL[pred]
            )

            current_labels.append(
                ID2LABEL[lab]
            )

        true_predictions.append(
            current_predictions
        )

        true_labels.append(
            current_labels
        )

    precision = precision_score(
        true_labels,
        true_predictions
    )

    recall = recall_score(
        true_labels,
        true_predictions
    )

    f1 = f1_score(
        true_labels,
        true_predictions
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# CLEAN OLD CHECKPOINTS
# ============================================================

def clean_old_checkpoints():

    if not os.path.exists(MODEL_DIR):
        return

    checkpoints = glob.glob(
        os.path.join(
            MODEL_DIR,
            "checkpoint-*"
        )
    )

    for checkpoint in checkpoints:

        if os.path.isdir(checkpoint):

            print(
                "Deleting old checkpoint:",
                checkpoint
            )

            shutil.rmtree(checkpoint)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HOSPITAL NER - HUGGING FACE TRAINING")
    print("=" * 60)

    print("\nBase model:")
    print(MODEL_NAME)

    # --------------------------------------------------------
    # CREATE MODEL DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DELETE OLD CHECKPOINTS
    # --------------------------------------------------------

    print("\nCleaning old checkpoints...")

    clean_old_checkpoints()

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print("Loading training data...")

    train_data = load_json(
        TRAIN_FILE
    )

    val_data = load_json(
        VAL_FILE
    )

    print(
        "Training examples:",
        len(train_data)
    )

    print(
        "Validation examples:",
        len(val_data)
    )

    train_dataset = prepare_dataset(
        train_data
    )

    val_dataset = prepare_dataset(
        val_data
    )

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    print("\nTokenizing datasets...")

    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    val_dataset = val_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nLoading pretrained model...")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # --------------------------------------------------------
    # DATA COLLATOR
    # --------------------------------------------------------

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    # --------------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=MODEL_DIR,

        num_train_epochs=EPOCHS,

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        weight_decay=WEIGHT_DECAY,

        # Evaluate after every epoch
        eval_strategy="epoch",

        # IMPORTANT:
        # Do NOT save checkpoints
        save_strategy="no",

        logging_strategy="epoch",

        report_to="none",

        fp16=False
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=val_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nStarting training...\n")

    trainer.train()

    # --------------------------------------------------------
    # FINAL MODEL SAVE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SAVING FINAL TRAINED MODEL")
    print("=" * 60)

    # Save ONLY the final trained model
    trainer.save_model(
        MODEL_DIR
    )

    # Save tokenizer
    tokenizer.save_pretrained(
        MODEL_DIR
    )

    # --------------------------------------------------------
    # FINAL CLEANUP
    # --------------------------------------------------------

    print("\nRemoving any checkpoint folders...")

    clean_old_checkpoints()

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("\nFinal model saved at:")
    print(MODEL_DIR)

    print("\nNo training checkpoints were saved.")

    print("\nYou can load the model using:")

    print(
        f'AutoModelForTokenClassification.from_pretrained("{MODEL_DIR}")'
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
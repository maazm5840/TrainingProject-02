import json

from src.pipeline import HospitalNERPipeline


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def evaluate_dataset(
    pipeline,
    data
):

    expected = set()

    predicted = set()

    for item in data:

        text = item["text"]

        # ----------------------------------------------------
        # Expected entities
        # ----------------------------------------------------

        for annotation in item.get(
            "annotations",
            []
        ):

            entity_text = annotation[0]

            label = annotation[1]

            start = text.find(
                entity_text
            )

            if start == -1:

                continue

            end = start + len(
                entity_text
            )

            expected.add(
                (
                    start,
                    end,
                    entity_text.lower(),
                    label
                )
            )

        # ----------------------------------------------------
        # Predicted entities
        # ----------------------------------------------------

        result = pipeline.process(
            text=text
        )

        for entity in result["entities"]:

            predicted.add(
                (
                    entity["start"],
                    entity["end"],
                    entity["text"].lower(),
                    entity["label"]
                )
            )

    true_positive = len(
        expected & predicted
    )

    false_positive = len(
        predicted - expected
    )

    false_negative = len(
        expected - predicted
    )

    if (
        true_positive + false_positive
        == 0
    ):

        precision = 0.0

    else:

        precision = (
            true_positive
            /
            (
                true_positive
                + false_positive
            )
        )

    if (
        true_positive + false_negative
        == 0
    ):

        recall = 0.0

    else:

        recall = (
            true_positive
            /
            (
                true_positive
                + false_negative
            )
        )

    if precision + recall == 0:

        f1 = 0.0

    else:

        f1 = (
            2
            * precision
            * recall
            /
            (precision + recall)
        )

    return precision, recall, f1


def main():

    print(
        "\n======================================"
    )

    print(
        "     HOSPITAL NER EVALUATION"
    )

    print(
        "======================================"
    )

    pipeline = HospitalNERPipeline()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_data = load_json(
        "dataset/val.json"
    )

    precision, recall, f1 = evaluate_dataset(
        pipeline,
        val_data
    )

    print(
        "\nValidation Results"
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "F1:",
        round(f1, 4)
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_data = load_json(
        "dataset/test.json"
    )

    precision, recall, f1 = evaluate_dataset(
        pipeline,
        test_data
    )

    print(
        "\nTest Results"
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "F1:",
        round(f1, 4)
    )


if __name__ == "__main__":

    main()
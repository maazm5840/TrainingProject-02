METADATA_FIELDS = {

    "MEDICATION": "medications",

    "PROCEDURE": "procedures",

    "DISEASE": "diseases",

    "SYMPTOM": "symptoms",

    "EQUIPMENT": "equipment",

    "DEPARTMENT": "departments",

    "PERSON_ROLE": "person_roles",

    "LOCATION": "locations",

    "LAB_TEST": "lab_tests",

    "DOSAGE": "dosages",

    "FREQUENCY": "frequencies",

    "DURATION": "durations",

    "POLICY": "policies"
}


def create_medical_metadata(entities):

    metadata = {
        field: []
        for field in METADATA_FIELDS.values()
    }

    for entity in entities:

        label = entity["label"]

        if label not in METADATA_FIELDS:
            continue

        field = METADATA_FIELDS[label]

        normalized = entity.get(
            "normalized",
            entity["text"].lower()
        )

        if normalized not in metadata[field]:

            metadata[field].append(
                normalized
            )

    return metadata
import re
from typing import List, Dict


# ---------------------------------------------------------
# Hospital / Medical Synonym Dictionary
# ---------------------------------------------------------

MEDICAL_TERMS = {
    "low bp": [
        "low blood pressure",
        "hypotension"
    ],

    "high bp": [
        "high blood pressure",
        "hypertension"
    ],

    "bp": [
        "blood pressure"
    ],

    "hr": [
        "heart rate"
    ],

    "icu": [
        "intensive care unit"
    ],

    "er": [
        "emergency room",
        "emergency department"
    ],

    "ppe": [
        "personal protective equipment"
    ],

    "sop": [
        "standard operating procedure"
    ],

    "cpr": [
        "cardiopulmonary resuscitation"
    ],

    "iv": [
        "intravenous"
    ],

    "admit": [
        "admission",
        "patient admission"
    ],

    "discharge": [
        "patient discharge",
        "discharge procedure"
    ],

    "infection control": [
        "infection prevention and control",
        "infection prevention",
        "hospital infection control"
    ],

    "hand washing": [
        "hand hygiene",
        "hand washing procedure"
    ]
}


# ---------------------------------------------------------
# Query Normalization
# ---------------------------------------------------------

def normalize_query(query: str) -> str:
    """
    Cleans and normalizes the user's query.
    """

    query = query.lower()

    # Remove extra spaces
    query = re.sub(r"\s+", " ", query)

    # Remove unnecessary punctuation
    query = re.sub(r"[^\w\s?-]", "", query)

    return query.strip()


# ---------------------------------------------------------
# Synonym Expansion
# ---------------------------------------------------------

def synonym_expansion(query: str) -> List[str]:
    """
    Generates queries using medical and hospital synonyms.
    """

    expanded_queries = []

    for term, synonyms in MEDICAL_TERMS.items():

        if term in query:

            for synonym in synonyms:

                new_query = query.replace(
                    term,
                    synonym
                )

                expanded_queries.append(
                    new_query
                )

    return expanded_queries


# ---------------------------------------------------------
# SOP Specific Expansion
# ---------------------------------------------------------

def sop_expansion(query: str) -> List[str]:
    """
    Creates hospital-SOP-oriented search queries.
    """

    queries = []

    queries.append(
        f"{query} hospital SOP"
    )

    queries.append(
        f"{query} hospital procedure"
    )

    queries.append(
        f"{query} clinical protocol"
    )

    return queries


# ---------------------------------------------------------
# Query Paraphrasing
# ---------------------------------------------------------

def paraphrase_expansion(query: str) -> List[str]:
    """
    Simple rule-based paraphrasing.

    This can later be replaced by a Hugging Face
    language model.
    """

    queries = []

    replacements = {
        "what is": [
            "what is the procedure for",
            "what is the protocol for"
        ],

        "how do i": [
            "what is the procedure to",
            "what are the steps to"
        ],

        "how should": [
            "what is the procedure for",
            "what is the protocol for"
        ]
    }

    for phrase, alternatives in replacements.items():

        if query.startswith(phrase):

            remaining_text = query[len(phrase):].strip()

            for alternative in alternatives:

                queries.append(
                    f"{alternative} {remaining_text}"
                )

    return queries


# ---------------------------------------------------------
# Duplicate Removal
# ---------------------------------------------------------

def remove_duplicates(
    queries: List[str]
) -> List[str]:

    unique_queries = []
    seen = set()

    for query in queries:

        query = query.strip()

        if not query:
            continue

        key = query.lower()

        if key not in seen:

            unique_queries.append(query)
            seen.add(key)

    return unique_queries


# ---------------------------------------------------------
# Main Query Expansion Function
# ---------------------------------------------------------

def expand_query(
    user_query: str,
    max_queries: int = 8
) -> Dict:

    # Original query
    original_query = user_query

    # Normalize
    normalized_query = normalize_query(
        user_query
    )

    # Always preserve original query
    queries = [
        normalized_query
    ]

    # Medical synonym expansion
    queries.extend(
        synonym_expansion(
            normalized_query
        )
    )

    # SOP-specific expansion
    queries.extend(
        sop_expansion(
            normalized_query
        )
    )

    # Paraphrase expansion
    queries.extend(
        paraphrase_expansion(
            normalized_query
        )
    )

    # Remove duplicates
    queries = remove_duplicates(
        queries
    )

    # Limit number of queries
    queries = queries[:max_queries]

    return {
        "original_query": original_query,
        "normalized_query": normalized_query,
        "expanded_queries": queries,
        "number_of_queries": len(queries)
    }


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    query = input(
        "Enter your hospital-related question: "
    )

    result = expand_query(query)

    print("\nOriginal Query:")
    print(result["original_query"])

    print("\nExpanded Queries:")

    for i, q in enumerate(
        result["expanded_queries"],
        start=1
    ):

        print(f"{i}. {q}")

    print(
        "\nTotal Queries:",
        result["number_of_queries"]
    )

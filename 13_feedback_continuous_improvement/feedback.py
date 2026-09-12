from database import save_feedback


def submit_feedback(
    question,
    answer,
    rating,
    feedback_type,
    comment=""
):

    if rating < 1 or rating > 5:
        raise ValueError(
            "Rating must be between 1 and 5"
        )

    save_feedback(
        question,
        answer,
        rating,
        feedback_type,
        comment
    )

    return "Feedback saved successfully!"
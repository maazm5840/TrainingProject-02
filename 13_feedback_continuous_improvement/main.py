from database import create_database
from feedback import submit_feedback
from analytics import show_analytics


# Create database
create_database()


print("===== RAG FEEDBACK SYSTEM =====")


question = input(
    "\nEnter your question: "
)


answer = input(
    "Enter the RAG answer: "
)


print("\nWas the answer helpful?")

print("1. 👍 Positive")

print("2. 👎 Negative")


choice = input(
    "Enter your choice: "
)


if choice == "1":

    feedback_type = "positive"

else:

    feedback_type = "negative"


rating = int(
    input(
        "Give rating from 1 to 5: "
    )
)


comment = input(
    "Enter your comment: "
)


result = submit_feedback(
    question,
    answer,
    rating,
    feedback_type,
    comment
)


print("\n", result)


show_analytics()
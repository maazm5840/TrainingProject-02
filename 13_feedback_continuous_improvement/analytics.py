import sqlite3

DATABASE_NAME = "feedback.db"


def show_analytics():

    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM feedback"
    )

    total = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM feedback "
        "WHERE feedback_type = 'positive'"
    )

    positive = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM feedback "
        "WHERE feedback_type = 'negative'"
    )

    negative = cursor.fetchone()[0]

    cursor.execute(
        "SELECT AVG(rating) FROM feedback"
    )

    average = cursor.fetchone()[0]

    connection.close()

    print("\n===== FEEDBACK ANALYTICS =====")

    print("Total Feedback:", total)

    print("Positive Feedback:", positive)

    print("Negative Feedback:", negative)

    if average:
        print(
            "Average Rating:",
            round(average, 2)
        )
    else:
        print("Average Rating: No ratings yet")
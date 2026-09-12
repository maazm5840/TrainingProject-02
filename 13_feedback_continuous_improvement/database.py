import sqlite3

DATABASE_NAME = "feedback.db"


def create_database():

    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            rating INTEGER,
            feedback_type TEXT,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def save_feedback(
    question,
    answer,
    rating,
    feedback_type,
    comment
):

    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO feedback
        (question, answer, rating, feedback_type, comment)
        VALUES (?, ?, ?, ?, ?)
    """, (
        question,
        answer,
        rating,
        feedback_type,
        comment
    ))

    connection.commit()
    connection.close()
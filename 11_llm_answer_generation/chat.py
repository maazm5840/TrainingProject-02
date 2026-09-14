from llm import HospitalLLM


def main():

    llm = HospitalLLM()

    context = """
    Hospital Emergency Admission SOP

    Emergency patients must first be registered
    at the emergency department.

    The triage nurse must perform an initial
    assessment.

    Patients are then directed to the appropriate
    treatment area according to the emergency protocol.

    Source: emergency_admission_sop.pdf
    """

    print("=" * 60)
    print("      HOSPITAL SOP LLM ASSISTANT")
    print("=" * 60)

    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            print("Assistant: Goodbye!")
            break

        try:

            answer = llm.generate_answer(
                question,
                context
            )

            print("\nAssistant:")
            print(answer)
            print()

        except Exception as e:

            print(
                "\nError:",
                e
            )


if __name__ == "__main__":
    main()
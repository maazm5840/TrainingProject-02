from llm import HospitalLLM


def main():

    llm = HospitalLLM()

    context = """
    Hospital Emergency Admission SOP

    1. Emergency patients must first be registered
       at the emergency department.

    2. The triage nurse must perform an initial
       assessment.

    3. Patients should then be directed to the
       appropriate treatment area according to
       the emergency protocol.

    Source: emergency_admission_sop.pdf
    """

    question = (
        "What is the procedure for admitting "
        "an emergency patient?"
    )

    answer = llm.generate_answer(
        question,
        context
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)


if __name__ == "__main__":
    main()
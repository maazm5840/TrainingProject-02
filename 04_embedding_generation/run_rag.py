from retrieve import retrieve


def main():

    print("=" * 55)
    print("           HOSPITAL SOP RAG")
    print("=" * 55)

    print("Type 'exit' to stop.\n")

    while True:

        question = input(
            "Enter your question: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit"
        }:

            print("\nExiting...")
            break

        if not question:

            print("Please enter a question.")
            continue

        try:

            results = retrieve(question)

            print("\nRelevant SOP Information")
            print("-" * 55)

            for result in results:

                print(
                    f"\nSource: {result['source']}"
                )

                print(
                    f"Similarity: {result['score']:.3f}"
                )

                print(
                    f"Chunk: {result['chunk']}"
                )

                print(
                    f"Content:\n{result['text']}"
                )

                print("-" * 55)

        except FileNotFoundError as error:

            print(f"\nError: {error}")

            break


if __name__ == "__main__":
    main()
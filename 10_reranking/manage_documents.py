from vector_store import (
    list_documents,
    delete_document,
    collection_count
)


def show_documents():

    documents = list_documents()

    print("\n" + "=" * 70)

    print("DOCUMENTS IN VECTOR DATABASE")

    print("=" * 70)


    if not documents:

        print("\nNo documents found.")

        return


    for number, document in enumerate(
        documents,
        start=1
    ):

        print(
            f"{number}. {document}"
        )


    print(
        f"\nTotal documents: "
        f"{len(documents)}"
    )


    print(
        f"Total chunks: "
        f"{collection_count()}"
    )


def delete_document_interactive():

    documents = list_documents()


    if not documents:

        print(
            "\nNo documents available."
        )

        return


    show_documents()


    source = input(
        "\nEnter exact document name to delete: "
    ).strip()


    if source not in documents:

        print(
            "\nDocument not found."
        )

        return


    deleted = delete_document(
        source
    )


    print(
        f"\nDeleted {deleted} chunks "
        f"from {source}"
    )


def main():

    while True:

        print("\n")

        print("=" * 70)

        print("DOCUMENT MANAGEMENT")

        print("=" * 70)

        print("1. List documents")

        print("2. Delete document")

        print("3. Exit")


        choice = input(
            "\nChoose option: "
        ).strip()


        if choice == "1":

            show_documents()


        elif choice == "2":

            delete_document_interactive()


        elif choice == "3":

            print(
                "\nExiting..."
            )

            break


        else:

            print(
                "\nInvalid option."
            )


if __name__ == "__main__":

    main()
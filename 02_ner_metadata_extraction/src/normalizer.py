class EntityNormalizer:

    def normalize(self, text, label):

        value = text.strip().lower()

        # Remove extra spaces
        value = " ".join(
            value.split()
        )

        return value
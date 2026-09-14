import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class HospitalLLM:

    def __init__(self):

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Add it to the .env file."
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model = os.getenv(
            "OPENAI_MODEL",
            "gpt-5.6-luna"
        )

    def generate_answer(self, question, context):

        prompt = f"""
You are an internal Hospital SOP Assistant.

Your job is to answer questions using ONLY
the hospital documents provided as context.

Rules:
1. Do not invent hospital procedures.
2. Do not provide medical diagnosis.
3. If the answer is not present in the context,
   clearly say that the information was not found.
4. Give a clear and concise answer.
5. Mention the source document when available.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""

        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        return response.output_text
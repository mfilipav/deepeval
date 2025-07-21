import os
from openai import OpenAI

class QABot:
    def __init__(
        self, 
        model: str = "gpt-4.1-mini", 
        system_prompt: str = "",
    ):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = system_prompt or (
            "You are an AI assistant for a period tracking app. Female users ask you a question "
            "about their cycle, period, sex life and general health advice. "
            "Given the user question `input` and user context `ctx`, answer user question. "
            "Context contains important information about user's menstrual cycle and health and "
            "demogrophic information. Your response should be casual, concise and polite. "
            "For serious medical questions, politely suggest the user to consult a doctor."
        )

    def answer(self, input: str, ctx: str) -> str:
        input_and_ctx = input + " \nContext: " + ctx
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_and_ctx},
        ]
        # print(f"  input_and_ctx: {input_and_ctx}")
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            messages=msgs,
        )

        content = response.choices[0].message.content.strip()
        return content


from ollama import Client


class LLMAdapter:
    def __init__(self, client: Client, model: str = "mistral"):
        self._client = client
        self._model = model
 
    def call(self, prompt: str) -> str:
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
import os

from ollama import Client
from openai import OpenAI
from litellm import completion

class OllamaLLMAdapter:
    def __init__(self, host: str, model: str):
        self._client = Client(host=host)
        self._model = model

    def call(self, prompt: str) -> str:
        response = self._client.chat(
            model=self._model,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return response['message']['content']

class LiteLLMAdapter:
    def __init__(self, host: str, model: str, api_key: str):
        self._client = OpenAI(api_key=api_key, base_url=host)
        self._model = model

    def call(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response.choices[0].message.content


def get_llm(provider, model, host, api_key):
    match provider:
        case 'ollama':
            host = host or os.environ.get('AEGIS_OLLAMA_HOST', 'http://localhost:11434')
            return OllamaLLMAdapter(host=host, model=model)
        case 'litellm':
            host = host or os.environ.get('AEGIS_LITELLM_HOST', 'https://litellm.labs.jb.gg/')
            return LiteLLMAdapter(host=host, model=model, api_key=api_key)
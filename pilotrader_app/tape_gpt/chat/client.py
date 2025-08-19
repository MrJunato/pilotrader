# file: tape_gpt/chat/client.py
from typing import List, Dict
from openai import OpenAI
from tape_gpt.config import settings

def get_openai_client() -> OpenAI:
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY não definido.")
    return OpenAI(api_key=settings.OPENAI_API_KEY)

def call_openai(client: OpenAI, model: str, messages: List[Dict], max_tokens: int = 800, temperature: float = 0.1) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content
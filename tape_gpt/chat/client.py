# file: tape_gpt/chat/client.py
from typing import List, Dict

def _render_messages_as_text(messages: List[Dict]) -> str:
    """
    Converte [{'role':..., 'content':...}] em um prompt textual único,
    adequado para a OpenAI Responses API.
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}:\n{content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)

def call_openai(api_key: str, model: str, messages: List[Dict], max_output_tokens: int = 800, temperature: float = 0.1) -> str:
    """
    Chama a OpenAI usando a Responses API (recomendado p/ gpt-4.1-mini).
    """
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não definido.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Pacote 'openai' não instalado. Execute: pip install openai") from e

    client = OpenAI(api_key=api_key)
    prompt = _render_messages_as_text(messages)
    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    return resp.output_text
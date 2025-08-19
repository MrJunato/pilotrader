# file: tape_gpt/chat/client.py
# (mantém o suporte a OpenAI para uso fora do Snowflake; não chama nada externo quando provider=disabled)
from typing import List, Dict
from tape_gpt.config import Settings

def _render_messages_as_text(messages: List[Dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}:\n{content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)

def call_llm(settings: Settings, messages: List[Dict], max_tokens: int = 800, temperature: float = 0.1) -> str:
    if settings.PROVIDER == "disabled":
        raise RuntimeError("LLM está desativado (provider=disabled).")
    elif settings.PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY não definido para provider=openai.")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("Pacote 'openai' não instalado. pip install openai") from e
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        prompt = _render_messages_as_text(messages)
        resp = client.responses.create(
            model=settings.OPENAI_MODEL,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.output_text
    else:
        raise ValueError(f"Provider desconhecido: {settings.PROVIDER}")
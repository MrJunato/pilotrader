# file: tape_gpt/chat/client.py
from typing import List, Dict
from tape_gpt.config import Settings

def _render_messages_as_text(messages: List[Dict]) -> str:
    """
    Converte o formato [{'role':..., 'content':...}] em um prompt textual simples,
    útil para providers que recebem apenas string (Cortex, OpenAI Responses).
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}:\n{content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)

def call_llm(settings: Settings, messages: List[Dict], max_tokens: int = 800, temperature: float = 0.1) -> str:
    """
    Roteia a chamada para o provider configurado.
    - cortex: SELECT snowflake.cortex.complete(model, prompt)
    - openai: Responses API (modelos gpt-4.1-*). Converte mensagens em texto único.
    """
    if settings.PROVIDER == "cortex":
        # Executa dentro do Streamlit no Snowflake sem EAI
        prompt = _render_messages_as_text(messages)
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        # SQL parametrizado: (model, prompt)
        df = session.sql(
            "select snowflake.cortex.complete(?, ?)",
            (settings.MODEL, prompt),
        )
        rows = df.collect()
        return rows[0][0] if rows else ""

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
            model=settings.MODEL,
            input=prompt,
            max_output_tokens=max_tokens,   # equivalente a max_tokens
            temperature=temperature,
        )
        return resp.output_text

    else:
        raise ValueError(f"Provider desconhecido: {settings.PROVIDER}")
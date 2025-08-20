# file: tape_gpt/chat/client.py
from typing import List, Dict, Optional

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

def call_openai(
    api_key: str,
    model: str,
    messages: List[Dict],
    max_output_tokens: int = 800,
    temperature: Optional[float] = None  # agora é opcional e só tentamos se vier
) -> str:
    """
    Chama a OpenAI usando a Responses API.
    Estratégia:
      - Se temperature for fornecido, tenta enviar com temperature.
      - Caso retorne erro 400 por parâmetro não suportado, re-tenta sem temperature.
    """
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não definido.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Pacote 'openai' não instalado. Execute: pip install openai") from e

    client = OpenAI(api_key=api_key)
    prompt = _render_messages_as_text(messages)

    # Monta payload sem 'temperature' por padrão
    base_req = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }

    # 1) Tenta com temperature (se fornecido)
    if temperature is not None:
        try:
            req = dict(base_req)
            req["temperature"] = float(temperature)
            resp = client.responses.create(**req)
            return resp.output_text
        except Exception as e:
            msg = str(e)
            # Fallback se o modelo não suportar 'temperature'
            if "Unsupported parameter" in msg and "'temperature'" in msg:
                # 2) Re-tenta sem temperature
                resp = client.responses.create(**base_req)
                return resp.output_text
            # Outros erros: propaga
            raise

    # Sem temperature: chamada direta
    resp = client.responses.create(**base_req)
    return resp.output_text
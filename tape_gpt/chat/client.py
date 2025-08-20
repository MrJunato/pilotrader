# file: tape_gpt/chat/client.py
from typing import List, Dict, Optional, Any

def _extract_text(resp: Any) -> str:
    # 1) caminho feliz
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    # 2) fallback: varre 'output' tentando blocos textuais
    out_chunks = []
    try:
        output = getattr(resp, "output", None) or []
        for item in output:
            # SDKs costumam expor .text ou .content (lista de blocos)
            t = getattr(item, "text", None)
            if isinstance(t, str) and t.strip():
                out_chunks.append(t)
                continue
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") in ("output_text", "text") and isinstance(block.get("text"), str):
                            out_chunks.append(block["text"])
    except Exception:
        pass
    return "\n".join([c for c in out_chunks if c]).strip()

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
    temperature: Optional[float] = None
) -> str:
    """
    Chama a OpenAI Responses API.
    - Usa input estruturado [{role, content}] em vez de colar tudo em um único texto.
    - Força response_format=text para popular output_text.
    - Para modelos GPT-5, seta reasoning.effort (evita ambiguidade de modo).
    - Faz fallback de extração se output_text vier vazio.
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
        "input": messages,  # <— usa a lista [{role, content}] diretamente
        "max_output_tokens": max_output_tokens,
        "response_format": {"type": "text"},
    }

    # Se for GPT-5, habilita um nível de reasoning explícito (evita variação de layout)
    if isinstance(model, str) and model.startswith("gpt-5"):
        base_req["reasoning"] = {"effort": "medium"}

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

    # 2) Sem temperature: chamada direta
    resp = client.responses.create(**base_req)

    text = getattr(resp, "output_text", None)
    if not text:
        try:
            # Fallback genérico: tenta juntar pedaços textuais do campo 'output'
            chunks = []
            for item in getattr(resp, "output", []) or []:
                t = getattr(item, "text", None) or getattr(item, "content", None)
                if isinstance(t, str):
                    chunks.append(t)
            text = "\n".join(chunks).strip()
        except Exception:
            text = ""
    return text
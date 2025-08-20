# file: tape_gpt/chat/client.py
from typing import List, Dict, Optional, Any
from openai import OpenAI

def _as_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    # objetos do SDK podem ter .model_dump() / .to_dict(), tentamos extrair
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    # fallback: tentar attrs conhecidos manualmente
    d = {}
    for k in ("type", "text", "content", "value"):
        if hasattr(obj, k):
            d[k] = getattr(obj, k)
    return d

def _extract_text(resp: Any) -> str:
    # 1) caminho feliz (Responses API)
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    chunks: List[str] = []

    # 2) Responses API — variantes (cobre 'text', 'output_text' e também 'reasoning')
    outputs = getattr(resp, "output", None) or getattr(resp, "outputs", None) or []
    try:
        for item in outputs:
            item_d = _as_dict(item)
            content = item_d.get("content", []) or []
            for block in content:
                b = _as_dict(block)
                btype = b.get("type")
                if btype in ("output_text", "text", "reasoning"):
                    t = b.get("text")
                    if isinstance(t, str) and t.strip():
                        chunks.append(t.strip())
                    else:
                        td = _as_dict(t)
                        val = td.get("value") or td.get("text")
                        if isinstance(val, str) and val.strip():
                            chunks.append(val.strip())
                if btype == "message":
                    inner = b.get("content", []) or []
                    for sb in inner:
                        sb_d = _as_dict(sb)
                        if sb_d.get("type") in ("output_text", "text", "reasoning"):
                            tt = sb_d.get("text")
                            if isinstance(tt, str) and tt.strip():
                                chunks.append(tt.strip())
                            else:
                                ttd = _as_dict(tt)
                                sval = ttd.get("value") or ttd.get("text")
                                if isinstance(sval, str) and sval.strip():
                                    chunks.append(sval.strip())
    except Exception:
        pass

    # 3) Fallback extra: Chat Completions shape (choices[].message.content)
    try:
        choices = getattr(resp, "choices", None)
        if choices:
            for ch in choices:
                msg = getattr(ch, "message", None)
                if msg:
                    c = getattr(msg, "content", None)
                    if isinstance(c, str) and c.strip():
                        chunks.append(c.strip())
    except Exception:
        pass

    return "\n".join([c for c in chunks if c]).strip()

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
    max_output_tokens: int = 1024,
    temperature: Optional[float] = None
) -> str:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não definido.")

    client = OpenAI(api_key=api_key)
    prompt = _render_messages_as_text(messages)

    # Requests para Responses API
    base_req = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }
    if isinstance(model, str) and model.startswith("gpt-5"):
        base_req["reasoning"] = {"effort": "low"}

    legacy_chat = isinstance(model, str) and model.startswith(("gpt-3.5", "gpt-4", "gpt-4o"))  # 

    def _responses_call() -> str:
        req = dict(base_req)
        if temperature is not None:
            req["temperature"] = float(temperature)
        # Corrigir para SEMPRE usar client.responses.create
        resp = client.responses.create(**req)  # use sempre o client.  
        txt = _extract_text(resp)
        if txt:
            return txt

        # Log de depuração opcional, útil se ainda vier vazio
        try:
            raw = getattr(resp, "model_dump_json", None)
            if callable(raw):
                print(raw(indent=2))
        except Exception:
            pass
        raise RuntimeError("Responses API retornou saída vazia (sem output_text nem blocos textuais).")  # 

    def _chat_completions_call() -> str:
        token_key = "max_completion_tokens" if str(model).startswith("gpt-5") else "max_tokens"
        req = {"model": model, "messages": messages, token_key: max_output_tokens}
        if temperature is not None:
            req["temperature"] = float(temperature)
        resp = client.chat.completions.create(**req)
        txt = _extract_text(resp)
        if txt:
            return txt

        # Se a mensagem vier só com tool_calls (sem content), opcionalmente serialize argumentos:
        try:
            ch0 = getattr(resp, "choices", [None])[0]
            msg = getattr(ch0, "message", None)
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                args_chunks = []
                for t in tool_calls:
                    fn = getattr(t, "function", None)
                    args = getattr(fn, "arguments", None) if fn else None
                    if isinstance(args, str) and args.strip():
                        args_chunks.append(args.strip())
                if args_chunks:
                    return "\n".join(args_chunks)
        except Exception:
            pass

        raise RuntimeError("Chat Completions retornou saída vazia (sem choices/message.content).")  # 

    # 2) EVITAR fallback para Chat quando o modelo é 4.1/5 (Responses-only)
    if isinstance(model, str) and model.startswith(("gpt-4.1", "gpt-5")):
        return _responses_call()

    # 1) Modelos legacy (3.5/4/4o) usam Chat primeiro
    if legacy_chat:
        try:
            return _chat_completions_call()
        except Exception:
            return _responses_call()

    # Default
    try:
        return _responses_call()
    except Exception:
        return _chat_completions_call()
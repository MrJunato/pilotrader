# file: tape_gpt/chat/client.py
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
    """
    - cortex: SELECT snowflake.cortex.complete(model, prompt)
      - tenta configurar cross-region na sessão (se permitido)
      - fallback para modelos alternativos se o escolhido não estiver disponível na região
    - openai: usa Responses API
    """
    if settings.PROVIDER == "cortex":
        prompt = _render_messages_as_text(messages)
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()

        # 1) Tenta habilitar cross region inference na sessão (best-effort)
        if settings.CORTEX_ALLOWED_REGIONS:
            regions_sql = ", ".join(f"'{r}'" for r in settings.CORTEX_ALLOWED_REGIONS)
            try:
                session.sql(f"alter session set CORTEX_REMOTE_INFERENCE_ALLOWED_REGIONS = ({regions_sql})").collect()
            except Exception:
                # Ignora se a conta não suportar esse parâmetro em sessão
                pass

        # 2) Função auxiliar para chamar o modelo
        def _complete_with_model(model_name: str) -> str:
            df = session.sql("select snowflake.cortex.complete(?, ?)", (model_name, prompt))
            rows = df.collect()
            return rows[0][0] if rows else ""

        # 3) Tenta modelo principal; se erro de região, faz fallback
        def _is_region_unavailable(err: Exception) -> bool:
            msg = str(err).lower()
            return "unavailable in your region" in msg or "cross region inference" in msg

        try:
            return _complete_with_model(settings.MODEL)
        except Exception as e:
            if not _is_region_unavailable(e):
                raise
            # tenta fallback em ordem
            for alt in settings.CORTEX_FALLBACK_MODELS:
                try:
                    return _complete_with_model(alt)
                except Exception as e2:
                    if _is_region_unavailable(e2):
                        continue
                    else:
                        raise
            # Se nada funcionou, propaga erro com instrução prática
            raise RuntimeError(
                "Nenhum modelo Cortex da lista está disponível na sua região e cross-region pode não estar habilitado. "
                "Tente: (a) habilitar cross region inference, ou (b) ajustar CORTEX_MODEL/CORTEX_FALLBACK_MODELS via env/secrets."
            ) from e

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
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.output_text

    else:
        raise ValueError(f"Provider desconhecido: {settings.PROVIDER}")
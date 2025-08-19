# file: tape_gpt/config.py
import os
from dataclasses import dataclass

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"   # modelo OpenAI (Responses API)
DEFAULT_CORTEX_MODEL = "llama3-8b"      # ajuste de acordo com sua conta/região

def _get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def _get_from_streamlit_secrets(name: str) -> str:
    try:
        import streamlit as st
        try:
            val = st.secrets[name]
            return str(val) if val is not None else ""
        except Exception:
            return ""
    except Exception:
        return ""

@dataclass(frozen=True)
class Settings:
    PROVIDER: str        # "cortex" ou "openai"
    MODEL: str           # modelo conforme o provider
    OPENAI_API_KEY: str  # usado apenas para provider="openai"

def get_settings() -> Settings:
    # Padrão: "cortex" (para funcionar no Snowflake trial)
    provider = (_get_from_streamlit_secrets("LLM_PROVIDER") or
                _get_env("LLM_PROVIDER", "cortex")).lower()
    if provider not in ("cortex", "openai"):
        provider = "cortex"

    if provider == "openai":
        model = (_get_from_streamlit_secrets("OPENAI_MODEL") or
                 _get_env("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
        api_key = (_get_from_streamlit_secrets("OPENAI_API_KEY") or
                   _get_env("OPENAI_API_KEY", ""))
    else:
        model = (_get_from_streamlit_secrets("CORTEX_MODEL") or
                 _get_env("CORTEX_MODEL", DEFAULT_CORTEX_MODEL))
        api_key = ""

    return Settings(PROVIDER=provider, MODEL=model, OPENAI_API_KEY=api_key)

# Mantém um objeto settings para compatibilidade com importações existentes,
# mas lembre que ele é resolvido em import-time.
settings = get_settings()
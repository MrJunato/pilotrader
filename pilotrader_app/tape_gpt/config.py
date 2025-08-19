# file: tape_gpt/config.py
import os
from dataclasses import dataclass

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"  # altere aqui quando quiser trocar

def _get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def _get_from_streamlit_secrets(name: str) -> str:
    try:
        import streamlit as st
        try:
            # acesso direto, protegido em try/except para não quebrar quando não existir
            val = st.secrets[name]
            return str(val) if val is not None else ""
        except Exception:
            return ""
    except Exception:
        return ""

@dataclass(frozen=True)
class Settings:
    OPENAI_API_KEY: str
    OPENAI_MODEL: str

def get_settings() -> Settings:
    # prioridade: secrets do app -> env -> vazio
    api_key = _get_from_streamlit_secrets("OPENAI_API_KEY") or _get_env("OPENAI_API_KEY", "")
    return Settings(
        OPENAI_API_KEY=api_key,
        OPENAI_MODEL=DEFAULT_OPENAI_MODEL,
    )

settings = get_settings()
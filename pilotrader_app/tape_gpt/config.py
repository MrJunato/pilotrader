# file: tape_gpt/config.py
import os
from dataclasses import dataclass
from typing import List

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

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
    PROVIDER: str        # "disabled", "openai" (padrão: disabled no Snowflake trial)
    OPENAI_MODEL: str
    OPENAI_API_KEY: str

def get_settings() -> Settings:
    provider = (_get_from_streamlit_secrets("LLM_PROVIDER") or
                _get_env("LLM_PROVIDER", "disabled")).lower()
    if provider not in ("disabled", "openai"):
        provider = "disabled"

    openai_model = (_get_from_streamlit_secrets("OPENAI_MODEL") or
                    _get_env("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    api_key = (_get_from_streamlit_secrets("OPENAI_API_KEY") or
               _get_env("OPENAI_API_KEY", ""))

    return Settings(
        PROVIDER=provider,
        OPENAI_MODEL=openai_model,
        OPENAI_API_KEY=api_key,
    )

settings = get_settings()
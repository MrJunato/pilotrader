# file: tape_gpt/config.py
import os
from dataclasses import dataclass
from typing import List

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_CORTEX_MODEL = "llama3-8b"  # ajuste conforme disponibilidade
DEFAULT_CORTEX_FALLBACKS = "llama3-70b,mistral-large,mistral-7b"

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

def _parse_csv_list(val: str) -> List[str]:
    return [x.strip() for x in val.split(",") if x.strip()]

@dataclass(frozen=True)
class Settings:
    PROVIDER: str            # "cortex" ou "openai"
    MODEL: str               # modelo conforme o provider
    OPENAI_API_KEY: str      # apenas para provider="openai"
    CORTEX_ALLOWED_REGIONS: List[str]  # ex.: ["ANY_REGION"] ou ["AWS_US","AZURE_US"]
    CORTEX_FALLBACK_MODELS: List[str]  # ordem de tentativa

def get_settings() -> Settings:
    provider = (_get_from_streamlit_secrets("LLM_PROVIDER") or
                _get_env("LLM_PROVIDER", "cortex")).lower()
    if provider not in ("cortex", "openai"):
        provider = "cortex"

    if provider == "openai":
        model = (_get_from_streamlit_secrets("OPENAI_MODEL") or
                 _get_env("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
        api_key = (_get_from_streamlit_secrets("OPENAI_API_KEY") or
                   _get_env("OPENAI_API_KEY", ""))
        allowed_regions = []
        fallbacks = []
    else:
        model = (_get_from_streamlit_secrets("CORTEX_MODEL") or
                 _get_env("CORTEX_MODEL", DEFAULT_CORTEX_MODEL))
        api_key = ""
        allowed_regions = _parse_csv_list(
            _get_from_streamlit_secrets("CORTEX_ALLOWED_REGIONS") or
            _get_env("CORTEX_ALLOWED_REGIONS", "ANY_REGION")
        )
        fallbacks = _parse_csv_list(
            _get_from_streamlit_secrets("CORTEX_FALLBACK_MODELS") or
            _get_env("CORTEX_FALLBACK_MODELS", DEFAULT_CORTEX_FALLBACKS)
        )

    return Settings(
        PROVIDER=provider,
        MODEL=model,
        OPENAI_API_KEY=api_key,
        CORTEX_ALLOWED_REGIONS=allowed_regions,
        CORTEX_FALLBACK_MODELS=fallbacks,
    )

settings = get_settings()
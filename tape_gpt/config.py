# file: tape_gpt/config.py
import os
from dataclasses import dataclass

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"  # modelo padrão para Responses API

def _get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def _get_from_streamlit_secrets(name: str) -> str:
    try:
        import streamlit as st
        try:
            val = st.secrets.get(name, "")
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
    # prioridade: secrets -> env -> vazio
    api_key = _get_from_streamlit_secrets("OPENAI_API_KEY") or _get_env("OPENAI_API_KEY", "")
    model = _get_from_streamlit_secrets("OPENAI_MODEL") or _get_env("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    return Settings(OPENAI_API_KEY=api_key, OPENAI_MODEL=model)

def require_openai_api_key() -> str:
    """
    Fallback interativo: se não houver secret/env, pede a chave via UI.
    Use no início do app para garantir a chave na sessão.
    """
    key = _get_from_streamlit_secrets("OPENAI_API_KEY") or _get_env("OPENAI_API_KEY", "")
    try:
        import streamlit as st
        if not key:
            key = st.session_state.get("OPENAI_API_KEY", "")
        if not key:
            st.info("Informe sua OpenAI API Key. Ela não será persistida em disco; ficará apenas na sessão.")
            st.session_state["OPENAI_API_KEY"] = st.text_input(
                "OpenAI API Key", type="password", value="", key="__openai_api_key_input"
            )
            st.stop()  # interrompe até o usuário preencher
        return key or st.session_state["OPENAI_API_KEY"]
    except Exception:
        return key
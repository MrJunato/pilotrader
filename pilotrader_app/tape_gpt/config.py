# file: tape_gpt/config.py
import os
from dataclasses import dataclass

# tenta ler de st.secrets quando em execução no Streamlit Cloud
try:
    import streamlit as st
    _SECRETS = st.secrets
except Exception:
    _SECRETS = {}

def get_secret(name: str, default: str = "") -> str:
    # prioridade: st.secrets -> env var -> default
    return _SECRETS.get(name, os.getenv(name, default))

@dataclass(frozen=True)
class Settings:
    OPENAI_API_KEY: str = get_secret("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = get_secret("OPENAI_MODEL", "gpt-4.1-mini")  # ajuste se quiser outro modelo

settings = Settings()
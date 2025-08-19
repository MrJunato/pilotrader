# file: tape_gpt/config.py
import os
from dataclasses import dataclass

# tenta usar st.secrets (se existir) e env var como fallback
try:
    import streamlit as st
    _SECRETS = st.secrets
except Exception:
    _SECRETS = {}

def _get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def _get_from_snowflake_secret(secret_fqn: str) -> str:
    """
    Lê um SECRET do Snowflake via SYSTEM$GET_SECRET('DB.SCHEMA.SECRET_NAME').
    Para TYPE=GENERIC_STRING, retorna 'secret_string'.
    Para TYPE=PASSWORD, retorna 'password'.
    """
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        row = session.sql(f"select system$get_secret('{secret_fqn}')").collect()[0]
        data = row[0]  # Variant -> dict
        return data.get("secret_string") or data.get("password") or ""
    except Exception:
        return ""

def _get_secret(name: str, default: str = "") -> str:
    # prioridade: st.secrets -> env var
    return _SECRETS.get(name, _get_env(name, default))

# FQN do secret no Snowflake (ajuste conforme você criou no SQL)
DEFAULT_OPENAI_SECRET_FQN = _get_env("OPENAI_SECRET_FQN", "SECURE_CFG.SECRETS.OPENAI_API")

@dataclass(frozen=True)
class Settings:
    # tenta primeiro UI/ENV; se vazio, busca no Snowflake Secret
    OPENAI_API_KEY: str = ""  # preenchido abaixo
    OPENAI_MODEL: str = _get_secret("OPENAI_MODEL", "gpt-4.1-mini")

# instancia e resolve API key
settings = Settings()
if not settings.OPENAI_API_KEY:
    # 1) tenta st.secrets / env
    api_key = _get_secret("OPENAI_API_KEY", "")
    # 2) se não tiver, busca no Snowflake Secret
    if not api_key:
        api_key = _get_from_snowflake_secret(DEFAULT_OPENAI_SECRET_FQN)
    # reconstroi Settings com a chave resolvida
    settings = Settings(OPENAI_API_KEY=api_key, OPENAI_MODEL=settings.OPENAI_MODEL)
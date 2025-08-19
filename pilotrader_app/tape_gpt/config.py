# file: tape_gpt/config.py
import os
import json
from dataclasses import dataclass

# Modelo fixo
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

def _get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def _get_from_streamlit_secrets(name: str) -> str:
    try:
        import streamlit as st
        try:
            val = st.secrets[name]  # acesso direto evita parse global
            return str(val) if val is not None else ""
        except Exception:
            return ""
    except Exception:
        return ""

def _lower_keys(d: dict) -> dict:
    return {str(k).lower(): v for k, v in d.items()}

def _parse_secret_payload(obj) -> str:
    # Aceita dict com chaves variadas (SECRET_STRING, secretString, secret_string) e PASSWORD
    if isinstance(obj, dict):
        dl = _lower_keys(obj)
        return (dl.get("secret_string")
                or dl.get("secretstring")
                or dl.get("password")
                or "")
    if isinstance(obj, str):
        try:
            d = json.loads(obj)
            return _parse_secret_payload(d)
        except Exception:
            return ""
    # Alguns tipos Variant podem se comportar como mapeáveis
    try:
        return _parse_secret_payload(dict(obj))
    except Exception:
        return ""

def _get_from_snowflake_secret(secret_fqn: str) -> str:
    if not secret_fqn:
        return ""
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        # Retorna VARIANT; em alguns casos é melhor serializar para JSON
        row = session.sql(f"select system$get_secret('{secret_fqn}')").collect()[0]
        payload = row[0]
        return _parse_secret_payload(payload)
    except Exception:
        return ""

def resolve_openai_key(secret_fqn_default: str) -> str:
    # 1) ENV
    val = _get_env("OPENAI_API_KEY", "")
    if val:
        return val
    # 2) st.secrets
    val = _get_from_streamlit_secrets("OPENAI_API_KEY")
    if val:
        return val
    # 3) Snowflake Secret
    fqn = _get_env("OPENAI_SECRET_FQN", secret_fqn_default)
    return _get_from_snowflake_secret(fqn) or ""

@dataclass(frozen=True)
class Settings:
    OPENAI_API_KEY: str
    OPENAI_MODEL: str

def get_settings() -> Settings:
    # ATENÇÃO: Ajuste o FQN abaixo para o nome exato do Secret que você criou no Snowflake
    secret_fqn_default = "SECURE_CFG.SECRETS.OPENAI_API"  # troque se necessário
    api_key = resolve_openai_key(secret_fqn_default)
    return Settings(
        OPENAI_API_KEY=api_key,
        OPENAI_MODEL=DEFAULT_OPENAI_MODEL,
    )

settings = get_settings()
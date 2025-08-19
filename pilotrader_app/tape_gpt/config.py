# file: tape_gpt/config.py
import os
import json
from dataclasses import dataclass

# ========== Helpers para resolver a OPENAI_API_KEY ==========
def _get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def _get_from_streamlit_secrets(name: str) -> str:
    try:
        import streamlit as st
        try:
            # acesso direto evita parse global de secrets inexistentes
            val = st.secrets[name]
            return str(val) if val is not None else ""
        except Exception:
            return ""
    except Exception:
        return ""

def _get_from_snowflake_secret(secret_fqn: str) -> str:
    """Lê SECRET do Snowflake via SYSTEM$GET_SECRET('DB.SCHEMA.NAME')."""
    if not secret_fqn:
        return ""
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        row = session.sql(f"select system$get_secret('{secret_fqn}')").collect()[0]
        data = row[0]  # VARIANT -> dict/json
        if isinstance(data, dict):
            return data.get("secret_string") or data.get("password") or ""
        if isinstance(data, str):
            try:
                d = json.loads(data)
                return d.get("secret_string") or d.get("password") or ""
            except Exception:
                return ""
        try:
            d = dict(data)
            return d.get("secret_string") or d.get("password") or ""
        except Exception:
            return ""
    except Exception:
        return ""

def resolve_value(name: str, default: str = "", snowflake_fqn_env: str | None = None, snowflake_fqn_default: str | None = None) -> str:
    # 1) ENV
    val = _get_env(name, "")
    if val:
        return val
    # 2) st.secrets
    val = _get_from_streamlit_secrets(name)
    if val:
        return val
    # 3) Snowflake Secret
    fqn = _get_env(snowflake_fqn_env or f"{name}_FQN", snowflake_fqn_default or "")
    if fqn:
        val = _get_from_snowflake_secret(fqn)
        if val:
            return val
    return default

# ========== Config do app ==========
# Modelo fixo em código — altere aqui quando quiser mudar
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

@dataclass(frozen=True)
class Settings:
    OPENAI_API_KEY: str
    OPENAI_MODEL: str

def get_settings() -> Settings:
    # Se você criou o secret com outro FQN, ajuste abaixo ou defina OPENAI_SECRET_FQN no app
    default_openai_fqn = os.getenv("OPENAI_SECRET_FQN", "SECURE_CFG.SECRETS.OPENAI_API")
    api_key = resolve_value("OPENAI_API_KEY", default="", snowflake_fqn_default=default_openai_fqn)
    return Settings(
        OPENAI_API_KEY=api_key,
        OPENAI_MODEL=DEFAULT_OPENAI_MODEL,  # fixo
    )

settings = get_settings()
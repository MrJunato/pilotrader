# file: tape_gpt/chat/chat_ui.py
import streamlit as st
from datetime import datetime
from typing import Optional, List, Dict

# Reuso dos módulos existentes
from tape_gpt.chat.prompts import assemble_messages
from tape_gpt.chat.client import call_openai
from tape_gpt.chat.summarizer import summarize_chat
from tape_gpt.data.preprocess import preprocess_ts, compute_imbalances
from tape_gpt.analysis.rule_based import analyze_tape

# Helpers de snapshot (migram de app.py para cá)
def _freeze_chat_snapshot(df_trades, offers_df=None):
    st.session_state.chat_frozen = True
    st.session_state.chat_snapshot_df = df_trades.copy() if df_trades is not None else None
    st.session_state.chat_snapshot_offers = offers_df.copy() if offers_df is not None else None
    st.session_state.chat_frozen_at = datetime.utcnow().isoformat()

def _reset_chat_snapshot():
    st.session_state.chat_frozen = False
    st.session_state.chat_snapshot_df = None
    st.session_state.chat_snapshot_offers = None
    st.session_state.chat_frozen_at = None
    st.session_state.chat_history = []
    st.session_state.chat_summary = ""

def _ensure_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # lista de {"user": "...", "assistant": "..."}
    if "chat_summary" not in st.session_state:
        st.session_state.chat_summary = ""
    if "chat_frozen" not in st.session_state:
        st.session_state.chat_frozen = False
        st.session_state.chat_snapshot_df = None
        st.session_state.chat_snapshot_offers = None
        st.session_state.chat_frozen_at = None

def render_chat_ui(
    *,
    uploaded_df,
    offers_df,
    settings,
    openai_api_key: str,
    max_history: int = 8
):
    _ensure_state()

    st.header("TapeGPT — Chatbot")
    if st.session_state.chat_frozen and st.session_state.chat_frozen_at:
        st.info(f"Contexto congelado em: {st.session_state.chat_frozen_at}")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Reiniciar conversa (nova foto)"):
                _reset_chat_snapshot()
                st.rerun()
        with col_b:
            if st.button("Descongelar (usar dados atuais)"):
                st.session_state.chat_frozen = False
                st.session_state.chat_snapshot_df = None
                st.session_state.chat_snapshot_offers = None
                st.session_state.chat_frozen_at = None
                st.rerun()

    # Render do histórico no estilo chat (auto-scrolling nativo)
    for turn in st.session_state.chat_history[-max_history:]:
        st.chat_message("user").write(turn["user"])
        st.chat_message("assistant").write(turn["assistant"])

    # Entrada do usuário no rodapé
    user_input = st.chat_input("Pergunte ao TapeGPT")
    if not user_input:
        return

    # 1) Congelar snapshot no envio (como já é feito hoje)
    if uploaded_df is not None:
        _freeze_chat_snapshot(uploaded_df, offers_df)

    # 2) Contexto do chat baseado no snapshot (se existir)
    df_chat = st.session_state.chat_snapshot_df if st.session_state.chat_snapshot_df is not None else uploaded_df
    insights_chat = None
    df_text = None
    if df_chat is not None and len(df_chat) > 0:
        df_proc = preprocess_ts(df_chat)
        imbs_chat = compute_imbalances(df_proc, window="1min")
        insights_chat = analyze_tape(df_proc, imbs_chat, freq="1min")
        try:
            last = df_chat.tail(200).to_csv(index=False)
            df_text = "Últimos trades (CSV heads):\n" + last[:10000]
        except Exception:
            pass

    # 3) Montagem de mensagens + chamada do modelo (reuso do pipeline atual)
    history_msgs: List[Dict] = []
    for turn in st.session_state.chat_history[-settings.MAX_HISTORY:]:
        history_msgs.append({"role": "user", "content": turn["user"]})
        history_msgs.append({"role": "assistant", "content": turn["assistant"]})

    messages = assemble_messages(
        user_text=user_input,
        df_sample_text=df_text,
        rule_based=insights_chat,
        history=history_msgs,
        chat_summary=st.session_state.chat_summary or None,
    )

    with st.spinner("Consultando modelo..."):
        try:
            assistant_text = call_openai(
                api_key=openai_api_key,
                model=settings.OPENAI_MODEL,
                messages=messages,
            )
        except Exception as e:
            st.error(f"Erro ao chamar a API: {e}")
            assistant_text = "Falha ao consultar o modelo."

    # 4) Atualiza histórico e resumo
    st.session_state.chat_history.append({"user": user_input, "assistant": assistant_text})
    try:
        # Concatena turns no formato esperado pelo summarizer
        hist_for_sum = []
        for t in st.session_state.chat_history[-(max_history*2):]:
            hist_for_sum.append({"role": "user", "content": t["user"]})
            hist_for_sum.append({"role": "assistant", "content": t["assistant"]})
        st.session_state.chat_summary = summarize_chat(
            api_key=openai_api_key,
            model=settings.OPENAI_MODEL,
            history=hist_for_sum,
            prior_summary=st.session_state.chat_summary or None,
            insights=insights_chat
        )
    except Exception:
        pass

    # 5) Escreve a última troca (aparece imediatamente no topo visual do chat)
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(assistant_text)
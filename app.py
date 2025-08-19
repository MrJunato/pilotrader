# file: app.py
import streamlit as st
import pandas as pd
from datetime import datetime

from tape_gpt.config import get_settings, require_openai_api_key
from tape_gpt.data.loaders import load_csv_ts, parse_profit_excel
from tape_gpt.data.preprocess import preprocess_ts, compute_imbalances
from tape_gpt.viz.charts import candle_volume_figure, buy_sell_imbalance_figures
from tape_gpt.chat.prompts import build_system_prompt, assemble_messages
from tape_gpt.chat.client import call_openai
from tape_gpt.chat.rule_based import analyze_tape, render_response

st.set_page_config(page_title="TapeGPT — Chatbot Tape Reading & TA", layout="wide")

# Config
settings = get_settings()
# Se a chave não estiver definida em env/secrets, pede via UI
openai_api_key = settings.OPENAI_API_KEY or require_openai_api_key()

# Cabeçalho
st.title("TapeGPT — Chatbot de Tape Reading e Análise Técnica (Day Trade)")
st.markdown("""
**Atenção:** Esta ferramenta é apenas para suporte à decisão. Não há garantias de resultado.
Teste em conta demo / backtest antes de operar ao vivo.
""")
st.caption(f"Modelo: {settings.OPENAI_MODEL}")

# Sidebar: upload / conexões
st.sidebar.header("Dados de mercado")
data_source = st.sidebar.selectbox(
    "Fonte de dados",
    ["Upload Excel (Profit Times in Trade)", "Conexão WebSocket (placeholder)"]
)

uploaded_df = None
offers_df = None

if data_source == "Upload Excel (Profit Times in Trade)":
    excel_file = st.sidebar.file_uploader("Envie XLSX do Profit (abas: ofertas, negocios)", type=["xlsx"])
    if excel_file:
        try:
            df_trades, df_offers = parse_profit_excel(excel_file)
            uploaded_df = df_trades
            offers_df = df_offers
            st.sidebar.success(f"XLSX carregado: {uploaded_df.shape[0]} negócios")
        except Exception as e:
            st.sidebar.error(f"Falha ao ler XLSX do Profit: {e}")
else:
    st.sidebar.info("WebSocket: inserir código do provedor e credenciais aqui. Atualmente é um placeholder.")

# Mostra / usa dados
imbs = None
if uploaded_df is not None:
    df = preprocess_ts(uploaded_df)

    agg_unit = st.selectbox("Agregação para plot (resolução)", ["1s","5s","15s","1min"])
    freq = agg_unit

    # --- Nova análise automática pela IA (rule_based) ---
    insights = analyze_tape(df, imbs, freq=freq)

    # --- Indicador visual principal antes dos gráficos ---
    def main_signal_indicator(insights):
        trend = insights.get("trend", "indefinida")
        reversal = insights.get("reversal_detected", False)
        imb = insights.get("imb_last", 0.0)
        # Definição do status
        if trend == "alta" and imb > 0.1 and not reversal:
            label = "Possível Alta"
            color = "green"
            icon = "⬆️"
            help_text = "O tape reading indica tendência de alta. Evite comprar no topo, prefira esperar correções."
        elif trend == "baixa" and imb < -0.1 and not reversal:
            label = "Possível Queda"
            color = "red"
            icon = "⬇️"
            help_text = "O tape reading indica tendência de baixa. Evite operar comprado, prefira esperar repiques."
        elif reversal:
            label = "Atenção: Possível Reversão"
            color = "orange"
            icon = "⚠️"
            help_text = "Há sinais de reversão. Evite operar até o mercado mostrar direção clara."
        elif trend == "lateral":
            label = "Estagnação / Lateralização"
            color = "gray"
            icon = "⏸️"
            help_text = "Mercado sem direção clara. O melhor é não operar ou usar posições pequenas."
        else:
            label = "Cenário Indefinido"
            color = "gray"
            icon = "❔"
            help_text = "Não há sinais claros no tape reading. Prefira não operar."
        st.markdown(
            f"<div style='border-radius:8px;padding:16px;background-color:{'rgba(0,200,0,0.08)' if color=='green' else 'rgba(200,0,0,0.08)' if color=='red' else 'rgba(255,165,0,0.08)' if color=='orange' else '#f0f0f0'};display:flex;align-items:center;'>"
            f"<span style='font-size:2em;margin-right:16px'>{icon}</span>"
            f"<span style='font-size:1.3em;font-weight:bold;color:{color}'>{label}</span>"
            f"</div>"
            f"<div style='font-size:1em;color:#666;margin-top:4px'>{help_text}</div>",
            unsafe_allow_html=True
        )

    main_signal_indicator(insights)

    # --- Gráficos ---
    fig_candle = candle_volume_figure(df, freq=freq)
    st.subheader("Gráfico de candles (agregação) e volume")
    st.plotly_chart(fig_candle, use_container_width=True)

    imbs = compute_imbalances(df, window=freq)
    fig_bs, fig_imb = buy_sell_imbalance_figures(imbs)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Volume Buy/Sell")
        st.plotly_chart(fig_bs, use_container_width=True)
    with col2:
        st.subheader("Imbalance")
        st.plotly_chart(fig_imb, use_container_width=True)

    st.subheader("Análise automática (heurística) dos dados")
    st.markdown(render_response(insights))

else:
    st.info("Carregue um XLSX do Profit para começar.")

# === Chat interface ===
st.sidebar.header("Chatbot")
user_name = st.sidebar.text_input("Seu nome (opcional)", value="Trader")
max_history = st.sidebar.slider("Mensagens de contexto (max)", min_value=2, max_value=20, value=8)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown('---')
st.subheader("Chat — pergunte ao especialista")

# Exibe histórico
for turn in st.session_state.chat_history[-max_history:]:
    with st.chat_message("user"):
        st.markdown(f"**{user_name}:** {turn['user']}")
    with st.chat_message("assistant"):
        st.markdown(f"**TapeGPT:** {turn['assistant']}")

# Entrada do usuário
user_input = st.chat_input("Digite sua mensagem e pressione Enter...")

if user_input:
    if not user_input.strip():
        st.warning("Escreva uma pergunta.")
    else:
        df_text = None
        if uploaded_df is not None:
            try:
                last = uploaded_df.tail(200).to_csv(index=False)
                df_text = "Últimos trades (CSV heads):\n" + last[:10000]
            except Exception:
                pass

        messages = assemble_messages(user_input, df_sample_text=df_text, system_prompt=build_system_prompt())

        with st.spinner("Consultando modelo..."):
            try:
                assistant_text = call_openai(
                    api_key=openai_api_key,
                    model=settings.OPENAI_MODEL,
                    messages=messages,
                )
            except Exception as e:
                st.error(f"Erro ao chamar a API: {e}")
                assistant_text = None

        if assistant_text:
            st.session_state.chat_history.append({
                "user": user_input,
                "assistant": assistant_text,
                "time": datetime.utcnow().isoformat()
            })
            st.rerun()

# Footer
st.markdown("---")
st.markdown("Entrada esperada: XLSX do Profit (abas `ofertas` e `negocios`).")
# file: app.py
import streamlit as st
import pandas as pd
from datetime import datetime
from tape_gpt.viz.indicators import render_main_signal_indicator
from tape_gpt.config import get_settings, require_openai_api_key
from tape_gpt.data.loaders import parse_profit_excel
from tape_gpt.data.preprocess import preprocess_ts, compute_imbalances
from tape_gpt.viz.charts import candle_volume_figure, buy_sell_imbalance_figures
from tape_gpt.viz.charts import top_aggressors_figure
from tape_gpt.analysis.orderflow import top_aggressors
from tape_gpt.chat.prompts import build_system_prompt, assemble_messages
from tape_gpt.chat.client import call_openai
from tape_gpt.analysis.rule_based import analyze_tape, render_response

st.set_page_config(page_title="TapeGPT — Chatbot Tape Reading & TA", layout="wide")

# Config
settings = get_settings()
# Se a chave não estiver definida em env/secrets, pede via UI
openai_api_key = settings.OPENAI_API_KEY or require_openai_api_key()

# Cabeçalho
st.title("Pilotrader — Análise automatizada de Tape Reading")
st.markdown("""
**Atenção:** Esta ferramenta é apenas para suporte à decisão. Não há garantias de resultado.
Teste em conta demo / backtest antes de operar ao vivo.
""")

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

    # 1) Imbalances (agora com aggr_diff/total_volume)
    imbs = compute_imbalances(df, window=freq)

    # 2) Análise heurística com pressão dos agressores
    insights = analyze_tape(df, imbs, freq=freq)

    # 2a) Top agressores (por agente) — usar DF “bruto” pois contém buyer/seller_agent 
    lookback = st.selectbox("Janela Top Agressores", ["10min","30min","60min"], index=1)
    try:
        top_buy_df, top_sell_df = top_aggressors(uploaded_df, lookback=lookback, top_n=5)
        fig_top = top_aggressors_figure(top_buy_df, top_sell_df)
        # Injetar no insights listas resumidas para exibição textual
        insights["top_buy_aggressors"] = list(zip(top_buy_df["agent"].tolist(), top_buy_df["volume"].tolist()))
        insights["top_sell_aggressors"] = list(zip(top_sell_df["agent"].tolist(), top_sell_df["volume"].tolist()))
    except Exception as e:
        fig_top = None
        st.warning(f"Falha ao calcular Top Agressores: {e}")

    # 3) Indicador principal
    render_main_signal_indicator(insights.get("main_signal", {}))  # 

    # 4) Gráficos
    fig_candle = candle_volume_figure(df, freq=freq)
    st.subheader("Gráfico de candles (agregação) e volume")
    st.plotly_chart(fig_candle, use_container_width=True)

    fig_bs, fig_imb = buy_sell_imbalance_figures(imbs)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Volume Buy/Sell")
        st.plotly_chart(fig_bs, use_container_width=True)
    with col2:
        st.subheader("Imbalance")
        st.plotly_chart(fig_imb, use_container_width=True)

    if fig_top is not None:
        st.subheader("Top Agressores (Tape Reading)")
        st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("Análise automática (heurística) dos dados")
    st.markdown(render_response(insights))  # texto com pressão dos agressores + ranking

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
st.caption(f"Modelo: {settings.OPENAI_MODEL}")

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
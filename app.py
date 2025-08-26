# file: app.py
import streamlit as st
import pandas as pd
from datetime import datetime
from tape_gpt.viz.indicators import render_main_signal_indicator
from tape_gpt.config import get_settings, require_openai_api_key
from tape_gpt.data.loaders import parse_profit_excel
from tape_gpt.data.preprocess import preprocess_ts, compute_imbalances
from tape_gpt.viz.charts import candle_volume_figure, buy_sell_imbalance_figures, top_aggressors_figure
from tape_gpt.analysis.orderflow import top_aggressors
from tape_gpt.chat.chat_ui import render_chat_ui
from tape_gpt.analysis.rule_based import analyze_tape, render_response
from tape_gpt.data.simulator import RealTimeSimulator
from streamlit_autorefresh import st_autorefresh
from tape_gpt.viz.order_book import order_book_figure
from tape_gpt.viz.time_sales import time_and_sales_figure

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

# --- Sidebar --- #
st.sidebar.header("Dados de mercado")

# --- Controle de abas (evita flicker no Chat enquanto Painel faz auto-refresh) ---
tab_label = st.sidebar.radio(
    "Navegação",
    options=["📊 Painel", "🤖 Chatbot"],
    index=0,
    help="Selecione a seção"
)
tab = "Painel" if tab_label.startswith("📊") else "Chatbot"

data_source = st.sidebar.selectbox(
    "Fonte de dados",
    ["Upload Excel (Profit Times in Trade)", "Conexão WebSocket (placeholder)", "Simular tempo real"]
)

agg_unit = st.sidebar.selectbox("Agregação para plot (resolução)", ["5s","1s","15s","1min"])
lookback = st.sidebar.selectbox("Janela Top Agressores", ["10min","30min","60min"], index=1)

uploaded_df = None
offers_df = None

#### Fonte 1: Simulador de tempo real
if "sim" not in st.session_state:
    st.session_state.sim = RealTimeSimulator(start_price=100000.0, tick_ms=5000, vol=2.0, max_rows=100)
    try:
        st.session_state.sim.seed_from_profit_xlsx("testes/exemplo_times_in_trade.xlsx")  # ponto de partida
    except Exception as e:
        st.warning(f"Falha ao semear simulador com XLSX: {e}")

# Controles do simulador
if data_source == "Simular tempo real":
    colA, colB, colC = st.sidebar.columns(3)
    with colA:
        running = st.toggle("Rodar", value=st.session_state.sim.is_running(), key="__sim_run_toggle")
    with colB:
        tick_ms = st.number_input("Tick (ms)", min_value=1000, max_value=10000, value=5000, step=500)  # 5s default
    with colC:
        vol = st.number_input("Vol (σ)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

    if running and not st.session_state.sim.is_running():
        st.session_state.sim.tick = tick_ms/1000.0
        st.session_state.sim.vol = float(vol)
        st.session_state.sim.start()
    elif (not running) and st.session_state.sim.is_running():
        st.session_state.sim.stop()
    else:
        st.session_state.sim.tick = tick_ms/1000.0
        st.session_state.sim.vol = float(vol)

    # Coleta dados correntes do simulador e usa o mesmo mapeamento do código atual
    sim_trades, sim_offers = st.session_state.sim.get_dataframes()
    if not sim_trades.empty:
        # NÃO force renomear se já existem as colunas internas
        uploaded_df = sim_trades.copy()
        if "price" not in uploaded_df.columns and "Valor" in uploaded_df.columns:
            uploaded_df = uploaded_df.rename(columns={"Valor": "price"})
        if "volume" not in uploaded_df.columns and "Quantidade" in uploaded_df.columns:
            uploaded_df = uploaded_df.rename(columns={"Quantidade": "volume"})
        offers_df = sim_offers.copy()

#### Fonte 2: Upload XLSX Profit
elif data_source == "Upload Excel (Profit Times in Trade)":
    excel_file = st.sidebar.file_uploader("Envie XLSX do Profit (abas: ofertas, negocios)", type=["xlsx"])
    if excel_file:
        try:
            df_trades, df_offers = parse_profit_excel(excel_file)
            uploaded_df = df_trades
            offers_df = df_offers
            st.sidebar.success(f"XLSX carregado: {uploaded_df.shape[0]} negócios")
        except Exception as e:
            st.sidebar.error(f"Falha ao ler XLSX do Profit: {e}")

#### Fonte 3: WebSocket (placeholder)
else:
    st.sidebar.info("WebSocket: inserir código do provedor e credenciais aqui. Atualmente é um placeholder.")

# ---------------- Snapshot (congelar contexto do chat) ----------------
# Ao enviar uma pergunta, congelaremos um snapshot do DF nesse instante.
if "chat_frozen" not in st.session_state:
    st.session_state.chat_frozen = False
if "chat_snapshot_df" not in st.session_state:
    st.session_state.chat_snapshot_df = None
if "chat_snapshot_offers" not in st.session_state:
    st.session_state.chat_snapshot_offers = None

def freeze_chat_snapshot(df_trades: pd.DataFrame, df_offers: pd.DataFrame | None):
    st.session_state.chat_frozen = True
    st.session_state.chat_snapshot_df = df_trades.copy() if df_trades is not None else None
    st.session_state.chat_snapshot_offers = df_offers.copy() if df_offers is not None else None
    st.session_state.chat_frozen_at = datetime.utcnow().isoformat()

def reset_chat_snapshot():
    st.session_state.chat_frozen = False
    st.session_state.chat_snapshot_df = None
    st.session_state.chat_snapshot_offers = None
    st.session_state.chat_frozen_at = None
    st.session_state.chat_history = []
    st.session_state.chat_summary = ""

# ---------------- Painel de análise/gráficos (tempo real) ----------------
imbs = None
if tab == "Painel":
    if uploaded_df is not None and len(uploaded_df) > 0:
        df = preprocess_ts(uploaded_df)
        freq = agg_unit

        # 1) Imbalances (agora com aggr_diff/total_volume)
        imbs = compute_imbalances(df, window=freq)

        # 2) Análise heurística com pressão dos agressores
        insights = analyze_tape(df, imbs, freq=freq)

        # 2a) Top agressores (por agente) — usar DF “bruto” pois contém buyer/seller_agent 
        try:
            top_buy_df, top_sell_df = top_aggressors(df, lookback=lookback, top_n=5)  # troque uploaded_df -> df  
            fig_top = top_aggressors_figure(top_buy_df, top_sell_df)  # espera cols agent/volume  
            insights["top_buy_aggressors"] = list(zip(top_buy_df["agent"].tolist(), top_buy_df["volume"].tolist()))
            insights["top_sell_aggressors"] = list(zip(top_sell_df["agent"].tolist(), top_sell_df["volume"].tolist()))
        except Exception as e:
            fig_top = None
            st.warning(f"Falha ao calcular Top Agressores: {e}")

        # 3) Indicador principal
        render_main_signal_indicator(insights.get("main_signal", {}))

        # 4) Gráficos
        st.subheader("Gráfico de candles (agregação) e volume")
        fig_candle = candle_volume_figure(df, freq=freq)
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
        st.markdown(render_response(insights))

        # 5) Time & Sales + Book
        st.subheader("Times & Trades")  
        if uploaded_df is not None and len(uploaded_df) > 0:
            st.plotly_chart(time_and_sales_figure(uploaded_df, limit=200), use_container_width=True)
        else:
            st.info("Sem dados de negócios disponíveis.")

        # Auto-refresh apenas quando a aba Painel está ativa
        if data_source == "Simular tempo real" and st.session_state.sim.is_running():
            # Atualiza mais rápido que o tick para dar tempo de redesenhar.
            refresh_ms = max(200, int(st.session_state.sim.tick * 1000 * 0.7))  # ~70% do tick
            st_autorefresh(interval=refresh_ms, key="rt_autorefresh")  # evita queue de updates 【】

    else:
        st.info("Carregue um XLSX ou ative a simulação para começar.")

# ---------------- Chatbot (congela o contexto no envio) ----------------
if tab == "Chatbot":
    render_chat_ui(
        uploaded_df=uploaded_df,
        offers_df=offers_df,
        settings=settings,
        openai_api_key=openai_api_key,
        max_history=8,
    )

# Footer
st.markdown("---")
st.markdown("Entrada esperada: XLSX do Profit (abas `ofertas` e `negocios`).")
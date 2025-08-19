# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go
import re
# opcional: websocket client para conexão com feed de mercado (placeholder)
# from websocket import create_connection

# === Config ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # você pode trocar

# Atualização para nova API OpenAI
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="TapeGPT — Chatbot Tape Reading & TA", layout="wide")

# Cabeçalho
st.title("TapeGPT — Chatbot de Tape Reading e Análise Técnica (Day Trade)")
st.markdown("""
**Atenção:** Esta ferramenta é apenas para *suporte à decisão*. Não há garantias de resultado.
Teste em conta demo / backtest antes de operar ao vivo.
""")

# Sidebar: upload / conexões
st.sidebar.header("Dados de mercado")
data_source = st.sidebar.selectbox("Fonte de dados", ["Upload CSV (time & sales)", "Conexão WebSocket (placeholder)"])

uploaded_df = None
if data_source == "Upload CSV (time & sales)":
    uploaded_file = st.sidebar.file_uploader("Envie CSV time & sales (colunas: timestamp,price,volume,side)", type=["csv"])
    if uploaded_file:
        uploaded_df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        st.sidebar.success(f"CSV carregado: {uploaded_df.shape[0]} linhas")
else:
    st.sidebar.info("WebSocket: inserir código do provedor e credenciais aqui. Atualmente é um placeholder.")

# Small helper to preprocess T&S
def preprocess_ts(df):
    df = df.copy()
    # normaliza nomes
    df.columns = [c.lower() for c in df.columns]
    # tenta converter timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    else:
        st.error("Coluna 'timestamp' não encontrada no CSV.")
    # inferir side se ausente (se tiver price e bid/ask não suportado aqui)
    if "side" not in df.columns:
        df["side"] = "unknown"
    return df

# Mostrar/usar dados
if uploaded_df is not None:
    df = preprocess_ts(uploaded_df)
    st.subheader("Amostra dos dados (time & sales)")
    st.dataframe(df.head(200))

    agg_unit = st.selectbox("Agregação para plot (resolução)", ["1s","5s","15s","1min"])
    if agg_unit.endswith("s"):
        freq = agg_unit
    else:
        freq = agg_unit

    df.set_index("timestamp", inplace=True)
    ohlc = df["price"].resample(freq).ohlc()
    vol = df["volume"].resample(freq).sum()
    merged = ohlc.join(vol)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Gráfico de candles (agregação) e volume")
        # Gráfico interativo de candles + volume
        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(
            x=merged.index,
            open=merged['open'],
            high=merged['high'],
            low=merged['low'],
            close=merged['close'],
            name='Candles'
        ))
        fig_candle.add_trace(go.Bar(
            x=merged.index,
            y=merged['volume'],
            name='Volume',
            marker_color='rgba(0,0,255,0.2)',
            yaxis='y2'
        ))
        fig_candle.update_layout(
            xaxis_title="Tempo",
            yaxis_title="Preço",
            yaxis2=dict(
                title="Volume",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(orientation="h"),
            height=350,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    with col2:
        st.subheader("Indicadores rápidos de Order Flow")
        # exemplo: trade imbalance por janela
        def compute_imbalances(df, window='1min'):
            vbuy = df[df['side'].str.lower()=="buy"].volume.resample(window).sum().fillna(0)
            vsell = df[df['side'].str.lower()=="sell"].volume.resample(window).sum().fillna(0)
            imbalance = (vbuy - vsell) / (vbuy + vsell + 1e-9)
            return pd.DataFrame({"vbuy":vbuy, "vsell":vsell, "imbalance":imbalance})
        try:
            imbs = compute_imbalances(df, window=freq)
            st.markdown("**Volume Buy/Sell**")
            # Gráfico interativo de Buy/Sell
            fig_bs = go.Figure()
            fig_bs.add_trace(go.Scatter(
                x=imbs.index, y=imbs["vbuy"], mode="lines", name="Buy", line=dict(color="green")
            ))
            fig_bs.add_trace(go.Scatter(
                x=imbs.index, y=imbs["vsell"], mode="lines", name="Sell", line=dict(color="red")
            ))
            fig_bs.update_layout(
                xaxis_title="Tempo",
                yaxis_title="Volume",
                legend=dict(orientation="h"),
                height=200,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_bs, use_container_width=True)

            st.markdown("**Imbalance**")
            fig_imb = go.Figure()
            fig_imb.add_trace(go.Scatter(
                x=imbs.index, y=imbs["imbalance"], mode="lines", name="Imbalance", line=dict(color="purple")
            ))
            fig_imb.update_layout(
                xaxis_title="Tempo",
                yaxis_title="Imbalance",
                height=200,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_imb, use_container_width=True)
        except Exception as e:
            st.warning("Não foi possível calcular imbalances: " + str(e))

else:
    st.info("Carregue um CSV de Time & Sales para começar.")

# === Chat interface ===
st.sidebar.header("Chatbot")
user_name = st.sidebar.text_input("Seu nome (opcional)", value="Trader")
max_history = st.sidebar.slider("Mensagens de contexto (max)", min_value=2, max_value=20, value=8)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def build_system_prompt(market="índice"):
    # Prompt de sistema orientador — ajuste conforme necessidade
    return (
        "Você é um especialista em tape reading (time & sales / order flow) e análise técnica intraday"
        " focado em mercados de índice (futuros e/ou CFDs). Responda de forma objetiva, com passos acionáveis,"
        " destaque níveis de suporte/resistência detectados, explique sinais no tape (grandes prints, agressividade"
        " de compradores/vendedores, imbalances) e conecte com contextos de price action e micro-estrutura."
        " Sempre inclua: (1) resumo em 1-2 linhas, (2) sinais observados (o que no tape levou à conclusão),"
        " (3) possíveis trade ideas com gerenciamento de risco (stop e target), e (4) notas sobre incertezas."
        " Não forneça garantias de resultado. Seja conservador e peça mais dados se necessário."
    )

def assemble_messages(user_text, df_sample_text=None):
    system = build_system_prompt()
    messages = [{"role":"system","content":system}]
    # few-shot examples curtos (poder ser estendidos)
    few_shot = [
        {"role":"user","content":"Resumo rápido do tape: houve um print grande comprador no topo da faixa, mas sem follow-through."},
        {"role":"assistant","content":"Resumo: Indecisão; sinal de possível exaustão de venda. Sinais: grande print comprador (volume elevado) numa resistência, mas sem sustentação. Trade idea: esperar pullback para avaliar entradas conservadoras; stop acima do pico do print. Incertezas: não houve confirmação em candles subsequentes."}
    ]
    messages += few_shot
    if df_sample_text:
        messages.append({"role":"user","content":"Aqui estão exemplos de leituras do tape:\n" + df_sample_text})
    messages.append({"role":"user","content": user_text})
    return messages

# Nova interface de chat
st.markdown('---')
st.subheader("Chat — pergunte ao especialista")

# Exibe o histórico de chat como uma tela de chat
for turn in st.session_state.chat_history[-max_history:]:
    with st.chat_message("user"):
        st.markdown(f"**{user_name}:** {turn['user']}")
    with st.chat_message("assistant"):
        st.markdown(f"**TapeGPT:** {turn['assistant']}")

# Campo de entrada de mensagem estilo chat
user_input = st.chat_input("Digite sua mensagem e pressione Enter...")

if user_input:
    if not OPENAI_API_KEY:
        st.error("Defina OPENAI_API_KEY como variável de ambiente antes de rodar.")
    elif not user_input.strip():
        st.warning("Escreva uma pergunta.")
    else:
        df_text = None
        if uploaded_df is not None:
            last = uploaded_df.tail(200).to_csv(index=False)
            df_text = "Últimos trades (CSV heads):\n" + last[:10000]
        messages = assemble_messages(user_input, df_text)
        with st.spinner("Consultando modelo..."):
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    max_tokens=800,
                    temperature=0.1,
                )
                assistant_text = resp.choices[0].message.content
            except Exception as e:
                st.error("Erro ao chamar a API OpenAI: " + str(e))
                assistant_text = None

        if assistant_text:
            st.session_state.chat_history.append({
                "user": user_input,
                "assistant": assistant_text,
                "time": datetime.utcnow().isoformat()
            })
            # Atualiza a tela para mostrar a nova mensagem
            st.rerun()

# Footer / download de exemplo
st.markdown("---")
st.markdown("Exemplo de CSV de time & sales esperado: colunas `timestamp,price,volume,side` (side = 'buy'/'sell').")
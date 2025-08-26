# file: tape_gpt/viz/charts.py
import plotly.graph_objects as go
import pandas as pd

def candle_volume_figure(df: pd.DataFrame, freq: str = "1min") -> go.Figure:
    temp = df.set_index("timestamp")
    ohlc = temp["price"].resample(freq).ohlc()
    vol = temp["volume"].resample(freq).sum()
    merged = ohlc.join(vol)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=merged.index,
        open=merged['open'],
        high=merged['high'],
        low=merged['low'],
        close=merged['close'],
        name='Candles'
    ))
    fig.add_trace(go.Bar(
        x=merged.index,
        y=merged['volume'],
        name='Volume',
        marker_color='rgba(0,0,255,0.2)',
        yaxis='y2'
    ))
    fig.update_layout(
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
    return fig

def buy_sell_imbalance_figures(imbs: pd.DataFrame):
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
        height=250,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    fig_imb = go.Figure()
    fig_imb.add_trace(go.Scatter(
        x=imbs.index, y=imbs["imbalance"], mode="lines", name="Imbalance", line=dict(color="purple")
    ))
    fig_imb.update_layout(
        xaxis_title="Tempo",
        yaxis_title="Imbalance",
        height=250,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig_bs, fig_imb

def top_aggressors_figure(top_buy: pd.DataFrame, top_sell: pd.DataFrame) -> go.Figure:
    tb = top_buy.copy() if top_buy is not None else pd.DataFrame()
    ts = top_sell.copy() if top_sell is not None else pd.DataFrame()

    for col in ("agent", "volume", "trades"):
        if col not in tb.columns:
            tb[col] = []
        if col not in ts.columns:
            ts[col] = []

    tb["agent"]  = tb["agent"].astype(str).fillna("")
    ts["agent"]  = ts["agent"].astype(str).fillna("")
    tb["volume"] = pd.to_numeric(tb["volume"], errors="coerce").fillna(0.0)
    ts["volume"] = pd.to_numeric(ts["volume"], errors="coerce").fillna(0.0)

    ts["volume_plot"] = -ts["volume"]

    y_sell = ts["agent"].tolist()
    x_sell = ts["volume_plot"].to_numpy()
    y_buy  = tb["agent"].tolist()
    x_buy  = tb["volume"].to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Bar(y=y_sell, x=x_sell, orientation="h",
                         name="Agressores de Venda", marker_color="rgba(200,0,0,0.7)"))
    fig.add_trace(go.Bar(y=y_buy, x=x_buy, orientation="h",
                         name="Agressores de Compra", marker_color="rgba(0,160,0,0.7)"))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Volume agredido",
        yaxis_title="Agente",
        legend=dict(orientation="h"),
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(zeroline=True)
    )
    return fig
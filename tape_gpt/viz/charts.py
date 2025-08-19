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
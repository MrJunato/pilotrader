# file: tape_gpt/viz/time_sales.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def time_and_sales_figure(trades_df: pd.DataFrame, limit: int = 150) -> go.Figure:
    if trades_df is None or len(trades_df) == 0:
        return go.Figure()

    cols_needed = {"timestamp","price","volume","side"}
    if not cols_needed.issubset(set(trades_df.columns)):
        return go.Figure()

    df = trades_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False).head(limit).iloc[::-1].reset_index(drop=True)

    # Colunas opcionais (agentes)
    buyer = df["buyer_agent"] if "buyer_agent" in df.columns else ""
    seller = df["seller_agent"] if "seller_agent" in df.columns else ""

    side = df["side"].astype(str).str.lower()
    colors = np.where(side.eq("buy"), "rgba(0,150,0,0.10)", np.where(side.eq("sell"), "rgba(200,0,0,0.10)", "rgba(0,0,0,0.03)"))

    table = pd.DataFrame({
        "Time": df["timestamp"].dt.strftime("%H:%M:%S"),
        "Price": df["price"].map(lambda x: f"{x:.2f}"),
        "Vol": df["volume"].astype(int),
        "Side": side.map({"buy":"BUY","sell":"SELL"}).fillna(""),
        "Buyer": buyer if isinstance(buyer, pd.Series) else pd.Series([""]*len(df)),
        "Seller": seller if isinstance(seller, pd.Series) else pd.Series([""]*len(df)),
    })

    fill_colors = [
        ["rgba(0,0,0,0)"]*len(df),  # Time
        ["rgba(0,0,0,0)"]*len(df),  # Price
        colors,                     # Vol (colore a linha conforme lado, efeito visual rápido)
        colors,                     # Side
        ["rgba(0,0,0,0)"]*len(df),  # Buyer
        ["rgba(0,0,0,0)"]*len(df),  # Seller
    ]

    fig = go.Figure(
        data=[
            go.Table(
                columnorder=[1,2,3,4,5,6],
                columnwidth=[1.0, 1.0, 0.8, 0.8, 1.2, 1.2],
                header=dict(
                    values=["Time", "Price", "Vol", "Side", "Buyer", "Seller"],
                    fill_color="rgba(240,240,240,1)",
                    align="center",
                    font=dict(size=12, color="#333")
                ),
                cells=dict(
                    values=[
                        table["Time"],
                        table["Price"],
                        table["Vol"],
                        table["Side"],
                        table["Buyer"],
                        table["Seller"],
                    ],
                    fill_color=fill_colors,
                    align=["center","right","right","center","left","left"],
                    height=24
                )
            )
        ]
    )
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=28*max(8, len(df)))
    return fig
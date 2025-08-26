# file: tape_gpt/viz/order_book.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def _col(df, *names):
    # Mapeia colunas de ofertas tanto do loader (buy_price, ...) quanto do simulador (bid, ...)
    names = list(names)
    for n in names:
        if n in df.columns:
            return n
    return None

def order_book_figure(offers_df: pd.DataFrame, depth: int = 10) -> go.Figure:
    if offers_df is None or len(offers_df) == 0:
        return go.Figure()

    df = offers_df.copy()

    # Colunas equivalentes (loader vs simulador)
    c_bid_price = _col(df, "bid", "buy_price", "Compra")
    c_ask_price = _col(df, "ask", "sell_price", "Venda")
    c_bid_qty   = _col(df, "qty_bid", "buy_qty", "Qtde_L")
    c_ask_qty   = _col(df, "qty_ask", "sell_qty", "Qtde_V")

    if not all([c_bid_price, c_ask_price, c_bid_qty, c_ask_qty]):
        return go.Figure()

    # Agrega por nível de preço (snapshot típico)
    bids = df[[c_bid_price, c_bid_qty]].dropna()
    asks = df[[c_ask_price, c_ask_qty]].dropna()

    bids = bids.groupby(c_bid_price, as_index=False)[c_bid_qty].sum()
    asks = asks.groupby(c_ask_price, as_index=False)[c_ask_qty].sum()

    # Ordenação estilo book
    bids = bids.sort_values(c_bid_price, ascending=False).head(depth)
    asks = asks.sort_values(c_ask_price, ascending=True).head(depth)

    # Normaliza shapes para montar tabela lado a lado
    max_len = max(len(bids), len(asks))
    bids = bids.reindex(range(max_len)).reset_index(drop=True)
    asks = asks.reindex(range(max_len)).reset_index(drop=True)

    # Renomeia para colunas finais
    table_df = pd.DataFrame({
        "Bid Qty": bids[c_bid_qty].fillna(0).astype("Int64"),
        "Bid Price": bids[c_bid_price],
        "Ask Price": asks[c_ask_price],
        "Ask Qty": asks[c_ask_qty].fillna(0).astype("Int64"),
    })

    # Cores: verde na coluna Bid Qty, vermelho na Ask Qty
    bid_colors = ["rgba(0,150,0,0.10)"] * max_len
    ask_colors = ["rgba(200,0,0,0.10)"] * max_len

    fill_colors = [
        bid_colors,           # Bid Qty
        ["rgba(0,0,0,0)"]*max_len,  # Bid Price
        ["rgba(0,0,0,0)"]*max_len,  # Ask Price
        ask_colors            # Ask Qty
    ]

    fig = go.Figure(
        data=[
            go.Table(
                columnorder=[1,2,3,4],
                columnwidth=[0.9, 1.1, 1.1, 0.9],
                header=dict(
                    values=["Bid Qty", "Bid Price", "Ask Price", "Ask Qty"],
                    fill_color="rgba(240,240,240,1)",
                    align="center",
                    font=dict(size=12, color="#333")
                ),
                cells=dict(
                    values=[
                        table_df["Bid Qty"],
                        table_df["Bid Price"].map(lambda x: f"{x:.2f}" if pd.notna(x) else ""),
                        table_df["Ask Price"].map(lambda x: f"{x:.2f}" if pd.notna(x) else ""),
                        table_df["Ask Qty"],
                    ],
                    fill_color=fill_colors,
                    align=["right","right","left","left"],
                    height=24
                )
            )
        ]
    )
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=28*max(6, max_len))
    return fig
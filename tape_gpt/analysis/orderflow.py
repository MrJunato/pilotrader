# file: tape_gpt/analysis/orderflow.py
import pandas as pd
import numpy as np
from typing import Tuple

def extract_aggressor_trades(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai (timestamp, side, volume, aggressor_agent) por negócio.
    side deve indicar o agressor (já vem do XLSX Profit: agressor -> buy/sell).
    buyer_agent/seller_agent também vêm do loader. 
    """
    df = df_raw.copy()
    # Normalizações defensivas
    for c in ("timestamp","side","volume"):
        if c not in df.columns:
            raise ValueError(f"Coluna '{c}' ausente para extrair agressores.")
    df["side"] = df["side"].astype(str).str.lower()
    if "buyer_agent" not in df.columns and "seller_agent" not in df.columns:
        # não há agentes: não há como ranquear "quem"
        df["aggressor_agent"] = np.nan
        return df[["timestamp","side","volume","aggressor_agent"]]

    buyer = df["buyer_agent"] if "buyer_agent" in df.columns else None
    seller = df["seller_agent"] if "seller_agent" in df.columns else None
    df["aggressor_agent"] = np.where(
        df["side"].eq("buy"),
        buyer if buyer is not None else np.nan,
        np.where(df["side"].eq("sell"), seller if seller is not None else np.nan, np.nan)
    )
    return df[["timestamp","side","volume","aggressor_agent"]]

def top_aggressors(df_raw: pd.DataFrame, lookback: str = "30min", top_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna dois DataFrames (top_buy, top_sell) com colunas: agent, volume, trades.
    """
    dfa = extract_aggressor_trades(df_raw).dropna(subset=["aggressor_agent"])
    dfa = dfa.sort_values("timestamp")
    if lookback:
        end = dfa["timestamp"].max()
        start = end - pd.to_timedelta(lookback)
        dfa = dfa[dfa["timestamp"].between(start, end)]

    grp = dfa.groupby(["side","aggressor_agent"]).agg(
        volume=("volume","sum"),
        trades=("volume","size")
    ).reset_index().sort_values(["side","volume"], ascending=[True, False])

    top_buy  = grp[grp["side"]=="buy"].nlargest(top_n, "volume")[["aggressor_agent","volume","trades"]]
    top_sell = grp[grp["side"]=="sell"].nlargest(top_n, "volume")[["aggressor_agent","volume","trades"]]
    top_buy  = top_buy.rename(columns={"aggressor_agent":"agent"})
    top_sell = top_sell.rename(columns={"aggressor_agent":"agent"})
    return top_buy, top_sell
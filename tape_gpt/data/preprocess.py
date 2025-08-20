# file: tape_gpt/data/preprocess.py
import pandas as pd

def preprocess_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns:
        raise ValueError("Coluna 'timestamp' não encontrada.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # normaliza side para buy/sell/unknown
    if "side" not in df.columns:
        df["side"] = "unknown"
    df["side"] = df["side"].astype(str).str.lower().map(
        lambda s: "buy" if s.startswith("c") or "buy" in s else ("sell" if s.startswith("v") or "sell" in s else "unknown")
    )

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["price","volume"])
    return df

def compute_imbalances(df: pd.DataFrame, window: str = "1min") -> pd.DataFrame:
    temp = df.set_index("timestamp")
    vbuy = temp[temp['side'] == "buy"].volume.resample(window).sum().fillna(0)
    vsell = temp[temp['side'] == "sell"].volume.resample(window).sum().fillna(0)
    imbalance = (vbuy - vsell) / (vbuy + vsell + 1e-9)

    # Diferença absoluta pedida (vendedores - compradores) e escala local
    aggr_diff = (vsell - vbuy)  # positivo => dom. vendedora; negativo => dom. compradora
    total_volume = (vbuy + vsell)

    out = pd.DataFrame({
        "vbuy": vbuy,
        "vsell": vsell,
        "imbalance": imbalance,
        "aggr_diff": aggr_diff,
        "total_volume": total_volume
    })
    out.index.name = "timestamp"
    return out
# file: tape_gpt/data/preprocess.py
import pandas as pd

def preprocess_ts(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["timestamp", "price", "volume"])

    df = df.copy()

    # Padroniza e remove colunas duplicadas (evita DataFrame em df["price"])
    df.columns = [str(c).strip() for c in df.columns]
    if getattr(df.columns, "duplicated", None) is not None and df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # Mapeia nomes comuns -> internos
    col_price_candidates  = ["price", "Preço", "preco", "Valor", "valor", "Price"]
    col_volume_candidates = ["volume", "Volume", "Quantidade", "qty", "Qty", "QTY"]

    price_col = next((c for c in col_price_candidates  if c in df.columns), None)
    vol_col   = next((c for c in col_volume_candidates if c in df.columns), None)

    # Falhas: cria colunas vazias para manter contrato de saída
    if price_col is None:
        df["price"] = pd.NA
    if vol_col is None:
        df["volume"] = pd.NA

    # Se price/volume retornarem DataFrame (colunas duplicadas upstream), reduz para Series
    def _series_or_first(colname: str):
        obj = df[colname]
        if isinstance(obj, pd.DataFrame):  # proteção para pandas com nomes duplicados
            return obj.iloc[:, 0]
        return obj

    price_series = _series_or_first(price_col) if price_col else df["price"]
    vol_series   = _series_or_first(vol_col)   if vol_col   else df["volume"]

    df["price"]  = pd.to_numeric(price_series, errors="coerce")
    df["volume"] = pd.to_numeric(vol_series,   errors="coerce")

    # timestamp: tenta várias convenções
    ts_candidates = ["timestamp", "Timestamp", "datahora", "DataHora", "time", "Time", "datetime", "DateTime"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    else:
        # Se não houver timestamp, cria sequência sintética
        ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="S")
    df["timestamp"] = ts

    # Campos opcionais que o pipeline aproveita
    # (side, buyer_agent, seller_agent podem ou não existir)
    for opt in ["side", "buyer_agent", "seller_agent"]:
        if opt not in df.columns:
            df[opt] = pd.NA

    # Limpa e ordena
    df = df.dropna(subset=["timestamp", "price", "volume"])
    df = df.sort_values("timestamp")
    return df[["timestamp", "price", "volume", "side", "buyer_agent", "seller_agent"]]

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
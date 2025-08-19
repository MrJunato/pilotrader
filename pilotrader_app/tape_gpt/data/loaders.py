# file: tape_gpt/data/loaders.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional

# --- Helpers ---
def _to_numeric(series: pd.Series) -> pd.Series:
    """Converte strings PT-BR (ponto milhar, vírgula decimal) em float."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    # Se tem vírgula decimal, remover pontos de milhar e trocar vírgula por ponto
    has_comma = s.str.contains(",", regex=False)
    s = s.where(~has_comma, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    # Caso contrário, tenta converter direto
    return pd.to_numeric(s, errors="coerce")

def _norm(s: str) -> str:
    s = str(s).strip().lower()
    for a,b in [("á","a"),("à","a"),("ã","a"),("â","a"),("é","e"),("ê","e"),("í","i"),("ó","o"),("ô","o"),("õ","o"),("ú","u"),("ç","c")]:
        s = s.replace(a,b)
    return s

def _find_first_col(df: pd.DataFrame, names) -> Optional[str]:
    targets = set(_norm(n) for n in names)
    for c in df.columns:
        if _norm(c) in targets:
            return c
    return None

# --- Public loaders ---
def load_csv_ts(file) -> pd.DataFrame:
    """Lê CSV com colunas timestamp,price,volume,side."""
    return pd.read_csv(file, parse_dates=["timestamp"])

def parse_profit_excel(file) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Lê XLSX do Profit (Times in Trade) e retorna:
    - df_trades: ['timestamp','price','volume','side','buyer_agent','seller_agent','aggressor']
    - df_offers: ['buyer_agent','buy_qty','buy_price','sell_price','sell_qty','seller_agent'] (opcional)
    """
    # Negócios
    try:
        df_negocios = pd.read_excel(file, sheet_name="negocios", header=1)
    except Exception:
        xls = pd.ExcelFile(file)
        negocios_name = next((s for s in xls.sheet_names if _norm(s).startswith("negoc")), None)
        if not negocios_name:
            raise ValueError("A planilha 'negocios' não foi encontrada.")
        df_negocios = pd.read_excel(file, sheet_name=negocios_name, header=1)

    if df_negocios is None or len(df_negocios) == 0:
        raise ValueError("A planilha 'negocios' está vazia.")

    col_buyer = _find_first_col(df_negocios, ["compradora","comprador","buyer"])
    col_price = _find_first_col(df_negocios, ["valor","preco","preço","price"])
    col_qty   = _find_first_col(df_negocios, ["quantidade","qtd","volume","qty"])
    col_seller= _find_first_col(df_negocios, ["vendedora","vendedor","seller"])
    col_aggr  = _find_first_col(df_negocios, ["agressor","aggressor","lado","side"])

    col_time  = _find_first_col(df_negocios, ["hora","horario","horário","time","timestamp"])
    col_date  = _find_first_col(df_negocios, ["data","date"])
    col_datetime = _find_first_col(df_negocios, ["data/hora","data hora","datetime"])

    df_tr = pd.DataFrame()
    if col_buyer:  df_tr["buyer_agent"]  = df_negocios[col_buyer]
    if col_seller: df_tr["seller_agent"] = df_negocios[col_seller]
    if col_aggr:   df_tr["aggressor"]    = df_negocios[col_aggr]

    if not col_price or not col_qty:
        raise ValueError("Colunas de preço/quantidade não foram encontradas em 'negocios' (esperado: Valor e Quantidade).")

    df_tr["price"]  = _to_numeric(df_negocios[col_price])
    df_tr["volume"] = _to_numeric(df_negocios[col_qty])

    def map_side(x):
        s = _norm(x)
        if s.startswith("c") or "compr" in s or "buy" in s or "bid" in s:
            return "buy"
        if s.startswith("v") or "vend" in s or "sell" in s or "ask" in s:
            return "sell"
        return "unknown"
    df_tr["side"] = df_tr["aggressor"].map(map_side) if "aggressor" in df_tr.columns else "unknown"

    # Timestamp
    ts = None
    if col_datetime:
        ts = pd.to_datetime(df_negocios[col_datetime], errors="coerce")
    elif col_date and col_time:
        ts = pd.to_datetime(
            df_negocios[col_date].astype(str).str.strip() + " " + df_negocios[col_time].astype(str).str.strip(),
            errors="coerce"
        )
    elif col_time:
        # Combina hora com data de hoje (timezone local do servidor)
        today = pd.Timestamp.today().normalize()
        parsed = pd.to_datetime(df_negocios[col_time], errors="coerce")
        fmt = parsed.dt.strftime("%H:%M:%S")
        ts = pd.to_datetime(today.date().isoformat() + " " + fmt, errors="coerce")

    if ts is None or ts.isna().all():
        # fallback sequencial (1s entre trades)
        base = pd.Timestamp.utcnow().floor("s")
        ts = base + pd.to_timedelta(np.arange(len(df_tr)), unit="s")

    df_tr["timestamp"] = ts
    df_tr = df_tr.sort_values("timestamp").reset_index(drop=True)
    df_tr = df_tr[["timestamp","price","volume","side"] + [c for c in ["buyer_agent","seller_agent","aggressor"] if c in df_tr.columns]]

    # Ofertas (opcional)
    df_off = None
    try:
        df_ofertas = pd.read_excel(file, sheet_name="ofertas", header=1)
    except Exception:
        try:
            xls = pd.ExcelFile(file)
            ofertas_name = next((s for s in xls.sheet_names if _norm(s).startswith("oferta")), None)
            df_ofertas = pd.read_excel(file, sheet_name=ofertas_name, header=1) if ofertas_name else None
        except Exception:
            df_ofertas = None

    if df_ofertas is not None and len(df_ofertas) > 0:
        # Estrutura típica: Agente | Qtde | Compra | Venda | Qtde | Agente (lado direito)
        # Duplicatas vêm como Agente, Agente.1 etc. Usamos heurística por nome base/sufixo.
        cols = list(df_ofertas.columns)
        # Lado comprador (primeiro conjunto)
        buyer_agent_col = _find_first_col(df_ofertas, ["agente"])
        buy_qty_col     = _find_first_col(df_ofertas, ["qtde","qtd"])
        buy_price_col   = _find_first_col(df_ofertas, ["compra","preco compra","preço compra","buy","bid"])

        # Lado vendedor (segundo conjunto) — procura as duplicatas remanescentes
        seller_agent_col = None
        sell_qty_col     = None
        sell_price_col   = _find_first_col(df_ofertas, ["venda","preco venda","preço venda","sell","ask"])

        # Identifica duplicatas por sufixo .1, .2
        bases = {}
        for c in cols:
            base = c.split(".")[0]
            bases.setdefault(base, []).append(c)
        # seleciona outra coluna "Agente" diferente da primeira
        if buyer_agent_col and len(bases.get(buyer_agent_col.split(".")[0], [])) > 1:
            for cand in bases[buyer_agent_col.split(".")[0]]:
                if cand != buyer_agent_col:
                    seller_agent_col = cand
                    break
        if buy_qty_col and len(bases.get(buy_qty_col.split(".")[0], [])) > 1:
            for cand in bases[buy_qty_col.split(".")[0]]:
                if cand != buy_qty_col:
                    sell_qty_col = cand
                    break

        df_off = pd.DataFrame()
        if buyer_agent_col:  df_off["buyer_agent"]  = df_ofertas[buyer_agent_col]
        if buy_qty_col:      df_off["buy_qty"]      = _to_numeric(df_ofertas[buy_qty_col])
        if buy_price_col:    df_off["buy_price"]    = _to_numeric(df_ofertas[buy_price_col])
        if sell_price_col:   df_off["sell_price"]   = _to_numeric(df_ofertas[sell_price_col])
        if sell_qty_col:     df_off["sell_qty"]     = _to_numeric(df_ofertas[sell_qty_col])
        if seller_agent_col: df_off["seller_agent"] = df_ofertas[seller_agent_col]

    return df_tr, df_off
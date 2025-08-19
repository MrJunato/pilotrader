# file: tape_gpt/chat/rule_based.py
import numpy as np
import pandas as pd

def _pct(a, b):
    try:
        if b == 0 or np.isnan(a) or np.isnan(b):
            return 0.0
        return 100.0 * (a - b) / b
    except Exception:
        return 0.0

def analyze_tape(df: pd.DataFrame, imbs: pd.DataFrame, freq: str = "1min") -> dict:
    out = {
        "summary": "Dados insuficientes.",
        "trend": "indefinida",
        "price_change_pct": 0.0,
        "imb_last": 0.0,
        "imb_5_sum": 0.0,
        "vbuy_last": 0.0,
        "vsell_last": 0.0,
        "vbuy_5": 0.0,
        "vsell_5": 0.0,
        "big_prints": [],
        "levels": [],
        "volatility": 0.0,
    }
    if df is None or len(df) < 10:
        return out

    df = df.copy()
    df = df.sort_values("timestamp")
    tail_trades = df.tail(min(500, len(df)))

    # Tendência simples por variação de preço nos últimos N negócios
    n = min(200, len(tail_trades))
    p0 = tail_trades["price"].iloc[-n]
    p1 = tail_trades["price"].iloc[-1]
    change_pct = _pct(p1, p0)
    trend = "alta" if change_pct > 0.1 else ("baixa" if change_pct < -0.1 else "lateral")

    # Volatilidade (desvio padrão normalizado) nos últimos N
    volat = float(tail_trades["price"].pct_change().std() * np.sqrt(60.0)) if n > 5 else 0.0

    # Imbalance e volumes por janela
    if imbs is not None and len(imbs) > 0:
        vbuy_last = float(imbs["vbuy"].iloc[-1])
        vsell_last = float(imbs["vsell"].iloc[-1])
        imb_last = float(imbs["imbalance"].iloc[-1])
        take = min(5, len(imbs))
        vbuy_5 = float(imbs["vbuy"].tail(take).sum())
        vsell_5 = float(imbs["vsell"].tail(take).sum())
    else:
        vbuy_last = vsell_last = vbuy_5 = vsell_5 = imb_last = 0.0

    # Prints grandes (percentil 95 no último bloco)
    vols = tail_trades["volume"].astype(float)
    thr = float(np.nanpercentile(vols, 95)) if len(vols) > 10 else float(vols.max())
    big = tail_trades[vols >= thr]
    big_prints = []
    for _, r in big.tail(10).iterrows():
        big_prints.append({
            "ts": r["timestamp"],
            "price": float(r["price"]),
            "volume": float(r["volume"]),
            "side": str(r.get("side", "unknown"))
        })

    # Níveis de S/R por OHLC na agregação escolhida
    try:
        ohlc = df.set_index("timestamp")["price"].resample(freq).ohlc().dropna()
        if len(ohlc) > 3:
            win = ohlc.tail(min(40, len(ohlc)))
            levels = [
                ("resistencia", float(win["high"].max())),
                ("suporte", float(win["low"].min())),
            ]
        else:
            levels = []
    except Exception:
        levels = []

    out.update({
        "summary": f"Tendência {trend} com variação de {change_pct:.2f}%. Imbalance último={imb_last:.2f}.",
        "trend": trend,
        "price_change_pct": change_pct,
        "imb_last": imb_last,
        "vbuy_last": vbuy_last,
        "vsell_last": vsell_last,
        "vbuy_5": vbuy_5,
        "vsell_5": vsell_5,
        "big_prints": big_prints,
        "levels": levels,
        "volatility": volat,
    })
    return out

def render_response(insights: dict, user_text: str = "") -> str:
    # Monta resposta no formato solicitado (resumo, sinais, ideias, incertezas)
    res = []
    res.append(f"Resumo: {insights['summary']}")
    sinais = []
    sinais.append(f"- Volumes (última janela): buy={insights['vbuy_last']:.0f}, sell={insights['vsell_last']:.0f}, imbalance={insights['imb_last']:.2f}")
    sinais.append(f"- Acúmulo (últimas janelas): buy={insights['vbuy_5']:.0f}, sell={insights['vsell_5']:.0f}")
    if insights["big_prints"]:
        bp = insights["big_prints"][-1]
        sinais.append(f"- Print grande recente: {bp['side']} vol={bp['volume']:.0f} @ {bp['price']:.2f} ({bp['ts']})")
    if insights["levels"]:
        lv = ", ".join([f"{t}:{v:.2f}" for t, v in insights["levels"]])
        sinais.append(f"- Níveis: {lv}")
    sinais.append(f"- Volatilidade (est.)={insights['volatility']:.3f}")
    res.append("Sinais observados:\n" + "\n".join(sinais))

    # Ideias de trade (genéricas e condicionais)
    ideas = []
    if insights["trend"] == "alta" and insights["imb_last"] > 0.1:
        ideas.append("- Tendência de alta com desequilíbrio comprador; plano: pullback até suporte próximo e retomada; stop abaixo do suporte; alvo no topo anterior.")
    elif insights["trend"] == "baixa" and insights["imb_last"] < -0.1:
        ideas.append("- Tendência de baixa com desequilíbrio vendedor; plano: pullback até resistência próxima; stop acima da resistência; alvo no fundo anterior.")
    else:
        ideas.append("- Contexto misto/lateral; plano: operar extremos (fade) com stops curtos; aguardar confirmação por volume.")
    res.append("Possíveis trade ideas (com risco):\n" + "\n".join(ideas))

    res.append("Incertezas: análise heurística sem LLM; resultados dependem da qualidade dos dados. Peça confirmações adicionais (ex.: volume por nível, delta por agressão).")
    if user_text:
        res.append(f"Nota: Pergunta do usuário foi considerada no contexto, mas a resposta é heurística: '{user_text[:200]}'")
    return "\n\n".join(res)
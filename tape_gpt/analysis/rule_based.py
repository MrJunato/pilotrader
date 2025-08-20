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
        # Novos campos para insights
        "big_prints_cluster": [],
        "volatility_rel": 0.0,
        "reversal_detected": False,
        "volume_concentration": "",
        "liquidity_comment": "",
        "main_signal": {"label": "Indefinido", "color": "gray", "icon": "❔", "help": ""},
    }
    if df is None or len(df) < 10:
        out["summary"] = "Poucos dados para análise. Evite operar até ter mais informações."
        out["main_signal"] = {
            "label": "Cenário Indefinido",
            "color": "gray",
            "icon": "❔",
            "help": "Não há dados suficientes para análise. Aguarde mais informações antes de operar."
        }
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

    # Volatilidade relativa (vs. últimos 1000 negócios)
    try:
        hist_n = min(1000, len(df))
        hist_vol = float(df["price"].tail(hist_n).pct_change().std() * np.sqrt(60.0))
        volatility_rel = volat / hist_vol if hist_vol > 0 else 1.0
    except Exception:
        volatility_rel = 1.0

    # Imbalance e volumes por janela (corrigido para evitar NaN/0 indevidos)
    if imbs is not None and len(imbs) > 0:
        imbs_valid = imbs.dropna(subset=["vbuy", "vsell", "imbalance"])
        if len(imbs_valid) > 0:
            vbuy_last = float(imbs_valid["vbuy"].iloc[-1]) if not pd.isna(imbs_valid["vbuy"].iloc[-1]) else 0.0
            vsell_last = float(imbs_valid["vsell"].iloc[-1]) if not pd.isna(imbs_valid["vsell"].iloc[-1]) else 0.0
            imb_last = float(imbs_valid["imbalance"].iloc[-1]) if not pd.isna(imbs_valid["imbalance"].iloc[-1]) else 0.0
            take = min(5, len(imbs_valid))
            vbuy_5 = float(imbs_valid["vbuy"].tail(take).sum())
            vsell_5 = float(imbs_valid["vsell"].tail(take).sum())
        else:
            vbuy_last = vsell_last = vbuy_5 = vsell_5 = imb_last = 0.0
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

    # Novos insights: cluster de prints grandes (>=3 em 2min)
    big_prints_cluster = []
    if len(big) >= 3:
        big_sorted = big.sort_values("timestamp")
        times = pd.to_datetime(big_sorted["timestamp"])
        for i in range(len(times) - 2):
            if (times.iloc[i+2] - times.iloc[i]).total_seconds() <= 120:
                cluster = big_sorted.iloc[i:i+3]
                big_prints_cluster.append({
                    "start": str(times.iloc[i]),
                    "end": str(times.iloc[i+2]),
                    "side": cluster["side"].mode()[0] if "side" in cluster else "unknown",
                    "vol_sum": float(cluster["volume"].sum()),
                    "prices": [float(x) for x in cluster["price"]]
                })
                break  # só reporta o primeiro cluster recente

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

    # Detecção de reversão recente (mudança de tendência nos últimos 30 trades)
    reversal_detected = False
    if len(tail_trades) > 30:
        last30 = tail_trades["price"].tail(30)
        if (last30.iloc[-1] > last30.iloc[0] and trend == "baixa") or (last30.iloc[-1] < last30.iloc[0] and trend == "alta"):
            reversal_detected = True

    # Concentração de volume: % do volume em prints grandes
    total_vol = float(tail_trades["volume"].sum())
    big_vol = float(big["volume"].sum())
    if total_vol > 0:
        perc_big = 100.0 * big_vol / total_vol
        if perc_big > 30:
            volume_concentration = f"Alta concentração de volume em prints grandes ({perc_big:.1f}%)"
        elif perc_big < 10:
            volume_concentration = f"Volume disperso, poucos prints grandes ({perc_big:.1f}%)"
        else:
            volume_concentration = f"Volume moderadamente concentrado em prints grandes ({perc_big:.1f}%)"
    else:
        volume_concentration = "Sem dados de volume."

    # Comentário sobre liquidez (volume médio por trade)
    avg_vol = float(tail_trades["volume"].mean())
    if avg_vol > 1000:
        liquidity_comment = f"Liquidez alta (volume médio por trade: {avg_vol:.0f})"
    elif avg_vol < 100:
        liquidity_comment = f"Liquidez baixa (volume médio por trade: {avg_vol:.0f})"
    else:
        liquidity_comment = f"Liquidez moderada (volume médio por trade: {avg_vol:.0f})"

    # Definição do sinal principal para indicador visual
    if trend == "alta" and imb_last > 0.1 and not reversal_detected:
        main_signal = {
            "label": "Possível Alta",
            "color": "green",
            "icon": "⬆️",
            "help": "O tape reading indica tendência de alta. Evite comprar no topo, prefira esperar correções."
        }
    elif trend == "baixa" and imb_last < -0.1 and not reversal_detected:
        main_signal = {
            "label": "Possível Queda",
            "color": "red",
            "icon": "⬇️",
            "help": "O tape reading indica tendência de baixa. Evite operar comprado, prefira esperar repiques."
        }
    elif reversal_detected:
        main_signal = {
            "label": "Atenção: Possível Reversão",
            "color": "orange",
            "icon": "⚠️",
            "help": "Há sinais de reversão. Evite operar até o mercado mostrar direção clara."
        }
    elif trend == "lateral":
        main_signal = {
            "label": "Estagnação / Lateralização",
            "color": "gray",
            "icon": "⏸️",
            "help": "Mercado sem direção clara. O melhor é não operar ou usar posições pequenas."
        }
    else:
        main_signal = {
            "label": "Cenário Indefinido",
            "color": "gray",
            "icon": "❔",
            "help": "Não há sinais claros no tape reading. Prefira não operar."
        }

    out.update({
        "summary": (
            f"Tendência: {trend.upper()} ({'subida' if trend=='alta' else 'queda' if trend=='baixa' else 'lateral'}), variação de {change_pct:.2f}%. "
            f"Imbalance atual: {imb_last:.2f}. "
            f"Volatilidade: {volat:.3f} ({'ALTA' if volatility_rel>1.2 else 'BAIXA' if volatility_rel<0.8 else 'normal'}). "
            f"{volume_concentration}. {liquidity_comment} "
            "⚠️ Lembre-se: proteger seu capital é mais importante que buscar ganhos rápidos."
        ),
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
        "big_prints_cluster": big_prints_cluster,
        "volatility_rel": volatility_rel,
        "reversal_detected": reversal_detected,
        "volume_concentration": volume_concentration,
        "liquidity_comment": liquidity_comment,
        "main_signal": main_signal,
    })
    return out

def render_response(insights: dict, user_text: str = "") -> str:
    res = []
    res.append(f"Resumo: {insights['summary']}")
    sinais = []
    sinais.append(f"- Volume recente: compra={insights['vbuy_last']:.0f}, venda={insights['vsell_last']:.0f}, imbalance={insights['imb_last']:.2f}")
    sinais.append(f"- Acúmulo (últimas janelas): compra={insights['vbuy_5']:.0f}, venda={insights['vsell_5']:.0f}")
    if insights["big_prints"]:
        bp = insights["big_prints"][-1]
        sinais.append(f"- Negócio grande recente: {bp['side']} volume={bp['volume']:.0f} @ {bp['price']:.2f} ({bp['ts']})")
    if insights.get("big_prints_cluster"):
        c = insights["big_prints_cluster"][0]
        sinais.append(f"- Sequência de grandes negócios: {c['side']} total={c['vol_sum']:.0f} entre {c['start']} e {c['end']}")
    if insights["levels"]:
        lv = ", ".join([f"{t}:{v:.2f}" for t, v in insights["levels"]])
        sinais.append(f"- Níveis importantes: {lv}")
    sinais.append(f"- Volatilidade estimada={insights['volatility']:.3f} (relativa: {insights['volatility_rel']:.2f})")
    if insights.get("reversal_detected"):
        sinais.append("- Atenção: possível reversão de tendência nos últimos negócios.")
    if insights.get("volume_concentration"):
        sinais.append(f"- {insights['volume_concentration']}")
    if insights.get("liquidity_comment"):
        sinais.append(f"- {insights['liquidity_comment']}")
    res.append("Sinais observados:\n" + "\n".join(sinais))

    # Recomendações sempre focadas em evitar perdas
    ideas = []
    if insights["trend"] == "alta" and insights["imb_last"] > 0.1:
        ideas.append(
            "O mercado está em tendência de alta, mas evite comprar após grandes subidas. "
            "Espere por uma correção (queda) até um suporte próximo antes de pensar em entrar. "
            "Nunca compre no topo. Use sempre stop-loss abaixo do suporte."
        )
    elif insights["trend"] == "baixa" and insights["imb_last"] < -0.1:
        ideas.append(
            "O mercado está em tendência de baixa. Evite operar comprado. "
            "Se pensar em vender, só faça após um repique (subida temporária) até uma resistência. "
            "Use sempre stop-loss acima da resistência. Não tente adivinhar fundo."
        )
    elif insights.get("reversal_detected"):
        ideas.append(
            "Há sinais de reversão. Não opere contra a tendência sem confirmação clara. "
            "Espere o mercado mostrar direção definida antes de entrar. Prefira não operar em momentos de dúvida."
        )
    elif insights.get("big_prints_cluster"):
        ideas.append(
            "Sequência de prints grandes pode indicar movimento forte. "
            "Evite entrar no meio do movimento. Espere o preço se acalmar e só opere se houver clara relação risco/retorno."
        )
    else:
        ideas.append(
            "Mercado lateral ou indefinido. O melhor é não operar ou usar posições pequenas. "
            "Evite operar em momentos de incerteza. Preserve seu capital."
        )
    res.append("Recomendações para evitar perdas:\n" + "\n".join(ideas))

    res.append(
        "⚠️ Dica importante: Sempre use stop-loss. Nunca opere com todo seu capital. "
        "Se não entender o cenário, prefira não operar. Proteger seu dinheiro é prioridade."
    )
    res.append(
        "Esta análise é baseada em regras simples e pode não captar todos os riscos. "
        "Consulte outras fontes e nunca opere apenas por este resumo."
    )
    if user_text:
        res.append(f"Nota: Sua pergunta foi considerada, mas a resposta é simplificada para facilitar o entendimento: '{user_text[:200]}'")
    return "\n\n".join(res)
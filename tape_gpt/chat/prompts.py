# file: tape_gpt/chat/prompts.py
from typing import List, Dict, Optional

def build_system_prompt(market: str = "índice", rule_based_summary: Optional[str] = None, main_signal: Optional[dict] = None) -> str:
    base = (
        "Você é um especialista em tape reading (leitura do fluxo de ordens) e análise técnica intraday, "
        "focado em mercados de índice (futuros e/ou CFDs). "
        "Seu objetivo é ajudar o usuário a entender o cenário de mercado de forma simples, didática e acessível, "
        "evitando termos técnicos sempre que possível. Se precisar usar termos técnicos, explique o significado de forma clara e curta. "
        "Sempre priorize a proteção do usuário contra perdas e oriente para decisões conservadoras. "
        "Responda com passos práticos e linguagem fácil, como se estivesse explicando para alguém iniciante. "
    )
    if main_signal:
        base += (
            f"\n\nResumo visual da análise automática: {main_signal.get('icon','')} {main_signal.get('label','')} — {main_signal.get('help','')}"
        )
    if rule_based_summary:
        base += (
            f"\n\nResumo detalhado da análise automática:\n{rule_based_summary}"
        )
    base += (
        "\n\nSempre inclua: (1) resumo em 1-2 frases simples, (2) sinais observados (explique o que significa), "
        "(3) possíveis ideias de operação com foco em evitar perdas, e (4) notas sobre incertezas. "
        "Se o usuário perguntar algo, responda de forma didática e sem jargões, explicando o que for necessário."
    )
    return base

def assemble_messages(
    user_text: str,
    df_sample_text: Optional[str] = None,
    system_prompt: Optional[str] = None,
    rule_based: Optional[dict] = None,
    history: Optional[List[Dict]] = None,
) -> List[Dict]:
    # Monta o prompt do sistema com dados do rule_based se disponíveis
    if rule_based:
        system = build_system_prompt(
            rule_based_summary=rule_based.get("summary"),
            main_signal=rule_based.get("main_signal")
        )
    else:
        system = system_prompt or build_system_prompt()

    messages = [{"role": "system", "content": system}]

    # Few-shot didático
    few_shot = [
        {"role": "user", "content": "Resumo rápido do tape: houve um print grande comprador no topo da faixa, mas sem follow-through."},
        {"role": "assistant", "content": "Resumo: Mercado mostrou indecisão, pode ser sinal de exaustão de venda. Sinais: grande negócio de compra (volume alto) numa região de resistência, mas sem continuidade. Ideia: esperar o preço cair um pouco antes de pensar em comprar; sempre use stop-loss. Incerteza: não houve confirmação em candles seguintes. (Obs: 'print grande' significa um negócio de volume muito acima da média, geralmente feito por participantes grandes.)"}
    ]
    messages += few_shot

    # Detalhes adicionais do rule_based no contexto
    if rule_based:
        if rule_based.get("levels"):
            levels = ", ".join([f"{t}: {v:.2f}" for t, v in rule_based["levels"]])
            messages.append({"role": "system", "content": f"Níveis importantes detectados: {levels}"})
        if rule_based.get("big_prints"):
            bp = rule_based["big_prints"][-1]
            messages.append({"role": "system", "content": f"Negócio grande recente: {bp['side']} volume={bp['volume']:.0f} @ {bp['price']:.2f} ({bp['ts']})"})
        
        # top agressores, se presentes no insights (preenchidos no app) 
        tb = rule_based.get("top_buy_aggressors") or []
        ts = rule_based.get("top_sell_aggressors") or []
        if tb:
            topb_txt = ", ".join([f"{a}({v:.0f})" for a, v in tb[:5]])
            messages.append({"role": "system", "content": f"Top agressores de COMPRA: {topb_txt}"})
        if ts:
            tops_txt = ", ".join([f"{a}({v:.0f})" for a, v in ts[:5]])
            messages.append({"role": "system", "content": f"Top agressores de VENDA: {tops_txt}"})

    if df_sample_text:
        messages.append({"role": "user", "content": "Aqui estão exemplos de leituras do tape:\n" + df_sample_text})

    # Histórico do chat (user/assistant) antes da pergunta atual
    if history:
        for h in history:
            # h deve ser {"role": "user"|"assistant", "content": "..."}
            messages.append(h)

    # Por último, a pergunta atual
    messages.append({"role": "user", "content": user_text})
    return messages
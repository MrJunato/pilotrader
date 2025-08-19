# file: tape_gpt/chat/prompts.py
from typing import List, Dict, Optional

def build_system_prompt(market: str = "índice") -> str:
    return (
        "Você é um especialista em tape reading (time & sales / order flow) e análise técnica intraday "
        "focado em mercados de índice (futuros e/ou CFDs). Responda de forma objetiva, com passos acionáveis, "
        "destaque níveis de suporte/resistência detectados, explique sinais no tape (grandes prints, agressividade "
        "de compradores/vendedores, imbalances) e conecte com contextos de price action e micro-estrutura. "
        "Sempre inclua: (1) resumo em 1-2 linhas, (2) sinais observados (o que no tape levou à conclusão), "
        "(3) possíveis trade ideas com gerenciamento de risco (stop e target), e (4) notas sobre incertezas. "
        "Não forneça garantias de resultado. Seja conservador e peça mais dados se necessário."
    )

def assemble_messages(user_text: str, df_sample_text: Optional[str] = None, system_prompt: Optional[str] = None) -> List[Dict]:
    system = system_prompt or build_system_prompt()
    messages = [{"role":"system","content":system}]
    few_shot = [
        {"role":"user","content":"Resumo rápido do tape: houve um print grande comprador no topo da faixa, mas sem follow-through."},
        {"role":"assistant","content":"Resumo: Indecisão; sinal de possível exaustão de venda. Sinais: grande print comprador (volume elevado) numa resistência, mas sem sustentação. Trade idea: esperar pullback para avaliar entradas conservadoras; stop acima do pico do print. Incertezas: não houve confirmação em candles subsequentes."}
    ]
    messages += few_shot
    if df_sample_text:
        messages.append({"role":"user","content":"Aqui estão exemplos de leituras do tape:\n" + df_sample_text})
    messages.append({"role":"user","content": user_text})
    return messages
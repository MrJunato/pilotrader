# file: tape_gpt/chat/summarizer.py
from typing import List, Dict, Optional
from .client import call_openai

def summarize_chat(
    api_key: str,
    model: str,
    history: List[Dict],
    prior_summary: Optional[str] = None,
    insights: Optional[dict] = None,
    max_turns: int = 12,
) -> str:
    """
    Gera/atualiza um resumo curto da conversa (memória de longo prazo).
    - history: lista [{"role":"user"/"assistant","content":"..."}]
    - prior_summary: resumo anterior (se houver)
    - insights: opcional, injeta um mini-resumo do cenário do AnalyzerAgent
    Retorna um texto de 2-4 frases.
    """
    # recorta últimos turnos para baratear o custo
    h = history[-max_turns:]

    sys = (
        "Você é um assistente especialista em resumir conversas de trading. "
        "Produza um resumo objetivo (2-4 frases) com: (1) objetivo do usuário, "
        "(2) contexto do mercado/tape, (3) recomendações/avisos já dados. "
        "Evite jargões e duplicações. Se o usuário mudou de tópico, mantenha apenas o essencial."
    )
    msgs = [{"role": "system", "content": sys}]

    if insights:
        mini = insights.get("summary") or ""
        main = insights.get("main_signal", {})
        main_txt = f"{main.get('icon','')} {main.get('label','')} — {main.get('help','')}".strip()
        if main_txt:
            mini = f"{mini}\nSinal: {main_txt}"
        if mini:
            msgs.append({"role": "system", "content": f"Contexto de análise automática:\n{mini[:1000]}"})

    if prior_summary:
        msgs.append({"role": "system", "content": f"Resumo anterior da conversa:\n{prior_summary[:1200]}"})

    # injeta últimos turnos
    msgs.extend(h)
    msgs.append({"role": "user", "content": "Atualize o resumo da conversa seguindo as instruções acima."})

    # Define temperatura condicional: None para gpt-5 (usa default=1), 0.2 para demais
    temp = None if isinstance(model, str) and model.startswith("gpt-5") else 0.2

    return call_openai(
        api_key=api_key,
        model=model,
        messages=msgs,
        max_output_tokens=240,
        temperature=temp
    )
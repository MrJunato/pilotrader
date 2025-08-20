# file: tape_gpt/viz/indicators.py
import streamlit as st

def render_main_signal_indicator(main_signal: dict):
    # Espera um dict como o retornado por analyze_tape: {"label","color","icon","help"}
    label = main_signal.get("label", "Cenário Indefinido")
    color = main_signal.get("color", "gray")
    icon  = main_signal.get("icon", "❔")
    help_text = main_signal.get("help", "Não há sinais claros no tape reading.")

    bg = {
        "green": "rgba(0,200,0,0.08)",
        "red": "rgba(200,0,0,0.08)",
        "orange": "rgba(255,165,0,0.08)",
        "gray": "#f0f0f0",
    }.get(color, "#f0f0f0")

    st.markdown(
        f"<div style='border-radius:8px;padding:16px;background-color:{bg};"
        f"display:flex;align-items:center;'>"
        f"<span style='font-size:2em;margin-right:16px'>{icon}</span>"
        f"<span style='font-size:1.3em;font-weight:bold;color:{color}'>{label}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.caption(help_text)
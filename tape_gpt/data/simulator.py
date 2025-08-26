# tape_gpt/data/simulator.py

import threading, time, random
from datetime import datetime, timezone
from collections import deque
import numpy as np
import pandas as pd
from tape_gpt.data.loaders import parse_profit_excel  # usa o mesmo parser do upload  

class RealTimeSimulator:
    """
    Gera dados 'como se fossem' em tempo real.
    - Negócios (negocios): Compradora, Valor, Quantidade, Vendedora, Agressor (+ colunas internas)
    - Ofertas (ofertas): Agente_L, Qtde_L, Compra, Venda, Qtde_V, Agente_V
      (mapeadas em: agent_bid, qty_bid, bid, ask, qty_ask, agent_ask)
    """
    def __init__(self, start_price: float = 100000.0, tick_ms: int = 5000, vol: float = 2.0, max_rows: int = 100):
        self.price = float(start_price)
        self.tick = max(10, int(tick_ms)) / 1000.0    # agora padrão = 1s
        self.vol = float(vol)
        self.max_rows = max_rows                      # janela fixa de 100 negócios
        self._running = False
        self._th = None
        self._lock = threading.Lock()

        self._negocios = deque(maxlen=self.max_rows)  # mantém sempre 100
        self._ofertas  = deque(maxlen=self.max_rows)
        self.agents = [f"AG{str(i).zfill(3)}" for i in range(1, 51)]

    def seed_from_profit_xlsx(self, path: str):
        """
        Lê testes/exemplo_times_in_trade.xlsx (abas 'negocios' e 'ofertas') e
        semeia o simulador com o mesmo formato usado no upload.  
        """
        df_trades, df_offers = parse_profit_excel(path)  # df_trades: price, volume, side, buyer_agent, seller_agent...
        # Mapeia para o schema interno do simulador (igual ao que ele já expõe hoje)  
        df_trades = df_trades.sort_values("timestamp").tail(self.max_rows).copy()
        for _, r in df_trades.iterrows():
            negocio = {
                "timestamp": pd.to_datetime(r["timestamp"], utc=True),
                "Valor": float(r["price"]),
                "Quantidade": int(r["volume"]),
                "Compradora": str(r.get("buyer_agent", "")) if pd.notna(r.get("buyer_agent", pd.NA)) else "",
                "Vendedora": str(r.get("seller_agent", "")) if pd.notna(r.get("seller_agent", pd.NA)) else "",
                "Agressor": ("Compradora" if str(r.get("side","")).lower().startswith("buy") else "Vendedora"),
                # colunas internas (já usadas no pipeline/indicadores)
                "price": float(r["price"]),
                "volume": int(r["volume"]),
                "side": "buy" if str(r.get("side","")).lower().startswith("buy") else "sell",
                "buyer_agent": str(r.get("buyer_agent", "")) if pd.notna(r.get("buyer_agent", pd.NA)) else "",
                "seller_agent": str(r.get("seller_agent", "")) if pd.notna(r.get("seller_agent", pd.NA)) else "",
            }
            self._negocios.append(negocio)
            self.price = negocio["price"]  # atualiza preço corrente com o último da semente

        if df_offers is not None and len(df_offers) > 0:
            df_offers = df_offers.tail(self.max_rows).copy()
            for _, r in df_offers.iterrows():
                oferta = {
                    "timestamp": pd.Timestamp.utcnow(),
                    "Agente_L": str(r.get("buyer_agent","")),
                    "Qtde_L": int(r.get("buy_qty", 0)),
                    "Compra": float(r.get("buy_price", self.price - 1)),
                    "Venda": float(r.get("sell_price", self.price + 1)),
                    "Qtde_V": int(r.get("sell_qty", 0)),
                    "Agente_V": str(r.get("seller_agent","")),
                    # mapeamento interno
                    "agent_bid": str(r.get("buyer_agent","")),
                    "qty_bid": int(r.get("buy_qty", 0)),
                    "bid": float(r.get("buy_price", self.price - 1)),
                    "ask": float(r.get("sell_price", self.price + 1)),
                    "qty_ask": int(r.get("sell_qty", 0)),
                    "agent_ask": str(r.get("seller_agent","")),
                }
                self._ofertas.append(oferta)

    def _step(self):
        # Gera próximo ponto (random walk) com base no último preço  
        dp = np.random.normal(0, self.vol)
        self.price = max(1.0, float(round(self.price + dp, 2)))
        ts = datetime.now(timezone.utc)  # agora tz-aware (UTC)  【】

        aggressor_side = random.choice(["Compradora", "Vendedora"])
        buyer  = random.choice(self.agents)
        seller = random.choice(self.agents)
        qty = int(max(1, np.random.exponential(scale=50)))

        spread = max(1.0, abs(np.random.normal(2.0, 1.0)))
        bid = round(self.price - spread/2, 2)
        ask = round(self.price + spread/2, 2)
        qty_bid = int(max(1, np.random.exponential(scale=100)))
        qty_ask = int(max(1, np.random.exponential(scale=100)))
        agent_bid = random.choice(self.agents)
        agent_ask = random.choice(self.agents)

        negocio = {
            "timestamp": ts,
            "Valor": float(self.price),
            "Quantidade": qty,
            "Compradora": buyer,
            "Vendedora": seller,
            "Agressor": aggressor_side,
            "price": float(self.price),
            "volume": qty,
            "side": "buy" if aggressor_side == "Compradora" else "sell",
            "buyer_agent": buyer,
            "seller_agent": seller,
        }
        oferta = {
            "timestamp": ts,
            "Agente_L": agent_bid, "Qtde_L": qty_bid, "Compra": float(bid),
            "Venda": float(ask),    "Qtde_V": qty_ask, "Agente_V": agent_ask,
            "agent_bid": agent_bid, "qty_bid": qty_bid, "bid": float(bid),
            "ask": float(ask), "qty_ask": qty_ask, "agent_ask": agent_ask,
        }
        with self._lock:
            self._negocios.append(negocio)  # deque maxlen=100 já exclui o mais antigo
            self._ofertas.append(oferta)

    def start(self):
        if self._running: return
        self._running = True
        def _loop():
            while self._running:
                self._step()
                time.sleep(self.tick)  # 1s por padrão
        self._th = threading.Thread(target=_loop, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def get_dataframes(self):
        with self._lock:
            df_tr = pd.DataFrame(list(self._negocios))
            df_of = pd.DataFrame(list(self._ofertas))
        return df_tr, df_of
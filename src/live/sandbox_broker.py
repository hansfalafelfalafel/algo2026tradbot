"""Брокер поверх песочницы Т-Банка для LiveTrader.

Реализует интерфейс Broker: оценка стоимости портфеля, текущие цены и
исполнение рыночных заявок на виртуальном счёте песочницы.
"""
from __future__ import annotations

import uuid
from typing import Dict, Optional


def _q2f(q) -> float:
    return q.units + q.nano / 1e9


class SandboxBroker:
    def __init__(self, client, account_id: str, figis: Dict[str, str],
                 lot_sizes: Optional[Dict[str, int]] = None):
        self.client = client
        self.account_id = account_id
        self.figis = figis                       # ticker -> figi
        self.rev = {v: k for k, v in figis.items()}  # figi -> ticker
        self.lot_sizes = lot_sizes or {t: 1 for t in figis}

    def portfolio_value(self) -> float:
        pf = self.client.sandbox.get_sandbox_portfolio(account_id=self.account_id)
        return _q2f(pf.total_amount_portfolio)

    def prices(self) -> Dict[str, float]:
        resp = self.client.get_last_prices(instrument_id=list(self.figis.values()))
        out: Dict[str, float] = {}
        for lp in resp.last_prices:
            tk = self.rev.get(lp.figi)
            if tk:
                out[tk] = _q2f(lp.price)
        return out

    def execute(self, order: dict, prices: Dict[str, float]) -> bool:
        from tinkoff.invest import OrderDirection, OrderType

        tk = order["ticker"]
        figi = self.figis.get(tk)
        if not figi:
            return False
        price = prices.get(tk)
        lots = order.get("lots")
        lot_size = self.lot_sizes.get(tk, 1)
        if lots is None and price:
            lots = int(order["rub"] / (price * lot_size)) if price * lot_size else 0
        if not lots:
            return False

        direction = (OrderDirection.ORDER_DIRECTION_BUY if lots > 0
                     else OrderDirection.ORDER_DIRECTION_SELL)
        try:
            self.client.sandbox.post_sandbox_order(
                instrument_id=figi,
                quantity=abs(int(lots)),
                direction=direction,
                account_id=self.account_id,
                order_type=OrderType.ORDER_TYPE_MARKET,
                order_id=str(uuid.uuid4()),
            )
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[sandbox_broker] Ошибка заявки {tk}: {e}")
            return False

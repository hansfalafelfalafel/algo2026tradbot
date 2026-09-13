"""Live-торговля с быстрым циклом, режимами исполнения и управлением из дашборда.

Связывает воедино: агента (целевые веса), ExecutionController (лимиты и режим),
брокера (песочница или dry-run), состояние для дашборда и Telegram.

Управление из дашборда — через файлы state/control.json (пауза, режим, flatten,
переопределение лимитов) и state/pending.json (очередь заявок на подтверждение).
"""
from __future__ import annotations

from typing import Dict, Optional, Protocol

from src.live.execution import ExecutionController, RiskLimits
from src.notify.telegram import TelegramNotifier
from src import state


class Broker(Protocol):
    def portfolio_value(self) -> float: ...
    def execute(self, order: dict, prices: Dict[str, float]) -> bool: ...


class DryRunBroker:
    """Симулятор исполнения (без обращения к API) — для тестов и обучения работе."""

    def __init__(self, capital: float, lot_sizes: Optional[Dict[str, int]] = None):
        self.cash = capital
        self.positions: Dict[str, int] = {}  # ticker -> лоты
        self.lot_sizes = lot_sizes or {}
        self._prices: Dict[str, float] = {}

    def set_prices(self, prices: Dict[str, float]) -> None:
        self._prices = dict(prices)

    def portfolio_value(self) -> float:
        val = self.cash
        for tk, lots in self.positions.items():
            price = self._prices.get(tk, 0.0)
            val += lots * self.lot_sizes.get(tk, 1) * price
        return val

    def execute(self, order: dict, prices: Dict[str, float]) -> bool:
        tk = order["ticker"]
        price = prices.get(tk)
        lots = order.get("lots")
        lot_size = self.lot_sizes.get(tk, 1)
        if not price:
            return False
        if lots is None:  # оценим лоты из суммы
            lots = int(order["rub"] / (price * lot_size)) if price * lot_size else 0
        if lots == 0:
            return False
        cost = lots * lot_size * price
        self.cash -= cost                       # BUY (+lots) уменьшает кэш, SELL (-lots) увеличивает
        self.positions[tk] = self.positions.get(tk, 0) + lots
        return True


class LiveTrader:
    def __init__(self, agent, cfg, broker: Broker, notifier: Optional[TelegramNotifier] = None):
        self.agent = agent
        self.cfg = cfg
        self.broker = broker
        self.notifier = notifier or TelegramNotifier()
        self.current_weights: Dict[str, float] = {}
        self.controller = ExecutionController(
            RiskLimits.from_config(cfg), mode=cfg.execution.get("mode", "manual")
        )

    # ------------------------------------------------------------- один шаг
    def step(self, target_weights: Dict[str, float], prices: Dict[str, float],
             lot_sizes: Optional[Dict[str, int]] = None,
             sentiment: float = 0.0, risk_off: bool = False) -> dict:
        """Обработать один цикл: команды -> лимиты -> исполнение -> состояние."""
        # 1. Команды управления из дашборда.
        ctrl = state.read_control()
        if ctrl.get("mode"):
            self.controller.mode = ctrl["mode"]
        self.controller.limits = RiskLimits.from_config(self.cfg, ctrl.get("limits_override"))

        capital = self.broker.portfolio_value()

        # 2. Пауза.
        if ctrl.get("paused"):
            self._write_state(capital, target_weights, sentiment, risk_off, note="пауза")
            return {"status": "paused", "capital": capital}

        # 3. Дневной стоп-лосс (kill-switch) -> закрыть всё.
        if self.controller.kill_switch_triggered(capital):
            self._flatten(prices, reason="дневной стоп-лосс")
            self._write_state(capital, {}, sentiment, risk_off, note="KILL-SWITCH")
            return {"status": "kill_switch", "capital": capital}

        # 4. Запрос «закрыть все позиции».
        if ctrl.get("flatten_requested"):
            self._flatten(prices, reason="ручной flatten")
            state.write_control({"flatten_requested": False})
            self._write_state(capital, {}, sentiment, risk_off, note="flatten")
            return {"status": "flattened", "capital": capital}

        # 5. Сначала исполняем ранее подтверждённые в дашборде заявки.
        self._execute_approved_pending(prices)

        # 6. Планируем новые заявки с учётом лимитов и режима.
        if not self.controller.can_trade_now():
            self._write_state(capital, target_weights, sentiment, risk_off, note="cooldown/лимит сделок")
            return {"status": "throttled", "capital": capital}

        orders = self.controller.plan_orders(
            capital=capital,
            current_weights=self.current_weights,
            target_weights=target_weights,
            prices=prices,
            rebalance_threshold=self.cfg.execution.get("rebalance_threshold", 0.03),
            lot_sizes=lot_sizes,
        )

        executed, pending = [], []
        for o in orders:
            if o["decision"] == "execute":
                if self.broker.execute(o, prices):
                    self.current_weights[o["ticker"]] = o["weight_to"]
                    self.controller.register_trade()
                    executed.append(o)
                    state.append_history({"event": "trade", **{k: o[k] for k in ("ticker", "side", "rub", "lots")}})
            elif o["decision"] == "pending":
                pending.append(o)

        # 7. Очередь на подтверждение — в дашборд.
        if pending:
            existing = state.read_pending()
            state.write_pending(existing + pending)

        # 8. Уведомления.
        if executed:
            self._notify_executed(executed)
        capital = self.broker.portfolio_value()
        self._write_state(capital, target_weights, sentiment, risk_off,
                          note=f"исполнено: {len(executed)}, на подтверждении: {len(pending)}")
        return {"status": "ok", "capital": capital, "executed": len(executed), "pending": len(pending)}

    # ---------------------------------------------------------- вспомогательное
    def _execute_approved_pending(self, prices: Dict[str, float]) -> None:
        pend = state.read_pending()
        if not pend:
            return
        keep = []
        for o in pend:
            if o.get("decision") == "approved":
                if self.broker.execute(o, prices):
                    self.current_weights[o["ticker"]] = o["weight_to"]
                    self.controller.register_trade()
                    trade = {
                        k: o.get(k)
                        for k in ("ticker", "side", "rub", "lots")
                    }
                    state.append_history({"event": "trade(approved)", **trade})
            elif o.get("decision") == "rejected":
                continue  # выкидываем
            else:
                keep.append(o)  # ещё ждёт решения
        state.write_pending(keep)

    def _flatten(self, prices: Dict[str, float], reason: str) -> None:
        for tk, w in list(self.current_weights.items()):
            if abs(w) < 1e-6:
                continue
            # заявка на закрытие в противоположную сторону
            order = {"ticker": tk, "side": "SELL" if w > 0 else "BUY",
                     "rub": -w * self.broker.portfolio_value(), "lots": None,
                     "weight_to": 0.0}
            self.broker.execute(order, prices)
            self.current_weights[tk] = 0.0
        msg = f"⛔️ Закрыты все позиции ({reason})."
        print("[live] " + msg)
        self.notifier.send(msg)

    def _notify_executed(self, executed: list) -> None:
        lines = ["<b>✅ Исполнены сделки</b>"]
        for o in executed:
            lines.append(f"{o['side']} {o['ticker']}: {abs(o['rub']):,.0f} ₽"
                         + (f" (~{o['lots']} лот.)" if o.get('lots') is not None else ""))
        self.notifier.send("\n".join(lines))

    def _write_state(self, capital, target_weights, sentiment, risk_off, note="") -> None:
        positions = [{"ticker": tk, "weight": self.current_weights.get(tk, 0.0),
                      "rub": self.current_weights.get(tk, 0.0) * capital}
                     for tk in target_weights] or \
                    [{"ticker": tk, "weight": w, "rub": w * capital}
                     for tk, w in self.current_weights.items()]
        state.write_latest({
            "capital": capital,
            "sentiment": sentiment,
            "risk_off": risk_off,
            "mode": self.controller.mode,
            "gross_exposure": sum(abs(p["weight"]) for p in positions),
            "cash_weight": max(0.0, 1.0 - sum(abs(p["weight"]) for p in positions)),
            "daily_drawdown": self.controller.daily_drawdown(capital),
            "trades_today": self.controller.day.trades_today if self.controller.day else 0,
            "note": note,
            "positions": positions,
        })
        state.append_equity(capital)

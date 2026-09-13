"""Движок исполнения: ручной / полуавто / авто режимы с жёсткими лимитами.

Отделяет «что хочет агент» (целевые веса) от «что реально исполняем» (заявки в
пределах риск-лимитов). Это уровень безопасности между моделью и деньгами.

Режимы:
  * manual — все заявки уходят в очередь на подтверждение (кнопка в дашборде);
  * semi   — авто-исполнение в пределах лимитов, крупные заявки — на подтверждение;
  * auto   — полностью автоматически в пределах лимитов.

Лимиты (config.yaml -> execution.limits) — «тонкая настройка»:
  сколько всего можно вложить автоматически, максимум на заявку и на инструмент,
  предел суммарной экспозиции, число сделок в день, антидребезг между заявками,
  дневной стоп-лосс (kill-switch), белый список инструментов.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RiskLimits:
    max_invest_rub: float = 300_000
    max_order_rub: float = 50_000
    max_position_weight: float = 0.35
    max_gross_exposure: float = 0.8
    max_trades_per_day: int = 20
    min_seconds_between_orders: float = 5
    max_daily_loss_pct: float = 0.05
    confirm_above_rub: float = 100_000
    allow_short: bool = False
    whitelist: List[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg, override: Optional[dict] = None) -> "RiskLimits":
        lim = dict(cfg.execution.get("limits", {}))
        if override:
            lim.update({k: v for k, v in override.items() if v is not None})
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        return cls(**{k: v for k, v in lim.items() if k in allowed})


@dataclass
class DayState:
    """Внутридневное состояние для лимитов и стоп-лосса."""
    day: str
    day_start_equity: float
    trades_today: int = 0
    last_order_ts: float = 0.0


class ExecutionController:
    def __init__(self, limits: RiskLimits, mode: str = "manual"):
        self.limits = limits
        self.mode = mode
        self.day: Optional[DayState] = None

    # ------------------------------------------------------------ учёт дня
    def start_day(self, equity: float, day: str) -> None:
        self.day = DayState(day=day, day_start_equity=equity)

    def _ensure_day(self, equity: float) -> None:
        today = time.strftime("%Y-%m-%d")
        if self.day is None or self.day.day != today:
            self.start_day(equity, today)

    def daily_drawdown(self, equity: float) -> float:
        if not self.day or self.day.day_start_equity <= 0:
            return 0.0
        return equity / self.day.day_start_equity - 1.0  # <=0 при убытке

    def kill_switch_triggered(self, equity: float) -> bool:
        """Сработал ли дневной стоп-лосс."""
        self._ensure_day(equity)
        return self.daily_drawdown(equity) <= -abs(self.limits.max_daily_loss_pct)

    def can_trade_now(self, now_ts: Optional[float] = None) -> bool:
        now_ts = now_ts if now_ts is not None else time.time()
        if not self.day:
            return True
        if self.day.trades_today >= self.limits.max_trades_per_day:
            return False
        if now_ts - self.day.last_order_ts < self.limits.min_seconds_between_orders:
            return False
        return True

    def register_trade(self, now_ts: Optional[float] = None) -> None:
        now_ts = now_ts if now_ts is not None else time.time()
        if self.day:
            self.day.trades_today += 1
            self.day.last_order_ts = now_ts

    # --------------------------------------------------- ограничение весов
    def clamp_target(self, target: Dict[str, float], capital: float) -> Dict[str, float]:
        """Применить лимиты к целевым весам портфеля."""
        L = self.limits
        w = {}
        for tk, val in target.items():
            if L.whitelist and tk not in L.whitelist:
                continue
            if not L.allow_short and val < 0:
                val = 0.0
            # предел доли на инструмент
            val = float(np.clip(val, -L.max_position_weight, L.max_position_weight))
            w[tk] = val

        gross = sum(abs(v) for v in w.values())
        # предел суммарной экспозиции
        if gross > L.max_gross_exposure and gross > 1e-9:
            k = L.max_gross_exposure / gross
            w = {tk: v * k for tk, v in w.items()}
            gross = L.max_gross_exposure
        # предел абсолютной суммы автоввода
        max_gross_by_rub = (L.max_invest_rub / capital) if capital > 0 else 0.0
        if gross > max_gross_by_rub and gross > 1e-9:
            k = max_gross_by_rub / gross
            w = {tk: v * k for tk, v in w.items()}
        return w

    # ------------------------------------------------------------- план
    def plan_orders(
        self,
        capital: float,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        rebalance_threshold: float = 0.03,
        lot_sizes: Optional[Dict[str, int]] = None,
    ) -> List[dict]:
        """Построить список заявок с учётом лимитов и режима.

        Каждая заявка: {id, ticker, side, rub, lots, weight_from, weight_to,
        decision}. decision ∈ {execute, pending, skip}.
        """
        self._ensure_day(capital)
        clamped = self.clamp_target(target_weights, capital)

        orders: List[dict] = []
        for tk, w_to in clamped.items():
            w_from = current_weights.get(tk, 0.0)
            dw = w_to - w_from
            if abs(dw) < rebalance_threshold:
                continue  # изменение слишком мало — не дёргаем рынок

            rub = dw * capital
            # ограничение размера одной заявки
            if abs(rub) > self.limits.max_order_rub:
                rub = np.sign(rub) * self.limits.max_order_rub

            lots = None
            if lot_sizes and tk in lot_sizes and lot_sizes[tk] > 0 and prices.get(tk):
                per_lot = prices[tk] * lot_sizes[tk]
                if per_lot > 0:
                    lots = int(rub / per_lot)
                    if lots == 0:
                        continue  # меньше одного лота — пропускаем

            # решение по режиму
            if self.mode == "auto":
                decision = "execute"
            elif self.mode == "semi":
                decision = "pending" if abs(rub) > self.limits.confirm_above_rub else "execute"
            else:  # manual
                decision = "pending"

            orders.append({
                "id": str(uuid.uuid4())[:8],
                "ticker": tk,
                "side": "BUY" if rub > 0 else "SELL",
                "rub": float(rub),
                "lots": lots,
                "weight_from": float(w_from),
                "weight_to": float(w_to),
                "decision": decision,
            })
        return orders

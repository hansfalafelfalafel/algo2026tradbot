"""Торговля обученного агента в песочнице Т-Банка (бумажная торговля).

Песочница (sandbox) — это тренировочный контур Invest API: сделки исполняются
на реальных рыночных ценах, но на виртуальные деньги. Это безопасный способ
проверить стратегию «в бою» без риска потерять реальный капитал.

Логика:
  1. Открываем (или переиспользуем) счёт песочницы и заводим виртуальные деньги.
  2. В цикле подтягиваем свежие свечи, считаем признаки, спрашиваем агента.
  3. Приводим фактическую позицию к целевой рыночными заявками.
"""
from __future__ import annotations

import time
import uuid
from datetime import timedelta

import numpy as np
import pandas as pd

from src.data.features import add_features, feature_columns


def _quotation_to_float(q) -> float:
    return q.units + q.nano / 1e9


class SandboxTrader:
    """Обёртка над SandboxClient для запуска агента в песочнице."""

    def __init__(self, model, cfg, token: str) -> None:
        self.model = model
        self.cfg = cfg
        self.token = token
        self.figi = cfg.data["figi"]
        self.lots_per_trade = int(cfg.sandbox["lots_per_trade"])
        self.window_size = int(cfg.features["window_size"])
        self.use_indicators = bool(cfg.features["use_indicators"])
        self.feature_cols = feature_columns(self.use_indicators)
        # Текущая позиция в «единицах стратегии»: -1 / 0 / +1
        self.position = 0
        self.account_id: str | None = None

    # -------------------------------------------------------- инициализация
    def setup_account(self, client) -> str:
        """Открыть счёт песочницы и завести виртуальные деньги."""
        from tinkoff.invest import MoneyValue

        account = client.sandbox.open_sandbox_account()
        self.account_id = account.account_id

        pay_in = float(self.cfg.sandbox["pay_in"])
        units = int(pay_in)
        nano = int(round((pay_in - units) * 1e9))
        client.sandbox.sandbox_pay_in(
            account_id=self.account_id,
            amount=MoneyValue(units=units, nano=nano, currency="rub"),
        )
        print(f"[sandbox] Открыт счёт {self.account_id}, заведено {pay_in:,.0f} руб.")
        return self.account_id

    # ------------------------------------------------------- рыночные данные
    def fetch_observation(self, client) -> np.ndarray | None:
        """Скачать свежие свечи и построить наблюдение для агента."""
        from tinkoff.invest import CandleInterval
        from tinkoff.invest.utils import now
        from src.data.loader import INTERVAL_MAP

        interval = getattr(CandleInterval, INTERVAL_MAP[self.cfg.data["interval"]])
        # Берём с запасом: окно + горизонт индикаторов.
        lookback_days = max(5, self.window_size // 4 + 5)

        rows = []
        for c in client.get_all_candles(
            instrument_id=self.figi,
            from_=now() - timedelta(days=lookback_days),
            interval=interval,
        ):
            rows.append(
                {
                    "time": c.time,
                    "open": _quotation_to_float(c.open),
                    "high": _quotation_to_float(c.high),
                    "low": _quotation_to_float(c.low),
                    "close": _quotation_to_float(c.close),
                    "volume": c.volume,
                }
            )
        df = pd.DataFrame(rows)
        feat = add_features(df, use_indicators=self.use_indicators)
        if len(feat) < self.window_size:
            print("[sandbox] Недостаточно свечей для наблюдения, пропускаю шаг.")
            return None

        window = feat[self.feature_cols].to_numpy(dtype=np.float64)[-self.window_size:]
        obs = np.concatenate([window.flatten(), [float(self.position)]])
        return obs.astype(np.float32)

    # ------------------------------------------------------------- торговля
    def rebalance(self, client, target_position: int) -> None:
        """Привести фактическую позицию к целевой рыночными заявками."""
        from tinkoff.invest import OrderDirection, OrderType

        target_lots = target_position * self.lots_per_trade
        current_lots = self.position * self.lots_per_trade
        delta = target_lots - current_lots
        if delta == 0:
            return

        direction = (
            OrderDirection.ORDER_DIRECTION_BUY
            if delta > 0
            else OrderDirection.ORDER_DIRECTION_SELL
        )
        resp = client.sandbox.post_sandbox_order(
            instrument_id=self.figi,
            quantity=abs(delta),
            direction=direction,
            account_id=self.account_id,
            order_type=OrderType.ORDER_TYPE_MARKET,
            order_id=str(uuid.uuid4()),
        )
        side = "BUY" if delta > 0 else "SELL"
        print(
            f"[sandbox] Заявка {side} {abs(delta)} лот(ов) -> "
            f"позиция {self.position:+d} => {target_position:+d} "
            f"(status={resp.execution_report_status})"
        )
        self.position = target_position

    def print_portfolio(self, client) -> float:
        """Показать текущее состояние виртуального портфеля и вернуть его стоимость."""
        pf = client.sandbox.get_sandbox_portfolio(account_id=self.account_id)
        total = _quotation_to_float(pf.total_amount_portfolio)
        print(f"[sandbox] Стоимость портфеля: {total:,.2f} руб.")
        # Пишем в общее состояние — это увидит дашборд (live-кривая капитала).
        try:
            from src.state import append_equity
            append_equity(total)
        except Exception:  # noqa: BLE001
            pass
        return total

    # ----------------------------------------------------------- основной цикл
    def run(self, n_iterations: int | None = None) -> None:
        """Запустить торговый цикл в песочнице.

        :param n_iterations: сколько итераций сделать (None = бесконечно, до Ctrl+C).
        """
        from tinkoff.invest.sandbox.client import SandboxClient

        poll = int(self.cfg.sandbox["poll_interval_sec"])
        with SandboxClient(self.token) as client:
            self.setup_account(client)
            i = 0
            try:
                while n_iterations is None or i < n_iterations:
                    obs = self.fetch_observation(client)
                    if obs is not None:
                        action, _ = self.model.predict(obs, deterministic=True)
                        from src.env.trading_env import ACTION_TO_POSITION
                        target = ACTION_TO_POSITION[int(action)]
                        if not self.cfg.env["allow_short"] and target == -1:
                            target = 0
                        self.rebalance(client, target)
                        self.print_portfolio(client)
                    i += 1
                    if n_iterations is None or i < n_iterations:
                        time.sleep(poll)
            except KeyboardInterrupt:
                print("\n[sandbox] Остановлено пользователем.")
            finally:
                if self.account_id:
                    print(f"[sandbox] Итоговое состояние счёта {self.account_id}:")
                    self.print_portfolio(client)

"""Стриминг котировок в реальном времени (для быстрой реакции).

Вместо периодического опроса цен (get_last_prices) подписываемся на поток
последних цен и держим их в кэше. Торговый цикл читает цену мгновенно, что
снижает задержку реакции с «периода опроса» до времени прихода тика.

Поток работает в фоновом потоке (thread), чтобы не блокировать основной цикл.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional


def _q2f(q) -> float:
    return q.units + q.nano / 1e9


class StreamingPriceFeed:
    """Фоновая подписка на последние цены по списку инструментов."""

    def __init__(self, client, figis: Dict[str, str]):
        self.client = client
        self.figis = figis                          # ticker -> figi
        self.rev = {v: k for k, v in figis.items()}  # figi -> ticker
        self._prices: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._manager = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from tinkoff.invest import LastPriceInstrument

        try:
            self._manager = self.client.create_market_data_stream()
            self._manager.last_price.subscribe(
                [LastPriceInstrument(instrument_id=figi) for figi in self.figis.values()]
            )
            for md in self._manager:
                if self._stop.is_set():
                    break
                lp = getattr(md, "last_price", None)
                if lp is not None:
                    tk = self.rev.get(lp.figi)
                    if tk:
                        with self._lock:
                            self._prices[tk] = _q2f(lp.price)
        except Exception as e:  # noqa: BLE001
            print(f"[stream] Поток котировок остановлен: {e}")

    def get_prices(self) -> Dict[str, float]:
        """Текущий снимок последних цен (может быть неполным на старте)."""
        with self._lock:
            return dict(self._prices)

    def ready(self) -> bool:
        """Есть ли уже цены по всем инструментам."""
        with self._lock:
            return all(t in self._prices for t in self.figis)

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._manager is not None:
                self._manager.stop()
        except Exception:  # noqa: BLE001
            pass

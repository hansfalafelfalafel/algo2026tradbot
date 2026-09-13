"""Коллектор биржевого стакана и ленты сделок (данные для слоя A / OFI).

Пишет 24/7 снимки стакана (глубина 10) и сделки по всем инструментам из
config.yaml в файлы data_cache/lob/<дата>_<тикер>_{book,trades}.csv.
Это «сырьё» для микроструктурных сигналов (order flow imbalance) — данных,
которых нет у большинства участников. Чем дольше работает, тем ценнее датасет.

Запуск (держать в tmux, работает постоянно):
    python scripts/30_collect_orderbook.py

Устойчив к обрывам: при ошибке стрима переподключается через 15 секунд.
Вне торговых часов данных просто не будет — это нормально.
"""
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

from dotenv import load_dotenv

from src.config import load_config

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))


def _q2f(q) -> float:
    return q.units + q.nano / 1e9


class DayWriter:
    """CSV-файлы по дням/тикерам с ленивым открытием и периодическим flush."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._files = {}

    def _get(self, kind: str, ticker: str, header):
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = (day, ticker, kind)
        if key not in self._files:
            path = self.root / f"{day}_{ticker}_{kind}.csv"
            # Заголовок нужен, если файла нет ИЛИ он пуст (заголовок мог не
            # доехать до диска при рестарте). Пишем и сразу сбрасываем на диск.
            new = (not path.exists()) or path.stat().st_size == 0
            f = open(path, "a", newline="", encoding="utf-8")
            w = csv.writer(f)
            if new:
                w.writerow(header)
                f.flush()
            self._files[key] = (f, w)
            # Закрываем файлы прошлых дней.
            for k in [k for k in self._files if k[0] != day]:
                self._files.pop(k)[0].close()
        return self._files[key][1], self._files[key][0]

    def book(self, ticker, ob):
        header = (["time"] + [f"bid_p{i}" for i in range(10)] + [f"bid_q{i}" for i in range(10)]
                  + [f"ask_p{i}" for i in range(10)] + [f"ask_q{i}" for i in range(10)])
        w, f = self._get("book", ticker, header)
        bids_p = [_q2f(b.price) for b in ob.bids[:10]] + [0.0] * (10 - len(ob.bids[:10]))
        bids_q = [b.quantity for b in ob.bids[:10]] + [0] * (10 - len(ob.bids[:10]))
        asks_p = [_q2f(a.price) for a in ob.asks[:10]] + [0.0] * (10 - len(ob.asks[:10]))
        asks_q = [a.quantity for a in ob.asks[:10]] + [0] * (10 - len(ob.asks[:10]))
        w.writerow([datetime.now(timezone.utc).isoformat()] + bids_p + bids_q + asks_p + asks_q)
        return f

    def trade(self, ticker, tr):
        w, f = self._get("trades", ticker, ["time", "price", "quantity", "direction"])
        w.writerow([datetime.now(timezone.utc).isoformat(), _q2f(tr.price),
                    tr.quantity, int(tr.direction)])
        return f


def main() -> None:
    from tinkoff.invest import Client, OrderBookInstrument, TradeInstrument

    cfg = load_config()
    token = os.environ.get("TINKOFF_TOKEN")
    if not token:
        print("Нужен TINKOFF_TOKEN в .env")
        sys.exit(1)

    figis = cfg.data["figis"]                     # ticker -> figi
    rev = {v: k for k, v in figis.items()}        # figi -> ticker
    writer = DayWriter(cfg.abs_path(cfg.data["cache_dir"], "lob"))
    print(f"[lob] Собираю стакан+ленту по {len(figis)} инструментам. Ctrl+C — стоп.")

    n_book = n_trade = 0
    last_flush = last_report = time.time()

    while True:  # цикл переподключения
        try:
            with Client(token) as client:
                stream = client.create_market_data_stream()
                stream.order_book.subscribe(
                    [OrderBookInstrument(instrument_id=f, depth=10) for f in figis.values()])
                stream.trades.subscribe(
                    [TradeInstrument(instrument_id=f) for f in figis.values()])
                for md in stream:
                    ob = getattr(md, "orderbook", None)
                    tr = getattr(md, "trade", None)
                    f = None
                    if ob is not None and ob.figi in rev:
                        f = writer.book(rev[ob.figi], ob); n_book += 1
                    elif tr is not None and tr.figi in rev:
                        f = writer.trade(rev[tr.figi], tr); n_trade += 1
                    now = time.time()
                    if f and now - last_flush > 10:
                        for fh, _ in writer._files.values():
                            fh.flush()
                        last_flush = now
                    if now - last_report > 300:
                        print(f"[lob] {datetime.now():%H:%M} снимков стакана: {n_book}, сделок: {n_trade}")
                        last_report = now
        except KeyboardInterrupt:
            print(f"\n[lob] Остановлено. Итог: {n_book} снимков, {n_trade} сделок.")
            return
        except Exception as e:  # noqa: BLE001
            print(f"[lob] Стрим оборвался ({e}); переподключение через 15 с...")
            time.sleep(15)


if __name__ == "__main__":
    main()

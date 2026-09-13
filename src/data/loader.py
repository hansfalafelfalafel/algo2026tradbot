"""Загрузка исторических свечей из Invest API Т-Банка (бывш. Tinkoff).

Модуль умеет:
  * находить FIGI по тикеру;
  * скачивать исторические свечи за N дней;
  * кэшировать их в CSV, чтобы не дёргать API повторно.

Данные качаются одинаково и для обучения, и для бэктеста — так мы гарантируем,
что модель тестируется на том же формате данных, что и в реальной торговле.
"""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Импорты SDK вынесены внутрь функций там, где это возможно, чтобы модуль можно
# было импортировать даже без установленного SDK (например, для юнит-тестов
# среды на синтетических данных).

INTERVAL_MAP = {
    "1min": "CANDLE_INTERVAL_1_MIN",
    "5min": "CANDLE_INTERVAL_5_MIN",
    "15min": "CANDLE_INTERVAL_15_MIN",
    "hour": "CANDLE_INTERVAL_HOUR",
    "day": "CANDLE_INTERVAL_DAY",
}


def _get_token() -> str:
    token = os.environ.get("TINKOFF_TOKEN")
    if not token:
        raise RuntimeError(
            "Не найден токен. Задайте переменную окружения TINKOFF_TOKEN "
            "(например, через файл .env). Токен создаётся в приложении Т-Банк "
            "Инвестиции -> Настройки -> Токен для Invest API."
        )
    return token


def _quotation_to_float(q) -> float:
    """Перевод Quotation/MoneyValue (units + nano) в обычный float."""
    return q.units + q.nano / 1e9


def find_figi_by_ticker(ticker: str, class_code: str = "TQBR") -> str:
    """Найти FIGI инструмента по тикеру.

    :param ticker: тикер, например "SBER".
    :param class_code: код режима торгов (TQBR — акции, SPBFUT — фьючерсы).
    """
    from tinkoff.invest import Client

    with Client(_get_token()) as client:
        resp = client.instruments.find_instrument(query=ticker)
        for inst in resp.instruments:
            if inst.ticker == ticker and inst.class_code == class_code:
                return inst.figi
        # если точного совпадения по class_code нет — вернём первый по тикеру
        for inst in resp.instruments:
            if inst.ticker == ticker:
                return inst.figi
    raise ValueError(f"Инструмент с тикером {ticker!r} не найден")


def download_candles(
    figi: str,
    interval: str = "hour",
    history_days: int = 500,
) -> pd.DataFrame:
    """Скачать исторические свечи и вернуть DataFrame.

    Колонки результата: time, open, high, low, close, volume.
    """
    import time as _time
    from tinkoff.invest import CandleInterval, Client
    from tinkoff.invest.exceptions import RequestError
    from tinkoff.invest.utils import now

    if interval not in INTERVAL_MAP:
        raise ValueError(f"Неизвестный интервал {interval!r}. Доступно: {list(INTERVAL_MAP)}")
    candle_interval = getattr(CandleInterval, INTERVAL_MAP[interval])

    # Максимальный период на один запрос по интервалу (дни).
    chunk_days = {"1min": 1, "5min": 1, "15min": 2, "hour": 7, "day": 365}.get(interval, 7)
    chunk = timedelta(days=chunk_days)

    end = now()
    cur = end - timedelta(days=history_days)
    rows = []
    with Client(_get_token()) as client:
        # Скачиваем частями с «притормаживанием», чтобы не упереться в лимит
        # запросов API (600/мин). При RESOURCE_EXHAUSTED ждём и повторяем.
        while cur < end:
            chunk_end = min(cur + chunk, end)
            for _attempt in range(8):
                try:
                    resp = client.market_data.get_candles(
                        instrument_id=figi, from_=cur, to=chunk_end,
                        interval=candle_interval,
                    )
                    for candle in resp.candles:
                        rows.append(
                            {
                                "time": candle.time,
                                "open": _quotation_to_float(candle.open),
                                "high": _quotation_to_float(candle.high),
                                "low": _quotation_to_float(candle.low),
                                "close": _quotation_to_float(candle.close),
                                "volume": candle.volume,
                            }
                        )
                    break
                except RequestError as e:  # noqa: PERF203
                    if "RESOURCE_EXHAUSTED" not in str(e):
                        raise
                    meta = getattr(e, "metadata", None)
                    reset = getattr(meta, "ratelimit_reset", None) if meta else None
                    wait = int(reset) + 2 if reset else 60
                    print(f"[loader] Лимит запросов API — жду {wait} c и продолжаю...")
                    _time.sleep(wait)
            else:
                raise RuntimeError("Не удалось скачать свечи: превышен лимит запросов.")
            cur = chunk_end
            _time.sleep(0.25)  # ~4 запроса/с — уверенно под лимитом 600/мин

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "API вернул пустой список свечей. Проверьте FIGI, интервал и то, "
            "что по инструменту есть история за выбранный период."
        )
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return df


def load_or_download(
    figi: str,
    interval: str,
    history_days: int,
    cache_dir: str | os.PathLike,
    force: bool = False,
) -> pd.DataFrame:
    """Вернуть свечи из кэша, либо скачать и сохранить в CSV."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{figi}_{interval}_{history_days}d.csv"

    if cache_file.exists() and not force:
        df = pd.read_csv(cache_file, parse_dates=["time"])
        return df

    df = download_candles(figi, interval, history_days)
    df.to_csv(cache_file, index=False)
    return df


def get_lot_sizes(figis: dict[str, str]) -> dict[str, int]:
    """Получить размер лота по каждому инструменту (сколько бумаг в 1 лоте)."""
    from tinkoff.invest import Client, InstrumentIdType

    result: dict[str, int] = {}
    with Client(_get_token()) as client:
        for ticker, figi in figis.items():
            try:
                resp = client.instruments.get_instrument_by(
                    id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi
                )
                result[ticker] = int(resp.instrument.lot)
            except Exception as e:  # noqa: BLE001
                print(f"[loader] Не удалось получить лот для {ticker}: {e}")
                result[ticker] = 1
    return result


def load_many(
    figis: dict[str, str],
    interval: str,
    history_days: int,
    cache_dir: str | os.PathLike,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Загрузить свечи для нескольких инструментов.

    :param figis: словарь {тикер -> FIGI}.
    :return: словарь {тикер -> DataFrame свечей}.
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker, figi in figis.items():
        try:
            result[ticker] = load_or_download(figi, interval, history_days, cache_dir, force)
        except Exception as e:  # noqa: BLE001
            print(f"[loader] Пропускаю {ticker} (нет данных за период): {e}")
    return result

"""Загрузка исторических микроданных MOEX AlgoPack по API-КЛЮЧУ.

Подготовка:
  1) Подписка AlgoPack куплена, API-ключ получен в кабинете data.moex.com.
  2) В .env добавить строку:  MOEX_APIKEY=твой_ключ
  3) pip install requests (уже стоит)

Запуск:
    python scripts/40_algopack_load.py                     # 6 мес, ликвидные тикеры
    python scripts/40_algopack_load.py --months 12
    python scripts/40_algopack_load.py --tickers SBER,GAZP

Результат: data_cache/algopack/<TICKER>_<stat>.csv
"""
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))

BASE = "https://apim.moex.com/iss/datashop/algopack/eq"
DEFAULT_TICKERS = ["SBER", "GAZP", "LKOH", "VTBR", "GMKN", "ROSN", "NVTK",
                   "TATN", "MGNT", "MTSS"]
STATS = ["tradestats", "obstats", "orderstats"]
CHUNK_DAYS = 9          # максимум дат на один запрос у ISS невелик — идём кусками
PAGE = 1000


def _arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def fetch_stat(sess: requests.Session, ticker: str, stat: str,
               start: date, end: date) -> pd.DataFrame:
    frames = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS), end)
        offset = 0
        while True:
            url = (f"{BASE}/{stat}/{ticker}.json"
                   f"?from={cur}&till={chunk_end}&start={offset}")
            r = sess.get(url, timeout=30)
            if r.status_code == 401:
                raise RuntimeError("401: ключ не принят — проверьте MOEX_APIKEY")
            if r.status_code == 403:
                raise RuntimeError("403: нет доступа — активна ли подписка AlgoPack?")
            r.raise_for_status()
            js = r.json()
            block = js.get("data") or {}
            cols, data = block.get("columns"), block.get("data")
            if not data:
                break
            frames.append(pd.DataFrame(data, columns=cols))
            offset += len(data)
            time.sleep(0.2)
        cur = chunk_end + timedelta(days=1)
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    key = os.environ.get("MOEX_APIKEY")
    if not key:
        print("Добавьте в .env строку: MOEX_APIKEY=ваш_ключ (кабинет data.moex.com)")
        sys.exit(1)

    months = int(_arg("--months", 6, float))
    tickers = [t.strip().upper() for t in
               _arg("--tickers", ",".join(DEFAULT_TICKERS)).split(",")]
    end = date.today()
    start = end - timedelta(days=30 * months)

    sess = requests.Session()
    sess.headers["Authorization"] = f"Bearer {key}"

    out_dir = Path(__file__).resolve().parents[1] / "data_cache" / "algopack"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Качаю {tickers} с {start} по {end}")

    for tk in tickers:
        for stat in STATS:
            try:
                df = fetch_stat(sess, tk, stat, start, end)
                if df.empty:
                    print(f"  {tk} {stat}: пусто (нет данных/доступа)")
                    continue
                out = out_dir / f"{tk}_{stat}.csv"
                df.to_csv(out, index=False)
                print(f"  {tk} {stat}: {len(df):,} строк -> {out.name}")
            except Exception as e:  # noqa: BLE001
                print(f"  {tk} {stat}: ОШИБКА {e}")

    print("\nГотово. Дальше: python scripts/41_algopack_model.py")


if __name__ == "__main__":
    main()

# Data

Полный `data_cache/` намеренно не хранится в GitHub: архив исходных свечей, стаканов и сделок занимает сотни мегабайт.

В `data/sample/` лежат небольшие фрагменты форматов, с которыми работает проект:

- `daily_candles_sample.csv` — дневные OHLCV-свечи;
- `lob_trades_sample.csv` — поток сделок;
- `orderbook_sample.csv` — 10 уровней стакана.

Полные данные создаются/загружаются скриптами `scripts/01_download_data.py`, `scripts/30_collect_orderbook.py` и загрузчиками в `src/data/` / `src/lob/`.

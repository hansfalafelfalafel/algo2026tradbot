# README для разработчика

## Быстрый маршрут по проекту

- `src/data/` — получение и подготовка свечных данных.
- `src/env/`, `src/agent/` — Gymnasium-среды и RL-агенты.
- `src/pairs/` — статистический арбитраж.
- `src/factor/` — факторные и momentum-подходы.
- `src/lob/` — микроструктура рынка / order book features.
- `src/backtest/` — backtesting, walk-forward и PBO.
- `src/live/` — sandbox/execution/risk-control.
- `src/notify/` — уведомления.
- `dashboard/` — Streamlit-мониторинг.
- `scripts/` — воспроизводимые эксперименты; карта находится в `scripts/README.md`.
- `results/` — небольшие итоговые артефакты, включённые в репозиторий.

## Проверки перед коммитом

```bash
python -m compileall -q src scripts tests dashboard
pytest -q
flake8 src tests dashboard --max-line-length=120
pylint src --max-line-length=120 --disable="C0103,C0114,C0115"
```

Некоторые эксперименты требуют полного локального `data_cache/`, T-Bank Invest API token и/или данных MOEX AlgoPack.

"""Общее хранилище состояния для дашборда и уведомлений.

Торговые скрипты пишут сюда текущее состояние (последняя рекомендация, кривая
капитала, история), а дашборд Streamlit — читает. Это разъединяет «мозг»
(агент) и «витрину» (дашборд): их можно запускать независимо.

Файлы (в папке state/):
  * latest.json   — последняя рекомендация и метрики (перезаписывается);
  * equity.csv    — история капитала (дозаписывается);
  * history.csv   — история рекомендаций/сделок (дозаписывается).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

STATE_DIR = Path(__file__).resolve().parents[1] / "state"


def _ensure_dir() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def write_latest(payload: Dict[str, Any]) -> Path:
    """Записать последнюю рекомендацию/состояние в latest.json."""
    _ensure_dir()
    payload = dict(payload)
    payload.setdefault("updated_at", now_iso())
    path = STATE_DIR / "latest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def read_latest() -> Dict[str, Any] | None:
    path = STATE_DIR / "latest.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_equity(equity: float, ts: str | None = None) -> None:
    _ensure_dir()
    row = pd.DataFrame([{"time": ts or now_iso(), "equity": equity}])
    path = STATE_DIR / "equity.csv"
    header = not path.exists()
    row.to_csv(path, mode="a", header=header, index=False)


def append_history(record: Dict[str, Any]) -> None:
    _ensure_dir()
    record = dict(record)
    record.setdefault("time", now_iso())
    path = STATE_DIR / "history.csv"
    header = not path.exists()
    pd.DataFrame([record]).to_csv(path, mode="a", header=header, index=False)


def read_equity() -> pd.DataFrame:
    path = STATE_DIR / "equity.csv"
    if not path.exists():
        return pd.DataFrame(columns=["time", "equity"])
    return pd.read_csv(path, parse_dates=["time"])


def read_history() -> pd.DataFrame:
    path = STATE_DIR / "history.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# --------------------------------------------------------------------- control
# Панель управления: дашборд пишет команды, торговый цикл их читает.
_DEFAULT_CONTROL = {
    "mode": None,               # None -> берётся из config; иначе manual/semi/auto
    "paused": False,            # пауза торговли
    "flatten_requested": False, # закрыть все позиции и выйти в кэш
    "limits_override": {},      # переопределение лимитов из дашборда
}


def read_control() -> Dict[str, Any]:
    path = STATE_DIR / "control.json"
    if not path.exists():
        return dict(_DEFAULT_CONTROL)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = dict(_DEFAULT_CONTROL)
    merged.update(data)
    return merged


def write_control(patch: Dict[str, Any]) -> Path:
    """Обновить (слить) команды управления."""
    _ensure_dir()
    cur = read_control()
    cur.update(patch)
    cur["updated_at"] = now_iso()
    path = STATE_DIR / "control.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)
    return path


# ----------------------------------------------------------------- pending orders
def read_pending() -> list:
    path = STATE_DIR / "pending.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_pending(orders: list) -> Path:
    _ensure_dir()
    path = STATE_DIR / "pending.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)
    return path

"""Агрегация новостей в числовой сигнал, выровненный со свечами.

Из «сырых» заголовков строим ежедневный индекс настроения рынка в [-1, 1],
сглаживаем его и вычисляем флаг ``risk_off`` (негативный фон -> снижаем риск).
Этот сигнал затем:
  * добавляется как признак в состояние агента;
  * используется риск-слоем среды для снижения экспозиции (предвосхищение спада).

Важно: открытые RSS-ленты отдают в основном свежие новости, поэтому для
исторического бэктеста глубокого архива нет. Для полноценной истории настроений
нужен новостной архив/платный API. В live-режиме (песочница) сигнал считается по
актуальным новостям и работает в полную силу.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from src.news.feeds import Headline
from src.news.sentiment import SentimentBackend


def score_headlines(headlines: List[Headline], backend: SentimentBackend) -> pd.DataFrame:
    """Оценить заголовки бэкендом и вернуть DataFrame [date, sentiment]."""
    if not headlines:
        return pd.DataFrame(columns=["date", "sentiment"])
    texts = [h.text for h in headlines]
    scores = backend.score_texts(texts)
    rows = []
    for h, s in zip(headlines, scores):
        day = h.published.date() if h.published else datetime.utcnow().date()
        rows.append({"date": pd.Timestamp(day), "sentiment": float(s)})
    df = pd.DataFrame(rows)
    # Среднее настроение за день.
    daily = df.groupby("date", as_index=False)["sentiment"].mean()
    return daily.sort_values("date").reset_index(drop=True)


def build_daily_series(
    daily: pd.DataFrame,
    ema_span: int = 3,
) -> pd.DataFrame:
    """Сгладить дневной сентимент EMA и добавить risk_off-флаг."""
    if daily.empty:
        return pd.DataFrame(columns=["date", "sentiment", "sentiment_smooth", "risk_off"])
    out = daily.copy()
    out["sentiment_smooth"] = out["sentiment"].ewm(span=ema_span, adjust=False).mean()
    return out


def align_to_df(
    candles: pd.DataFrame,
    daily: pd.DataFrame,
    risk_off_threshold: float = -0.2,
) -> pd.DataFrame:
    """Выровнять дневной сентимент с таймфреймом свечей.

    Для каждой свечи берём сглаженный сентимент за её день (или последний
    известный до неё). Отсутствующие значения -> 0 (нейтрально).

    Возвращает копию candles с колонками sentiment и risk_off (0/1).
    """
    df = candles.copy().reset_index(drop=True)
    # Приводим к единому разрешению datetime64[ns], иначе merge_asof падает.
    df["_date"] = pd.to_datetime(df["time"]).dt.normalize().astype("datetime64[ns]")

    if daily is None or daily.empty:
        df["sentiment"] = 0.0
        df["risk_off"] = 0.0
        return df.drop(columns="_date")

    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize().astype("datetime64[ns]")
    d = d.sort_values("date")
    col = "sentiment_smooth" if "sentiment_smooth" in d.columns else "sentiment"

    # merge_asof: для каждой свечи берём последний известный сентимент.
    merged = pd.merge_asof(
        df.sort_values("_date"),
        d[["date", col]].rename(columns={col: "sentiment"}),
        left_on="_date",
        right_on="date",
        direction="backward",
    )
    merged["sentiment"] = merged["sentiment"].fillna(0.0)
    merged["risk_off"] = (merged["sentiment"] < risk_off_threshold).astype(float)
    merged = merged.sort_index().drop(columns=["_date", "date"], errors="ignore")
    return merged.reset_index(drop=True)


def attach_sentiment(
    candles: pd.DataFrame,
    headlines: List[Headline],
    backend: SentimentBackend,
    ema_span: int = 3,
    risk_off_threshold: float = -0.2,
) -> pd.DataFrame:
    """Полный конвейер: заголовки -> сентимент -> колонки в датафрейме свечей."""
    daily = score_headlines(headlines, backend)
    daily = build_daily_series(daily, ema_span=ema_span)
    return align_to_df(candles, daily, risk_off_threshold=risk_off_threshold)


def cache_path(cache_dir: str | Path, tag: str) -> Path:
    p = Path(cache_dir) / f"news_{tag}.csv"
    return p

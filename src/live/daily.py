"""Формирование и отправка рекомендации «портфель на сегодня».

Общая логика, которую используют и разовый скрипт (08_recommend.py), и
планировщик (10_scheduler.py). Здесь же — обновление состояния для дашборда.
"""
from __future__ import annotations

import os

import pandas as pd

from src.agent.ensemble import ENSEMBLE_ALGOS, EnsembleAgent
from src.data.loader import get_lot_sizes, load_many
from src.data.portfolio_data import build_portfolio_arrays
from src.live.recommend import get_recommendation
from src.notify.telegram import TelegramNotifier, format_recommendation
from src.state import append_equity, write_latest


def _news_df(cfg, refresh: bool = False, sample: bool = False):
    if refresh or sample:
        from src.news.feeds import fetch_headlines, sample_headlines
        from src.news.sentiment import get_backend
        from src.news.signal import build_daily_series, score_headlines

        headlines = sample_headlines() if sample else fetch_headlines(cfg.news.get("feeds") or None)
        backend = get_backend(cfg.news.get("backend", "lexicon"))
        daily = build_daily_series(score_headlines(headlines, backend),
                                   ema_span=cfg.news.get("ema_span", 3))
        daily.to_csv(cfg.abs_path(cfg.data["cache_dir"], "news_signal.csv"), index=False)

    path = cfg.abs_path(cfg.data["cache_dir"], "news_signal.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    col = "sentiment_smooth" if "sentiment_smooth" in df.columns else "sentiment"
    return df[["date", col]].rename(columns={"date": "time", col: "sentiment"})


def load_ensemble(cfg):
    model_dir = cfg.abs_path(cfg.agent["model_dir"])
    paths = {n: model_dir / f"{cfg.agent['model_name']}_{n}.zip"
             for n in ENSEMBLE_ALGOS
             if (model_dir / f"{cfg.agent['model_name']}_{n}.zip").exists()}
    if not paths:
        raise FileNotFoundError("Модели ансамбля не найдены. Запустите scripts/06_train_ensemble.py.")
    models = {n: ENSEMBLE_ALGOS[n].load(p) for n, p in paths.items()}
    return EnsembleAgent(models, {n: 1.0 / len(models) for n in models})


def send_daily_recommendation(cfg, refresh_news: bool = True, sample_news: bool = False) -> dict:
    """Посчитать рекомендацию, записать состояние и отправить в Telegram."""
    candle_dfs = load_many(cfg.data["figis"], cfg.data["interval"],
                           cfg.data["history_days"], cfg.abs_path(cfg.data["cache_dir"]))
    news_df = _news_df(cfg, refresh=refresh_news, sample=sample_news) if cfg.news.get("enabled") else None
    arrays = build_portfolio_arrays(candle_dfs, cfg.features["use_indicators"], news_df)

    # Та же нормализация признаков, что при обучении.
    from src.data.normalize import load_stats, normalize_apply
    _stats = load_stats(cfg.abs_path(cfg.agent["model_dir"], f"{cfg.agent['model_name']}_norm.npz"))
    if _stats is not None:
        normalize_apply(arrays, _stats)

    agent = load_ensemble(cfg)
    capital = float(cfg.sandbox.get("pay_in", 1_000_000))

    lot_sizes = None
    if os.environ.get("TINKOFF_TOKEN"):
        try:
            lot_sizes = get_lot_sizes(cfg.data["figis"])
        except Exception:  # noqa: BLE001
            pass

    rec = get_recommendation(agent, arrays, cfg, capital, lot_sizes)
    write_latest(rec)
    append_equity(capital)

    msg = format_recommendation(rec)
    TelegramNotifier().send(msg)
    return rec

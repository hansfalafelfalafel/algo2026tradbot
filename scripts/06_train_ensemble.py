"""Обучить ансамбль PPO+SAC+A2C на портфеле инструментов (с новостным сигналом).

Запуск:
    python scripts/06_train_ensemble.py

Данные берутся из кэша (сначала 01_download_data.py для каждого инструмента и,
опционально, 05_fetch_news.py). Модели ансамбля сохраняются в models/.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.agent.ensemble import train_ensemble
from src.config import load_config
from src.data.loader import load_many
from src.data.portfolio_data import build_portfolio_arrays

load_dotenv()


def _load_news_df(cfg):
    path = cfg.abs_path(cfg.data["cache_dir"], "news_signal.csv")
    if not path.exists():
        print("[news] Файл сигнала не найден — обучаю без новостей (сентимент=0).")
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    col = "sentiment_smooth" if "sentiment_smooth" in df.columns else "sentiment"
    return df[["date", col]].rename(columns={"date": "time", col: "sentiment"})


def main() -> None:
    cfg = load_config()

    candle_dfs = load_many(
        figis=cfg.data["figis"],
        interval=cfg.data["interval"],
        history_days=cfg.data["history_days"],
        cache_dir=cfg.abs_path(cfg.data["cache_dir"]),
    )
    news_df = _load_news_df(cfg) if cfg.news.get("enabled", False) else None

    arrays = build_portfolio_arrays(
        candle_dfs,
        use_indicators=cfg.features["use_indicators"],
        sentiment_df=news_df,
    )
    print(f"Инструменты: {arrays['assets']} | баров: {len(arrays['prices'])}")

    # Разбиение по времени: train | (валидация+тест позже в бэктесте).
    split = int(len(arrays["prices"]) * cfg.data["train_ratio"])

    # Нормализация признаков по обучающей части (без подглядывания в будущее).
    from src.data.normalize import normalize_fit, save_stats
    stats = normalize_fit(arrays, split)
    norm_path = cfg.abs_path(cfg.agent["model_dir"], f"{cfg.agent['model_name']}_norm.npz")
    save_stats(norm_path, stats)
    print(f"Нормализатор сохранён: {norm_path.name}")

    def slice_arrays(a: dict, lo: int, hi: int) -> dict:
        out = dict(a)
        out["prices"] = a["prices"][lo:hi]
        out["features"] = a["features"][lo:hi]
        out["sentiment"] = a["sentiment"][lo:hi]
        return out

    arrays_train = slice_arrays(arrays, 0, split)
    paths = train_ensemble(arrays_train, cfg)
    print("\nОбучены модели:")
    for name, p in paths.items():
        print(f"  {name}: {p.name}")


if __name__ == "__main__":
    main()

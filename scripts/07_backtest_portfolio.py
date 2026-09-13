"""Бэктест ансамбля на отложенной (test) выборке; веса ансамбля — по валидации.

Запуск:
    python scripts/07_backtest_portfolio.py

Печатает метрики и сохраняет график reports/equity_ensemble.png.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.agent.ensemble import ENSEMBLE_ALGOS, EnsembleAgent
from src.backtest.backtest import plot_results
from src.backtest.portfolio_backtest import format_portfolio_report, run_portfolio_backtest
from src.config import load_config
from src.data.loader import load_many
from src.data.portfolio_data import build_portfolio_arrays

load_dotenv()


def _load_news_df(cfg):
    path = cfg.abs_path(cfg.data["cache_dir"], "news_signal.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    col = "sentiment_smooth" if "sentiment_smooth" in df.columns else "sentiment"
    return df[["date", col]].rename(columns={"date": "time", col: "sentiment"})


def _slice(a: dict, lo: int, hi: int) -> dict:
    out = dict(a)
    for k in ("prices", "features", "sentiment"):
        out[k] = a[k][lo:hi]
    return out


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
        candle_dfs, use_indicators=cfg.features["use_indicators"], sentiment_df=news_df
    )

    # Применяем ту же нормализацию признаков, что была при обучении.
    from src.data.normalize import load_stats, normalize_apply
    stats = load_stats(cfg.abs_path(cfg.agent["model_dir"], f"{cfg.agent['model_name']}_norm.npz"))
    if stats is not None:
        normalize_apply(arrays, stats)

    n = len(arrays["prices"])
    tr = int(n * cfg.data["train_ratio"])
    va = int(n * (cfg.data["train_ratio"] + cfg.data["val_ratio"]))
    arrays_val = _slice(arrays, tr, va)
    arrays_test = _slice(arrays, va, n)
    print(f"Валидация: {va - tr} баров | Тест: {n - va} баров")

    # Собираем пути моделей ансамбля.
    model_dir = cfg.abs_path(cfg.agent["model_dir"])
    paths = {
        name: model_dir / f"{cfg.agent['model_name']}_{name}.zip"
        for name in ENSEMBLE_ALGOS
        if (model_dir / f"{cfg.agent['model_name']}_{name}.zip").exists()
    }
    if not paths:
        print("Модели ансамбля не найдены. Сначала запустите 06_train_ensemble.py.")
        sys.exit(1)

    agent = EnsembleAgent.load_and_weight(paths, arrays_val, cfg)
    result = run_portfolio_backtest(agent, arrays_test, cfg)

    print("\n" + format_portfolio_report(result, cfg))

    # Сравнение с простыми бейзлайнами на том же тестовом периоде.
    from src.backtest.baselines import compare_baselines, format_baselines
    ws = cfg.features["window_size"]
    base = compare_baselines(arrays_test["prices"][ws:], cfg.env["initial_balance"],
                             cfg.data["interval"])
    print("\n" + format_baselines(base, rl_metrics=result["metrics"]))

    plot_path = cfg.abs_path("reports", "equity_ensemble.png")
    plot_results(result, cfg, plot_path)
    print(f"\nГрафик сохранён: {plot_path}")


if __name__ == "__main__":
    main()

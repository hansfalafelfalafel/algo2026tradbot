"""Честная проверка: walk-forward переобучение + оценка переобучения (PBO).

Запуск:
    python scripts/11_walkforward.py

Внимание: обучает ансамбль несколько раз (по числу фолдов) — это долго.
Для ускорения уменьшите agent.total_timesteps или n_folds.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.backtest.backtest import compute_metrics
from src.backtest.walkforward import run_walkforward
from src.config import load_config
from src.data.loader import load_many
from src.data.portfolio_data import build_portfolio_arrays

load_dotenv()


def _news_df(cfg):
    path = cfg.abs_path(cfg.data["cache_dir"], "news_signal.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    col = "sentiment_smooth" if "sentiment_smooth" in df.columns else "sentiment"
    return df[["date", col]].rename(columns={"date": "time", col: "sentiment"})


def main() -> None:
    cfg = load_config()
    candle_dfs = load_many(cfg.data["figis"], cfg.data["interval"],
                           cfg.data["history_days"], cfg.abs_path(cfg.data["cache_dir"]))
    arrays = build_portfolio_arrays(candle_dfs, cfg.features["use_indicators"],
                                    _news_df(cfg) if cfg.news.get("enabled") else None)

    n_folds = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 4
    res = run_walkforward(arrays, cfg, n_folds=n_folds)

    m = compute_metrics(res["oos_equity"], cfg.data["interval"])
    print("\n" + "=" * 58)
    print("WALK-FORWARD (out-of-sample) РЕЗУЛЬТАТ")
    print("=" * 58)
    print(f"OOS баров:            {res['n_oos_bars']}")
    print(f"Итоговая доходность:  {m['total_return']*100:+.2f}%")
    print(f"Коэффициент Шарпа:    {m['sharpe']:.2f}")
    print(f"Макс. просадка:       {m['max_drawdown']*100:+.2f}%")
    print("-" * 58)
    pbo = res["pbo"]
    verdict = ("НИЗКИЙ — стратегия устойчива" if pbo < 0.2
               else "СРЕДНИЙ — осторожно" if pbo < 0.4
               else "ВЫСОКИЙ — вероятно переобучение!")
    print(f"PBO (вероятность переобучения): {pbo:.2f}  [{verdict}]")
    print("=" * 58)


if __name__ == "__main__":
    main()

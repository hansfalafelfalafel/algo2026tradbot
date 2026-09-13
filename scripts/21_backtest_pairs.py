"""Маркет-нейтральный бэктест парной стратегии (возврат к среднему).

Запуск:
    python scripts/21_backtest_pairs.py                 # лучшая пара из pairs.csv
    python scripts/21_backtest_pairs.py SBER GAZP        # конкретная пара
    python scripts/21_backtest_pairs.py --entry 2.5 --exit 0.3

Сначала запустите scripts/20_find_pairs.py. Стратегия оценивается на отложенном
(out-of-sample) периоде — смотрите колонку «Тест (OOS)».
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.config import load_config
from src.data.loader import load_many
from src.pairs.backtest_pairs import backtest_pair, format_pair_report

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))


def _arg(flag, default):
    if flag in sys.argv:
        return float(sys.argv[sys.argv.index(flag) + 1])
    return default


def main() -> None:
    cfg = load_config()
    candle_dfs = load_many(cfg.data["figis"], cfg.data["interval"],
                           cfg.data["history_days"], cfg.abs_path(cfg.data["cache_dir"]))

    # Пара: из аргументов или лучшая из pairs.csv.
    tickers = [a for a in sys.argv[1:] if not a.startswith("--")
               and not a.replace(".", "").isdigit()]
    if len(tickers) >= 2:
        y_name, x_name = tickers[0].upper(), tickers[1].upper()
    else:
        pairs_path = cfg.abs_path(cfg.data["cache_dir"], "pairs.csv")
        if not pairs_path.exists():
            print("Нет pairs.csv — сначала запустите scripts/20_find_pairs.py.")
            sys.exit(1)
        pairs = pd.read_csv(pairs_path)
        best = pairs.iloc[0]
        y_name, x_name = best["y"], best["x"]

    if y_name not in candle_dfs or x_name not in candle_dfs:
        print(f"Нет данных по {y_name} или {x_name}. Проверьте config.yaml.")
        sys.exit(1)

    # Выравниваем по общему времени.
    y = candle_dfs[y_name].set_index("time")["close"]
    x = candle_dfs[x_name].set_index("time")["close"]
    joined = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()

    res = backtest_pair(
        joined["y"].to_numpy(), joined["x"].to_numpy(),
        train_frac=cfg.data["train_ratio"],
        entry=_arg("--entry", 2.0), exit=_arg("--exit", 0.5),
        cost=cfg.env["commission"] + cfg.env["slippage"],
        initial_balance=cfg.env["initial_balance"],
        interval=cfg.data["interval"],
    )
    print(format_pair_report(y_name, x_name, res))


if __name__ == "__main__":
    main()

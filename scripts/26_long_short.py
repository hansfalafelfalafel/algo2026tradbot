"""Лонг-шорт моментум — последний свечной тест (маркет-нейтральный).

Запуск:
    python scripts/26_long_short.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.config import load_config
from src.data.loader import load_many
from src.factor.longshort import format_ls_report, long_short_backtest
from src.factor.momentum import align_prices

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))


def main() -> None:
    cfg = load_config()
    candle_dfs = load_many(cfg.data["figis"], cfg.data["interval"],
                           cfg.data["history_days"], cfg.abs_path(cfg.data["cache_dir"]))
    prices = align_prices(candle_dfs)
    print(f"Вселенная: {prices.shape[1]} акций, {len(prices)} дн.")
    res = long_short_backtest(
        prices, cost=cfg.env["commission"] + cfg.env["slippage"],
        cap=cfg.env["initial_balance"])
    print("\n" + format_ls_report(res))


if __name__ == "__main__":
    main()

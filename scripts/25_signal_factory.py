"""Фабрика сигналов (моментум + разворот + низкая волатильность) + мета-фильтр.

Запуск:
    python scripts/25_signal_factory.py
    python scripts/25_signal_factory.py --p 0.6
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.config import load_config
from src.data.loader import load_many
from src.factor.factory import factory_backtest, format_factory_report
from src.factor.momentum import align_prices

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))


def _arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main() -> None:
    cfg = load_config()
    candle_dfs = load_many(cfg.data["figis"], cfg.data["interval"],
                           cfg.data["history_days"], cfg.abs_path(cfg.data["cache_dir"]))
    prices = align_prices(candle_dfs)
    print(f"Вселенная: {prices.shape[1]} акций, {len(prices)} дн.")

    res = factory_backtest(
        prices, cap=cfg.env["initial_balance"],
        p_threshold=_arg("--p", 0.55),
        cost=cfg.env["commission"] + cfg.env["slippage"],
    )
    print("\n" + format_factory_report(res))


if __name__ == "__main__":
    main()

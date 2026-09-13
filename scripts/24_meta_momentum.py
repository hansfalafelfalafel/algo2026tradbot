"""Слой C: мета-лейблинг поверх моментум-сигналов (сравнение B vs B+C на OOS).

Запуск:
    python scripts/24_meta_momentum.py
    python scripts/24_meta_momentum.py --p 0.6      # строже фильтр

Требует scikit-learn: pip install scikit-learn
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.config import load_config
from src.data.loader import load_many
from src.factor.meta import format_meta_report, meta_momentum_backtest
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

    res = meta_momentum_backtest(
        prices, cap=cfg.env["initial_balance"],
        p_threshold=_arg("--p", 0.55),
        cost=cfg.env["commission"] + cfg.env["slippage"],
    )
    print("\n" + format_meta_report(res))


if __name__ == "__main__":
    main()

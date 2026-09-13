"""Бэктест портфеля из нескольких коинтегрированных пар (диверсификация).

Запуск:
    python scripts/22_pairs_portfolio.py                 # авто-отбор пар из pairs.csv
    python scripts/22_pairs_portfolio.py --topk 4         # взять 4 лучшие
    python scripts/22_pairs_portfolio.py --max-hl 200     # фильтр по полураспаду

Отбор по умолчанию: p-value < 0.05 и период полураспада 5..250 баров (быстрый и
устойчивый возврат к среднему). Сначала запустите scripts/20_find_pairs.py.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.config import load_config
from src.data.loader import load_many
from src.pairs.portfolio_pairs import backtest_portfolio, format_portfolio

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))


def _arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main() -> None:
    cfg = load_config()
    pairs_path = cfg.abs_path(cfg.data["cache_dir"], "pairs.csv")
    if not pairs_path.exists():
        print("Нет pairs.csv — сначала запустите scripts/20_find_pairs.py.")
        sys.exit(1)

    df = pd.read_csv(pairs_path)
    max_pv = _arg("--max-pvalue", 0.05)
    max_hl = _arg("--max-hl", 250.0)
    min_hl = _arg("--min-hl", 5.0)
    topk = int(_arg("--topk", 5))

    max_leg = int(_arg("--max-per-leg", 2))  # не больше N пар на один инструмент
    cand = df[(df["pvalue"] < max_pv) & (df["half_life"] > min_hl)
              & (df["half_life"] < max_hl)].sort_values("pvalue")

    # Жадный отбор с ограничением концентрации: пропускаем пару, если любой её
    # инструмент уже участвует в max_leg парах. Так корзина честно разнородна.
    counts: dict = {}
    pairs = []
    for _, r in cand.iterrows():
        if len(pairs) >= topk:
            break
        if counts.get(r["y"], 0) >= max_leg or counts.get(r["x"], 0) >= max_leg:
            continue
        pairs.append((r["y"], r["x"]))
        counts[r["y"]] = counts.get(r["y"], 0) + 1
        counts[r["x"]] = counts.get(r["x"], 0) + 1

    if not pairs:
        print("После фильтров не осталось пар. Ослабьте --max-pvalue / --max-hl.")
        sys.exit(1)
    print(f"Портфель из {len(pairs)} пар: {[f'{y}/{x}' for y, x in pairs]}")

    candle_dfs = load_many(cfg.data["figis"], cfg.data["interval"],
                           cfg.data["history_days"], cfg.abs_path(cfg.data["cache_dir"]))
    res = backtest_portfolio(
        candle_dfs, pairs,
        train_frac=cfg.data["train_ratio"],
        entry=_arg("--entry", 2.0), exit=_arg("--exit", 0.5),
        cost=cfg.env["commission"] + cfg.env["slippage"],
        cap=cfg.env["initial_balance"], interval=cfg.data["interval"],
    )
    print("\n" + format_portfolio(res))


if __name__ == "__main__":
    main()

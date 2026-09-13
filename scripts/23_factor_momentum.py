"""Слой B: бэктест моментум-портфеля с vol-targeting и режимным фильтром.

Запуск (нужны дневные данные: в config.yaml interval: day):
    python scripts/23_factor_momentum.py
    python scripts/23_factor_momentum.py --top 10 --target-vol 0.12

Ничего не обучается — сигналы считаются только по прошлому окну, поэтому весь
период по построению out-of-sample.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.config import load_config
from src.data.loader import load_many
from src.factor.momentum import align_prices, format_momentum_report, momentum_backtest

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))


def _arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main() -> None:
    cfg = load_config()
    if cfg.data["interval"] != "day":
        print(f"Внимание: interval={cfg.data['interval']}; слой B рассчитан на дневки.")

    candle_dfs = load_many(cfg.data["figis"], cfg.data["interval"],
                           cfg.data["history_days"], cfg.abs_path(cfg.data["cache_dir"]))
    prices = align_prices(candle_dfs)
    print(f"Вселенная: {prices.shape[1]} акций, {len(prices)} дн. "
          f"({prices.index[0].date()} — {prices.index[-1].date()})")

    res = momentum_backtest(
        prices,
        top_n=int(_arg("--top", 8)),
        target_vol=_arg("--target-vol", 0.15),
        cost=cfg.env["commission"] + cfg.env["slippage"],
        cap=cfg.env["initial_balance"],
    )
    print("\n" + format_momentum_report(res))

    # График.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(res["times"], res["equity"], label="Слой B (моментум)", lw=1.6)
        ax.plot(res["times"], res["buyhold"][1:len(res["equity"]) + 1],
                label="Buy&Hold (равновзв.)", lw=1.2, alpha=0.8)
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylabel("капитал, руб.")
        out = cfg.abs_path("reports", "equity_momentum.png")
        out.parent.mkdir(exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=120)
        print(f"График: {out}")
    except Exception as e:  # noqa: BLE001
        print(f"(график не построен: {e})")


if __name__ == "__main__":
    main()

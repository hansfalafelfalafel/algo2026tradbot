"""Маркет-нейтральный бэктест возврата к среднему для пары инструментов.

Стратегия (dollar-neutral): держим равные по деньгам противоположные позиции в
двух инструментах. Сигнал — z-оценка лог-отношения цен:
  * z > entry  -> спред «слишком высок» -> шорт спреда (шорт y, лонг x);
  * z < -entry -> лонг спреда (лонг y, шорт x);
  * |z| < exit -> закрываем позицию.

ВАЖНО (честность): среднее и std для z-оценки берутся ТОЛЬКО из обучающего
периода, затем применяются к тесту (out-of-sample). Иначе — подглядывание.
"""
from __future__ import annotations

import numpy as np

from src.backtest.backtest import compute_metrics


def backtest_pair(
    y_prices: np.ndarray,
    x_prices: np.ndarray,
    train_frac: float = 0.6,
    entry: float = 2.0,
    exit: float = 0.5,
    cost: float = 0.0007,
    initial_balance: float = 100_000.0,
    interval: str = "hour",
) -> dict:
    """Прогнать парную стратегию, оценить на train и out-of-sample тесте."""
    y = np.log(np.asarray(y_prices, dtype=np.float64))
    x = np.log(np.asarray(x_prices, dtype=np.float64))
    n = len(y)
    split = int(n * train_frac)

    # Хедж-коэффициент и статистики спреда — по обучающей части.
    beta = np.polyfit(x[:split], y[:split], 1)[0]
    spread = y - beta * x
    mu = spread[:split].mean()
    sigma = spread[:split].std()
    if sigma < 1e-9:
        sigma = 1.0
    z = (spread - mu) / sigma

    # Доходности инструментов (в процентах).
    ry = np.diff(np.exp(y)) / np.exp(y)[:-1]
    rx = np.diff(np.exp(x)) / np.exp(x)[:-1]

    # Формируем позицию по спреду с гистерезисом (entry/exit).
    pos = np.zeros(n)
    cur = 0
    for t in range(n):
        if cur == 0:
            if z[t] > entry:
                cur = -1
            elif z[t] < -entry:
                cur = 1
        else:
            if abs(z[t]) < exit:
                cur = 0
        pos[t] = cur

    # Доходность dollar-neutral портфеля: +0.5 в одну ногу, -0.5 в другую.
    strat_ret = np.zeros(n - 1)
    for t in range(1, n):
        s = pos[t - 1]
        gross_ret = 0.5 * (ry[t - 1] - rx[t - 1])  # long y / short x на единицу спреда
        turnover = abs(pos[t - 1] - (pos[t - 2] if t >= 2 else 0.0))
        strat_ret[t - 1] = s * gross_ret - turnover * cost

    equity = initial_balance * np.cumprod(1 + strat_ret)

    # Метрики отдельно на train и test (out-of-sample).
    tr_eq = initial_balance * np.cumprod(1 + strat_ret[: split - 1])
    te_eq = initial_balance * np.cumprod(1 + strat_ret[split - 1:])

    return {
        "beta": float(beta),
        "n_trades": int((np.abs(np.diff(pos)) > 0).sum()),
        "equity": equity,
        "z": z,
        "positions": pos,
        "metrics_all": compute_metrics(equity, interval),
        "metrics_train": compute_metrics(tr_eq, interval),
        "metrics_test": compute_metrics(te_eq, interval),
        "split": split,
    }


def format_pair_report(y_name: str, x_name: str, res: dict) -> str:
    m, tr, te = res["metrics_all"], res["metrics_train"], res["metrics_test"]

    def pct(x):
        return f"{x*100:+.2f}%"

    return "\n".join([
        "=" * 60,
        f"ПАРНЫЙ БЭКТЕСТ: LONG {y_name} / SHORT {x_name}  (beta={res['beta']:.2f})",
        f"Сделок: {res['n_trades']}",
        "=" * 60,
        f"{'Метрика':<24}{'Train (IS)':>16}{'Тест (OOS)':>16}",
        "-" * 60,
        f"{'Доходность':<24}{pct(tr['total_return']):>16}{pct(te['total_return']):>16}",
        f"{'Коэф. Шарпа':<24}{tr['sharpe']:>16.2f}{te['sharpe']:>16.2f}",
        f"{'Макс. просадка':<24}{pct(tr['max_drawdown']):>16}{pct(te['max_drawdown']):>16}",
        "=" * 60,
        "Главное — колонка «Тест (OOS)»: там данные, которых стратегия не видела.",
    ])

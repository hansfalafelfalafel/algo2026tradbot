"""Портфель из нескольких парных стратегий (диверсификация).

Торгуем несколько коинтегрированных пар одновременно, равным риском на каждую.
Даже если отдельная пара «ломается» вне выборки, диверсификация сглаживает
итоговую кривую — обычно это заметно повышает коэффициент Шарпа по сравнению с
одиночной парой.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.backtest.backtest import compute_metrics


def _pair_returns(y_prices: np.ndarray, x_prices: np.ndarray, split: int,
                  entry: float, exit: float, cost: float) -> np.ndarray:
    """Пошаговые доходности одной парной стратегии (статистики z — по train)."""
    y = np.log(np.asarray(y_prices, dtype=np.float64))
    x = np.log(np.asarray(x_prices, dtype=np.float64))
    beta = np.polyfit(x[:split], y[:split], 1)[0]
    spread = y - beta * x
    mu, sigma = spread[:split].mean(), spread[:split].std()
    if sigma < 1e-9:
        sigma = 1.0
    z = (spread - mu) / sigma

    ry = np.diff(np.exp(y)) / np.exp(y)[:-1]
    rx = np.diff(np.exp(x)) / np.exp(x)[:-1]

    n = len(y)
    pos = np.zeros(n)
    cur = 0
    for t in range(n):
        if cur == 0:
            if z[t] > entry:
                cur = -1
            elif z[t] < -entry:
                cur = 1
        elif abs(z[t]) < exit:
            cur = 0
        pos[t] = cur

    r = np.zeros(n - 1)
    for t in range(1, n):
        s = pos[t - 1]
        turnover = abs(pos[t - 1] - (pos[t - 2] if t >= 2 else 0.0))
        r[t - 1] = s * 0.5 * (ry[t - 1] - rx[t - 1]) - turnover * cost
    return r


def _align(candle_dfs, names) -> pd.DataFrame:
    closes = {}
    for name in names:
        s = candle_dfs[name].copy()
        s["time"] = pd.to_datetime(s["time"], utc=True)
        closes[name] = s.set_index("time")["close"]
    return pd.DataFrame(closes).dropna()


def backtest_portfolio(
    candle_dfs, pairs: List[Tuple[str, str]],
    train_frac: float = 0.6, entry: float = 2.0, exit: float = 0.5,
    cost: float = 0.0007, cap: float = 100_000.0, interval: str = "hour",
) -> dict:
    """Бэктест равновзвешенного портфеля пар. Возвращает метрики IS/OOS."""
    involved = sorted({n for pair in pairs for n in pair})
    prices = _align(candle_dfs, involved)
    n = len(prices)
    split = int(n * train_frac)

    pair_rets, per_pair = [], []
    for y, x in pairs:
        r = _pair_returns(prices[y].to_numpy(), prices[x].to_numpy(), split, entry, exit, cost)
        pair_rets.append(r)
        oos_eq = cap * np.cumprod(1 + r[split - 1:])
        per_pair.append({"pair": f"{y}/{x}", "oos": compute_metrics(oos_eq, interval)})

    R = np.mean(np.vstack(pair_rets), axis=0)  # равный вес по парам
    equity = cap * np.cumprod(1 + R)
    is_eq = cap * np.cumprod(1 + R[: split - 1])
    oos_eq = cap * np.cumprod(1 + R[split - 1:])

    return {
        "equity": equity,
        "metrics_is": compute_metrics(is_eq, interval),
        "metrics_oos": compute_metrics(oos_eq, interval),
        "per_pair": per_pair,
        "n_pairs": len(pairs),
        "split": split,
    }


def format_portfolio(res: dict) -> str:
    def pct(x):
        return f"{x*100:+.2f}%"

    io, oo = res["metrics_is"], res["metrics_oos"]
    lines = [
        "=" * 62,
        f"ПОРТФЕЛЬ ИЗ {res['n_pairs']} ПАР (равный вес, маркет-нейтральный)",
        "=" * 62,
        f"{'Метрика':<24}{'Train (IS)':>18}{'Тест (OOS)':>18}",
        "-" * 62,
        f"{'Доходность':<24}{pct(io['total_return']):>18}{pct(oo['total_return']):>18}",
        f"{'Коэф. Шарпа':<24}{io['sharpe']:>18.2f}{oo['sharpe']:>18.2f}",
        f"{'Макс. просадка':<24}{pct(io['max_drawdown']):>18}{pct(oo['max_drawdown']):>18}",
        "-" * 62,
        "Вклад пар (OOS Шарп):",
    ]
    for p in res["per_pair"]:
        lines.append(f"  {p['pair']:<16} доходность {pct(p['oos']['total_return']):>9}  "
                     f"Шарп {p['oos']['sharpe']:>6.2f}")
    lines.append("=" * 62)
    return "\n".join(lines)

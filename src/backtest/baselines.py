"""Простые (не-RL) стратегии-бейзлайны для сравнения с ансамблем.

Смысл: если RL-агент не обыгрывает даже примитивное правило (моментум,
пересечение средних), значит он пока не добавляет ценности. Это важная точка
отсчёта и для диплома, и для понимания, куда копать.

Все бейзлайны — long-only, равновзвешенные, с учётом комиссий.
"""
from __future__ import annotations

import numpy as np

from src.backtest.backtest import compute_metrics


def _run_weights(prices: np.ndarray, weights_fn, cap: float, cost: float) -> np.ndarray:
    """Прогнать стратегию, заданную функцией весов, вернуть кривую капитала."""
    T, N = prices.shape
    equity = [cap]
    w_prev = np.zeros(N)
    for t in range(1, T):
        w = weights_fn(t - 1)  # веса на основе данных до момента t-1
        turnover = np.sum(np.abs(w - w_prev))
        ret = np.dot(w, prices[t] / prices[t - 1] - 1.0) - turnover * cost
        equity.append(equity[-1] * (1.0 + ret))
        w_prev = w
    return np.asarray(equity)


def buy_hold(prices: np.ndarray, cap: float) -> np.ndarray:
    norm = prices / prices[0]
    return cap * norm.mean(axis=1)


def momentum(prices: np.ndarray, cap: float, lookback: int = 20, cost: float = 0.0007) -> np.ndarray:
    T, N = prices.shape

    def wfn(t):
        if t < lookback:
            return np.zeros(N)
        r = prices[t] / prices[t - lookback] - 1.0
        pos = r > 0
        k = pos.sum()
        if k == 0:
            return np.zeros(N)
        w = np.zeros(N)
        w[pos] = 1.0 / k  # равный вес среди «растущих», gross = 1
        return w

    return _run_weights(prices, wfn, cap, cost)


def ma_crossover(prices: np.ndarray, cap: float, short: int = 10, long: int = 30,
                 cost: float = 0.0007) -> np.ndarray:
    import pandas as pd
    T, N = prices.shape
    df = pd.DataFrame(prices)
    sma_s = df.rolling(short).mean().to_numpy()
    sma_l = df.rolling(long).mean().to_numpy()

    def wfn(t):
        if t < long:
            return np.zeros(N)
        sig = (sma_s[t] > sma_l[t]).astype(float)
        k = sig.sum()
        return sig / k if k > 0 else np.zeros(N)

    return _run_weights(prices, wfn, cap, cost)


def all_cash(prices: np.ndarray, cap: float) -> np.ndarray:
    return np.full(prices.shape[0], cap)


def compare_baselines(prices: np.ndarray, cap: float, interval: str) -> dict:
    """Посчитать метрики всех бейзлайнов на данной ценовой матрице."""
    strategies = {
        "Buy&Hold": buy_hold(prices, cap),
        "Моментум": momentum(prices, cap),
        "MA-кроссовер": ma_crossover(prices, cap),
        "Только кэш": all_cash(prices, cap),
    }
    return {name: compute_metrics(eq, interval) for name, eq in strategies.items()}


def format_baselines(metrics: dict, rl_metrics: dict | None = None) -> str:
    rows = ["=" * 66, "СРАВНЕНИЕ СО СТРАТЕГИЯМИ-БЕЙЗЛАЙНАМИ", "=" * 66,
            f"{'Стратегия':<20}{'Доходность':>14}{'Шарп':>10}{'Просадка':>14}"]
    rows.append("-" * 66)
    if rl_metrics:
        rows.append(f"{'RL-ансамбль':<20}{rl_metrics['total_return']*100:>13.2f}%"
                    f"{rl_metrics['sharpe']:>10.2f}{rl_metrics['max_drawdown']*100:>13.2f}%")
    for name, m in metrics.items():
        rows.append(f"{name:<20}{m['total_return']*100:>13.2f}%"
                    f"{m['sharpe']:>10.2f}{m['max_drawdown']*100:>13.2f}%")
    rows.append("=" * 66)
    return "\n".join(rows)

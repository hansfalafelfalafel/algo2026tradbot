"""Лонг-шорт моментум: маркет-нейтральный фактор (последний свечной тест).

Классическая академическая конструкция: покупаем сильнейшие по моментуму акции
и одновременно шортим слабейшие, равным риском на ногу (dollar-neutral).
Направление рынка выпадает из уравнения — стратегия зарабатывает, только если
победители продолжают обгонять проигравших. Ничего не обучается — весь период
по построению out-of-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.backtest import compute_metrics


def long_short_backtest(
    prices: pd.DataFrame,
    lookback=126, skip=5, n_side=6, vol_win=60, rebalance=5,
    cost=0.0007, cap=100_000.0, ann=252, gross_per_side=0.5,
) -> dict:
    P = prices.to_numpy(dtype=np.float64)
    T, N = P.shape
    rets = np.diff(P, axis=0) / P[:-1]
    index = (P / P[0]).mean(axis=1)

    start = max(lookback + skip, vol_win) + 1
    w = np.zeros(N)
    out = []
    for t in range(start, T - 1):
        if (t - start) % rebalance == 0:
            vol = rets[t - vol_win:t].std(axis=0) * np.sqrt(ann)
            frozen = (np.abs(rets[t - vol_win:t]) < 1e-12).mean(axis=0)
            active = np.where((frozen < 0.3) & (vol > 0.03))[0]
            new_w = np.zeros(N)
            if len(active) >= 2 * n_side:
                mom = P[t - skip] / P[t - lookback] - 1.0
                ranked = active[np.argsort(mom[active])[::-1]]
                longs, shorts = ranked[:n_side], ranked[-n_side:]
                for side, sign in ((longs, 1.0), (shorts, -1.0)):
                    inv = np.array([1.0 / max(vol[i], 0.05) for i in side])
                    ww = inv / inv.sum() * gross_per_side
                    for i, wi in zip(side, ww):
                        new_w[i] = sign * wi
            tc = np.abs(new_w - w).sum() * cost
            w = new_w
        else:
            tc = 0.0
        out.append(float(w @ rets[t]) - tc)

    r = np.asarray(out)
    equity = cap * np.cumprod(1 + r)
    split = int(len(r) * 0.7)
    bh = cap * (index[start: start + len(r) + 1] / index[start])
    return {
        "metrics": compute_metrics(equity, "day"),
        "metrics_recent": compute_metrics(cap * np.cumprod(1 + r[split:]), "day"),
        "bh_metrics": compute_metrics(bh, "day"),
        "equity": equity,
    }


def format_ls_report(res: dict) -> str:
    def pct(x):
        return f"{x*100:+.2f}%"

    m, rec, bh = res["metrics"], res["metrics_recent"], res["bh_metrics"]
    return "\n".join([
        "=" * 64,
        "ЛОНГ-ШОРТ МОМЕНТУМ (маркет-нейтральный, dollar-neutral)",
        "=" * 64,
        f"{'Метрика':<26}{'Стратегия':>12}{'Посл. треть':>13}{'Buy&Hold':>12}",
        "-" * 64,
        f"{'Доходность':<26}{pct(m['total_return']):>12}{pct(rec['total_return']):>13}{pct(bh['total_return']):>12}",
        f"{'Коэф. Шарпа':<26}{m['sharpe']:>12.2f}{rec['sharpe']:>13.2f}{bh['sharpe']:>12.2f}",
        f"{'Макс. просадка':<26}{pct(m['max_drawdown']):>12}{pct(rec['max_drawdown']):>13}{pct(bh['max_drawdown']):>12}",
        "=" * 64,
    ])

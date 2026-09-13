"""Текущий целевой портфель по «выигрышной» логике слоя B.

Та же самая логика, что в бэктесте momentum_backtest (моментум top-N +
веса 1/волатильность + режимный фильтр по 200-дневной средней + volatility
targeting), применённая к последнему бару — «что держать на следующей неделе».

Ничего не обучается; параметры стандартные из литературы, те же, что дали
+8.6% против +1.9% у рынка в бэктесте 2021–2026.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def current_target_portfolio(
    prices: pd.DataFrame,
    lookback: int = 126,
    skip: int = 5,
    top_n: int = 8,
    vol_win: int = 60,
    target_vol: float = 0.15,
    regime_win: int = 200,
    ann: int = 252,
) -> Dict:
    """Целевые веса портфеля на сегодня + диагностика режима."""
    names = list(prices.columns)
    P = prices.to_numpy(dtype=np.float64)
    T = len(P)
    if T < max(lookback + skip, regime_win, vol_win) + 2:
        raise ValueError("Недостаточно истории для расчёта портфеля.")

    rets = np.diff(P, axis=0) / P[:-1]
    index = (P / P[0]).mean(axis=1)
    index_sma = pd.Series(index).rolling(regime_win).mean().to_numpy()
    bull = bool(index[-1] > index_sma[-1])
    regime_strength = float(index[-1] / index_sma[-1] - 1.0)

    vol = rets[-vol_win:].std(axis=0) * np.sqrt(ann)
    frozen = (np.abs(rets[-vol_win:]) < 1e-12).mean(axis=0)
    active = (frozen < 0.3) & (vol > 0.03)

    mom = P[-1 - skip] / P[-1 - lookback] - 1.0
    order = np.argsort(mom)[::-1]
    chosen = [i for i in order if mom[i] > 0 and active[i]][:top_n]

    weights: Dict[str, float] = {}
    basket_vol = 0.0
    scale = 0.0
    if chosen and bull:
        inv = np.array([1.0 / max(vol[i], 0.05) for i in chosen])
        ww = inv / inv.sum()
        basket = rets[-vol_win:][:, chosen] @ ww
        basket_vol = float(basket.std() * np.sqrt(ann))
        scale = float(min(1.0, target_vol / max(basket_vol, 1e-4)))
        for i, wi in zip(chosen, ww):
            weights[names[i]] = float(wi * scale)

    return {
        "weights": weights,                      # тикер -> доля капитала
        "bull_regime": bull,                     # False -> весь портфель в кэше
        "regime_strength": regime_strength,      # индекс/SMA200 - 1
        "basket_vol": basket_vol,
        "vol_scale": scale,
        "cash_weight": float(1.0 - sum(weights.values())),
        "momentum": {names[i]: float(mom[i]) for i in chosen},
        "asof": str(prices.index[-1].date()),
        "n_active": int(active.sum()),
    }

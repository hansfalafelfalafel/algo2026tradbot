"""Поиск коинтегрированных пар для маркет-нейтрального (парного) трейдинга.

Идея парного трейдинга: если два инструмента исторически ходят вместе, их спред
(разница) колеблется вокруг среднего. Когда спред сильно отклоняется — открываем
позицию против отклонения (лонг отставший + шорт убежавший) и зарабатываем на
возврате к среднему. Такая стратегия **не зависит от направления рынка** — ей всё
равно, растёт рынок или падает, что напрямую лечит проблему прошлого подхода.

Здесь мы отбираем пары по тесту коинтеграции (Энгла–Грейнджера): чем ниже
p-value, тем устойчивее связь и пригоднее пара для возврата к среднему.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd


def _align_closes(candle_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Свести цены закрытия всех инструментов в одну таблицу по общему времени."""
    series = {}
    for name, df in candle_dfs.items():
        s = df.copy()
        s["time"] = pd.to_datetime(s["time"], utc=True)
        series[name] = s.set_index("time")["close"]
    prices = pd.DataFrame(series).dropna()
    return prices


def _half_life(spread: np.ndarray) -> float:
    """Период полураспада отклонения (за сколько баров спред возвращается вдвое)."""
    s = np.asarray(spread, dtype=np.float64)
    lag = s[:-1]
    delta = np.diff(s)
    beta = np.polyfit(lag - lag.mean(), delta, 1)[0]
    if beta >= 0:
        return np.inf  # нет возврата к среднему
    return float(-np.log(2) / beta)


def find_cointegrated_pairs(
    candle_dfs: Dict[str, pd.DataFrame],
    max_pvalue: float = 0.05,
) -> pd.DataFrame:
    """Найти и ранжировать пары по коинтеграции.

    :return: DataFrame с колонками y, x, pvalue, corr, beta, half_life —
        отсортированный по возрастанию p-value (лучшие пары сверху).
    """
    from statsmodels.tsa.stattools import coint

    prices = _align_closes(candle_dfs)
    logp = np.log(prices)
    names = list(prices.columns)

    rows: List[dict] = []
    for a, b in combinations(names, 2):
        y, x = logp[a].to_numpy(), logp[b].to_numpy()
        try:
            _, pvalue, _ = coint(y, x)
        except Exception:  # noqa: BLE001
            continue
        # Хедж-коэффициент через МНК: y = alpha + beta*x.
        beta = np.polyfit(x, y, 1)[0]
        spread = y - beta * x
        corr = float(np.corrcoef(prices[a], prices[b])[0, 1])
        hl = _half_life(spread)
        rows.append({
            "y": a, "x": b, "pvalue": float(pvalue), "corr": corr,
            "beta": float(beta), "half_life": hl,
        })

    df = pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)
    return df

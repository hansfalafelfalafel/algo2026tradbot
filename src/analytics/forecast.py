"""Прогноз роста капитала методом Monte-Carlo.

Честный подход: вместо одной «красивой линии» строим множество сценариев,
пересэмплируя историческую доходность стратегии (bootstrap), и показываем
доверительные коридоры (перцентили). Это отражает реальную неопределённость.
"""
from __future__ import annotations

import numpy as np


def monte_carlo_forecast(
    returns: np.ndarray,
    capital: float,
    horizon: int,
    n_sims: int = 2000,
    percentiles=(5, 50, 95),
    block: int = 5,
    seed: int = 42,
) -> dict:
    """Симуляция траекторий капитала на horizon шагов вперёд.

    :param returns: исторические пошаговые доходности стратегии (доли).
    :param capital: текущий капитал.
    :param horizon: горизонт прогноза в шагах (барах).
    :param n_sims: число сценариев.
    :param percentiles: какие перцентили строить (коридоры).
    :param block: длина блока для block-bootstrap (сохраняет автокорреляцию).
    :return: dict с массивами перцентильных траекторий и сводными метриками.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        raise ValueError("Недостаточно исторических доходностей для прогноза.")

    rng = np.random.default_rng(seed)
    # Block bootstrap: набираем горизонт кусками по block, чтобы сохранить
    # локальную структуру ряда (тренды/кластеры волатильности).
    n_blocks = int(np.ceil(horizon / block))
    max_start = len(r) - block
    paths = np.empty((n_sims, horizon))
    for s in range(n_sims):
        if max_start > 0:
            starts = rng.integers(0, max_start + 1, size=n_blocks)
            seq = np.concatenate([r[i : i + block] for i in starts])[:horizon]
        else:
            seq = rng.choice(r, size=horizon, replace=True)
        paths[s] = capital * np.cumprod(1.0 + seq)

    # Перцентильные коридоры на каждом шаге.
    pct = {p: np.percentile(paths, p, axis=0) for p in percentiles}
    finals = paths[:, -1]

    median_final = float(np.median(finals))
    # Ожидаемая доходность за горизонт и её приведение к «годовым» (по медиане).
    total_return = median_final / capital - 1.0
    prob_loss = float(np.mean(finals < capital))

    return {
        "horizon": horizon,
        "percentiles": {int(p): pct[p] for p in percentiles},
        "median_path": pct.get(50, np.median(paths, axis=0)),
        "final_median": median_final,
        "final_p5": float(np.percentile(finals, 5)),
        "final_p95": float(np.percentile(finals, 95)),
        "expected_total_return": float(total_return),
        "prob_loss": prob_loss,
        "capital": float(capital),
    }

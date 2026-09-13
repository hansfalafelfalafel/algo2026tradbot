"""Слой B: кросс-секционный моментум + volatility targeting + режимный фильтр.

Три документированных компонента (см. docs/EDGE_ARCHITECTURE.md):
  * Моментум: раз в `rebalance` дней покупаем top-N акций по доходности за
    последние ~6 месяцев (пропуская последнюю неделю — классический skip против
    краткосрочного разворота). Берём только акции с положительным моментумом.
  * Веса ∝ 1/волатильность (risk parity внутри корзины).
  * Режимный фильтр: если равновзвешенный индекс нашей вселенной ниже своей
    200-дневной средней — уходим в кэш (медвежий режим; наши данные показали,
    что в обвал кэш — лучший актив).
  * Volatility targeting: масштабируем экспозицию так, чтобы ожидаемая
    волатильность портфеля была ~target_vol годовых (без плеча, максимум 1.0).

Ключевое: НИЧЕГО не обучается — все решения используют только прошлое окно,
поэтому весь бэктест по построению out-of-sample (параметры стандартны из
литературы, не подбирались под наши данные).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.backtest.backtest import compute_metrics


def align_prices(candle_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = {}
    for name, df in candle_dfs.items():
        s = df.copy()
        s["time"] = pd.to_datetime(s["time"], utc=True)
        series[name] = s.set_index("time")["close"]
    # outer join + ffill: у бумаг бывают пропуски отдельных дней
    prices = pd.DataFrame(series).sort_index().ffill().dropna()
    return prices


def momentum_backtest(
    prices: pd.DataFrame,
    lookback: int = 126,      # ~6 месяцев
    skip: int = 5,            # пропустить последнюю неделю
    top_n: int = 8,
    vol_win: int = 60,
    target_vol: float = 0.15, # целевая годовая волатильность 15%
    rebalance: int = 5,       # ребаланс раз в неделю
    regime_win: int = 200,    # 200-дневная средняя индекса
    cost: float = 0.0007,
    cap: float = 100_000.0,
    ann: int = 252,
) -> dict:
    P = prices.to_numpy(dtype=np.float64)
    T, N = P.shape
    rets = np.diff(P, axis=0) / P[:-1]                 # (T-1, N)
    index = (P / P[0]).mean(axis=1)                    # равновзвешенный индекс
    index_sma = pd.Series(index).rolling(regime_win).mean().to_numpy()

    start = max(lookback + skip, regime_win, vol_win) + 1
    w = np.zeros(N)
    port_ret = []
    weights_log = []
    exposure_log = []

    for t in range(start, T - 1):
        if (t - start) % rebalance == 0:
            # --- моментум-скор ---
            mom = P[t - skip] / P[t - lookback] - 1.0
            order = np.argsort(mom)[::-1]
            chosen = [i for i in order[:top_n] if mom[i] > 0]

            new_w = np.zeros(N)
            if chosen and index[t] > index_sma[t]:     # режим: только бычий
                vol = rets[t - vol_win:t].std(axis=0) * np.sqrt(ann)
                inv = np.array([1.0 / max(vol[i], 1e-4) for i in chosen])
                ww = inv / inv.sum()
                # --- volatility targeting (по волатильности корзины) ---
                basket = rets[t - vol_win:t][:, chosen] @ ww
                basket_vol = basket.std() * np.sqrt(ann)
                scale = min(1.0, target_vol / max(basket_vol, 1e-4))
                for i, wi in zip(chosen, ww):
                    new_w[i] = wi * scale
            turnover = np.abs(new_w - w).sum()
            w = new_w
            tc = turnover * cost
        else:
            tc = 0.0
        port_ret.append(float(w @ rets[t]) - tc)
        weights_log.append(w.copy())
        exposure_log.append(w.sum())

    r = np.asarray(port_ret)
    equity = cap * np.cumprod(1 + r)

    # Бенчмарки на том же отрезке.
    bh = cap * (index[start: start + len(r) + 1] / index[start])
    split = int(len(r) * 0.7)  # последняя треть — «свежий» период отдельно

    return {
        "equity": equity,
        "returns": r,
        "buyhold": bh,
        "metrics": compute_metrics(equity, "day"),
        "metrics_recent": compute_metrics(cap * np.cumprod(1 + r[split:]), "day"),
        "bh_metrics": compute_metrics(bh, "day"),
        "avg_exposure": float(np.mean(exposure_log)),
        "time_in_market": float(np.mean(np.asarray(exposure_log) > 0.01)),
        "times": prices.index[start + 1: start + 1 + len(r)],
    }


def format_momentum_report(res: dict) -> str:
    def pct(x):
        return f"{x*100:+.2f}%"

    m, rec, bh = res["metrics"], res["metrics_recent"], res["bh_metrics"]
    return "\n".join([
        "=" * 64,
        "СЛОЙ B: МОМЕНТУМ + VOL-TARGETING + РЕЖИМНЫЙ ФИЛЬТР (дневки)",
        "=" * 64,
        f"{'Метрика':<26}{'Стратегия':>12}{'Посл. треть':>13}{'Buy&Hold':>12}",
        "-" * 64,
        f"{'Доходность':<26}{pct(m['total_return']):>12}{pct(rec['total_return']):>13}{pct(bh['total_return']):>12}",
        f"{'Коэф. Шарпа':<26}{m['sharpe']:>12.2f}{rec['sharpe']:>13.2f}{bh['sharpe']:>12.2f}",
        f"{'Макс. просадка':<26}{pct(m['max_drawdown']):>12}{pct(rec['max_drawdown']):>13}{pct(bh['max_drawdown']):>12}",
        "-" * 64,
        f"Средняя экспозиция: {res['avg_exposure']:.2f} | Время в рынке: {res['time_in_market']*100:.0f}%",
        "Параметры стандартные из литературы (не подбирались под данные).",
        "=" * 64,
    ])

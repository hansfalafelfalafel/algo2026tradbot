"""Слой C: мета-лейблинг (Лопес де Прадо) поверх сигналов слоя B.

Идея: базовая стратегия (моментум) генерирует сигналы «купить акцию i».
Мета-модель (градиентный бустинг) обучается предсказывать, окажется ли
КОНКРЕТНЫЙ сигнал прибыльным, по признакам контекста: сила моментума,
волатильность бумаги и рынка, режим, относительная сила сигнала и т.п.
Торгуются только сигналы с вероятностью успеха выше порога.

Честность: модель обучается walk-forward (расширяющееся окно), прогнозы всегда
только по прошлым данным; сравнение «слой B vs слой B+C» делается на одном и
том же out-of-sample отрезке.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.backtest.backtest import compute_metrics


def _build_events_and_run(
    prices: pd.DataFrame,
    lookback=126, skip=5, top_n=8, vol_win=60, target_vol=0.15,
    rebalance=5, regime_win=200, cost=0.0007, ann=252,
):
    """Один проход слоя B с записью событий-сигналов и их признаков."""
    P = prices.to_numpy(dtype=np.float64)
    T, N = P.shape
    rets = np.diff(P, axis=0) / P[:-1]
    index = (P / P[0]).mean(axis=1)
    index_sma = pd.Series(index).rolling(regime_win).mean().to_numpy()
    index_ret = np.diff(index) / index[:-1]

    start = max(lookback + skip, regime_win, vol_win) + 1
    events = []          # (t, asset, features, forward_ret)
    schedule = []        # ребаланс-даты и выбранные активы с весами до фильтра

    for t in range(start, T - 1, rebalance):
        mom = P[t - skip] / P[t - lookback] - 1.0
        order = np.argsort(mom)[::-1]
        chosen = [i for i in order[:top_n] if mom[i] > 0]
        if not chosen or index[t] <= index_sma[t]:
            schedule.append((t, [], {}))
            continue

        vol = rets[t - vol_win:t].std(axis=0) * np.sqrt(ann)
        inv = np.array([1.0 / max(vol[i], 1e-4) for i in chosen])
        ww = inv / inv.sum()
        basket = rets[t - vol_win:t][:, chosen] @ ww
        scale = min(1.0, target_vol / max(basket.std() * np.sqrt(ann), 1e-4))
        weights = {i: float(w * scale) for i, w in zip(chosen, ww)}

        mom_med = float(np.median(mom[np.isfinite(mom)]))
        idx_vol = float(index_ret[t - vol_win:t].std() * np.sqrt(ann))
        regime_str = float(index[t] / index_sma[t] - 1.0)
        h = min(rebalance, T - 1 - t)
        for rank, i in enumerate(chosen):
            fwd = float(P[t + h, i] / P[t, i] - 1.0) - cost
            feats = [
                float(mom[i]), float(mom[i] - mom_med), float(rank),
                float(vol[i]), idx_vol, regime_str,
                float(rets[t - 5:t, i].sum()),      # краткосрочный разворот
                float(np.corrcoef(rets[t - vol_win:t, i], index_ret[t - vol_win:t])[0, 1]),
            ]
            events.append({"t": t, "asset": i, "x": feats, "y": fwd})
        schedule.append((t, chosen, weights))

    return {"events": events, "schedule": schedule, "rets": rets,
            "start": start, "T": T, "rebalance": rebalance, "cost": cost}


def meta_momentum_backtest(
    prices: pd.DataFrame, cap: float = 100_000.0,
    p_threshold: float = 0.55, train_frac: float = 0.6,
    retrain_every: int = 10, **kw,
) -> Dict:
    """Сравнение слоя B и слоя B+C (мета-фильтр) на одном OOS-отрезке."""
    from sklearn.ensemble import GradientBoostingClassifier

    ctx = _build_events_and_run(prices, **kw)
    events, schedule = ctx["events"], ctx["schedule"]
    rets, rebalance, cost = ctx["rets"], ctx["rebalance"], ctx["cost"]

    reb_ts = [t for t, _, _ in schedule]
    t_split = reb_ts[int(len(reb_ts) * train_frac)]

    # --- walk-forward обучение мета-модели ---
    events_sorted = sorted(events, key=lambda e: e["t"])
    probs: Dict[tuple, float] = {}
    model, last_train_size = None, 0
    hist_X, hist_y = [], []
    ptr = 0
    for e in events_sorted:
        # добавляем в историю события, чьи исходы уже известны (t + rebalance <= e[t])
        while ptr < len(events_sorted) and events_sorted[ptr]["t"] + rebalance <= e["t"]:
            hist_X.append(events_sorted[ptr]["x"])
            hist_y.append(1 if events_sorted[ptr]["y"] > 0 else 0)
            ptr += 1
        if e["t"] < t_split:
            continue
        if len(hist_y) >= 50 and (model is None or len(hist_y) - last_train_size >= retrain_every):
            if len(set(hist_y)) > 1:
                model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                   random_state=42)
                model.fit(np.array(hist_X), np.array(hist_y))
                last_train_size = len(hist_y)
        probs[(e["t"], e["asset"])] = (
            float(model.predict_proba([e["x"]])[0, 1]) if model is not None else 1.0
        )

    # --- симуляция обоих портфелей на OOS (t >= t_split) ---
    def run(filtered: bool):
        N = rets.shape[1]
        w = np.zeros(N)
        out = []
        for k, (t, chosen, weights) in enumerate(schedule):
            if t < t_split:
                continue
            new_w = np.zeros(N)
            for i in chosen:
                wi = weights[i]
                if filtered:
                    p = probs.get((t, i), 1.0)
                    if p < p_threshold:
                        continue
                new_w[i] = wi
            tc = np.abs(new_w - w).sum() * cost
            w = new_w
            t_end = schedule[k + 1][0] if k + 1 < len(schedule) else ctx["T"] - 1
            for tt in range(t, min(t_end, rets.shape[0])):
                r = float(w @ rets[tt]) - (tc if tt == t else 0.0)
                out.append(r)
        return np.asarray(out)

    r_base = run(filtered=False)
    r_meta = run(filtered=True)

    n_oos = len([e for e in events_sorted if e["t"] >= t_split])
    n_kept = len([1 for k, p in probs.items() if p >= p_threshold])
    return {
        "base_metrics": compute_metrics(cap * np.cumprod(1 + r_base), "day"),
        "meta_metrics": compute_metrics(cap * np.cumprod(1 + r_meta), "day"),
        "n_signals_oos": n_oos,
        "n_signals_kept": n_kept,
        "oos_days": len(r_base),
    }


def format_meta_report(res: Dict) -> str:
    def pct(x):
        return f"{x*100:+.2f}%"

    b, m = res["base_metrics"], res["meta_metrics"]
    kept = res["n_signals_kept"] / max(1, res["n_signals_oos"]) * 100
    return "\n".join([
        "=" * 64,
        "СЛОЙ C: МЕТА-ЛЕЙБЛИНГ ПОВЕРХ МОМЕНТУМА — сравнение на одном OOS",
        "=" * 64,
        f"OOS дней: {res['oos_days']} | сигналов: {res['n_signals_oos']} | "
        f"мета-фильтр пропустил: {kept:.0f}%",
        "-" * 64,
        f"{'Метрика':<26}{'Слой B (без C)':>18}{'Слой B + C':>18}",
        "-" * 64,
        f"{'Доходность':<26}{pct(b['total_return']):>18}{pct(m['total_return']):>18}",
        f"{'Коэф. Шарпа':<26}{b['sharpe']:>18.2f}{m['sharpe']:>18.2f}",
        f"{'Макс. просадка':<26}{pct(b['max_drawdown']):>18}{pct(m['max_drawdown']):>18}",
        "=" * 64,
    ])

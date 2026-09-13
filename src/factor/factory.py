"""Фабрика сигналов: несколько независимых типов сигналов + общий мета-фильтр.

Три источника (по литературе — слабо коррелированные между собой):
  * MOM — кросс-секционный моментум (6 мес, работает в трендовом рынке);
  * REV — краткосрочный недельный разворот (лонг перепроданных, работает в
    боковике/панике — противофаза моментуму);
  * LOWVOL — защитный: лонг самых низковолатильных бумаг (аномалия низкой
    волатильности, живуча в слабых рынках).

Каждый сигнал — «событие» с признаками контекста и типом. Мета-модель
(walk-forward, только прошлые данные) предсказывает вероятность успеха события;
торгуются события с P >= порога. Сравниваем пул без фильтра vs с фильтром на
одном OOS-отрезке.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.backtest.backtest import compute_metrics

SIGNAL_TYPES = ["MOM", "REV", "LOWVOL"]


def generate_events(
    prices: pd.DataFrame,
    lookback=126, skip=5, top_n=6, rev_win=5, rev_n=4, lowvol_n=5,
    vol_win=60, rebalance=5, regime_win=200, cost=0.0007, ann=252,
):
    P = prices.to_numpy(dtype=np.float64)
    T, N = P.shape
    rets = np.diff(P, axis=0) / P[:-1]
    index = (P / P[0]).mean(axis=1)
    index_sma = pd.Series(index).rolling(regime_win).mean().to_numpy()
    index_ret = np.diff(index) / index[:-1]

    start = max(lookback + skip, regime_win, vol_win) + 1
    events: List[dict] = []
    reb_dates: List[int] = []

    for t in range(start, T - 1, rebalance):
        reb_dates.append(t)
        vol = rets[t - vol_win:t].std(axis=0) * np.sqrt(ann)
        # Гигиена: исключаем «замороженные»/неликвидные бумаги (много дней без
        # движения или волатильность ~0 из-за ffill) — иначе 1/vol даёт им
        # гигантский вес и портфель перестаёт торговать.
        frozen = (np.abs(rets[t - vol_win:t]) < 1e-12).mean(axis=0)
        active = (frozen < 0.3) & (vol > 0.03)
        idx_vol = float(index_ret[t - vol_win:t].std() * np.sqrt(ann))
        regime_str = float(index[t] / index_sma[t] - 1.0)
        bull = index[t] > index_sma[t]
        h = min(rebalance, T - 1 - t)

        cands: List[tuple] = []  # (type, asset, score)
        # MOM — только в бычьем режиме
        if bull:
            mom = P[t - skip] / P[t - lookback] - 1.0
            order = np.argsort(mom)[::-1]
            for rank, i in enumerate([i for i in order[:top_n]
                                      if mom[i] > 0 and active[i]]):
                cands.append(("MOM", i, float(mom[i]), rank))
        # REV — перепроданные за неделю (в любом режиме)
        rev = P[t] / P[t - rev_win] - 1.0
        order = [i for i in np.argsort(rev) if active[i]]
        for rank, i in enumerate(order[:rev_n]):
            if rev[i] < -0.02:  # только заметная просадка
                cands.append(("REV", i, float(rev[i]), rank))
        # LOWVOL — самые «тихие» из ЖИВЫХ бумаг (в любом режиме)
        order = [i for i in np.argsort(vol) if active[i]]
        for rank, i in enumerate(order[:lowvol_n]):
            cands.append(("LOWVOL", i, float(vol[i]), rank))

        for typ, i, score, rank in cands:
            fwd = float(P[t + h, i] / P[t, i] - 1.0) - cost
            corr = float(np.corrcoef(rets[t - vol_win:t, i],
                                     index_ret[t - vol_win:t])[0, 1])
            x = ([1.0 if typ == s else 0.0 for s in SIGNAL_TYPES]
                 + [score, float(rank), float(vol[i]), idx_vol, regime_str,
                    float(rets[t - 5:t, i].sum()), corr, 1.0 if bull else 0.0])
            # Очистка: у «замороженных» бумаг (нет торгов) corr/vol дают NaN.
            x = [float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for v in x]
            events.append({"t": t, "asset": i, "type": typ, "x": x, "y": fwd,
                           "vol": float(max(vol[i], 1e-4))})

    return {"events": events, "reb_dates": reb_dates, "rets": rets,
            "T": T, "rebalance": rebalance, "cost": cost}


def factory_backtest(
    prices: pd.DataFrame, cap: float = 100_000.0,
    p_threshold: float = 0.55, train_frac: float = 0.6,
    retrain_every: int = 20, max_gross: float = 1.0, per_signal_w: float = 0.12,
    **kw,
) -> Dict:
    from sklearn.ensemble import GradientBoostingClassifier

    ctx = generate_events(prices, **kw)
    events = sorted(ctx["events"], key=lambda e: e["t"])
    rets, rebalance, cost = ctx["rets"], ctx["rebalance"], ctx["cost"]
    reb_dates = ctx["reb_dates"]
    t_split = reb_dates[int(len(reb_dates) * train_frac)]

    # --- walk-forward мета-модель на пуле событий всех типов ---
    probs: Dict[tuple, float] = {}
    hist_X, hist_y = [], []
    model, last_n, ptr = None, 0, 0
    for e in events:
        while ptr < len(events) and events[ptr]["t"] + rebalance <= e["t"]:
            hist_X.append(events[ptr]["x"])
            hist_y.append(1 if events[ptr]["y"] > 0 else 0)
            ptr += 1
        if e["t"] < t_split:
            continue
        if len(hist_y) >= 80 and (model is None or len(hist_y) - last_n >= retrain_every):
            if len(set(hist_y)) > 1:
                model = GradientBoostingClassifier(n_estimators=120, max_depth=3,
                                                   random_state=42)
                model.fit(np.array(hist_X), np.array(hist_y))
                last_n = len(hist_y)
        probs[(e["t"], e["asset"], e["type"])] = (
            float(model.predict_proba([e["x"]])[0, 1]) if model is not None else 1.0
        )

    by_date: Dict[int, list] = {}
    for e in events:
        if e["t"] >= t_split:
            by_date.setdefault(e["t"], []).append(e)

    def run(filtered: bool):
        N = rets.shape[1]
        w = np.zeros(N)
        out = []
        dates = [t for t in reb_dates if t >= t_split]
        for k, t in enumerate(dates):
            sel = by_date.get(t, [])
            if filtered:
                sel = [e for e in sel
                       if probs.get((t, e["asset"], e["type"]), 0) >= p_threshold]
            new_w = np.zeros(N)
            if sel:
                inv = np.array([1.0 / max(e["vol"], 0.05) for e in sel])
                ww = inv / inv.sum() * min(max_gross, per_signal_w * len(sel))
                for e, wi in zip(sel, ww):
                    new_w[e["asset"]] += wi
            tc = np.abs(new_w - w).sum() * cost
            w = new_w
            t_end = dates[k + 1] if k + 1 < len(dates) else ctx["T"] - 1
            for tt in range(t, min(t_end, rets.shape[0])):
                out.append(float(w @ rets[tt]) - (tc if tt == t else 0.0))
        return np.asarray(out)

    r_base, r_meta = run(False), run(True)
    n_oos = sum(len(v) for v in by_date.values())
    n_kept = sum(1 for p in probs.values() if p >= p_threshold)

    # разбивка по типам сигналов (доля пропущенных фильтром)
    type_stats = {}
    for typ in SIGNAL_TYPES:
        tot = sum(1 for k in probs if k[2] == typ)
        kept = sum(1 for k, p in probs.items() if k[2] == typ and p >= p_threshold)
        type_stats[typ] = (tot, kept)

    return {
        "base_metrics": compute_metrics(cap * np.cumprod(1 + r_base), "day"),
        "meta_metrics": compute_metrics(cap * np.cumprod(1 + r_meta), "day"),
        "n_signals_oos": n_oos, "n_signals_kept": n_kept,
        "oos_days": len(r_base), "type_stats": type_stats,
    }


def format_factory_report(res: Dict) -> str:
    def pct(x):
        return f"{x*100:+.2f}%"

    b, m = res["base_metrics"], res["meta_metrics"]
    lines = [
        "=" * 64,
        "ФАБРИКА СИГНАЛОВ (MOM+REV+LOWVOL) + МЕТА-ФИЛЬТР — один OOS",
        "=" * 64,
        f"OOS дней: {res['oos_days']} | сигналов: {res['n_signals_oos']} | "
        f"пропущено фильтром: {res['n_signals_kept']}",
        "-" * 64,
        f"{'Метрика':<26}{'Пул без фильтра':>18}{'Пул + мета':>18}",
        "-" * 64,
        f"{'Доходность':<26}{pct(b['total_return']):>18}{pct(m['total_return']):>18}",
        f"{'Коэф. Шарпа':<26}{b['sharpe']:>18.2f}{m['sharpe']:>18.2f}",
        f"{'Макс. просадка':<26}{pct(b['max_drawdown']):>18}{pct(m['max_drawdown']):>18}",
        "-" * 64,
        "Фильтр по типам сигналов (всего -> пропущено):",
    ]
    for typ, (tot, kept) in res["type_stats"].items():
        lines.append(f"  {typ:<8} {tot:>4} -> {kept}")
    lines.append("=" * 64)
    return "\n".join(lines)

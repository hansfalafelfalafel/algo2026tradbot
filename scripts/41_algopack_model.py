"""Оценка OFI-сигнала на ИСТОРИИ AlgoPack (5-мин суперсвечи, месяцы данных).

Тот же протокол, что для собранного стакана (31), но на длинной истории:
  * признаки — микроструктурные поля tradestats/obstats/orderstats
    (дисбалансы покупок/продаж, дисбалансы стакана, статистика заявок);
  * метка — направление цены через horizon баров (бар = 5 минут);
  * оценка — walk-forward ПО НЕДЕЛЯМ: модель для недели N обучается только
    на неделях < N. Печатает AUC и чистый edge по неделям.

Запуск (после 40_algopack_load.py):
    python scripts/41_algopack_model.py
    python scripts/41_algopack_model.py --horizon 3
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

# Кандидаты в признаки (берём те, что реально есть в данных).
CAND_FEATURES = [
    # tradestats: дисбаланс сделок
    "disb", "vol_b", "vol_s", "trades_b", "trades_s", "pr_std", "pr_change",
    # obstats: дисбаланс стакана и спреды
    "imbalance_vol_bbo", "imbalance_val_bbo", "imbalance_vol", "imbalance_val",
    "spread_bbo", "spread_lv10", "spread_1mio", "levels_b", "levels_s",
    "vol_b_l1", "vol_s_l1",
    # orderstats: поток заявок
    "put_orders_b", "put_orders_s", "put_vol_b", "put_vol_s",
    "cancel_orders_b", "cancel_orders_s", "cancel_vol_b", "cancel_vol_s",
]


def _arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def _load_ticker(ap_dir: Path, tk: str):
    frames = []
    for stat in ("tradestats", "obstats", "orderstats"):
        p = ap_dir / f"{tk}_{stat}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        if "tradedate" in df.columns and "tradetime" in df.columns:
            df["ts"] = pd.to_datetime(df["tradedate"].astype(str) + " "
                                      + df["tradetime"].astype(str))
        else:
            tcol = next((c for c in ("ts", "systime", "begin", "time")
                         if c in df.columns), None)
            if tcol is None:
                continue
            df["ts"] = pd.to_datetime(df[tcol])
        df = df.drop_duplicates("ts").set_index("ts").sort_index()
        df = df.drop(columns=[c for c in ("tradedate", "tradetime", "secid", "systime")
                              if c in df.columns])
        frames.append(df)
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        f = f.drop(columns=[c for c in f.columns if c in out.columns])
        out = out.join(f, how="outer")
    return out


def main() -> None:
    ap_dir = ROOT / "data_cache" / "algopack"
    horizon = int(_arg("--horizon", 1))
    if not ap_dir.exists() or not list(ap_dir.glob("*_tradestats.csv")):
        print("Нет данных AlgoPack — сначала scripts/40_algopack_load.py")
        sys.exit(0)

    tickers = sorted({p.name.split("_")[0] for p in ap_dir.glob("*_tradestats.csv")})
    rows = []
    for tk in tickers:
        df = _load_ticker(ap_dir, tk)
        if df is None or len(df) < 500:
            continue
        price_col = next((c for c in ("pr_close", "close", "pr_vwap") if c in df.columns), None)
        if price_col is None:
            continue
        feats = [c for c in CAND_FEATURES if c in df.columns]
        d = df[[price_col] + feats].copy()
        d = d.apply(pd.to_numeric, errors="coerce")
        # нормализация объёмных признаков в z-оценки по скользящему окну
        for c in feats:
            s = d[c].rolling(100, min_periods=30)
            d[c] = ((d[c] - s.mean()) / s.std().replace(0, np.nan)).clip(-10, 10)
        d["fwd_ret"] = d[price_col].shift(-horizon) / d[price_col] - 1.0
        d["ticker"] = tk
        d = d.rename(columns={price_col: "price"})
        rows.append(d.reset_index())

    if not rows:
        print("Не удалось собрать признаки — пришлите вывод: "
              "head -3 data_cache/algopack/SBER_tradestats.csv")
        sys.exit(0)

    df = pd.concat(rows, ignore_index=True).dropna(subset=["fwd_ret"])
    feats = [c for c in CAND_FEATURES if c in df.columns]
    df[feats] = df[feats].fillna(0.0)
    df = df.sort_values("ts").reset_index(drop=True)
    print(f"Датасет AlgoPack: {len(df):,} 5-мин баров, {df.ticker.nunique()} тикеров, "
          f"{df.ts.dt.date.nunique()} торг. дней, признаков: {len(feats)}")
    print(f"Период: {df.ts.min()} — {df.ts.max()} | горизонт {horizon * 5} мин")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    y = (df["fwd_ret"] > 0).astype(int).to_numpy()
    X = df[feats].to_numpy()
    weeks = df.ts.dt.to_period("W")
    uniq_weeks = weeks.unique()

    cost_bp = 2 * (1.5 + 0.5)  # полуспред ~1.5 б.п. (ликвидные) + комиссия, x2
    print(f"Допущение издержек: {cost_bp:.1f} б.п. на круг (рыночные заявки)\n")
    print(f"{'Неделя':<12}{'баров':>8}{'AUC':>8}{'edge вниз':>12}{'edge вверх':>12}")
    print("-" * 54)

    aucs, e_dns, e_ups = [], [], []
    for i, w in enumerate(uniq_weeks):
        tr_m = (weeks < w).to_numpy()
        te_m = (weeks == w).to_numpy()
        if tr_m.sum() < 5000 or te_m.sum() < 500:
            continue
        yd = y[te_m]
        if len(set(yd)) < 2:
            continue
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=42)
        m.fit(X[tr_m], y[tr_m])
        p = m.predict_proba(X[te_m])[:, 1]
        a = roc_auc_score(yd, p)
        g = df[te_m].copy()
        g["p"] = p
        e_dn = -g[g.p <= g.p.quantile(0.1)].fwd_ret.mean() * 1e4 - cost_bp
        e_up = g[g.p >= g.p.quantile(0.9)].fwd_ret.mean() * 1e4 - cost_bp
        aucs.append(a); e_dns.append(e_dn); e_ups.append(e_up)
        print(f"{str(w):<12}{int(te_m.sum()):>8}{a:>8.3f}{e_dn:>+12.1f}{e_up:>+12.1f}")

    if aucs:
        print("-" * 54)
        pos_weeks = sum(1 for a in aucs if a > 0.55)
        print(f"ИТОГО: недель {len(aucs)} | средний AUC {np.mean(aucs):.3f} | "
              f"недель с AUC>0.55: {pos_weeks}/{len(aucs)}")
        print(f"Средний edge: вниз {np.mean(e_dns):+.1f} б.п., вверх {np.mean(e_ups):+.1f} б.п.")
        print("\nКритерий: средний AUC>0.55 И средний edge>0 на большинстве недель "
              "-> сигнал реален, строим исполнение.")


if __name__ == "__main__":
    main()

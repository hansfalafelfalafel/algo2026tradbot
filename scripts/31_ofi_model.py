"""Слой A: модель на дисбалансе потока заявок (OFI) из собранного стакана.

Запуск:
    python scripts/31_ofi_model.py
    python scripts/31_ofi_model.py --horizon 10

Считает ДВЕ модели (A/B): базовые 7 признаков и расширенные 14 — по мере
накопления дней данные сами покажут, какой набор сильнее. Печатает AUC,
чистый edge после издержек и стабильность по дням.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.config import load_config
from src.lob.dataset import FEATURES, FEATURES_BASIC, build_dataset

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))


def _arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main() -> None:
    cfg = load_config()
    lob_dir = cfg.abs_path(cfg.data["cache_dir"], "lob")
    horizon = int(_arg("--horizon", 5))

    if not lob_dir.exists() or not list(lob_dir.glob("*_book.csv")):
        print("Данных стакана ещё нет — коллектор должен поработать в торговые часы.")
        sys.exit(0)

    print("Строю датасет из собранного стакана...")
    df = build_dataset(lob_dir, horizon_min=horizon)
    if df.empty or len(df) < 2000:
        print(f"Пока мало данных: {len(df)} баров (нужно >= 2000). Коллектор копит.")
        sys.exit(0)

    days = df["time"].dt.date.nunique()
    print(f"Датасет: {len(df):,} баров (основная сессия), "
          f"{df.ticker.nunique()} тикеров, {days} торг. дней, горизонт {horizon} мин.")

    ms = _arg("--max-spread", 0)
    if ms:
        n0 = len(df)
        df = df[df.spread_bp <= ms]
        print(f"Фильтр ликвидности: спред <= {ms:g} б.п. -> {len(df):,} из {n0:,} баров, "
              f"{df.ticker.nunique()} тикеров")
    df = df.sort_values("time").reset_index(drop=True)
    split = int(len(df) * 0.6)
    y = (df["fwd_ret"] > 0).astype(int).to_numpy()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    fees = 0.5
    results = {}
    for name, feats in (("базовые-7", FEATURES_BASIC), ("расшир-14", FEATURES)):
        X = df[feats].to_numpy()
        model = HistGradientBoostingClassifier(max_depth=4, max_iter=200,
                                               random_state=42)
        model.fit(X[:split], y[:split])
        p = model.predict_proba(X[split:])[:, 1]
        oos = df.iloc[split:].copy()
        oos["p"] = p
        auc = roc_auc_score(y[split:], p)
        half_spread = oos.spread_bp.median() / 2
        cost = 2 * (half_spread + fees)
        up = oos[oos.p >= np.quantile(p, 0.9)].fwd_ret.mean() * 1e4 - cost
        dn = -oos[oos.p <= np.quantile(p, 0.1)].fwd_ret.mean() * 1e4 - cost
        results[name] = {"auc": auc, "up": up, "dn": dn, "oos": oos, "cost": cost}

    print("\n" + "=" * 66)
    print(f"{'Модель':<12}{'OOS AUC':>10}{'edge вверх':>14}{'edge вниз':>14}   (чистый, б.п.)")
    print("-" * 66)
    for name, r in results.items():
        print(f"{name:<12}{r['auc']:>10.3f}{r['up']:>+14.1f}{r['dn']:>+14.1f}")
    print(f"Издержки на круг: ~{results['базовые-7']['cost']:.1f} б.п. "
          f"(рыночные заявки; лимитные — заметно меньше)")
    print("-" * 66)

    # Стабильность по дням — для лучшей по AUC модели.
    best = max(results, key=lambda k: results[k]["auc"])
    oos = results[best]["oos"]
    cost = results[best]["cost"]
    oos["date"] = oos["time"].dt.date
    daily = []
    print(f"Стабильность по дням (модель {best}):")
    for d, g in oos.groupby("date"):
        if len(g) < 200 or g.fwd_ret.gt(0).nunique() < 2:
            continue
        a = roc_auc_score((g.fwd_ret > 0).astype(int), g.p)
        e_dn = -g[g.p <= g.p.quantile(0.1)].fwd_ret.mean() * 1e4 - cost
        e_up = g[g.p >= g.p.quantile(0.9)].fwd_ret.mean() * 1e4 - cost
        daily.append({"date": str(d), "bars": int(len(g)), "auc": round(float(a), 3),
                      "edge_dn": round(float(e_dn), 1), "edge_up": round(float(e_up), 1)})
        print(f"  {d}: баров {len(g):5d} | AUC {a:.3f} | edge вниз {e_dn:+.1f} | "
              f"вверх {e_up:+.1f} б.п.")
    print("=" * 66)

    # Сохраняем сводку для дашборда (вкладка «Слой A»).
    import json
    from src.state import STATE_DIR, now_iso
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "updated_at": now_iso(), "days": int(days), "bars": int(len(df)),
        "horizon_min": horizon, "cost_bp": round(float(cost), 1), "best": best,
        "models": {n: {"auc": round(float(r["auc"]), 3),
                       "edge_up": round(float(r["up"]), 1),
                       "edge_dn": round(float(r["dn"]), 1)} for n, r in results.items()},
        "daily": daily,
    }
    with open(STATE_DIR / "ofi_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Сводка сохранена для дашборда: state/ofi_results.json")


if __name__ == "__main__":
    main()

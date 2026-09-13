"""DL-модель (GRU) по ПОСЛЕДОВАТЕЛЬНОСТЯМ микроструктуры AlgoPack.

Отличие от 41 (бустинг): модель видит не один бар, а последние SEQ баров —
то есть «фильм» о том, как разворачивается поток заявок, а не один кадр.

Протокол: обучение на первых 60% времени, оценка на последних 40% с
разбивкой по неделям. Метрики те же: AUC и чистый edge топ-децилей после
издержек (4 б.п. на круг).

Требует PyTorch (CPU):
    pip install torch --index-url https://download.pytorch.org/whl/cpu

Запуск:
    python scripts/42_dl_model.py                # горизонт 5 мин
    python scripts/42_dl_model.py --horizon 6    # 30 мин
    python scripts/42_dl_model.py --epochs 5
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
SEQ = 24  # длина последовательности: 24 бара = 2 часа

CAND_FEATURES = [
    "disb", "vol_b", "vol_s", "trades_b", "trades_s", "pr_std", "pr_change",
    "imbalance_vol_bbo", "imbalance_val_bbo", "imbalance_vol", "imbalance_val",
    "spread_bbo", "spread_lv10", "spread_1mio", "levels_b", "levels_s",
    "vol_b_l1", "vol_s_l1",
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
            continue
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


def build_sequences(horizon: int):
    ap_dir = ROOT / "data_cache" / "algopack"
    tickers = sorted({p.name.split("_")[0] for p in ap_dir.glob("*_tradestats.csv")})
    X_list, y_list, t_list, r_list = [], [], [], []
    feats_ref = None
    for tk in tickers:
        df = _load_ticker(ap_dir, tk)
        if df is None or len(df) < SEQ + horizon + 100:
            continue
        price_col = next((c for c in ("pr_close", "close", "pr_vwap") if c in df.columns), None)
        if price_col is None:
            continue
        feats = [c for c in CAND_FEATURES if c in df.columns]
        if feats_ref is None:
            feats_ref = feats
        else:
            feats = [c for c in feats_ref if c in df.columns]
            if len(feats) != len(feats_ref):
                continue
        d = df[[price_col] + feats].apply(pd.to_numeric, errors="coerce")
        for c in feats:
            s = d[c].rolling(100, min_periods=30)
            d[c] = ((d[c] - s.mean()) / s.std().replace(0, np.nan)).clip(-10, 10)
        d["fwd_ret"] = d[price_col].shift(-horizon) / d[price_col] - 1.0
        d = d.dropna(subset=["fwd_ret"])
        F = d[feats].fillna(0.0).to_numpy(dtype=np.float32)
        R = d["fwd_ret"].to_numpy(dtype=np.float32)
        T = d.index.to_numpy()
        # скользящие окна длиной SEQ
        for i in range(SEQ, len(d)):
            X_list.append(F[i - SEQ:i])
            y_list.append(1.0 if R[i] > 0 else 0.0)
            r_list.append(R[i])
            t_list.append(T[i])
    X = np.stack(X_list)
    y = np.asarray(y_list, dtype=np.float32)
    r = np.asarray(r_list, dtype=np.float32)
    t = pd.to_datetime(pd.Series(t_list))
    order = np.argsort(t.values)
    return X[order], y[order], r[order], t.iloc[order].reset_index(drop=True), feats_ref


def main() -> None:
    horizon = int(_arg("--horizon", 1))
    epochs = int(_arg("--epochs", 4))

    try:
        import torch
        import torch.nn as nn
    except Exception:
        print("Нужен PyTorch (CPU): pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu")
        sys.exit(1)

    print("Собираю последовательности из AlgoPack...")
    X, y, r, t, feats = build_sequences(horizon)
    print(f"Последовательностей: {len(X):,} | окно {SEQ} баров x {X.shape[2]} признаков "
          f"| горизонт {horizon*5} мин")

    split = int(len(X) * 0.6)
    torch.manual_seed(42)
    dev = "cpu"

    class Net(nn.Module):
        def __init__(self, nf):
            super().__init__()
            self.gru = nn.GRU(nf, 32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1]).squeeze(-1)

    model = Net(X.shape[2]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xtr = torch.from_numpy(X[:split])
    ytr = torch.from_numpy(y[:split])
    n = len(Xtr)
    bs = 512
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print(f"эпоха {ep+1}/{epochs}: loss {tot/n:.4f}")

    # OOS-оценка
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(split, len(X), 4096):
            out = model(torch.from_numpy(X[i:i + 4096]))
            ps.append(torch.sigmoid(out).numpy())
    p = np.concatenate(ps)
    yo, ro, to = y[split:], r[split:], t.iloc[split:].reset_index(drop=True)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(yo, p)
    cost = 4.0
    q_hi, q_lo = np.quantile(p, 0.9), np.quantile(p, 0.1)
    e_up = ro[p >= q_hi].mean() * 1e4 - cost
    e_dn = -ro[p <= q_lo].mean() * 1e4 - cost

    print("\n" + "=" * 60)
    print(f"DL (GRU по последовательностям)  |  OOS AUC: {auc:.3f}")
    print(f"Чистый edge (издержки {cost} б.п.): вверх {e_up:+.1f} | вниз {e_dn:+.1f} б.п.")
    print("-" * 60)
    weeks = to.dt.to_period("W")
    print(f"{'Неделя':<12}{'AUC':>8}{'вниз':>8}{'вверх':>8}")
    for w in weeks.unique():
        m = (weeks == w).to_numpy()
        if m.sum() < 500 or len(set(yo[m])) < 2:
            continue
        a = roc_auc_score(yo[m], p[m])
        ed = -ro[m][p[m] <= np.quantile(p[m], 0.1)].mean() * 1e4 - cost
        eu = ro[m][p[m] >= np.quantile(p[m], 0.9)].mean() * 1e4 - cost
        print(f"{str(w):<12}{a:>8.3f}{ed:>+8.1f}{eu:>+8.1f}")
    print("=" * 60)
    print("Критерий (объявлен заранее): чистый edge >= +1 б.п. стабильно по неделям "
          "-> строим исполнение; иначе ветка DL-на-агрегатах закрывается.")


if __name__ == "__main__":
    main()

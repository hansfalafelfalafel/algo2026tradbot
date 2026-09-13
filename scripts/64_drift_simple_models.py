from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path("/root/rl-trading-tbank")
DATA = ROOT / "state" / "ofi_dataset_h5.pkl"

RET5_MEAN = -0.0000156187
RET5_STD  =  0.0020025922
RET1_MEAN = -0.0000033215
RET1_STD  =  0.0008796108

TRAIN_WINDOWS = [
    ("LONG",   "2026-07-28", "2026-08-25"),
    ("RECENT", "2026-08-11", "2026-08-25"),
    ("LAST7",  "2026-08-18", "2026-08-25"),
]

VAL_FROM = pd.Timestamp("2026-08-25", tz="UTC")
VAL_TO   = pd.Timestamp("2026-09-01", tz="UTC")

FEATURES = [
    "abs_score",
    "spread_bp",
    "ret_1m",
    "ret_5m",
    "ofi",
    "tfi",
    "imb1",
    "imb5",
    "micro_dev",
    "n_snap",
]

print("=" * 110)
print("DRIFT + SIMPLE MODEL DIAGNOSTICS")
print("=" * 110)

df = pd.read_pickle(DATA)

df["time"] = pd.to_datetime(
    df["time"],
    utc=True,
    errors="coerce",
)

for c in [
    "mid", "spread_bp", "ret_1m", "ret_5m",
    "ofi", "tfi", "imb1", "imb5",
    "micro_dev", "n_snap",
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(
    subset=["time", "ticker", "mid", "ret_1m", "ret_5m"]
).copy()

z5 = (df["ret_5m"] - RET5_MEAN) / RET5_STD
z1 = (df["ret_1m"] - RET1_MEAN) / RET1_STD

df["score_mr"] = -0.75*z5 - 0.25*z1
df["abs_score"] = df["score_mr"].abs()

# Используем уже существующий fwd_ret как быстрый диагностический target.
df["fwd_ret"] = pd.to_numeric(
    df["fwd_ret"],
    errors="coerce",
)

# MR side
df["side"] = np.where(
    df["score_mr"] > 0,
    1,
    -1,
)

# 10bp hurdle diagnostic.
df["future_bp"] = (
    df["side"]
    * df["fwd_ret"]
    * 10000
)

df["target"] = (
    df["future_bp"] > 10.0
).astype(int)

# сильные сигналы
df = df[
    (df["abs_score"] >= 1.30)
    & (df["spread_bp"] <= 2.0)
].copy()

val = df[
    (df["time"] >= VAL_FROM)
    & (df["time"] < VAL_TO)
].copy()

print("VAL rows:", len(val))
print("VAL prevalence:", f"{val['target'].mean():.3f}")

print()
print("=" * 110)
print("FEATURE DRIFT: TRAIN LONG vs VALIDATION")
print("=" * 110)

train_long = df[
    (df["time"] >= pd.Timestamp("2026-07-28", tz="UTC"))
    & (df["time"] < pd.Timestamp("2026-08-25", tz="UTC"))
].copy()

for f in FEATURES:
    a = pd.to_numeric(train_long[f], errors="coerce")
    b = pd.to_numeric(val[f], errors="coerce")

    am = a.median()
    bm = b.median()

    scale = a.std()

    shift = (
        (bm - am) / scale
        if pd.notna(scale) and scale > 0
        else np.nan
    )

    print(
        f"{f:15s} "
        f"train_med={am:+.6f} "
        f"val_med={bm:+.6f} "
        f"shift_sigma={shift:+.2f}"
    )

print()
print("=" * 110)
print("MODEL STABILITY")
print("=" * 110)

rows = []

for name, start, end in TRAIN_WINDOWS:
    train = df[
        (df["time"] >= pd.Timestamp(start, tz="UTC"))
        & (df["time"] < pd.Timestamp(end, tz="UTC"))
    ].copy()

    med = train[FEATURES].median()

    Xtr = train[FEATURES].fillna(med)
    ytr = train["target"]

    Xv = val[FEATURES].fillna(med)
    yv = val["target"]

    models = {
        "LOGIT": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                max_iter=1000,
                class_weight="balanced",
            )
        ),
        "HGB": HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=100,
            max_depth=2,
            min_samples_leaf=100,
            l2_regularization=5.0,
            random_state=42,
        ),
    }

    for model_name, model in models.items():
        model.fit(Xtr, ytr)

        ptr = model.predict_proba(Xtr)[:, 1]
        pv = model.predict_proba(Xv)[:, 1]

        tr_auc = roc_auc_score(ytr, ptr)
        va_auc = roc_auc_score(yv, pv)

        va_pr = average_precision_score(yv, pv)

        rows.append({
            "train_window": name,
            "model": model_name,
            "train_rows": len(train),
            "train_auc": tr_auc,
            "val_auc": va_auc,
            "val_auc_inverted": 1.0 - va_auc,
            "val_pr_auc": va_pr,
        })

out = pd.DataFrame(rows)

print(
    out.to_string(
        index=False,
        formatters={
            "train_auc": "{:.3f}".format,
            "val_auc": "{:.3f}".format,
            "val_auc_inverted": "{:.3f}".format,
            "val_pr_auc": "{:.3f}".format,
        }
    )
)

print()
print("=" * 110)
print("BEST VALIDATION")
print("=" * 110)

print(
    out.sort_values(
        "val_auc",
        ascending=False
    ).head(10).to_string(
        index=False,
        formatters={
            "train_auc": "{:.3f}".format,
            "val_auc": "{:.3f}".format,
            "val_auc_inverted": "{:.3f}".format,
            "val_pr_auc": "{:.3f}".format,
        }
    )
)

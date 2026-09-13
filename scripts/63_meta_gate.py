from pathlib import Path
import gc
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

ROOT = Path("/root/rl-trading-tbank")

HIST_FILE = ROOT / "state" / "ofi_dataset_h5.pkl"
LIVE_FILE = ROOT / "state" / "shadow_observer_features.csv"

OUT = ROOT / "state" / "meta_gate_results.csv"
OUT_TRADES = ROOT / "state" / "meta_gate_forward_trades.csv"

# ============================================================
# FROZEN NORMALIZATION
# ============================================================

RET5_MEAN = -0.0000156187
RET5_STD  =  0.0020025922

RET1_MEAN = -0.0000033215
RET1_STD  =  0.0008796108

# Наблюдаемая round-trip комиссия sandbox
FEE_BP = 10.0

# ============================================================
# SPLITS
# ============================================================

TRAIN_FROM = pd.Timestamp("2026-07-28", tz="UTC")
TRAIN_TO   = pd.Timestamp("2026-08-25", tz="UTC")

VAL_FROM = pd.Timestamp("2026-08-25", tz="UTC")
VAL_TO   = pd.Timestamp("2026-09-01", tz="UTC")

FWD_FROM = pd.Timestamp("2026-09-06", tz="UTC")

HORIZONS = [60, 90, 120]
MODES = ["MR", "MOM"]

# Не пытаемся классифицировать каждую минуту рынка.
# Gate работает поверх достаточно сильного alpha-сигнала.
BASE_SCORE_THRESHOLD = 1.30

PROB_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
]

FEATURES = [
    "abs_score",
    "signed_score",
    "spread_bp",
    "ret_1m",
    "ret_5m",
    "ofi",
    "tfi",
    "imb1",
    "imb5",
    "micro_dev",
    "breadth",
    "share_up",
    "n_snap",
    "tod_sin",
    "tod_cos",
]


# ============================================================
# LOAD
# ============================================================

def rebuild_ret5(live):
    live = (
        live.sort_values(["ticker", "time"])
        .reset_index(drop=True)
    )

    five_ns = pd.Timedelta(minutes=5).value
    tolerance_ns = pd.Timedelta(minutes=2).value

    pieces = []

    for ticker, g in live.groupby(
        "ticker",
        sort=False,
    ):
        g = g.copy()

        ts = (
            g["time"]
            .dt.tz_convert(None)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )

        mid = pd.to_numeric(
            g["mid"],
            errors="coerce",
        ).to_numpy(dtype=float)

        target = ts - five_ns

        j = np.searchsorted(
            ts,
            target,
            side="right",
        ) - 1

        ret5 = np.full(
            len(g),
            np.nan,
            dtype=np.float32,
        )

        valid = j >= 0
        rows = np.where(valid)[0]
        jj = j[valid]

        close = (
            target[rows] - ts[jj]
            <= tolerance_ns
        )

        rows = rows[close]
        jj = jj[close]

        good = (
            np.isfinite(mid[rows])
            & np.isfinite(mid[jj])
            & (mid[jj] > 0)
        )

        rows = rows[good]
        jj = jj[good]

        ret5[rows] = (
            mid[rows] / mid[jj] - 1.0
        )

        g["ret_5m"] = ret5
        pieces.append(g)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def load_data():
    print("=" * 120)
    print("LOAD DATA")
    print("=" * 120)

    hist = pd.read_pickle(HIST_FILE)

    wanted = [
        "time",
        "ticker",
        "mid",
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

    hist = hist[
        [c for c in wanted if c in hist.columns]
    ].copy()

    hist["time"] = pd.to_datetime(
        hist["time"],
        utc=True,
        errors="coerce",
    )

    print("historical:", f"{len(hist):,}")

    live_cols = [
        "time",
        "ticker",
        "mid",
        "spread_bp",
        "ret_1m",
        "ofi",
        "tfi",
        "imb1",
        "imb5",
        "micro_dev",
        "n_snap",
    ]

    live = pd.read_csv(
        LIVE_FILE,
        usecols=lambda c: c in set(live_cols),
    )

    live["time"] = pd.to_datetime(
        live["time"],
        utc=True,
        errors="coerce",
    )

    print("live:", f"{len(live):,}")

    live = rebuild_ret5(live)

    print(
        "live ret_5m:",
        f"{live['ret_5m'].notna().sum():,}",
        "/",
        f"{len(live):,}",
    )

    df = pd.concat(
        [hist, live],
        ignore_index=True,
        sort=False,
    )

    del hist, live
    gc.collect()

    numeric = [
        "mid",
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

    for c in numeric:
        if c not in df.columns:
            df[c] = np.nan

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    # Некоторые order-flow признаки могут отсутствовать
    # на отдельных строках — оставляем NaN:
    # HistGradientBoosting умеет с ними работать.
    df = df.dropna(
        subset=[
            "time",
            "ticker",
            "mid",
            "spread_bp",
            "ret_1m",
            "ret_5m",
        ]
    )

    df = df[
        (df["mid"] > 0)
        & (df["spread_bp"] >= 0)
    ]

    df = (
        df.sort_values(
            ["ticker", "time"]
        )
        .drop_duplicates(
            ["ticker", "time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Alpha scores
    # --------------------------------------------------------

    z5 = (
        df["ret_5m"] - RET5_MEAN
    ) / RET5_STD

    z1 = (
        df["ret_1m"] - RET1_MEAN
    ) / RET1_STD

    df["score_mr"] = (
        -0.75 * z5
        -0.25 * z1
    ).astype("float32")

    df["score_mom"] = (
        -df["score_mr"]
    ).astype("float32")

    # --------------------------------------------------------
    # Breadth
    # --------------------------------------------------------

    up = (
        df["ret_5m"] > 0
    ).astype("float32")

    share_up = (
        pd.DataFrame({
            "time": df["time"],
            "up": up,
        })
        .groupby("time")["up"]
        .mean()
    )

    df["share_up"] = (
        df["time"]
        .map(share_up)
        .astype("float32")
    )

    df["breadth"] = np.maximum(
        df["share_up"],
        1.0 - df["share_up"],
    ).astype("float32")

    # --------------------------------------------------------
    # Intraday time, cyclic
    # --------------------------------------------------------

    msk = df["time"].dt.tz_convert(
        "Europe/Moscow"
    )

    minute = (
        msk.dt.hour * 60
        + msk.dt.minute
    )

    angle = (
        2.0
        * np.pi
        * minute
        / 1440.0
    )

    df["tod_sin"] = np.sin(angle).astype(
        "float32"
    )

    df["tod_cos"] = np.cos(angle).astype(
        "float32"
    )

    # Save memory
    for c in [
        "mid",
        "spread_bp",
        "ret_1m",
        "ret_5m",
        "ofi",
        "tfi",
        "imb1",
        "imb5",
        "micro_dev",
        "n_snap",
        "share_up",
        "breadth",
    ]:
        df[c] = df[c].astype("float32")

    print(
        "usable:",
        f"{len(df):,}"
    )

    print(
        "range:",
        df["time"].min(),
        "->",
        df["time"].max(),
    )

    return df


# ============================================================
# FUTURE EXIT
# ============================================================

def add_exit(df, horizon):
    pieces = []

    delta_ns = pd.Timedelta(
        minutes=horizon
    ).value

    tolerance_ns = pd.Timedelta(
        minutes=3
    ).value

    keep = [
        "time",
        "ticker",
        "mid",
        "spread_bp",
        "ret_1m",
        "ret_5m",
        "ofi",
        "tfi",
        "imb1",
        "imb5",
        "micro_dev",
        "n_snap",
        "share_up",
        "breadth",
        "tod_sin",
        "tod_cos",
        "score_mr",
        "score_mom",
    ]

    base = df[keep]

    for ticker, g in base.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values("time").copy()

        ts = (
            g["time"]
            .dt.tz_convert(None)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )

        mid = g["mid"].to_numpy(
            dtype=float
        )

        spr = g["spread_bp"].to_numpy(
            dtype=float
        )

        target = ts + delta_ns

        j = np.searchsorted(
            ts,
            target,
            side="left",
        )

        exit_mid = np.full(
            len(g),
            np.nan,
            dtype=np.float32,
        )

        exit_spread = np.full(
            len(g),
            np.nan,
            dtype=np.float32,
        )

        exit_ns = np.full(
            len(g),
            np.iinfo(np.int64).min,
            dtype=np.int64,
        )

        valid = j < len(g)

        rows = np.where(valid)[0]
        jj = j[valid]

        close = (
            ts[jj] - target[rows]
            <= tolerance_ns
        )

        rows = rows[close]
        jj = jj[close]

        exit_mid[rows] = mid[jj]
        exit_spread[rows] = spr[jj]
        exit_ns[rows] = ts[jj]

        g["exit_mid"] = exit_mid
        g["exit_spread"] = exit_spread

        exit_time = pd.Series(
            pd.NaT,
            index=g.index,
            dtype="datetime64[ns, UTC]",
        )

        if len(rows):
            exit_time.iloc[rows] = (
                pd.to_datetime(
                    exit_ns[rows],
                    utc=True,
                )
            )

        g["exit_time"] = exit_time

        pieces.append(g)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


# ============================================================
# CANDIDATES
# ============================================================

def make_candidates(hdf, mode):
    score_col = (
        "score_mr"
        if mode == "MR"
        else "score_mom"
    )

    c = hdf[
        hdf["exit_mid"].notna()
        & hdf["exit_time"].notna()
        & (hdf["spread_bp"] <= 2.0)
        & (
            hdf[score_col].abs()
            >= BASE_SCORE_THRESHOLD
        )
    ].copy()

    c["side"] = np.where(
        c[score_col] > 0,
        1,
        -1,
    ).astype(np.int8)

    c["abs_score"] = (
        c[score_col].abs()
        .astype("float32")
    )

    # signed_score всегда положителен в направлении сделки,
    # но magnitude сохраняет силу alpha.
    c["signed_score"] = (
        c[score_col]
        * c["side"]
    ).astype("float32")

    c["gross_bp"] = (
        c["side"]
        * (
            c["exit_mid"] / c["mid"]
            - 1.0
        )
        * 10000.0
    )

    c["cost_bp"] = (
        FEE_BP
        + 0.5 * c["spread_bp"]
        + 0.5 * c["exit_spread"]
    )

    c["net_bp"] = (
        c["gross_bp"]
        - c["cost_bp"]
    )

    # Classification target:
    # сделка уже после реальных расходов > 0
    c["target"] = (
        c["net_bp"] > 0
    ).astype(np.int8)

    return c


# ============================================================
# NON-OVERLAP
# ============================================================

def apply_gate(c, probability, threshold):
    x = c.copy()

    x["p_good"] = probability

    x = x[
        x["p_good"] >= threshold
    ].copy()

    if x.empty:
        return x

    # В одну минуту максимум один самый уверенный сигнал.
    # Сильно режем turnover.
    x = (
        x.sort_values(
            ["time", "p_good"],
            ascending=[True, False],
        )
        .groupby(
            "time",
            group_keys=False,
        )
        .head(1)
    )

    # Non-overlap per ticker
    keep = []

    for ticker, g in x.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values("time")

        busy_until = None

        for idx, r in g.iterrows():
            t = r["time"]

            if (
                busy_until is not None
                and t < busy_until
            ):
                continue

            keep.append(idx)
            busy_until = r["exit_time"]

    if not keep:
        return x.iloc[:0]

    return x.loc[keep].copy()


# ============================================================
# METRICS
# ============================================================

def pnl_metrics(x):
    if x.empty:
        return {
            "trades": 0,
            "net": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "wr": np.nan,
            "days": 0,
            "positive_days": 0,
            "without_best_day": np.nan,
            "without_best_ticker": np.nan,
        }

    x = x.copy()

    x["date_msk"] = (
        x["time"]
        .dt.tz_convert("Europe/Moscow")
        .dt.date
    )

    total = x["net_bp"].sum()

    daily = (
        x.groupby("date_msk")
        ["net_bp"]
        .sum()
    )

    ticker = (
        x.groupby("ticker")
        ["net_bp"]
        .sum()
    )

    return {
        "trades": int(len(x)),
        "net": float(total),
        "mean": float(
            x["net_bp"].mean()
        ),
        "median": float(
            x["net_bp"].median()
        ),
        "wr": float(
            100
            * (x["net_bp"] > 0).mean()
        ),
        "days": int(len(daily)),
        "positive_days": int(
            (daily > 0).sum()
        ),
        "without_best_day": float(
            total - daily.max()
        ),
        "without_best_ticker": float(
            total - ticker.max()
        ),
    }


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    return (
        roc_auc_score(y, p),
        average_precision_score(y, p),
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 120)
print("META-GATE LAB")
print("=" * 120)
print(
    "Train:",
    TRAIN_FROM,
    "->",
    TRAIN_TO,
)
print(
    "Validation:",
    VAL_FROM,
    "->",
    VAL_TO,
)
print(
    "Forward:",
    FWD_FROM,
    "-> latest",
)
print(
    "Fee assumption:",
    FEE_BP,
    "bp round-trip + spread",
)

df = load_data()

results = []
forward_trade_parts = []

for horizon in HORIZONS:
    print()
    print("=" * 120)
    print(f"HORIZON {horizon} MIN")
    print("=" * 120)

    hdf = add_exit(
        df,
        horizon,
    )

    for mode in MODES:
        print()
        print("-" * 100)
        print(mode)
        print("-" * 100)

        c = make_candidates(
            hdf,
            mode,
        )

        train = c[
            (c["time"] >= TRAIN_FROM)
            & (c["time"] < TRAIN_TO)
        ].copy()

        val = c[
            (c["time"] >= VAL_FROM)
            & (c["time"] < VAL_TO)
        ].copy()

        fwd = c[
            c["time"] >= FWD_FROM
        ].copy()

        print(
            "candidates:",
            "train",
            len(train),
            "| val",
            len(val),
            "| fwd",
            len(fwd),
        )

        if (
            len(train) < 200
            or len(val) < 20
        ):
            print("SKIP: insufficient samples")
            continue

        # Impute only using TRAIN medians.
        medians = (
            train[FEATURES]
            .median()
        )

        X_train = (
            train[FEATURES]
            .fillna(medians)
            .to_numpy(dtype=np.float32)
        )

        y_train = (
            train["target"]
            .to_numpy(dtype=np.int8)
        )

        X_val = (
            val[FEATURES]
            .fillna(medians)
            .to_numpy(dtype=np.float32)
        )

        y_val = (
            val["target"]
            .to_numpy(dtype=np.int8)
        )

        X_fwd = (
            fwd[FEATURES]
            .fillna(medians)
            .to_numpy(dtype=np.float32)
        )

        # Fairly conservative model.
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_depth=4,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=42,
        )

        model.fit(
            X_train,
            y_train,
        )

        p_train = model.predict_proba(
            X_train
        )[:, 1]

        p_val = model.predict_proba(
            X_val
        )[:, 1]

        p_fwd = (
            model.predict_proba(X_fwd)[:, 1]
            if len(fwd)
            else np.array([])
        )

        train_auc, train_pr = safe_auc(
            y_train,
            p_train,
        )

        val_auc, val_pr = safe_auc(
            y_val,
            p_val,
        )

        print(
            "AUC:",
            f"train={train_auc:.3f}",
            f"val={val_auc:.3f}",
        )

        print(
            "PR-AUC:",
            f"train={train_pr:.3f}",
            f"val={val_pr:.3f}",
        )

        # ====================================================
        # Select gate threshold ON VALIDATION ONLY
        # ====================================================

        threshold_rows = []

        for prob_cut in PROB_THRESHOLDS:
            gated = apply_gate(
                val,
                p_val,
                prob_cut,
            )

            m = pnl_metrics(gated)

            threshold_rows.append({
                "prob_cut": prob_cut,
                **m,
            })

        thresholds = pd.DataFrame(
            threshold_rows
        )

        eligible = thresholds[
            (thresholds["trades"] >= 15)
            & (thresholds["net"] > 0)
        ].copy()

        if eligible.empty:
            # Still report the least bad validation option,
            # but do NOT call it a winner.
            selected = (
                thresholds.sort_values(
                    ["net", "mean"],
                    ascending=False,
                )
                .iloc[0]
            )

            val_pass = False

        else:
            # Reward PnL + per-trade edge + robustness.
            eligible["selection_score"] = (
                eligible["mean"]
                + 0.01 * eligible["net"]
                + 0.005
                * eligible["without_best_day"].fillna(-999)
                + 0.005
                * eligible["without_best_ticker"].fillna(-999)
            )

            selected = (
                eligible.sort_values(
                    "selection_score",
                    ascending=False,
                )
                .iloc[0]
            )

            val_pass = True

        prob_cut = float(
            selected["prob_cut"]
        )

        vg = apply_gate(
            val,
            p_val,
            prob_cut,
        )

        fg = apply_gate(
            fwd,
            p_fwd,
            prob_cut,
        )

        vm = pnl_metrics(vg)
        fm = pnl_metrics(fg)

        print()
        print(
            "selected probability:",
            f"{prob_cut:.2f}",
            "| validation pass:",
            val_pass,
        )

        print(
            "VAL:",
            f"trades={vm['trades']}",
            f"net={vm['net']:+.1f}bp",
            f"mean={vm['mean']:+.2f}",
            f"WR={vm['wr']:.1f}%",
            f"days={vm['positive_days']}/{vm['days']}",
        )

        print(
            "FWD:",
            f"trades={fm['trades']}",
            (
                f"net={fm['net']:+.1f}bp"
                if np.isfinite(fm["net"])
                else "net=NaN"
            ),
            (
                f"mean={fm['mean']:+.2f}"
                if np.isfinite(fm["mean"])
                else "mean=NaN"
            ),
            (
                f"WR={fm['wr']:.1f}%"
                if np.isfinite(fm["wr"])
                else "WR=NaN"
            ),
            f"days={fm['positive_days']}/{fm['days']}",
        )

        results.append({
            "mode": mode,
            "horizon": horizon,

            "train_candidates": len(train),
            "val_candidates": len(val),
            "fwd_candidates": len(fwd),

            "train_auc": train_auc,
            "train_pr_auc": train_pr,
            "val_auc": val_auc,
            "val_pr_auc": val_pr,

            "prob_cut": prob_cut,
            "val_pass": val_pass,

            "val_trades": vm["trades"],
            "val_net_bp": vm["net"],
            "val_mean_bp": vm["mean"],
            "val_median_bp": vm["median"],
            "val_wr": vm["wr"],
            "val_pos_days": vm["positive_days"],
            "val_days": vm["days"],
            "val_without_best_day":
                vm["without_best_day"],
            "val_without_best_ticker":
                vm["without_best_ticker"],

            "fwd_trades": fm["trades"],
            "fwd_net_bp": fm["net"],
            "fwd_mean_bp": fm["mean"],
            "fwd_median_bp": fm["median"],
            "fwd_wr": fm["wr"],
            "fwd_pos_days": fm["positive_days"],
            "fwd_days": fm["days"],
            "fwd_without_best_day":
                fm["without_best_day"],
            "fwd_without_best_ticker":
                fm["without_best_ticker"],
        })

        if not fg.empty:
            fg = fg.copy()

            fg["mode"] = mode
            fg["horizon"] = horizon
            fg["prob_cut"] = prob_cut

            forward_trade_parts.append(
                fg[
                    [
                        "time",
                        "ticker",
                        "mode",
                        "horizon",
                        "side",
                        "p_good",
                        "prob_cut",
                        "gross_bp",
                        "cost_bp",
                        "net_bp",
                    ]
                ]
            )

        del (
            c,
            train,
            val,
            fwd,
            X_train,
            X_val,
            X_fwd,
            y_train,
            y_val,
            p_train,
            p_val,
            p_fwd,
            model,
        )

        gc.collect()

    del hdf
    gc.collect()

# ============================================================
# OUTPUT
# ============================================================

res = pd.DataFrame(results)

res.to_csv(
    OUT,
    index=False,
)

if forward_trade_parts:
    forward_trades = pd.concat(
        forward_trade_parts,
        ignore_index=True,
    )

    forward_trades.to_csv(
        OUT_TRADES,
        index=False,
    )

print()
print("=" * 120)
print("META-GATE RESULTS")
print("=" * 120)

cols = [
    "mode",
    "horizon",
    "val_auc",
    "val_pr_auc",
    "prob_cut",
    "val_pass",

    "val_trades",
    "val_net_bp",
    "val_mean_bp",
    "val_wr",
    "val_pos_days",
    "val_days",
    "val_without_best_day",
    "val_without_best_ticker",

    "fwd_trades",
    "fwd_net_bp",
    "fwd_mean_bp",
    "fwd_wr",
    "fwd_pos_days",
    "fwd_days",
    "fwd_without_best_day",
    "fwd_without_best_ticker",
]

print(
    res[cols]
    .sort_values(
        "fwd_net_bp",
        ascending=False,
    )
    .to_string(
        index=False,
        formatters={
            "val_auc": "{:.3f}".format,
            "val_pr_auc": "{:.3f}".format,

            "val_net_bp": "{:+.1f}".format,
            "val_mean_bp": "{:+.2f}".format,
            "val_wr": "{:.1f}".format,
            "val_without_best_day":
                "{:+.1f}".format,
            "val_without_best_ticker":
                "{:+.1f}".format,

            "fwd_net_bp": "{:+.1f}".format,
            "fwd_mean_bp": "{:+.2f}".format,
            "fwd_wr": "{:.1f}".format,
            "fwd_without_best_day":
                "{:+.1f}".format,
            "fwd_without_best_ticker":
                "{:+.1f}".format,
        },
    )
)

print()
print("=" * 120)
print("STRICT SURVIVORS")
print("=" * 120)

strict = res[
    (res["val_pass"] == True)
    & (res["fwd_trades"] >= 10)
    & (res["fwd_net_bp"] > 0)
    & (res["fwd_mean_bp"] > 0)
].copy()

if strict.empty:
    print("NO STRICT SURVIVORS")
else:
    print(
        strict[cols]
        .sort_values(
            "fwd_net_bp",
            ascending=False,
        )
        .to_string(
            index=False,
            formatters={
                "val_auc": "{:.3f}".format,
                "val_pr_auc": "{:.3f}".format,
                "val_net_bp": "{:+.1f}".format,
                "val_mean_bp": "{:+.2f}".format,
                "val_wr": "{:.1f}".format,
                "val_without_best_day":
                    "{:+.1f}".format,
                "val_without_best_ticker":
                    "{:+.1f}".format,
                "fwd_net_bp": "{:+.1f}".format,
                "fwd_mean_bp": "{:+.2f}".format,
                "fwd_wr": "{:.1f}".format,
                "fwd_without_best_day":
                    "{:+.1f}".format,
                "fwd_without_best_ticker":
                    "{:+.1f}".format,
            },
        )
    )

print()
print("Saved:")
print(OUT)
print(OUT_TRADES)

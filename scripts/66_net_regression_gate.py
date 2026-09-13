from pathlib import Path
import gc
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path("/root/rl-trading-tbank")

HIST_FILE = ROOT / "state" / "ofi_dataset_h5.pkl"
LIVE_FILE = ROOT / "state" / "shadow_observer_features.csv"

RET5_MEAN = -0.0000156187
RET5_STD  =  0.0020025922
RET1_MEAN = -0.0000033215
RET1_STD  =  0.0008796108

TRAIN_FROM = pd.Timestamp("2026-07-28", tz="UTC")
TRAIN_TO   = pd.Timestamp("2026-08-25", tz="UTC")

VAL_FROM = pd.Timestamp("2026-08-25", tz="UTC")
VAL_TO   = pd.Timestamp("2026-09-01", tz="UTC")

FWD_FROM = pd.Timestamp("2026-09-06", tz="UTC")

FEE_BP = 10.0

HORIZONS = [60, 90, 120]
QUANTILES = [0.80, 0.90, 0.95]

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


def rebuild_ret5(live):
    five_ns = pd.Timedelta(minutes=5).value
    tol_ns = pd.Timedelta(minutes=2).value

    parts = []

    for ticker, g in live.groupby("ticker", sort=False):
        g = g.sort_values("time").copy()

        ts = (
            g["time"]
            .dt.tz_convert(None)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )

        mid = g["mid"].to_numpy(dtype=float)

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

        ok = j >= 0

        rows = np.where(ok)[0]
        jj = j[ok]

        close = (
            target[rows] - ts[jj]
            <= tol_ns
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
        parts.append(g)

    return pd.concat(
        parts,
        ignore_index=True,
    )


def load_data():
    print("=" * 110)
    print("NET-BP REGRESSION GATE — LOAD DATA")
    print("=" * 110)

    hist = pd.read_pickle(HIST_FILE)

    cols = [
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
        [c for c in cols if c in hist.columns]
    ].copy()

    hist["time"] = pd.to_datetime(
        hist["time"],
        utc=True,
        errors="coerce",
    )

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

    for c in live_cols:
        if c not in ["time", "ticker"] and c in live.columns:
            live[c] = pd.to_numeric(
                live[c],
                errors="coerce",
            )

    live = rebuild_ret5(live)

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

    df = (
        df.sort_values(["ticker", "time"])
        .drop_duplicates(
            ["ticker", "time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

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

    df["abs_score"] = (
        df["score_mr"].abs()
        .astype("float32")
    )

    return df


def add_exit(df, horizon):
    pieces = []

    delta_ns = pd.Timedelta(
        minutes=horizon
    ).value

    tol_ns = pd.Timedelta(
        minutes=3
    ).value

    for ticker, g in df.groupby(
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

        mid = g["mid"].to_numpy(dtype=float)
        spr = g["spread_bp"].to_numpy(dtype=float)

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

        exit_time_ns = np.full(
            len(g),
            np.iinfo(np.int64).min,
            dtype=np.int64,
        )

        valid = j < len(g)

        rows = np.where(valid)[0]
        jj = j[valid]

        close = (
            ts[jj] - target[rows]
            <= tol_ns
        )

        rows = rows[close]
        jj = jj[close]

        exit_mid[rows] = mid[jj]
        exit_spread[rows] = spr[jj]
        exit_time_ns[rows] = ts[jj]

        g["exit_mid"] = exit_mid
        g["exit_spread"] = exit_spread

        et = pd.Series(
            pd.NaT,
            index=g.index,
            dtype="datetime64[ns, UTC]",
        )

        if len(rows):
            et.iloc[rows] = pd.to_datetime(
                exit_time_ns[rows],
                utc=True,
            )

        g["exit_time"] = et

        pieces.append(g)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def make_candidates(df):
    c = df[
        (df["abs_score"] >= 1.30)
        & (df["spread_bp"] <= 2.0)
        & df["exit_mid"].notna()
        & df["exit_time"].notna()
    ].copy()

    c["side"] = np.where(
        c["score_mr"] > 0,
        1,
        -1,
    )

    c["gross_bp"] = (
        c["side"]
        * (
            c["exit_mid"] / c["mid"] - 1
        )
        * 10000
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

    c["target"] = (
        c["net_bp"] > 0
    ).astype(np.int8)

    return c


def non_overlap(c):
    if c.empty:
        return c

    keep = []

    for ticker, g in c.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values("time")

        busy_until = None

        for idx, row in g.iterrows():
            if (
                busy_until is not None
                and row["time"] < busy_until
            ):
                continue

            keep.append(idx)
            busy_until = row["exit_time"]

    if not keep:
        return c.iloc[:0]

    return c.loc[keep].copy()


def choose_top(c, probabilities, cutoff):
    x = c.copy()
    x["p"] = probabilities

    x = x[
        x["p"] >= cutoff
    ].copy()

    if x.empty:
        return x

    # максимум одна сделка в момент времени
    x = (
        x.sort_values(
            ["time", "p"],
            ascending=[True, False],
        )
        .groupby("time", group_keys=False)
        .head(1)
    )

    return non_overlap(x)


def metrics(x):
    if x.empty:
        return {
            "trades": 0,
            "net": np.nan,
            "mean": np.nan,
            "wr": np.nan,
            "pos_days": 0,
            "days": 0,
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
        "trades": len(x),
        "net": total,
        "mean": x["net_bp"].mean(),
        "wr": 100 * (x["net_bp"] > 0).mean(),
        "pos_days": int((daily > 0).sum()),
        "days": len(daily),
        "without_best_day": total - daily.max(),
        "without_best_ticker": total - ticker.max(),
    }


df = load_data()

rows = []

for horizon in HORIZONS:
    print()
    print("=" * 110)
    print("HORIZON", horizon)
    print("=" * 110)

    hdf = add_exit(df, horizon)
    c = make_candidates(hdf)

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

    med = train[FEATURES].median()

    Xtr = train[FEATURES].fillna(med)
    ytr = train["target"]

    Xv = val[FEATURES].fillna(med)
    yv = val["target"]

    Xf = fwd[FEATURES].fillna(med)

    # ========================================================
    # REGRESSION: predict economic outcome directly
    # ========================================================

    # Winsorize TRAIN target only.
    # Prevent a few extreme returns from dominating fitting.
    lo = train["net_bp"].quantile(0.01)
    hi = train["net_bp"].quantile(0.99)

    ytr_reg = (
        train["net_bp"]
        .clip(lo, hi)
        .astype(float)
    )

    model = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=150,
        max_depth=2,
        min_samples_leaf=100,
        l2_regularization=10.0,
        loss="squared_error",
        random_state=42,
    )

    model.fit(Xtr, ytr_reg)

    pv = model.predict(Xv)
    pf = model.predict(Xf)

    train_pred = model.predict(Xtr)

    train_mae = mean_absolute_error(
        ytr_reg,
        train_pred,
    )

    val_mae = mean_absolute_error(
        val["net_bp"],
        pv,
    )

    # Rank correlation is more important than absolute calibration.
    val_rank_corr = pd.Series(pv).corr(
        pd.Series(val["net_bp"].to_numpy()),
        method="spearman",
    )

    print(
        f"TRAIN MAE={train_mae:.2f} bp "
        f"| VAL MAE={val_mae:.2f} bp "
        f"| VAL Spearman={val_rank_corr:+.3f}"
    )

    for q in QUANTILES:
        cutoff = np.quantile(pv, q)

        vg = choose_top(
            val,
            pv,
            cutoff,
        )

        fg = choose_top(
            fwd,
            pf,
            cutoff,
        )

        vm = metrics(vg)
        fm = metrics(fg)

        print()
        print(
            f"TOP {(1-q)*100:.0f}% "
            f"| cutoff={cutoff:.4f}"
        )

        print(
            "VAL:",
            f"{vm['trades']} trades",
            f"net={vm['net']:+.1f}",
            f"mean={vm['mean']:+.2f}",
            f"WR={vm['wr']:.1f}%",
            f"days={vm['pos_days']}/{vm['days']}",
            f"no_best_day={vm['without_best_day']:+.1f}",
        )

        print(
            "FWD:",
            f"{fm['trades']} trades",
            (
                f"net={fm['net']:+.1f}"
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
            f"days={fm['pos_days']}/{fm['days']}",
            (
                f"no_best_day={fm['without_best_day']:+.1f}"
                if np.isfinite(fm["without_best_day"])
                else "no_best_day=NaN"
            ),
        )

        rows.append({
            "horizon": horizon,
            "quantile": q,
            "cutoff": cutoff,

            "val_mae": val_mae,
            "val_rank_corr": val_rank_corr,

            "val_trades": vm["trades"],
            "val_net": vm["net"],
            "val_mean": vm["mean"],
            "val_wr": vm["wr"],
            "val_pos_days": vm["pos_days"],
            "val_days": vm["days"],
            "val_without_best_day":
                vm["without_best_day"],
            "val_without_best_ticker":
                vm["without_best_ticker"],

            "fwd_trades": fm["trades"],
            "fwd_net": fm["net"],
            "fwd_mean": fm["mean"],
            "fwd_wr": fm["wr"],
            "fwd_pos_days": fm["pos_days"],
            "fwd_days": fm["days"],
            "fwd_without_best_day":
                fm["without_best_day"],
            "fwd_without_best_ticker":
                fm["without_best_ticker"],
        })

    del hdf, c, train, val, fwd, model
    gc.collect()

res = pd.DataFrame(rows)

out = ROOT / "state" / "ranked_meta_gate_results.csv"

res.to_csv(
    out,
    index=False,
)

print()
print("=" * 110)
print("NET REGRESSION SUMMARY")
print("=" * 110)

print(
    res.sort_values(
        "fwd_net",
        ascending=False,
    ).to_string(
        index=False,
        formatters={
            "cutoff": "{:.4f}".format,
            "val_mae": "{:.2f}".format,
            "val_rank_corr": "{:+.3f}".format,
            "val_net": "{:+.1f}".format,
            "val_mean": "{:+.2f}".format,
            "fwd_net": "{:+.1f}".format,
            "fwd_mean": "{:+.2f}".format,
        }
    )
)

print()
print("=" * 110)
print("STRICT REGRESSION SURVIVORS")
print("=" * 110)

strict = res[
    (res["val_trades"] >= 15)
    & (res["val_net"] > 0)
    & (res["val_without_best_day"] > 0)
    & (res["fwd_trades"] >= 10)
    & (res["fwd_net"] > 0)
].copy()

if strict.empty:
    print("NO STRICT SURVIVORS")
else:
    print(
        strict.sort_values(
            "fwd_net",
            ascending=False,
        ).to_string(index=False)
    )

print()
print("Saved:", out)

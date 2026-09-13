from pathlib import Path
import gc
import numpy as np
import pandas as pd

ROOT = Path("/root/rl-trading-tbank")

HIST_FILE = ROOT / "state" / "ofi_dataset_h5.pkl"
LIVE_FILE = ROOT / "state" / "shadow_observer_features.csv"

RET5_MEAN = -0.0000156187
RET5_STD  =  0.0020025922
RET1_MEAN = -0.0000033215
RET1_STD  =  0.0008796108

FEE_BP = 10.0

WINDOWS = [
    ("W1", pd.Timestamp("2026-08-11", tz="UTC"), pd.Timestamp("2026-08-18", tz="UTC")),
    ("W2", pd.Timestamp("2026-08-18", tz="UTC"), pd.Timestamp("2026-08-25", tz="UTC")),
    ("W3", pd.Timestamp("2026-08-25", tz="UTC"), pd.Timestamp("2026-09-01", tz="UTC")),
]

FWD_FROM = pd.Timestamp("2026-09-06", tz="UTC")

HORIZONS = [60, 90, 120]

# Regime-switch:
# calm -> MR
# trend -> MOMENTUM
REGIME_BREADTHS = [0.60, 0.65, 0.70]

THRESHOLDS = [1.3, 1.5, 1.75, 2.0]

# cross-sectional top/bottom k
KS = [1, 2, 3]


def load_data():
    print("Loading historical...")
    hist = pd.read_pickle(HIST_FILE)[
        ["time", "ticker", "mid", "spread_bp", "ret_1m", "ret_5m"]
    ].copy()

    hist["time"] = pd.to_datetime(hist["time"], utc=True, errors="coerce")

    print("historical:", f"{len(hist):,}")

    print("Loading live...")
    live = pd.read_csv(
        LIVE_FILE,
        usecols=["time", "ticker", "mid", "spread_bp", "ret_1m"],
    )

    live["time"] = pd.to_datetime(live["time"], utc=True, errors="coerce")

    for c in ["mid", "spread_bp", "ret_1m"]:
        live[c] = pd.to_numeric(live[c], errors="coerce")

    live = (
        live.dropna(subset=["time", "ticker", "mid", "ret_1m"])
        .sort_values(["ticker", "time"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # rebuild ret_5m using PAST ONLY
    # --------------------------------------------------------
    five_ns = pd.Timedelta(minutes=5).value
    tol_ns = pd.Timedelta(minutes=2).value

    parts = []

    for ticker, g in live.groupby("ticker", sort=False):
        g = g.copy()

        ts = (
            g["time"]
            .dt.tz_convert(None)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )

        mid = g["mid"].to_numpy(dtype=float)

        target = ts - five_ns

        j = np.searchsorted(ts, target, side="right") - 1

        ret5 = np.full(len(g), np.nan)

        valid = j >= 0
        rows = np.where(valid)[0]
        jj = j[valid]

        close = target[rows] - ts[jj] <= tol_ns

        rows = rows[close]
        jj = jj[close]

        good = (
            np.isfinite(mid[rows])
            & np.isfinite(mid[jj])
            & (mid[jj] > 0)
        )

        rows = rows[good]
        jj = jj[good]

        ret5[rows] = mid[rows] / mid[jj] - 1

        g["ret_5m"] = ret5
        parts.append(g)

    live = pd.concat(parts, ignore_index=True)

    print(
        "live ret_5m:",
        f"{live['ret_5m'].notna().sum():,}",
        "/",
        f"{len(live):,}",
    )

    df = pd.concat([hist, live], ignore_index=True, sort=False)

    del hist, live, parts
    gc.collect()

    for c in ["mid", "spread_bp", "ret_1m", "ret_5m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=["time", "ticker", "mid", "spread_bp", "ret_1m", "ret_5m"]
    )

    df = df[df["mid"] > 0]

    df = (
        df.sort_values(["ticker", "time"])
        .drop_duplicates(["ticker", "time"], keep="last")
        .reset_index(drop=True)
    )

    z5 = (df["ret_5m"] - RET5_MEAN) / RET5_STD
    z1 = (df["ret_1m"] - RET1_MEAN) / RET1_STD

    df["score_mr"] = (-0.75 * z5 - 0.25 * z1).astype("float32")
    df["score_mom"] = (-df["score_mr"]).astype("float32")

    # market breadth
    up = (df["ret_5m"] > 0).astype("float32")

    share_up = (
        pd.DataFrame({
            "time": df["time"],
            "up": up,
        })
        .groupby("time")["up"]
        .mean()
    )

    df["share_up"] = df["time"].map(share_up).astype("float32")

    df["breadth"] = np.maximum(
        df["share_up"],
        1.0 - df["share_up"],
    ).astype("float32")

    print(
        "range:",
        df["time"].min(),
        "->",
        df["time"].max(),
    )

    return df


def add_exit(df, horizon):
    pieces = []

    delta_ns = pd.Timedelta(minutes=horizon).value
    tol_ns = pd.Timedelta(minutes=3).value

    cols = [
        "time",
        "ticker",
        "mid",
        "spread_bp",
        "score_mr",
        "score_mom",
        "breadth",
        "share_up",
    ]

    base = df[cols]

    for ticker, g in base.groupby("ticker", sort=False):
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
        j = np.searchsorted(ts, target, side="left")

        exit_mid = np.full(len(g), np.nan, dtype="float32")
        exit_spread = np.full(len(g), np.nan, dtype="float32")
        exit_ns = np.full(
            len(g),
            np.iinfo(np.int64).min,
            dtype=np.int64,
        )

        valid = j < len(g)

        rows = np.where(valid)[0]
        jj = j[valid]

        close = ts[jj] - target[rows] <= tol_ns

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
            exit_time.iloc[rows] = pd.to_datetime(
                exit_ns[rows],
                utc=True,
            )

        g["exit_time"] = exit_time
        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)


def non_overlap(c):
    if c.empty:
        return c

    keep = []

    for ticker, g in c.groupby("ticker", sort=False):
        g = g.sort_values("time")
        busy_until = None

        for idx, r in g.iterrows():
            if busy_until is not None and r["time"] < busy_until:
                continue

            keep.append(idx)
            busy_until = r["exit_time"]

    return c.loc[keep].copy()


def add_pnl(c):
    if c.empty:
        return c

    c = c.copy()

    c["gross_bp"] = (
        c["side"]
        * (c["exit_mid"] / c["mid"] - 1.0)
        * 10000
    )

    c["cost_bp"] = (
        FEE_BP
        + 0.5 * c["spread_bp"]
        + 0.5 * c["exit_spread"]
    )

    c["net_bp"] = c["gross_bp"] - c["cost_bp"]

    c["date_msk"] = (
        c["time"]
        .dt.tz_convert("Europe/Moscow")
        .dt.date
    )

    return c


def regime_trades(df, threshold, regime_cut):
    """
    calm:
        mean reversion

    trending:
        momentum in prevailing market direction
    """

    c = df[
        df["exit_mid"].notna()
        & df["exit_time"].notna()
        & (df["spread_bp"] <= 2.0)
    ].copy()

    if c.empty:
        return c

    calm = c["breadth"] <= regime_cut
    trend = c["breadth"] > regime_cut

    pieces = []

    # CALM -> MR
    a = c[
        calm
        & (c["score_mr"].abs() >= threshold)
    ].copy()

    if not a.empty:
        a["side"] = np.where(a["score_mr"] > 0, 1, -1)
        a["mode"] = "MR"
        pieces.append(a)

    # TREND -> MOMENTUM,
    # only in prevailing market direction
    b = c[
        trend
        & (c["score_mom"].abs() >= threshold)
    ].copy()

    if not b.empty:
        b["side"] = np.where(b["score_mom"] > 0, 1, -1)

        market_side = np.where(
            b["share_up"] >= 0.5,
            1,
            -1,
        )

        b = b[b["side"] == market_side].copy()
        b["mode"] = "MOM"
        pieces.append(b)

    if not pieces:
        return c.iloc[:0]

    out = pd.concat(pieces, ignore_index=False)

    # Only strongest signal per minute
    out["_strength"] = np.where(
        out["mode"] == "MR",
        out["score_mr"].abs(),
        out["score_mom"].abs(),
    )

    out = (
        out.sort_values(
            ["time", "_strength"],
            ascending=[True, False],
        )
        .groupby("time", group_keys=False)
        .head(1)
    )

    out = non_overlap(out)

    return add_pnl(out)


def cross_sectional_trades(df, k, mode):
    """
    Relative selection at each minute.

    MR:
      long most negative recent movers via high score_mr
      short strongest opposite MR score

    MOM:
      long strongest momentum
      short weakest momentum
    """

    score_col = (
        "score_mr"
        if mode == "MR"
        else "score_mom"
    )

    c = df[
        df["exit_mid"].notna()
        & df["exit_time"].notna()
        & (df["spread_bp"] <= 2.0)
    ].copy()

    if c.empty:
        return c

    selected = []

    for _, g in c.groupby("time", sort=False):
        # avoid tiny universes
        if len(g) < 10:
            continue

        g = g.sort_values(score_col)

        shorts = g.head(k).copy()
        longs = g.tail(k).copy()

        shorts["side"] = -1
        longs["side"] = 1

        selected.append(shorts)
        selected.append(longs)

    if not selected:
        return c.iloc[:0]

    out = pd.concat(selected, ignore_index=False)

    out = non_overlap(out)

    return add_pnl(out)


def metrics(c):
    if c.empty:
        return {
            "trades": 0,
            "net": np.nan,
            "mean": np.nan,
            "wr": np.nan,
            "days": 0,
            "pos_days": 0,
            "without_best_day": np.nan,
            "without_best_ticker": np.nan,
        }

    total = c["net_bp"].sum()

    daily = c.groupby("date_msk")["net_bp"].sum()
    tick = c.groupby("ticker")["net_bp"].sum()

    return {
        "trades": len(c),
        "net": total,
        "mean": c["net_bp"].mean(),
        "wr": 100 * (c["net_bp"] > 0).mean(),
        "days": len(daily),
        "pos_days": int((daily > 0).sum()),
        "without_best_day": total - daily.max(),
        "without_best_ticker": total - tick.max(),
    }


def evaluate_periods(trades):
    row = {}

    pre_parts = []

    for name, start, end in WINDOWS:
        w = trades[
            (trades["time"] >= start)
            & (trades["time"] < end)
        ].copy()

        m = metrics(w)

        row[f"{name}_trades"] = m["trades"]
        row[f"{name}_net"] = m["net"]

        if not w.empty:
            pre_parts.append(w)

    if pre_parts:
        pre = pd.concat(pre_parts, ignore_index=True)
    else:
        pre = trades.iloc[:0]

    pm = metrics(pre)

    fwd = trades[
        trades["time"] >= FWD_FROM
    ].copy()

    fm = metrics(fwd)

    row.update({
        "pre_trades": pm["trades"],
        "pre_net": pm["net"],
        "pre_mean": pm["mean"],
        "pre_wr": pm["wr"],
        "pre_days": pm["days"],
        "pre_pos_days": pm["pos_days"],
        "pre_without_best_day": pm["without_best_day"],
        "pre_without_best_ticker": pm["without_best_ticker"],

        "fwd_trades": fm["trades"],
        "fwd_net": fm["net"],
        "fwd_mean": fm["mean"],
        "fwd_wr": fm["wr"],
        "fwd_days": fm["days"],
        "fwd_pos_days": fm["pos_days"],
        "fwd_without_best_day": fm["without_best_day"],
        "fwd_without_best_ticker": fm["without_best_ticker"],
    })

    return row


print("=" * 120)
print("REGIME + CROSS-SECTIONAL LAB")
print("=" * 120)

df = load_data()

results = []

for horizon in HORIZONS:
    print()
    print("=" * 120)
    print("HORIZON", horizon)
    print("=" * 120)

    hdf = add_exit(df, horizon)

    # --------------------------------------------------------
    # REGIME SWITCH
    # --------------------------------------------------------
    for threshold in THRESHOLDS:
        for regime_cut in REGIME_BREADTHS:

            t = regime_trades(
                hdf,
                threshold,
                regime_cut,
            )

            row = {
                "family": "REGIME",
                "horizon": horizon,
                "param1": threshold,
                "param2": regime_cut,
            }

            row.update(
                evaluate_periods(t)
            )

            results.append(row)

    # --------------------------------------------------------
    # CROSS SECTIONAL
    # --------------------------------------------------------
    for mode in ["MR", "MOM"]:
        for k in KS:

            t = cross_sectional_trades(
                hdf,
                k,
                mode,
            )

            row = {
                "family": f"XSEC_{mode}",
                "horizon": horizon,
                "param1": k,
                "param2": np.nan,
            }

            row.update(
                evaluate_periods(t)
            )

            results.append(row)

    del hdf
    gc.collect()

    print("finished", horizon)

res = pd.DataFrame(results)

out = ROOT / "state" / "regime_cross_sectional_results.csv"
res.to_csv(out, index=False)

# ------------------------------------------------------------
# PRE-FORWARD ROBUST FILTER
# ------------------------------------------------------------

robust = res[
    (res["pre_trades"] >= 30)
    & (res["pre_net"] > 0)
    & (res["pre_without_best_day"] > 0)
    & (res["pre_without_best_ticker"] > 0)
].copy()

robust["positive_windows"] = (
    (robust["W1_net"] > 0).astype(int)
    + (robust["W2_net"] > 0).astype(int)
    + (robust["W3_net"] > 0).astype(int)
)

robust = robust[
    robust["positive_windows"] >= 2
].copy()

robust["score"] = (
    robust["pre_mean"]
    + 0.01 * robust["pre_net"]
    + 0.01 * robust["pre_without_best_day"]
    + 0.01 * robust["pre_without_best_ticker"]
    + 2 * robust["positive_windows"]
)

robust = robust.sort_values(
    "score",
    ascending=False,
)

cols = [
    "family",
    "horizon",
    "param1",
    "param2",

    "W1_trades",
    "W1_net",
    "W2_trades",
    "W2_net",
    "W3_trades",
    "W3_net",

    "pre_trades",
    "pre_net",
    "pre_mean",
    "pre_wr",
    "positive_windows",
    "pre_without_best_day",
    "pre_without_best_ticker",

    "fwd_trades",
    "fwd_net",
    "fwd_mean",
    "fwd_wr",
    "fwd_pos_days",
    "fwd_days",
    "fwd_without_best_day",
    "fwd_without_best_ticker",
]

fmt = {
    c: "{:+.1f}".format
    for c in [
        "W1_net",
        "W2_net",
        "W3_net",
        "pre_net",
        "pre_mean",
        "pre_without_best_day",
        "pre_without_best_ticker",
        "fwd_net",
        "fwd_mean",
        "fwd_without_best_day",
        "fwd_without_best_ticker",
    ]
}

print()
print("=" * 120)
print("ROBUST PRE-FORWARD")
print("=" * 120)

if robust.empty:
    print("NONE")
else:
    print(
        robust[cols]
        .head(20)
        .to_string(
            index=False,
            formatters=fmt,
        )
    )

print()
print("=" * 120)
print("ROBUST + FORWARD POSITIVE")
print("=" * 120)

survive = robust[
    (robust["fwd_trades"] >= 10)
    & (robust["fwd_net"] > 0)
].copy()

if survive.empty:
    print("NO SURVIVORS")
else:
    print(
        survive[cols]
        .head(20)
        .to_string(
            index=False,
            formatters=fmt,
        )
    )

print()
print("Saved:", out)

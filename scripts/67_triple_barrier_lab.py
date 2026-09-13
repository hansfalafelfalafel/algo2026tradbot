from pathlib import Path
import gc
import itertools
import numpy as np
import pandas as pd

ROOT = Path("/root/rl-trading-tbank")

HIST_FILE = ROOT / "state" / "ofi_dataset_h5.pkl"
LIVE_FILE = ROOT / "state" / "shadow_observer_features.csv"

OUT_ALL = ROOT / "state" / "triple_barrier_all.csv"
OUT_TOP = ROOT / "state" / "triple_barrier_top.csv"

# =====================================================================
# Frozen MR normalization
# =====================================================================

RET5_MEAN = -0.0000156187
RET5_STD  =  0.0020025922

RET1_MEAN = -0.0000033215
RET1_STD  =  0.0008796108

# Actual-ish T-Bank commission observed in sandbox.
# Spread is ADDED separately.
COMMISSION_BP = 10.0

# =====================================================================
# SEARCH SPACE
# =====================================================================

SCORE_THRESHOLDS = [1.30, 1.50, 1.75]

BREADTH_LIMITS = [0.60, 0.65]

SPREAD_LIMITS = [1.50, 2.00]

TAKE_PROFITS = [20.0, 30.0, 40.0, 60.0]
STOP_LOSSES  = [15.0, 25.0, 35.0, 50.0]

MAX_HOLDS = [60, 90, 120]

# =====================================================================
# PRE-SEPTEMBER WINDOWS
#
# ALL selection happens here.
# Sep 6-9 is NOT used to choose parameters.
# =====================================================================

WINDOWS = [
    (
        "W0",
        pd.Timestamp("2026-07-28", tz="UTC"),
        pd.Timestamp("2026-08-04", tz="UTC"),
    ),
    (
        "W1",
        pd.Timestamp("2026-08-04", tz="UTC"),
        pd.Timestamp("2026-08-11", tz="UTC"),
    ),
    (
        "W2",
        pd.Timestamp("2026-08-11", tz="UTC"),
        pd.Timestamp("2026-08-18", tz="UTC"),
    ),
    (
        "W3",
        pd.Timestamp("2026-08-18", tz="UTC"),
        pd.Timestamp("2026-08-25", tz="UTC"),
    ),
    (
        "W4",
        pd.Timestamp("2026-08-25", tz="UTC"),
        pd.Timestamp("2026-09-01", tz="UTC"),
    ),
]

PRE_FROM = WINDOWS[0][1]
PRE_TO = WINDOWS[-1][2]

# Already repeatedly inspected. Stress-check only.
OLD_FORWARD_FROM = pd.Timestamp("2026-09-06", tz="UTC")

# NEW untouched forward:
# 2026-09-10 00:00 Moscow == 2026-09-09 21:00 UTC
FRESH_FORWARD_FROM = pd.Timestamp(
    "2026-09-10 00:00:00",
    tz="Europe/Moscow",
).tz_convert("UTC")


# =====================================================================
# DATA
# =====================================================================

def rebuild_ret5(live):
    five_ns = pd.Timedelta(minutes=5).value
    tolerance_ns = pd.Timedelta(minutes=2).value

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
        parts.append(g)

    return pd.concat(
        parts,
        ignore_index=True,
    )


def load_data():
    print("=" * 120)
    print("TRIPLE BARRIER LAB — LOAD DATA")
    print("=" * 120)

    hist = pd.read_pickle(HIST_FILE)[
        [
            "time",
            "ticker",
            "mid",
            "spread_bp",
            "ret_1m",
            "ret_5m",
        ]
    ].copy()

    hist["time"] = pd.to_datetime(
        hist["time"],
        utc=True,
        errors="coerce",
    )

    print("historical:", f"{len(hist):,}")

    live = pd.read_csv(
        LIVE_FILE,
        usecols=[
            "time",
            "ticker",
            "mid",
            "spread_bp",
            "ret_1m",
        ],
    )

    live["time"] = pd.to_datetime(
        live["time"],
        utc=True,
        errors="coerce",
    )

    for c in [
        "mid",
        "spread_bp",
        "ret_1m",
    ]:
        live[c] = pd.to_numeric(
            live[c],
            errors="coerce",
        )

    live = live.dropna(
        subset=[
            "time",
            "ticker",
            "mid",
            "spread_bp",
            "ret_1m",
        ]
    ).copy()

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

    for c in [
        "mid",
        "spread_bp",
        "ret_1m",
        "ret_5m",
    ]:
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

    # ---------------------------------------------------------
    # MR SCORE
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Breadth proxy, known at entry
    # ---------------------------------------------------------

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

    breadth = np.maximum(
        share_up,
        1.0 - share_up,
    )

    df["breadth"] = (
        df["time"]
        .map(breadth)
        .astype("float32")
    )

    for c in [
        "mid",
        "spread_bp",
        "ret_1m",
        "ret_5m",
        "score_mr",
        "breadth",
    ]:
        df[c] = df[c].astype("float32")

    print("usable:", f"{len(df):,}")

    print(
        "range:",
        df["time"].min(),
        "->",
        df["time"].max(),
    )

    print(
        "fresh forward starts:",
        FRESH_FORWARD_FROM,
    )

    return df


# =====================================================================
# PRECOMPUTE TICKER ARRAYS
# =====================================================================

def make_market_arrays(df):
    out = {}

    for ticker, g in df.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values("time")

        out[ticker] = {
            "time": g["time"].to_numpy(),
            "time_ns": (
                g["time"]
                .dt.tz_convert(None)
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            ),
            "mid": g["mid"].to_numpy(dtype=float),
            "spread": g["spread_bp"].to_numpy(dtype=float),
        }

    return out


# =====================================================================
# FIRST TOUCH ENGINE
# =====================================================================

def simulate_one(
    row,
    market,
    tp_bp,
    sl_bp,
    max_hold_min,
):
    """
    LONG ONLY.

    First-touch:
      TP if signed/gross mid return >= +tp
      SL if signed/gross mid return <= -sl
      otherwise TIME at first observation >= max_hold.

    No future information used at entry.
    """

    ticker = row.ticker

    a = market[ticker]

    ts = a["time_ns"]
    mids = a["mid"]
    spreads = a["spread"]

    entry_ns = pd.Timestamp(
        row.time
    ).tz_convert("UTC").tz_localize(None).value

    start = np.searchsorted(
        ts,
        entry_ns,
        side="left",
    )

    if start >= len(ts):
        return None

    entry_mid = float(row.mid)

    if not np.isfinite(entry_mid) or entry_mid <= 0:
        return None

    end_target = (
        entry_ns
        + pd.Timedelta(
            minutes=max_hold_min
        ).value
    )

    end = np.searchsorted(
        ts,
        end_target,
        side="left",
    )

    # Need at least some future observation
    if start + 1 >= len(ts):
        return None

    stop = min(
        end + 1,
        len(ts),
    )

    if stop <= start + 1:
        return None

    chosen = None
    reason = None

    # Start AFTER entry row
    for j in range(
        start + 1,
        stop,
    ):
        px = mids[j]

        if not np.isfinite(px):
            continue

        gross_bp = (
            px / entry_mid - 1.0
        ) * 10000.0

        if gross_bp >= tp_bp:
            chosen = j
            reason = "TP"
            break

        if gross_bp <= -sl_bp:
            chosen = j
            reason = "SL"
            break

    # No barrier touched -> time stop
    if chosen is None:
        if end >= len(ts):
            return None

        # Require reasonably close observation to requested time-stop
        if (
            ts[end] - end_target
            > pd.Timedelta(minutes=3).value
        ):
            return None

        chosen = end
        reason = "TIME"

    exit_mid = float(mids[chosen])
    exit_spread = float(spreads[chosen])

    if (
        not np.isfinite(exit_mid)
        or not np.isfinite(exit_spread)
    ):
        return None

    gross_bp = (
        exit_mid / entry_mid - 1.0
    ) * 10000.0

    cost_bp = (
        COMMISSION_BP
        + 0.5 * float(row.spread_bp)
        + 0.5 * exit_spread
    )

    net_bp = gross_bp - cost_bp

    return {
        "time": row.time,
        "exit_time": pd.Timestamp(
            a["time"][chosen]
        ),
        "ticker": ticker,
        "entry_mid": entry_mid,
        "exit_mid": exit_mid,
        "entry_spread": float(row.spread_bp),
        "exit_spread": exit_spread,
        "score_mr": float(row.score_mr),
        "breadth": float(row.breadth),
        "reason": reason,
        "gross_bp": gross_bp,
        "cost_bp": cost_bp,
        "net_bp": net_bp,
    }


# =====================================================================
# NON-OVERLAP
# =====================================================================

def non_overlap(trades):
    if trades.empty:
        return trades

    keep = []

    for ticker, g in trades.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values("time")

        busy_until = None

        for idx, r in g.iterrows():
            if (
                busy_until is not None
                and r["time"] < busy_until
            ):
                continue

            keep.append(idx)
            busy_until = r["exit_time"]

    if not keep:
        return trades.iloc[:0]

    return trades.loc[keep].copy()


# =====================================================================
# METRICS
# =====================================================================

def metrics(x):
    if x.empty:
        return {
            "trades": 0,
            "net": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "wr": np.nan,
            "days": 0,
            "pos_days": 0,
            "without_best_day": np.nan,
            "without_best_ticker": np.nan,
            "tp_pct": np.nan,
            "sl_pct": np.nan,
            "time_pct": np.nan,
        }

    x = x.copy()

    x["date_msk"] = (
        x["time"]
        .dt.tz_convert("Europe/Moscow")
        .dt.date
    )

    total = float(
        x["net_bp"].sum()
    )

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

    reason = (
        x["reason"]
        .value_counts(normalize=True)
    )

    return {
        "trades": int(len(x)),
        "net": total,
        "mean": float(x["net_bp"].mean()),
        "median": float(x["net_bp"].median()),
        "wr": float(
            100 * (x["net_bp"] > 0).mean()
        ),
        "days": int(len(daily)),
        "pos_days": int(
            (daily > 0).sum()
        ),
        "without_best_day": float(
            total - daily.max()
        ),
        "without_best_ticker": float(
            total - ticker.max()
        ),
        "tp_pct": float(
            100 * reason.get("TP", 0)
        ),
        "sl_pct": float(
            100 * reason.get("SL", 0)
        ),
        "time_pct": float(
            100 * reason.get("TIME", 0)
        ),
    }


# =====================================================================
# MAIN
# =====================================================================

df = load_data()

market = make_market_arrays(df)

# Maximum possible entry universe.
# LONG MR only.
base_entries = df[
    (df["score_mr"] >= min(SCORE_THRESHOLDS))
    & (df["spread_bp"] <= max(SPREAD_LIMITS))
    & (df["breadth"] <= max(BREADTH_LIMITS))
].copy()

print()
print(
    "base LONG MR entries:",
    f"{len(base_entries):,}",
)

# -------------------------------------------------------------
# Precompute FIRST TOUCH outcome for barrier settings.
#
# Threshold/breadth/spread filters are applied AFTER,
# avoiding recalculating future paths unnecessarily.
# -------------------------------------------------------------

barrier_cache = {}

barrier_grid = list(
    itertools.product(
        MAX_HOLDS,
        TAKE_PROFITS,
        STOP_LOSSES,
    )
)

print(
    "barrier combinations:",
    len(barrier_grid),
)

for n, (
    hold,
    tp,
    sl,
) in enumerate(
    barrier_grid,
    start=1,
):
    key = (
        hold,
        tp,
        sl,
    )

    rows = []

    for r in base_entries.itertuples(
        index=False
    ):
        sim = simulate_one(
            r,
            market,
            tp_bp=tp,
            sl_bp=sl,
            max_hold_min=hold,
        )

        if sim is not None:
            rows.append(sim)

    if rows:
        trades = pd.DataFrame(rows)

        trades["time"] = pd.to_datetime(
            trades["time"],
            utc=True,
        )

        trades["exit_time"] = pd.to_datetime(
            trades["exit_time"],
            utc=True,
        )
    else:
        trades = pd.DataFrame()

    barrier_cache[key] = trades

    if (
        n % 8 == 0
        or n == len(barrier_grid)
    ):
        print(
            f"first-touch precompute: "
            f"{n}/{len(barrier_grid)}"
        )

print()
print("=" * 120)
print("GRID SEARCH — PRE-SEPTEMBER ONLY")
print("=" * 120)

results = []

config_grid = list(
    itertools.product(
        SCORE_THRESHOLDS,
        BREADTH_LIMITS,
        SPREAD_LIMITS,
        MAX_HOLDS,
        TAKE_PROFITS,
        STOP_LOSSES,
    )
)

print(
    "total configs:",
    len(config_grid),
)

for n, (
    score_cut,
    breadth_cut,
    spread_cut,
    hold,
    tp,
    sl,
) in enumerate(
    config_grid,
    start=1,
):
    source = barrier_cache[
        (
            hold,
            tp,
            sl,
        )
    ]

    if source.empty:
        continue

    t = source[
        (source["score_mr"] >= score_cut)
        & (source["breadth"] <= breadth_cut)
        & (source["entry_spread"] <= spread_cut)
    ].copy()

    t = non_overlap(t)

    row = {
        "score_cut": score_cut,
        "breadth_cut": breadth_cut,
        "spread_cut": spread_cut,
        "hold_min": hold,
        "tp_bp": tp,
        "sl_bp": sl,
    }

    pre_parts = []
    positive_windows = 0
    usable_windows = 0

    for name, start, end in WINDOWS:
        w = t[
            (t["time"] >= start)
            & (t["time"] < end)
        ]

        m = metrics(w)

        row[f"{name}_trades"] = m["trades"]
        row[f"{name}_net"] = m["net"]

        if m["trades"] > 0:
            usable_windows += 1

            if m["net"] > 0:
                positive_windows += 1

            pre_parts.append(w)

    if pre_parts:
        pre = pd.concat(
            pre_parts,
            ignore_index=True,
        )
    else:
        pre = t.iloc[:0].copy()

    pm = metrics(pre)

    # ---------------------------------------------------------
    # Sep 6-9 — OBSERVATIONAL ONLY
    # ---------------------------------------------------------

    old_fwd = t[
        (t["time"] >= OLD_FORWARD_FROM)
        & (t["time"] < FRESH_FORWARD_FROM)
    ]

    om = metrics(old_fwd)

    # ---------------------------------------------------------
    # Sep 10+ — TRUE NEW FORWARD
    # ---------------------------------------------------------

    fresh = t[
        t["time"] >= FRESH_FORWARD_FROM
    ]

    fm = metrics(fresh)

    row.update({
        "pre_trades": pm["trades"],
        "pre_net": pm["net"],
        "pre_mean": pm["mean"],
        "pre_median": pm["median"],
        "pre_wr": pm["wr"],
        "pre_days": pm["days"],
        "pre_pos_days": pm["pos_days"],
        "pre_positive_windows":
            positive_windows,
        "pre_usable_windows":
            usable_windows,
        "pre_without_best_day":
            pm["without_best_day"],
        "pre_without_best_ticker":
            pm["without_best_ticker"],
        "pre_tp_pct": pm["tp_pct"],
        "pre_sl_pct": pm["sl_pct"],
        "pre_time_pct": pm["time_pct"],

        "old_fwd_trades": om["trades"],
        "old_fwd_net": om["net"],
        "old_fwd_mean": om["mean"],
        "old_fwd_wr": om["wr"],
        "old_fwd_pos_days":
            om["pos_days"],
        "old_fwd_days":
            om["days"],
        "old_fwd_without_best_day":
            om["without_best_day"],
        "old_fwd_without_best_ticker":
            om["without_best_ticker"],

        "fresh_trades": fm["trades"],
        "fresh_net": fm["net"],
        "fresh_mean": fm["mean"],
        "fresh_wr": fm["wr"],
        "fresh_pos_days":
            fm["pos_days"],
        "fresh_days":
            fm["days"],
        "fresh_without_best_day":
            fm["without_best_day"],
        "fresh_without_best_ticker":
            fm["without_best_ticker"],
    })

    results.append(row)

    if n % 100 == 0:
        print(
            f"configs: {n}/{len(config_grid)}"
        )

res = pd.DataFrame(results)

res.to_csv(
    OUT_ALL,
    index=False,
)

# =====================================================================
# STRICT SELECTION — ABSOLUTELY NO SEP6+ INVOLVED
# =====================================================================

strict = res[
    (res["pre_trades"] >= 40)
    & (res["pre_usable_windows"] == 5)
    & (res["pre_positive_windows"] >= 4)
    & (res["pre_net"] > 0)
    & (res["pre_mean"] > 0)
    & (res["pre_without_best_day"] > 0)
    & (res["pre_without_best_ticker"] > 0)
].copy()

if not strict.empty:
    strict["robust_score"] = (
        strict["pre_mean"]
        + 0.01 * strict["pre_net"]
        + 0.01
        * strict["pre_without_best_day"]
        + 0.01
        * strict["pre_without_best_ticker"]
        + 3.0
        * strict["pre_positive_windows"]
    )

    strict = strict.sort_values(
        "robust_score",
        ascending=False,
    )

strict.to_csv(
    OUT_TOP,
    index=False,
)

cols = [
    "score_cut",
    "breadth_cut",
    "spread_cut",
    "hold_min",
    "tp_bp",
    "sl_bp",

    "W0_net",
    "W1_net",
    "W2_net",
    "W3_net",
    "W4_net",

    "pre_trades",
    "pre_net",
    "pre_mean",
    "pre_wr",
    "pre_positive_windows",
    "pre_without_best_day",
    "pre_without_best_ticker",
    "pre_tp_pct",
    "pre_sl_pct",
    "pre_time_pct",

    "old_fwd_trades",
    "old_fwd_net",
    "old_fwd_mean",
    "old_fwd_wr",
    "old_fwd_pos_days",
    "old_fwd_days",
    "old_fwd_without_best_day",
    "old_fwd_without_best_ticker",

    "fresh_trades",
    "fresh_net",
    "fresh_mean",
    "fresh_wr",
    "fresh_pos_days",
    "fresh_days",
]

fmt = {
    c: "{:+.1f}".format
    for c in [
        "W0_net",
        "W1_net",
        "W2_net",
        "W3_net",
        "W4_net",
        "pre_net",
        "pre_mean",
        "pre_without_best_day",
        "pre_without_best_ticker",
        "old_fwd_net",
        "old_fwd_mean",
        "old_fwd_without_best_day",
        "old_fwd_without_best_ticker",
        "fresh_net",
        "fresh_mean",
    ]
}

print()
print("=" * 120)
print("STRICT PRE-SEPTEMBER WINNERS")
print("=" * 120)

if strict.empty:
    print("NO STRICT WINNERS")
else:
    print(
        strict[
            cols
        ].head(20).to_string(
            index=False,
            formatters=fmt,
        )
    )

print()
print("=" * 120)
print("OLD SEP6-9 STRESS CHECK")
print("NOT USED FOR SELECTION")
print("=" * 120)

if strict.empty:
    print("NO PRE-SELECTED STRATEGIES")
else:
    stress = strict[
        strict["old_fwd_trades"] >= 5
    ].copy()

    stress = stress.sort_values(
        "old_fwd_net",
        ascending=False,
    )

    if stress.empty:
        print("NO STRATEGIES WITH >=5 OLD-FORWARD TRADES")
    else:
        print(
            stress[
                cols
            ].head(20).to_string(
                index=False,
                formatters=fmt,
            )
        )

print()
print("=" * 120)
print("FRESH FORWARD — SEP10+ MOSCOW")
print("THIS IS THE NEW UNTOUCHED TEST")
print("=" * 120)

if strict.empty:
    print("NO PRE-SELECTED STRATEGIES")
else:
    fresh_show = strict[
        strict["fresh_trades"] > 0
    ].copy()

    if fresh_show.empty:
        print(
            "NO FRESH TRADES YET — THIS IS EXPECTED "
            "UNTIL SEP10 MARKET DATA ARRIVES"
        )
    else:
        print(
            fresh_show[
                cols
            ].sort_values(
                "fresh_net",
                ascending=False,
            ).head(20).to_string(
                index=False,
                formatters=fmt,
            )
        )

print()
print("Saved:")
print(OUT_ALL)
print(OUT_TOP)

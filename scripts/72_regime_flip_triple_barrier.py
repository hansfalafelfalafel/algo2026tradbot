from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/rl-trading-tbank")

HIST_FILE = ROOT / "state" / "ofi_dataset_h5.pkl"
LIVE_FILE = ROOT / "state" / "shadow_observer_features.csv"

OUT = ROOT / "state" / "regime_flip_results.csv"
OUT_TRADES = ROOT / "state" / "regime_flip_trades.csv"

RET5_MEAN = -0.0000156187
RET5_STD  =  0.0020025922
RET1_MEAN = -0.0000033215
RET1_STD  =  0.0008796108

COMMISSION_BP = 6.0

SCORE_MIN = 1.50
SPREAD_MAX = 2.0
BREADTH_MAX = 0.60

TP_BP = 60.0
SL_BP = 50.0
MAX_HOLD = 120

PRE_FROM = pd.Timestamp("2026-07-28", tz="UTC")
PRE_TO   = pd.Timestamp("2026-09-01", tz="UTC")

SEP_FROM = pd.Timestamp("2026-09-06", tz="UTC")

FRESH_FROM = pd.Timestamp(
    "2026-09-10 00:00:00",
    tz="Europe/Moscow"
).tz_convert("UTC")

WINDOWS = [
    ("W0", "2026-07-28", "2026-08-04"),
    ("W1", "2026-08-04", "2026-08-11"),
    ("W2", "2026-08-11", "2026-08-18"),
    ("W3", "2026-08-18", "2026-08-25"),
    ("W4", "2026-08-25", "2026-09-01"),
]

WINDOWS = [
    (
        name,
        pd.Timestamp(a, tz="UTC"),
        pd.Timestamp(b, tz="UTC"),
    )
    for name, a, b in WINDOWS
]

# Only parameter selected on PRE.
FLIP_THRESHOLDS = [
    0.40,
    0.425,
    0.45,
    0.475,
    0.50,
]


# ============================================================
# RET5
# ============================================================

def rebuild_ret5(live):
    five_ns = pd.Timedelta(minutes=5).value
    tol_ns = pd.Timedelta(minutes=2).value

    pieces = []

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
        pieces.append(g)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


# ============================================================
# LOAD
# ============================================================

def load_data():
    print("=" * 120)
    print("REGIME FLIP TRIPLE BARRIER")
    print("=" * 120)

    hist = pd.read_pickle(HIST_FILE)

    wanted = [
        "time",
        "ticker",
        "mid",
        "spread_bp",
        "ret_1m",
        "ret_5m",
    ]

    hist = hist[
        [c for c in wanted if c in hist.columns]
    ].copy()

    hist["time"] = pd.to_datetime(
        hist["time"],
        utc=True,
        errors="coerce",
    )

    live = pd.read_csv(
        LIVE_FILE,
        usecols=lambda c: c in {
            "time",
            "ticker",
            "mid",
            "spread_bp",
            "ret_1m",
        },
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

    live = rebuild_ret5(live)

    df = pd.concat(
        [hist, live],
        ignore_index=True,
        sort=False,
    )

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
    )

    # directional breadth
    share_up = (
        df.assign(
            _up=(df["ret_5m"] > 0).astype(float)
        )
        .groupby("time")["_up"]
        .mean()
    )

    df["share_up"] = df["time"].map(
        share_up
    )

    df["breadth"] = np.maximum(
        df["share_up"],
        1.0 - df["share_up"],
    )

    print(
        "rows:",
        f"{len(df):,}",
    )

    print(
        "range:",
        df["time"].min(),
        "->",
        df["time"].max(),
    )

    return df


# ============================================================
# ARRAYS
# ============================================================

def market_arrays(df):
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
            "mid": g["mid"].to_numpy(
                dtype=float
            ),
            "spread": g["spread_bp"].to_numpy(
                dtype=float
            ),
        }

    return out


# ============================================================
# FIRST TOUCH — GENERAL SIDE
# ============================================================

def simulate(
    row,
    arrays,
    side,
):
    a = arrays[row.ticker]

    ts = a["time_ns"]
    mids = a["mid"]
    spreads = a["spread"]

    entry_time = pd.Timestamp(
        row.time
    )

    entry_ns = (
        entry_time
        .tz_convert("UTC")
        .tz_localize(None)
        .value
    )

    start = np.searchsorted(
        ts,
        entry_ns,
        side="left",
    )

    if start >= len(ts) - 1:
        return None

    entry_mid = float(row.mid)

    target_ns = (
        entry_ns
        + pd.Timedelta(
            minutes=MAX_HOLD
        ).value
    )

    end = np.searchsorted(
        ts,
        target_ns,
        side="left",
    )

    stop = min(
        end + 1,
        len(ts),
    )

    chosen = None
    reason = None

    for j in range(
        start + 1,
        stop,
    ):
        px = mids[j]

        if not np.isfinite(px):
            continue

        directional_bp = (
            side
            * (
                px / entry_mid - 1.0
            )
            * 10000.0
        )

        if directional_bp >= TP_BP:
            chosen = j
            reason = "TP"
            break

        if directional_bp <= -SL_BP:
            chosen = j
            reason = "SL"
            break

    if chosen is None:
        if end >= len(ts):
            return None

        if (
            ts[end] - target_ns
            > pd.Timedelta(minutes=3).value
        ):
            return None

        chosen = end
        reason = "TIME"

    exit_mid = float(mids[chosen])
    exit_spread = float(
        spreads[chosen]
    )

    if (
        not np.isfinite(exit_mid)
        or not np.isfinite(exit_spread)
    ):
        return None

    gross_bp = (
        side
        * (
            exit_mid / entry_mid - 1.0
        )
        * 10000.0
    )

    cost_bp = (
        COMMISSION_BP
        + 0.5 * float(row.spread_bp)
        + 0.5 * exit_spread
    )

    return {
        "time": row.time,
        "exit_time": pd.Timestamp(
            a["time"][chosen]
        ),
        "ticker": row.ticker,
        "side": side,
        "score_mr": float(row.score_mr),
        "share_up": float(row.share_up),
        "breadth": float(row.breadth),
        "entry_spread": float(
            row.spread_bp
        ),
        "exit_spread": exit_spread,
        "reason": reason,
        "gross_bp": gross_bp,
        "cost_bp": cost_bp,
        "net_bp": gross_bp - cost_bp,
    }


# ============================================================
# NON OVERLAP
# ============================================================

def non_overlap(x):
    if x.empty:
        return x

    keep = []

    for ticker, g in x.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values("time")
        busy = None

        for idx, r in g.iterrows():
            if (
                busy is not None
                and r["time"] < busy
            ):
                continue

            keep.append(idx)
            busy = r["exit_time"]

    return x.loc[keep].copy()


# ============================================================
# METRICS
# ============================================================

def metrics(x):
    if x.empty:
        return {
            "trades": 0,
            "net": np.nan,
            "mean": np.nan,
            "wr": np.nan,
            "days": 0,
            "pos_days": 0,
            "no_best_day": np.nan,
            "no_best_ticker": np.nan,
        }

    x = x.copy()

    x["date_msk"] = (
        x["time"]
        .dt.tz_convert(
            "Europe/Moscow"
        )
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
        "net": float(total),
        "mean": float(
            x["net_bp"].mean()
        ),
        "wr": float(
            100
            * (x["net_bp"] > 0).mean()
        ),
        "days": len(daily),
        "pos_days": int(
            (daily > 0).sum()
        ),
        "no_best_day": float(
            total - daily.max()
        ),
        "no_best_ticker": float(
            total - ticker.max()
        ),
    }


# ============================================================
# MAIN
# ============================================================

df = load_data()
arrays = market_arrays(df)

# Same class of downward shock events.
entries = df[
    (df["score_mr"] >= SCORE_MIN)
    & (df["spread_bp"] <= SPREAD_MAX)
    & (df["breadth"] <= BREADTH_MAX)
].copy()

print(
    "candidate events:",
    len(entries),
)

# Simulate BOTH directions once.
rows = []

for i, r in enumerate(
    entries.itertuples(index=False),
    start=1,
):
    for side in [1, -1]:
        s = simulate(
            r,
            arrays,
            side,
        )

        if s is not None:
            rows.append(s)

    if i % 100 == 0:
        print(
            "simulate:",
            i,
            "/",
            len(entries),
        )

tr = pd.DataFrame(rows)

tr["time"] = pd.to_datetime(
    tr["time"],
    utc=True,
)

tr["exit_time"] = pd.to_datetime(
    tr["exit_time"],
    utc=True,
)

print(
    "directional simulations:",
    len(tr),
)

result_rows = []
export_parts = []

for flip in FLIP_THRESHOLDS:

    # One row per event:
    #
    # weak market -> SHORT continuation
    # otherwise   -> LONG mean reversion
    selected = tr[
        (
            (
                (tr["share_up"] < flip)
                & (tr["side"] == -1)
            )
            |
            (
                (tr["share_up"] >= flip)
                & (tr["side"] == 1)
            )
        )
    ].copy()

    selected = non_overlap(
        selected
    )

    selected["flip_threshold"] = (
        flip
    )

    export_parts.append(
        selected
    )

    row = {
        "flip_threshold": flip,
    }

    positive_windows = 0
    usable_windows = 0

    for name, a, b in WINDOWS:
        w = selected[
            (selected["time"] >= a)
            & (selected["time"] < b)
        ]

        m = metrics(w)

        row[f"{name}_trades"] = (
            m["trades"]
        )
        row[f"{name}_net"] = (
            m["net"]
        )

        if m["trades"] > 0:
            usable_windows += 1

            if m["net"] > 0:
                positive_windows += 1

    pre = selected[
        (selected["time"] >= PRE_FROM)
        & (selected["time"] < PRE_TO)
    ]

    sep = selected[
        (selected["time"] >= SEP_FROM)
        & (selected["time"] < FRESH_FROM)
    ]

    fresh = selected[
        selected["time"] >= FRESH_FROM
    ]

    pm = metrics(pre)
    sm = metrics(sep)
    fm = metrics(fresh)

    row.update({
        "positive_windows":
            positive_windows,
        "usable_windows":
            usable_windows,

        "pre_trades":
            pm["trades"],
        "pre_net":
            pm["net"],
        "pre_mean":
            pm["mean"],
        "pre_wr":
            pm["wr"],
        "pre_pos_days":
            pm["pos_days"],
        "pre_days":
            pm["days"],
        "pre_no_best_day":
            pm["no_best_day"],
        "pre_no_best_ticker":
            pm["no_best_ticker"],

        "sep_trades":
            sm["trades"],
        "sep_net":
            sm["net"],
        "sep_mean":
            sm["mean"],
        "sep_wr":
            sm["wr"],
        "sep_pos_days":
            sm["pos_days"],
        "sep_days":
            sm["days"],
        "sep_no_best_day":
            sm["no_best_day"],
        "sep_no_best_ticker":
            sm["no_best_ticker"],

        "fresh_trades":
            fm["trades"],
        "fresh_net":
            fm["net"],
        "fresh_mean":
            fm["mean"],
        "fresh_pos_days":
            fm["pos_days"],
        "fresh_days":
            fm["days"],
    })

    result_rows.append(row)

res = pd.DataFrame(
    result_rows
)

res.to_csv(
    OUT,
    index=False,
)

if export_parts:
    pd.concat(
        export_parts,
        ignore_index=True,
    ).to_csv(
        OUT_TRADES,
        index=False,
    )


# ============================================================
# TABLE
# ============================================================

cols = [
    "flip_threshold",

    "W0_net",
    "W1_net",
    "W2_net",
    "W3_net",
    "W4_net",

    "pre_trades",
    "pre_net",
    "pre_mean",
    "pre_wr",
    "positive_windows",
    "pre_pos_days",
    "pre_days",
    "pre_no_best_day",
    "pre_no_best_ticker",

    "sep_trades",
    "sep_net",
    "sep_mean",
    "sep_wr",
    "sep_pos_days",
    "sep_days",
    "sep_no_best_day",
    "sep_no_best_ticker",

    "fresh_trades",
    "fresh_net",
    "fresh_mean",
]

print()
print("=" * 120)
print("REGIME FLIP RESULTS")
print("=" * 120)

print(
    res[cols]
    .sort_values(
        "pre_net",
        ascending=False,
    )
    .to_string(
        index=False,
        formatters={
            c: "{:+.1f}".format
            for c in [
                "W0_net",
                "W1_net",
                "W2_net",
                "W3_net",
                "W4_net",
                "pre_net",
                "pre_no_best_day",
                "pre_no_best_ticker",
                "sep_net",
                "sep_no_best_day",
                "sep_no_best_ticker",
                "fresh_net",
            ]
        }
        | {
            "pre_mean":
                "{:+.2f}".format,
            "sep_mean":
                "{:+.2f}".format,
            "fresh_mean":
                "{:+.2f}".format,
        }
    )
)


# ============================================================
# PRE SELECT ONLY
# ============================================================

strict = res[
    (res["pre_trades"] >= 50)
    & (res["usable_windows"] == 5)
    & (res["positive_windows"] >= 4)
    & (res["pre_net"] > 0)
    & (res["pre_mean"] > 0)
    & (res["pre_no_best_day"] > 0)
    & (res["pre_no_best_ticker"] > 0)
].copy()

print()
print("=" * 120)
print("PRE-ROBUST REGIME FLIPS")
print("=" * 120)

if strict.empty:
    print("NONE")
else:
    print(
        strict[
            cols
        ]
        .sort_values(
            "pre_net",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )


print()
print("=" * 120)
print("PRE-ROBUST + SEP6-9 POSITIVE")
print("=" * 120)

if strict.empty:
    print("NONE")
else:
    survived = strict[
        (strict["sep_trades"] >= 5)
        & (strict["sep_net"] > 0)
        & (strict["sep_no_best_day"] > 0)
        & (strict["sep_no_best_ticker"] > 0)
    ]

    if survived.empty:
        print("NONE")
    else:
        print(
            survived[
                cols
            ]
            .sort_values(
                "sep_net",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )


print()
print("=" * 120)
print("FRESH SEP10+")
print("=" * 120)

fresh = strict[
    strict["fresh_trades"] > 0
]

if fresh.empty:
    print("NO FRESH TRADES YET")
else:
    print(
        fresh[
            cols
        ]
        .sort_values(
            "fresh_net",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

print()
print("Saved:")
print(OUT)
print(OUT_TRADES)

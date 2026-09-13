from pathlib import Path
import itertools
import gc
import numpy as np
import pandas as pd

ROOT = Path("/root/rl-trading-tbank")

HIST_FILE = ROOT / "state" / "ofi_dataset_h5.pkl"
LIVE_FILE = ROOT / "state" / "shadow_observer_features.csv"

OUT = ROOT / "state" / "flow_momentum_results.csv"
OUT_TRADES = ROOT / "state" / "flow_momentum_trades.csv"

# ============================================================
# REAL ECONOMICS
# ============================================================

COMMISSION_BP = 10.0

RET5_MEAN = -0.0000156187
RET5_STD  =  0.0020025922

RET1_MEAN = -0.0000033215
RET1_STD  =  0.0008796108

# ============================================================
# SPLITS
# ============================================================

PRE_FROM = pd.Timestamp("2026-07-28", tz="UTC")
PRE_TO   = pd.Timestamp("2026-09-01", tz="UTC")

SEP_FROM = pd.Timestamp("2026-09-06", tz="UTC")

FRESH_FROM = pd.Timestamp(
    "2026-09-10 00:00:00",
    tz="Europe/Moscow",
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

# ============================================================
# SMALL SEARCH SPACE
# ============================================================

SCORE_CUTS = [
    1.30,
    1.50,
    1.75,
    2.00,
]

FLOW_VOTES = [
    1,
    2,
    3,
]

MARKET_ALIGN = [
    False,
    True,
]

HORIZONS = [
    30,
    60,
    90,
]

SPREAD_MAX = 2.0


# ============================================================
# LIVE RET5
# ============================================================

def rebuild_ret5(live):
    five_ns = pd.Timedelta(minutes=5).value
    tol_ns = pd.Timedelta(minutes=2).value

    parts = []

    for ticker, g in live.groupby(
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
        parts.append(g)

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ============================================================
# LOAD
# ============================================================

def load_data():
    print("=" * 120)
    print("FLOW-CONFIRMED MOMENTUM — LOAD")
    print("=" * 120)

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

    hist = pd.read_pickle(
        HIST_FILE
    )

    hist = hist[
        [c for c in wanted if c in hist.columns]
    ].copy()

    hist["time"] = pd.to_datetime(
        hist["time"],
        utc=True,
        errors="coerce",
    )

    live_wanted = [
        c
        for c in wanted
        if c != "ret_5m"
    ]

    live = pd.read_csv(
        LIVE_FILE,
        usecols=lambda c:
            c in set(live_wanted),
    )

    live["time"] = pd.to_datetime(
        live["time"],
        utc=True,
        errors="coerce",
    )

    live = rebuild_ret5(
        live
    )

    df = pd.concat(
        [hist, live],
        ignore_index=True,
        sort=False,
    )

    del hist, live
    gc.collect()

    numeric = [
        c
        for c in wanted
        if c not in [
            "time",
            "ticker",
        ]
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
    # MOMENTUM SCORE
    #
    # Positive = upward momentum
    # Negative = downward momentum
    # --------------------------------------------------------

    z5 = (
        df["ret_5m"] - RET5_MEAN
    ) / RET5_STD

    z1 = (
        df["ret_1m"] - RET1_MEAN
    ) / RET1_STD

    df["score_mom"] = (
        0.75 * z5
        + 0.25 * z1
    ).astype("float32")

    # --------------------------------------------------------
    # PRICE CONFIRMATION
    # --------------------------------------------------------

    df["price_agree"] = (
        np.sign(df["ret_1m"])
        == np.sign(df["ret_5m"])
    )

    # --------------------------------------------------------
    # FLOW VOTES
    #
    # Positive raw feature is treated as buy pressure.
    # Negative as sell pressure.
    #
    # We do not require magnitude yet — only direction.
    # --------------------------------------------------------

    side = np.sign(
        df["score_mom"]
    )

    votes = np.zeros(
        len(df),
        dtype=np.int8,
    )

    for c in [
        "ofi",
        "tfi",
        "imb1",
        "imb5",
        "micro_dev",
    ]:
        v = pd.to_numeric(
            df[c],
            errors="coerce",
        ).fillna(0).to_numpy()

        votes += (
            side.to_numpy()
            * v
            > 0
        ).astype(np.int8)

    df["flow_votes"] = votes

    # --------------------------------------------------------
    # SIGNED MARKET DIRECTION
    # --------------------------------------------------------

    share_up = (
        df.assign(
            _up=(
                df["ret_5m"] > 0
            ).astype(float)
        )
        .groupby("time")["_up"]
        .mean()
    )

    df["share_up"] = (
        df["time"]
        .map(share_up)
        .astype("float32")
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
# FUTURE EXIT
# ============================================================

def add_exit(df, horizon):
    pieces = []

    delta_ns = pd.Timedelta(
        minutes=horizon
    ).value

    tol_ns = pd.Timedelta(
        minutes=3
    ).value

    keep = [
        "time",
        "ticker",
        "mid",
        "spread_bp",
        "ret_1m",
        "ret_5m",
        "score_mom",
        "price_agree",
        "flow_votes",
        "share_up",
    ]

    base = df[keep]

    for ticker, g in base.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values(
            "time"
        ).copy()

        ts = (
            g["time"]
            .dt.tz_convert(None)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )

        mid = g["mid"].to_numpy(
            dtype=float
        )

        spr = g[
            "spread_bp"
        ].to_numpy(
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

        rows = np.where(
            valid
        )[0]

        jj = j[valid]

        close = (
            ts[jj]
            - target[rows]
            <= tol_ns
        )

        rows = rows[close]
        jj = jj[close]

        exit_mid[rows] = (
            mid[jj]
        )

        exit_spread[rows] = (
            spr[jj]
        )

        exit_ns[rows] = (
            ts[jj]
        )

        g["exit_mid"] = (
            exit_mid
        )

        g["exit_spread"] = (
            exit_spread
        )

        et = pd.Series(
            pd.NaT,
            index=g.index,
            dtype="datetime64[ns, UTC]",
        )

        if len(rows):
            et.iloc[rows] = (
                pd.to_datetime(
                    exit_ns[rows],
                    utc=True,
                )
            )

        g["exit_time"] = et

        pieces.append(g)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


# ============================================================
# NON-OVERLAP
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

    if not keep:
        return x.iloc[:0]

    return x.loc[
        keep
    ].copy()


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

    total = float(
        x["net_bp"].sum()
    )

    daily = (
        x.groupby(
            "date_msk"
        )["net_bp"]
        .sum()
    )

    ticker = (
        x.groupby(
            "ticker"
        )["net_bp"]
        .sum()
    )

    return {
        "trades":
            int(len(x)),

        "net":
            total,

        "mean":
            float(
                x["net_bp"].mean()
            ),

        "wr":
            float(
                100
                * (
                    x["net_bp"] > 0
                ).mean()
            ),

        "days":
            int(len(daily)),

        "pos_days":
            int(
                (daily > 0).sum()
            ),

        "no_best_day":
            float(
                total
                - daily.max()
            ),

        "no_best_ticker":
            float(
                total
                - ticker.max()
            ),
    }


# ============================================================
# MAIN
# ============================================================

df = load_data()

results = []
trade_parts = []

for horizon in HORIZONS:

    print()
    print("=" * 120)
    print(
        "HORIZON",
        horizon,
    )
    print("=" * 120)

    h = add_exit(
        df,
        horizon,
    )

    base = h[
        h["exit_mid"].notna()
        & h["exit_time"].notna()
        & (
            h["spread_bp"]
            <= SPREAD_MAX
        )
        & h["price_agree"]
    ].copy()

    base["side"] = np.where(
        base["score_mom"] > 0,
        1,
        -1,
    ).astype(np.int8)

    base["gross_bp"] = (
        base["side"]
        * (
            base["exit_mid"]
            / base["mid"]
            - 1.0
        )
        * 10000.0
    )

    base["cost_bp"] = (
        COMMISSION_BP
        + 0.5
        * base["spread_bp"]
        + 0.5
        * base["exit_spread"]
    )

    base["net_bp"] = (
        base["gross_bp"]
        - base["cost_bp"]
    )

    for (
        score_cut,
        flow_cut,
        align_market,
    ) in itertools.product(
        SCORE_CUTS,
        FLOW_VOTES,
        MARKET_ALIGN,
    ):

        x = base[
            base[
                "score_mom"
            ].abs()
            >= score_cut
        ].copy()

        x = x[
            x["flow_votes"]
            >= flow_cut
        ]

        if align_market:

            long_ok = (
                (x["side"] == 1)
                & (
                    x["share_up"]
                    >= 0.50
                )
            )

            short_ok = (
                (x["side"] == -1)
                & (
                    x["share_up"]
                    <= 0.50
                )
            )

            x = x[
                long_ok | short_ok
            ]

        x = non_overlap(
            x
        )

        row = {
            "horizon":
                horizon,

            "score_cut":
                score_cut,

            "flow_votes":
                flow_cut,

            "market_align":
                align_market,
        }

        positive_windows = 0
        usable_windows = 0

        for name, a, b in WINDOWS:

            w = x[
                (x["time"] >= a)
                & (x["time"] < b)
            ]

            m = metrics(w)

            row[
                f"{name}_trades"
            ] = m["trades"]

            row[
                f"{name}_net"
            ] = m["net"]

            if m["trades"] > 0:

                usable_windows += 1

                if m["net"] > 0:
                    positive_windows += 1

        pre = x[
            (x["time"] >= PRE_FROM)
            & (x["time"] < PRE_TO)
        ]

        sep = x[
            (x["time"] >= SEP_FROM)
            & (x["time"] < FRESH_FROM)
        ]

        fresh = x[
            x["time"] >= FRESH_FROM
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

            "fresh_wr":
                fm["wr"],

            "fresh_pos_days":
                fm["pos_days"],

            "fresh_days":
                fm["days"],

            "fresh_no_best_day":
                fm["no_best_day"],

            "fresh_no_best_ticker":
                fm["no_best_ticker"],
        })

        results.append(row)

    del h, base
    gc.collect()


res = pd.DataFrame(
    results
)

res.to_csv(
    OUT,
    index=False,
)


# ============================================================
# SELECT ONLY ON PRE
# ============================================================

strict = res[
    (res["pre_trades"] >= 40)
    & (
        res["usable_windows"]
        == 5
    )
    & (
        res["positive_windows"]
        >= 4
    )
    & (
        res["pre_net"] > 0
    )
    & (
        res["pre_mean"] > 0
    )
    & (
        res[
            "pre_no_best_day"
        ] > 0
    )
    & (
        res[
            "pre_no_best_ticker"
        ] > 0
    )
].copy()

if not strict.empty:

    strict[
        "selection_score"
    ] = (
        strict["pre_mean"]
        + 0.01
        * strict["pre_net"]
        + 0.01
        * strict[
            "pre_no_best_day"
        ]
        + 0.01
        * strict[
            "pre_no_best_ticker"
        ]
        + 5
        * strict[
            "positive_windows"
        ]
    )

    strict = strict.sort_values(
        "selection_score",
        ascending=False,
    )


cols = [
    "horizon",
    "score_cut",
    "flow_votes",
    "market_align",

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
    "fresh_wr",
    "fresh_pos_days",
    "fresh_days",

    "fresh_no_best_day",
    "fresh_no_best_ticker",
]

formatters = {
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
        "fresh_no_best_day",
        "fresh_no_best_ticker",
    ]
}

formatters.update({
    "pre_mean":
        "{:+.2f}".format,

    "sep_mean":
        "{:+.2f}".format,

    "fresh_mean":
        "{:+.2f}".format,
})


print()
print("=" * 120)
print("PRE-ROBUST FLOW MOMENTUM")
print("=" * 120)

if strict.empty:
    print("NONE")
else:
    print(
        strict[
            cols
        ]
        .head(25)
        .to_string(
            index=False,
            formatters=formatters,
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
        & (
            strict[
                "sep_no_best_day"
            ] > 0
        )
        & (
            strict[
                "sep_no_best_ticker"
            ] > 0
        )
    ].copy()

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
            .head(25)
            .to_string(
                index=False,
                formatters=formatters,
            )
        )


print()
print("=" * 120)
print("PRE + SEP ROBUST + FRESH POSITIVE")
print("=" * 120)

if strict.empty:

    print("NONE")

else:

    final = strict[
        (strict["sep_trades"] >= 5)
        & (strict["sep_net"] > 0)
        & (
            strict[
                "sep_no_best_day"
            ] > 0
        )
        & (
            strict[
                "sep_no_best_ticker"
            ] > 0
        )
        & (
            strict[
                "fresh_trades"
            ] >= 5
        )
        & (
            strict[
                "fresh_net"
            ] > 0
        )
        & (
            strict[
                "fresh_no_best_day"
            ] > 0
        )
        & (
            strict[
                "fresh_no_best_ticker"
            ] > 0
        )
    ].copy()

    if final.empty:
        print("NONE")
    else:
        print(
            final[
                cols
            ]
            .sort_values(
                "fresh_net",
                ascending=False,
            )
            .head(25)
            .to_string(
                index=False,
                formatters=formatters,
            )
        )


print()
print("=" * 120)
print("BEST FRESH AMONG PRE-ROBUST")
print("FRESH NOT USED FOR PRE-SELECTION")
print("=" * 120)

if strict.empty:

    print("NONE")

else:

    q = strict[
        strict[
            "fresh_trades"
        ] > 0
    ].copy()

    if q.empty:
        print("NONE")
    else:
        print(
            q[
                cols
            ]
            .sort_values(
                "fresh_net",
                ascending=False,
            )
            .head(20)
            .to_string(
                index=False,
                formatters=formatters,
            )
        )


print()
print("Saved:")
print(OUT)

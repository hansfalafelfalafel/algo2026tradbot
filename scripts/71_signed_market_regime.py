from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/rl-trading-tbank")

TRADES = ROOT / "state" / "exact_cost_candidate_trades.csv"

OUT_ENRICHED = (
    ROOT / "state" / "signed_regime_trades.csv"
)

OUT_RESULTS = (
    ROOT / "state" / "signed_regime_results.csv"
)

COMMISSION_BP = 6.0

PRE_FROM = pd.Timestamp(
    "2026-07-28",
    tz="UTC",
)

PRE_TO = pd.Timestamp(
    "2026-09-01",
    tz="UTC",
)

SEP_FROM = pd.Timestamp(
    "2026-09-06",
    tz="UTC",
)

FRESH_FROM = pd.Timestamp(
    "2026-09-10 00:00:00",
    tz="Europe/Moscow",
).tz_convert("UTC")


# ============================================================
# LOAD 67 FUNCTIONS
# ============================================================

src = (
    ROOT / "scripts" / "67_triple_barrier_lab.py"
).read_text()

marker = (
    "# =====================================================================\n"
    "# MAIN\n"
    "# ====================================================================="
)

defs = src.split(marker)[0]

ns = {}

exec(
    compile(
        defs,
        "67_triple_barrier_lab.py",
        "exec",
    ),
    ns,
)

load_data = ns["load_data"]


# ============================================================
# LOAD MARKET
# ============================================================

print("=" * 120)
print("SIGNED MARKET REGIME")
print("=" * 120)

df = load_data()

# ------------------------------------------------------------
# Directional market state
#
# IMPORTANT:
# breadth=max(share_up,1-share_up) loses direction.
# We now preserve it.
# ------------------------------------------------------------

market = (
    df.groupby("time")
    .agg(
        share_up=(
            "ret_5m",
            lambda s:
                float((s > 0).mean())
        ),

        market_ret5_median=(
            "ret_5m",
            "median",
        ),

        market_ret1_median=(
            "ret_1m",
            "median",
        ),

        market_score_median=(
            "score_mr",
            "median",
        ),

        market_n=(
            "ticker",
            "nunique",
        ),
    )
    .reset_index()
)

market["signed_breadth"] = (
    2.0 * market["share_up"] - 1.0
)

print(
    "market states:",
    f"{len(market):,}"
)


# ============================================================
# LOAD EXACT TRADES
# ============================================================

t = pd.read_csv(TRADES)

t["time"] = pd.to_datetime(
    t["time"],
    utc=True,
)

t["exit_time"] = pd.to_datetime(
    t["exit_time"],
    utc=True,
)

t = t[
    t["strategy"] == "MR_TB_SCORE_150"
].copy()

t["spread_cost_bp"] = (
    0.5 * t["entry_spread"]
    + 0.5 * t["exit_spread"]
)

t["net6"] = (
    t["gross_bp"]
    - t["spread_cost_bp"]
    - COMMISSION_BP
)

t = t.merge(
    market,
    on="time",
    how="left",
)

msk = t["time"].dt.tz_convert(
    "Europe/Moscow"
)

t["date_msk"] = msk.dt.date
t["hour_msk"] = msk.dt.hour

t["period"] = np.select(
    [
        (t["time"] >= PRE_FROM)
        & (t["time"] < PRE_TO),

        (t["time"] >= SEP_FROM)
        & (t["time"] < FRESH_FROM),

        t["time"] >= FRESH_FROM,
    ],
    [
        "PRE",
        "SEP6_9",
        "FRESH",
    ],
    default="OTHER",
)

t.to_csv(
    OUT_ENRICHED,
    index=False,
)


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

    daily = (
        x.groupby("date_msk")
        ["net6"]
        .sum()
    )

    ticker = (
        x.groupby("ticker")
        ["net6"]
        .sum()
    )

    total = x["net6"].sum()

    return {
        "trades": len(x),
        "net": total,
        "mean": x["net6"].mean(),
        "wr": (
            100
            * (x["net6"] > 0).mean()
        ),
        "days": len(daily),
        "pos_days": int(
            (daily > 0).sum()
        ),
        "no_best_day": (
            total - daily.max()
        ),
        "no_best_ticker": (
            total - ticker.max()
        ),
    }


# ============================================================
# RAW REGIME COMPARISON
# ============================================================

print()
print("=" * 120)
print("RAW MARKET REGIME COMPARISON")
print("=" * 120)

cols = [
    "share_up",
    "signed_breadth",
    "market_ret5_median",
    "market_ret1_median",
    "market_score_median",
]

for period in [
    "PRE",
    "SEP6_9",
]:
    z = t[
        t["period"] == period
    ]

    print()
    print(period)
    print("-" * 80)

    print(
        "trades:",
        len(z),
        "| net:",
        f"{z['net6'].sum():+.1f}",
    )

    for c in cols:
        print(
            f"{c:24s}",
            "median=",
            f"{z[c].median():+.6f}",
            "mean=",
            f"{z[c].mean():+.6f}",
        )


# ============================================================
# MARKET DIRECTION BINS
# ============================================================

print()
print("=" * 120)
print("SHARE_UP BINS")
print("=" * 120)

bins = [
    0.00,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    1.00,
]

t["share_up_bin"] = pd.cut(
    t["share_up"],
    bins=bins,
    include_lowest=True,
)

z = (
    t[
        t["period"].isin(
            ["PRE", "SEP6_9"]
        )
    ]
    .groupby(
        [
            "period",
            "share_up_bin",
        ],
        observed=True,
    )
    .agg(
        trades=("net6", "size"),
        net=("net6", "sum"),
        mean=("net6", "mean"),
        wr=(
            "net6",
            lambda s:
                100 * (s > 0).mean()
        ),
    )
    .reset_index()
)

print(
    z.to_string(
        index=False,
        formatters={
            "net": "{:+.1f}".format,
            "mean": "{:+.2f}".format,
            "wr": "{:.1f}".format,
        }
    )
)


# ============================================================
# SIMPLE PRE-ONLY GATES
#
# Small, interpretable family.
# We select ONLY using PRE.
# ============================================================

print()
print("=" * 120)
print("PRE-ONLY SIMPLE GATE SEARCH")
print("=" * 120)

SHARE_MIN = [
    0.40,
    0.45,
    0.50,
    0.55,
]

MARKET_RET5_MIN = [
    -0.003,
    -0.002,
    -0.001,
    0.000,
]

START_HOURS = [
    10,
    12,
    13,
    14,
]

SCORE_MAX = [
    1.70,
    1.90,
    2.20,
    99.0,
]

rows = []

pre_all = t[
    t["period"] == "PRE"
].copy()

sep_all = t[
    t["period"] == "SEP6_9"
].copy()

fresh_all = t[
    t["period"] == "FRESH"
].copy()

# Weekly pre windows identical to earlier experiment.
W = [
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

for share_min in SHARE_MIN:
    for market_ret5_min in MARKET_RET5_MIN:
        for start_hour in START_HOURS:
            for score_max in SCORE_MAX:

                def gate(x):
                    return x[
                        (x["share_up"] >= share_min)
                        & (
                            x["market_ret5_median"]
                            >= market_ret5_min
                        )
                        & (
                            x["hour_msk"]
                            >= start_hour
                        )
                        & (
                            x["score_mr"]
                            <= score_max
                        )
                    ].copy()

                gp = gate(pre_all)

                positive_windows = 0
                usable_windows = 0

                weekly = {}

                for name, a, b in W:
                    q = gp[
                        (gp["time"] >= a)
                        & (gp["time"] < b)
                    ]

                    qm = metrics(q)

                    weekly[name] = qm["net"]

                    if qm["trades"] > 0:
                        usable_windows += 1

                        if qm["net"] > 0:
                            positive_windows += 1

                pm = metrics(gp)

                gs = gate(sep_all)
                sm = metrics(gs)

                gf = gate(fresh_all)
                fm = metrics(gf)

                rows.append({
                    "share_min":
                        share_min,

                    "market_ret5_min":
                        market_ret5_min,

                    "start_hour":
                        start_hour,

                    "score_max":
                        score_max,

                    **{
                        f"{k}_net": v
                        for k, v
                        in weekly.items()
                    },

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

                    "pre_days":
                        pm["days"],

                    "pre_pos_days":
                        pm["pos_days"],

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

                    "sep_days":
                        sm["days"],

                    "sep_pos_days":
                        sm["pos_days"],

                    "sep_no_best_day":
                        sm["no_best_day"],

                    "fresh_trades":
                        fm["trades"],

                    "fresh_net":
                        fm["net"],

                    "fresh_mean":
                        fm["mean"],
                })

res = pd.DataFrame(rows)

res.to_csv(
    OUT_RESULTS,
    index=False,
)


# ============================================================
# SELECT USING PRE ONLY
# ============================================================

strict_pre = res[
    (res["pre_trades"] >= 30)
    & (res["usable_windows"] == 5)
    & (res["positive_windows"] >= 4)
    & (res["pre_net"] > 0)
    & (res["pre_mean"] > 0)
    & (res["pre_no_best_day"] > 0)
    & (res["pre_no_best_ticker"] > 0)
].copy()

if not strict_pre.empty:
    strict_pre["selection_score"] = (
        strict_pre["pre_mean"]
        + 0.01
        * strict_pre["pre_net"]
        + 0.01
        * strict_pre["pre_no_best_day"]
        + 0.01
        * strict_pre["pre_no_best_ticker"]
        + 5.0
        * strict_pre["positive_windows"]
    )

    strict_pre = strict_pre.sort_values(
        "selection_score",
        ascending=False,
    )


show = [
    "share_min",
    "market_ret5_min",
    "start_hour",
    "score_max",

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
    "sep_pos_days",
    "sep_days",
    "sep_no_best_day",

    "fresh_trades",
    "fresh_net",
    "fresh_mean",
]

print()
print("=" * 120)
print("STRICT PRE-SELECTED SIGNED REGIME GATES")
print("=" * 120)

if strict_pre.empty:
    print("NONE")
else:
    print(
        strict_pre[
            show
        ]
        .head(20)
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


print()
print("=" * 120)
print("PRE-SELECTED + SEP6-9 POSITIVE")
print("SEP NOT USED FOR PARAMETER SELECTION")
print("=" * 120)

if strict_pre.empty:
    print("NONE")
else:
    survivors = strict_pre[
        (strict_pre["sep_trades"] >= 5)
        & (strict_pre["sep_net"] > 0)
        & (
            strict_pre[
                "sep_no_best_day"
            ] > 0
        )
    ].copy()

    if survivors.empty:
        print("NONE")
    else:
        print(
            survivors[
                show
            ]
            .sort_values(
                "sep_net",
                ascending=False,
            )
            .head(20)
            .to_string(
                index=False
            )
        )


print()
print("=" * 120)
print("FRESH SEP10+")
print("=" * 120)

if strict_pre.empty:
    print("NO PRE-SELECTED GATES")
else:
    q = strict_pre[
        strict_pre["fresh_trades"] > 0
    ]

    if q.empty:
        print("NO FRESH TRADES YET")
    else:
        print(
            q[
                show
            ]
            .sort_values(
                "fresh_net",
                ascending=False,
            )
            .head(20)
            .to_string(
                index=False
            )
        )

print()
print("Saved:")
print(OUT_ENRICHED)
print(OUT_RESULTS)

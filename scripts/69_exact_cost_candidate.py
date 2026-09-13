from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/rl-trading-tbank")

# ============================================================
# Reuse FUNCTIONS from 67 without running its MAIN section
# ============================================================

src = (
    ROOT / "scripts" / "67_triple_barrier_lab.py"
).read_text()

marker = "# =====================================================================\n# MAIN\n# ====================================================================="

if marker not in src:
    raise RuntimeError("MAIN marker not found in script 67")

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
make_market_arrays = ns["make_market_arrays"]
simulate_one = ns["simulate_one"]
non_overlap = ns["non_overlap"]

# ============================================================
# EXACT CANDIDATES
# ============================================================

CANDIDATES = [
    {
        "name": "MR_TB_SCORE_130",
        "score": 1.30,
    },
    {
        "name": "MR_TB_SCORE_150",
        "score": 1.50,
    },
]

BREADTH = 0.60
SPREAD = 2.00

HOLD = 120
TP = 60.0
SL = 50.0

COSTS = [
    0.0,
    2.0,
    4.0,
    6.0,
    8.0,
    10.0,
]

WINDOWS = ns["WINDOWS"]

OLD_FWD_FROM = ns["OLD_FORWARD_FROM"]
FRESH_FWD_FROM = ns["FRESH_FORWARD_FROM"]

OUT_TRADES = (
    ROOT
    / "state"
    / "exact_cost_candidate_trades.csv"
)

OUT_SUMMARY = (
    ROOT
    / "state"
    / "exact_cost_candidate_summary.csv"
)


def exact_metrics(x, commission):
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
        }

    z = x.copy()

    # simulate_one was originally computed with:
    #
    # cost = 10bp commission
    #        + entry half-spread
    #        + exit half-spread
    #
    # Recover spread component EXACTLY.
    z["spread_cost_bp"] = (
        0.5 * z["entry_spread"]
        + 0.5 * z["exit_spread"]
    )

    z["net_exact"] = (
        z["gross_bp"]
        - z["spread_cost_bp"]
        - commission
    )

    z["date_msk"] = (
        z["time"]
        .dt.tz_convert("Europe/Moscow")
        .dt.date
    )

    total = z["net_exact"].sum()

    daily = (
        z.groupby("date_msk")
        ["net_exact"]
        .sum()
    )

    ticker = (
        z.groupby("ticker")
        ["net_exact"]
        .sum()
    )

    return {
        "trades": int(len(z)),
        "net": float(total),
        "mean": float(
            z["net_exact"].mean()
        ),
        "median": float(
            z["net_exact"].median()
        ),
        "wr": float(
            100
            * (z["net_exact"] > 0).mean()
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
    }


print("=" * 120)
print("EXACT COST ROBUSTNESS")
print("=" * 120)

df = load_data()

market = make_market_arrays(df)

# Need only max universe used by the 2 candidates.
entries = df[
    (df["score_mr"] >= 1.30)
    & (df["breadth"] <= BREADTH)
    & (df["spread_bp"] <= SPREAD)
].copy()

print(
    "candidate entry universe:",
    len(entries),
)

# ============================================================
# Simulate barrier path ONCE
# ============================================================

rows = []

for i, r in enumerate(
    entries.itertuples(index=False),
    start=1,
):
    s = simulate_one(
        r,
        market,
        tp_bp=TP,
        sl_bp=SL,
        max_hold_min=HOLD,
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

trades_all = pd.DataFrame(rows)

trades_all["time"] = pd.to_datetime(
    trades_all["time"],
    utc=True,
)

trades_all["exit_time"] = pd.to_datetime(
    trades_all["exit_time"],
    utc=True,
)

print(
    "simulated trades before filters/non-overlap:",
    len(trades_all),
)

all_summaries = []
all_trade_exports = []

for cfg in CANDIDATES:

    print()
    print("=" * 120)
    print(cfg["name"])
    print("=" * 120)

    t = trades_all[
        trades_all["score_mr"]
        >= cfg["score"]
    ].copy()

    # Important:
    # non-overlap AFTER applying score threshold,
    # separately for each strategy.
    t = non_overlap(t)

    print(
        "non-overlap trades:",
        len(t),
    )

    t["strategy"] = cfg["name"]

    t["spread_cost_bp"] = (
        0.5 * t["entry_spread"]
        + 0.5 * t["exit_spread"]
    )

    all_trade_exports.append(t)

    for cost in COSTS:

        print()
        print("-" * 100)
        print(
            f"COMMISSION = {cost:.1f} BP"
        )
        print("-" * 100)

        positive_windows = 0
        usable_windows = 0

        pre_parts = []

        row = {
            "strategy": cfg["name"],
            "score": cfg["score"],
            "breadth": BREADTH,
            "spread": SPREAD,
            "hold": HOLD,
            "tp": TP,
            "sl": SL,
            "commission_bp": cost,
        }

        for name, start, end in WINDOWS:

            w = t[
                (t["time"] >= start)
                & (t["time"] < end)
            ].copy()

            m = exact_metrics(
                w,
                cost,
            )

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

                pre_parts.append(w)

            print(
                f"{name}:",
                f"trades={m['trades']:3d}",
                (
                    f"net={m['net']:+8.1f}"
                    if np.isfinite(m["net"])
                    else "net=NaN"
                )
            )

        if pre_parts:
            pre = pd.concat(
                pre_parts,
                ignore_index=True,
            )
        else:
            pre = t.iloc[:0]

        pm = exact_metrics(
            pre,
            cost,
        )

        old = t[
            (t["time"] >= OLD_FWD_FROM)
            & (t["time"] < FRESH_FWD_FROM)
        ].copy()

        om = exact_metrics(
            old,
            cost,
        )

        fresh = t[
            t["time"] >= FRESH_FWD_FROM
        ].copy()

        fm = exact_metrics(
            fresh,
            cost,
        )

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

            "pre_median":
                pm["median"],

            "pre_wr":
                pm["wr"],

            "pre_pos_days":
                pm["pos_days"],

            "pre_days":
                pm["days"],

            "pre_without_best_day":
                pm["without_best_day"],

            "pre_without_best_ticker":
                pm["without_best_ticker"],

            "old_trades":
                om["trades"],

            "old_net":
                om["net"],

            "old_mean":
                om["mean"],

            "old_pos_days":
                om["pos_days"],

            "old_days":
                om["days"],

            "old_without_best_day":
                om["without_best_day"],

            "old_without_best_ticker":
                om["without_best_ticker"],

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

            "fresh_without_best_day":
                fm["without_best_day"],

            "fresh_without_best_ticker":
                fm["without_best_ticker"],
        })

        all_summaries.append(row)

        print()
        print(
            "PRE:",
            f"trades={pm['trades']}",
            f"net={pm['net']:+.1f}",
            f"mean={pm['mean']:+.2f}",
            f"windows={positive_windows}/5",
            f"days={pm['pos_days']}/{pm['days']}",
        )

        print(
            "     no best day:",
            f"{pm['without_best_day']:+.1f}",
        )

        print(
            "     no best ticker:",
            f"{pm['without_best_ticker']:+.1f}",
        )

        print(
            "SEP6-9:",
            f"trades={om['trades']}",
            (
                f"net={om['net']:+.1f}"
                if np.isfinite(om["net"])
                else "net=NaN"
            ),
            (
                f"mean={om['mean']:+.2f}"
                if np.isfinite(om["mean"])
                else "mean=NaN"
            ),
            f"days={om['pos_days']}/{om['days']}",
        )

        print(
            "SEP10+:",
            f"trades={fm['trades']}",
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
            f"days={fm['pos_days']}/{fm['days']}",
        )

summary = pd.DataFrame(
    all_summaries
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

if all_trade_exports:
    trade_export = pd.concat(
        all_trade_exports,
        ignore_index=True,
    )

    trade_export.to_csv(
        OUT_TRADES,
        index=False,
    )

print()
print("=" * 120)
print("EXACT VIABILITY TABLE")
print("=" * 120)

cols = [
    "strategy",
    "commission_bp",

    "pre_trades",
    "pre_net",
    "pre_mean",

    "positive_windows",

    "pre_pos_days",
    "pre_days",

    "pre_without_best_day",
    "pre_without_best_ticker",

    "old_trades",
    "old_net",
    "old_mean",
    "old_pos_days",
    "old_days",

    "old_without_best_day",
    "old_without_best_ticker",

    "fresh_trades",
    "fresh_net",
    "fresh_mean",
]

print(
    summary[
        cols
    ].to_string(
        index=False,
        formatters={
            "pre_net":
                "{:+.1f}".format,
            "pre_mean":
                "{:+.2f}".format,
            "pre_without_best_day":
                "{:+.1f}".format,
            "pre_without_best_ticker":
                "{:+.1f}".format,

            "old_net":
                "{:+.1f}".format,
            "old_mean":
                "{:+.2f}".format,
            "old_without_best_day":
                "{:+.1f}".format,
            "old_without_best_ticker":
                "{:+.1f}".format,

            "fresh_net":
                "{:+.1f}".format,
            "fresh_mean":
                "{:+.2f}".format,
        },
    )
)

print()
print("=" * 120)
print("STRICT ECONOMICALLY VIABLE")
print("=" * 120)

strict = summary[
    (summary["pre_trades"] >= 40)
    & (summary["positive_windows"] >= 4)
    & (summary["pre_net"] > 0)
    & (summary["pre_mean"] > 0)
    & (
        summary[
            "pre_without_best_day"
        ] > 0
    )
    & (
        summary[
            "pre_without_best_ticker"
        ] > 0
    )
].copy()

if strict.empty:
    print("NONE")
else:
    print(
        strict[
            cols
        ].to_string(
            index=False,
            formatters={
                "pre_net":
                    "{:+.1f}".format,
                "pre_mean":
                    "{:+.2f}".format,
                "pre_without_best_day":
                    "{:+.1f}".format,
                "pre_without_best_ticker":
                    "{:+.1f}".format,
                "old_net":
                    "{:+.1f}".format,
                "old_mean":
                    "{:+.2f}".format,
                "old_without_best_day":
                    "{:+.1f}".format,
                "old_without_best_ticker":
                    "{:+.1f}".format,
                "fresh_net":
                    "{:+.1f}".format,
                "fresh_mean":
                    "{:+.2f}".format,
            },
        )
    )

print()
print("Saved:")
print(OUT_SUMMARY)
print(OUT_TRADES)

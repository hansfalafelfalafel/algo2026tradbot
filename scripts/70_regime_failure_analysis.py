from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/rl-trading-tbank")

TRADES = ROOT / "state" / "exact_cost_candidate_trades.csv"
OUT = ROOT / "state" / "regime_failure_analysis.csv"

PRE_FROM = pd.Timestamp("2026-07-28", tz="UTC")
PRE_TO   = pd.Timestamp("2026-09-01", tz="UTC")

OLD_FROM = pd.Timestamp("2026-09-06", tz="UTC")
OLD_TO = pd.Timestamp(
    "2026-09-10 00:00:00",
    tz="Europe/Moscow",
).tz_convert("UTC")

COMMISSION = 6.0

df = pd.read_csv(TRADES)

df["time"] = pd.to_datetime(
    df["time"],
    utc=True,
)

df["exit_time"] = pd.to_datetime(
    df["exit_time"],
    utc=True,
)

# Only the stronger pre-selected candidate
df = df[
    df["strategy"] == "MR_TB_SCORE_150"
].copy()

df["spread_cost_bp"] = (
    0.5 * df["entry_spread"]
    + 0.5 * df["exit_spread"]
)

df["net6"] = (
    df["gross_bp"]
    - df["spread_cost_bp"]
    - COMMISSION
)

msk = df["time"].dt.tz_convert(
    "Europe/Moscow"
)

df["date_msk"] = msk.dt.date
df["hour_msk"] = msk.dt.hour

df["period"] = np.select(
    [
        (df["time"] >= PRE_FROM)
        & (df["time"] < PRE_TO),

        (df["time"] >= OLD_FROM)
        & (df["time"] < OLD_TO),
    ],
    [
        "PRE",
        "SEP6_9",
    ],
    default="OTHER",
)

df = df[
    df["period"].isin(
        ["PRE", "SEP6_9"]
    )
].copy()


def print_summary(name, x):
    print()
    print("=" * 115)
    print(name)
    print("=" * 115)

    if x.empty:
        print("NO DATA")
        return

    print(
        "trades:",
        len(x),
        "| net6:",
        f"{x['net6'].sum():+.1f} bp",
        "| mean:",
        f"{x['net6'].mean():+.2f}",
        "| median:",
        f"{x['net6'].median():+.2f}",
        "| WR:",
        f"{100*(x['net6'] > 0).mean():.1f}%"
    )

    print(
        "score median:",
        f"{x['score_mr'].median():.3f}",
        "| breadth median:",
        f"{x['breadth'].median():.3f}",
        "| entry spread median:",
        f"{x['entry_spread'].median():.3f}",
    )

    print()
    print("EXIT REASONS")

    r = (
        x.groupby("reason")
        .agg(
            trades=("net6", "size"),
            net=("net6", "sum"),
            mean=("net6", "mean"),
        )
        .sort_values("net", ascending=False)
    )

    print(
        r.to_string(
            formatters={
                "net": "{:+.1f}".format,
                "mean": "{:+.2f}".format,
            }
        )
    )


print("=" * 115)
print("REGIME FAILURE ANALYSIS — MR TB SCORE 1.50 @ 6BP")
print("=" * 115)

for period in [
    "PRE",
    "SEP6_9",
]:
    print_summary(
        period,
        df[df["period"] == period],
    )


print()
print("=" * 115)
print("DAILY")
print("=" * 115)

daily = (
    df.groupby(
        ["period", "date_msk"]
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
        score=("score_mr", "median"),
        breadth=("breadth", "median"),
        spread=("entry_spread", "median"),
    )
    .reset_index()
)

print(
    daily.to_string(
        index=False,
        formatters={
            "net": "{:+.1f}".format,
            "mean": "{:+.2f}".format,
            "wr": "{:.1f}".format,
            "score": "{:.3f}".format,
            "breadth": "{:.3f}".format,
            "spread": "{:.3f}".format,
        }
    )
)


print()
print("=" * 115)
print("TICKERS")
print("=" * 115)

tick = (
    df.groupby(
        ["period", "ticker"]
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

for period in [
    "PRE",
    "SEP6_9",
]:
    print()
    print("---", period, "---")

    z = (
        tick[
            tick["period"] == period
        ]
        .sort_values(
            "net",
            ascending=False,
        )
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


print()
print("=" * 115)
print("HOUR OF DAY — MOSCOW")
print("=" * 115)

hourly = (
    df.groupby(
        ["period", "hour_msk"]
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
    hourly.to_string(
        index=False,
        formatters={
            "net": "{:+.1f}".format,
            "mean": "{:+.2f}".format,
            "wr": "{:.1f}".format,
        }
    )
)


print()
print("=" * 115)
print("FEATURE BINS")
print("=" * 115)

for feature in [
    "score_mr",
    "breadth",
    "entry_spread",
]:
    print()
    print("---", feature, "---")

    # Bins are derived from PRE only.
    pre = df[
        df["period"] == "PRE"
    ][feature].dropna()

    edges = np.unique(
        pre.quantile(
            [0, .25, .5, .75, 1]
        ).to_numpy()
    )

    if len(edges) < 3:
        print("Not enough unique values")
        continue

    df["_bin"] = pd.cut(
        df[feature],
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )

    z = (
        df.groupby(
            ["period", "_bin"],
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


print()
print("=" * 115)
print("PRE WINNERS VS LOSERS")
print("=" * 115)

pre = df[
    df["period"] == "PRE"
].copy()

for label, x in [
    (
        "WIN",
        pre[pre["net6"] > 0],
    ),
    (
        "LOSS",
        pre[pre["net6"] <= 0],
    ),
]:
    print(
        label,
        "n=", len(x),
        "| score=",
        f"{x['score_mr'].median():.3f}",
        "| breadth=",
        f"{x['breadth'].median():.3f}",
        "| spread=",
        f"{x['entry_spread'].median():.3f}",
    )


daily.to_csv(
    OUT,
    index=False,
)

print()
print("Saved:", OUT)

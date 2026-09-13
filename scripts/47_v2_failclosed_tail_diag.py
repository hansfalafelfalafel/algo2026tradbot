from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/rl-trading-tbank")
STATE = ROOT / "state"

WF_FILE = STATE / "net_edge_v2_walkforward.csv"
TRADES_FILE = STATE / "net_edge_v2_test_trades.csv"

OUT_FAILCLOSED = STATE / "net_edge_v2_failclosed_trades.csv"
OUT_DAILY = STATE / "net_edge_v2_failclosed_daily.csv"
OUT_TAILS = STATE / "net_edge_v2_tail_trades.csv"


def winsorized_mean(s: pd.Series, q=0.05):
    if s.empty:
        return np.nan

    lo = s.quantile(q)
    hi = s.quantile(1 - q)

    return float(
        s.clip(lo, hi).mean()
    )


def main():
    print("=" * 110)
    print("V2 FAIL-CLOSED + TAIL DIAGNOSTICS")
    print("=" * 110)

    if not WF_FILE.exists():
        raise SystemExit(
            f"Missing {WF_FILE}"
        )

    if not TRADES_FILE.exists():
        raise SystemExit(
            f"Missing {TRADES_FILE}"
        )

    wf = pd.read_csv(WF_FILE)
    trades = pd.read_csv(TRADES_FILE)

    if "wf_test_date" not in trades.columns:
        raise SystemExit(
            "wf_test_date missing in trades file"
        )

    trades["wf_test_date"] = (
        trades["wf_test_date"]
        .astype(str)
    )

    wf["test_date"] = (
        wf["test_date"]
        .astype(str)
    )

    # ------------------------------------------------------------
    # 1. FAIL-CLOSED FILTER
    # ------------------------------------------------------------

    passed_dates = set(
        wf.loc[
            wf["selected_passed"] == True,
            "test_date",
        ]
    )

    failed_dates = set(
        wf.loc[
            wf["selected_passed"] != True,
            "test_date",
        ]
    )

    print()
    print("passed test dates:")
    print(sorted(passed_dates))

    print()
    print("rejected test dates:")
    print(sorted(failed_dates))

    fc = trades[
        trades["wf_test_date"].isin(
            passed_dates
        )
    ].copy()

    rejected = trades[
        trades["wf_test_date"].isin(
            failed_dates
        )
    ].copy()

    print()
    print("=" * 110)
    print("FAIL-CLOSED AGGREGATE")
    print("=" * 110)

    print("all historical test trades:", len(trades))
    print("accepted by validation gate:", len(fc))
    print("rejected by validation gate:", len(rejected))

    if fc.empty:
        print("NO FAIL-CLOSED TRADES")
        return

    print()
    print("longs:", int((fc["side"] == 1).sum()))
    print("shorts:", int((fc["side"] == -1).sum()))

    print()
    print(
        "gross mean:",
        f"{fc['gross_bp'].mean():+.3f} bp",
    )

    print(
        "cost mean:",
        f"{fc['execution_cost_bp'].mean():.3f} bp",
    )

    print(
        "net mean:",
        f"{fc['net_bp'].mean():+.3f} bp",
    )

    print(
        "net median:",
        f"{fc['net_bp'].median():+.3f} bp",
    )

    print(
        "winsor 5%:",
        f"{winsorized_mean(fc['net_bp']):+.3f} bp",
    )

    print(
        "net sum:",
        f"{fc['net_bp'].sum():+.3f} bp",
    )

    print(
        "hit rate:",
        f"{100*(fc['net_bp'] > 0).mean():.2f}%",
    )

    # ------------------------------------------------------------
    # 2. DAILY
    # ------------------------------------------------------------

    daily = (
        fc.groupby("wf_test_date")
        .agg(
            trades=("net_bp", "size"),
            longs=("side", lambda s: (s == 1).sum()),
            shorts=("side", lambda s: (s == -1).sum()),
            gross_mean=("gross_bp", "mean"),
            cost_mean=("execution_cost_bp", "mean"),
            net_mean=("net_bp", "mean"),
            net_median=("net_bp", "median"),
            net_sum=("net_bp", "sum"),
            hit_rate=("net_bp", lambda s: (s > 0).mean()),
            worst=("net_bp", "min"),
            best=("net_bp", "max"),
        )
        .reset_index()
    )

    print()
    print("=" * 110)
    print("FAIL-CLOSED DAILY")
    print("=" * 110)

    print(
        daily.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    positive_days = int(
        (daily["net_sum"] > 0).sum()
    )

    print()
    print(
        "positive days:",
        f"{positive_days}/{len(daily)}",
    )

    print(
        "positive day share:",
        f"{100*positive_days/len(daily):.1f}%",
    )

    print(
        "median daily net mean:",
        f"{daily['net_mean'].median():+.3f} bp",
    )

    # ------------------------------------------------------------
    # 3. TAIL ROBUSTNESS
    # ------------------------------------------------------------

    print()
    print("=" * 110)
    print("TAIL ROBUSTNESS")
    print("=" * 110)

    ordered = (
        fc["net_bp"]
        .sort_values(
            ascending=False
        )
    )

    for k in [0, 1, 2, 3, 5]:
        if len(ordered) <= k:
            continue

        s = (
            ordered.iloc[k:]
            if k
            else ordered
        )

        print(
            f"drop best {k}: "
            f"n={len(s):4d} "
            f"mean={s.mean():+.3f} bp"
        )

    print()

    for q in [0.01, 0.025, 0.05, 0.10]:
        print(
            f"winsor {100*q:4.1f}%: "
            f"{winsorized_mean(fc['net_bp'], q):+.3f} bp"
        )

    # ------------------------------------------------------------
    # 4. EXTREME LOSSES
    # ------------------------------------------------------------

    print()
    print("=" * 110)
    print("WORST 20 FAIL-CLOSED TRADES")
    print("=" * 110)

    base_cols = [
        "wf_test_date",
        "time",
        "ticker",
        "side",
        "gross_bp",
        "execution_cost_bp",
        "net_bp",
    ]

    optional_cols = [
        "spread_bp",
        "future_spread_bp",
        "expected_cost_bp",
        "p_up",
        "p_long_edge",
        "p_short_edge",
        "ofi",
        "tfi",
        "imb1",
        "imb5",
        "micro_dev",
        "ret_1m",
        "ret_5m",
    ]

    cols = [
        c
        for c in base_cols + optional_cols
        if c in fc.columns
    ]

    worst = (
        fc.sort_values(
            "net_bp"
        )
        .head(20)
        [cols]
        .copy()
    )

    print(
        worst.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------
    # 5. TAIL BUCKETS
    # ------------------------------------------------------------

    fc["loss_bucket"] = pd.cut(
        fc["net_bp"],
        bins=[
            -np.inf,
            -50,
            -25,
            -10,
            0,
            10,
            25,
            50,
            np.inf,
        ],
        labels=[
            "<=-50",
            "-50..-25",
            "-25..-10",
            "-10..0",
            "0..10",
            "10..25",
            "25..50",
            ">50",
        ],
    )

    buckets = (
        fc.groupby(
            "loss_bucket",
            observed=False,
        )
        .agg(
            trades=("net_bp", "size"),
            net_mean=("net_bp", "mean"),
            net_sum=("net_bp", "sum"),
        )
        .reset_index()
    )

    print()
    print("=" * 110)
    print("NET PNL BUCKETS")
    print("=" * 110)

    print(
        buckets.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # ------------------------------------------------------------
    # 6. BY SIDE
    # ------------------------------------------------------------

    side = (
        fc.groupby("side")
        .agg(
            trades=("net_bp", "size"),
            gross_mean=("gross_bp", "mean"),
            cost_mean=("execution_cost_bp", "mean"),
            net_mean=("net_bp", "mean"),
            net_median=("net_bp", "median"),
            net_sum=("net_bp", "sum"),
            hit_rate=("net_bp", lambda s: (s > 0).mean()),
            worst=("net_bp", "min"),
        )
        .reset_index()
    )

    print()
    print("=" * 110)
    print("BY SIDE")
    print("=" * 110)

    print(
        side.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------
    # 7. BY TICKER — DESCRIPTIVE ONLY
    # ------------------------------------------------------------

    ticker = (
        fc.groupby("ticker")
        .agg(
            trades=("net_bp", "size"),
            net_mean=("net_bp", "mean"),
            net_median=("net_bp", "median"),
            net_sum=("net_bp", "sum"),
            hit_rate=("net_bp", lambda s: (s > 0).mean()),
            worst=("net_bp", "min"),
        )
        .reset_index()
        .sort_values(
            "net_sum"
        )
    )

    print()
    print("=" * 110)
    print("BY TICKER — DESCRIPTIVE ONLY")
    print("=" * 110)

    print(
        ticker.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------
    # 8. SAVE
    # ------------------------------------------------------------

    fc.to_csv(
        OUT_FAILCLOSED,
        index=False,
    )

    daily.to_csv(
        OUT_DAILY,
        index=False,
    )

    worst.to_csv(
        OUT_TAILS,
        index=False,
    )

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)

    print(OUT_FAILCLOSED)
    print(OUT_DAILY)
    print(OUT_TAILS)

    print()
    print(
        "No thresholds or models were changed."
    )


if __name__ == "__main__":
    main()

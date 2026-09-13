from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/rl-trading-tbank")
STATE = ROOT / "state"

HIST_FILE = STATE / "ofi_dataset_h5.pkl"
FORWARD_FILE = STATE / "shadow_observer_features.csv"
FREEZE_CONFIG = STATE / "shadow_strategy_v1.json"
SOURCE_40 = ROOT / "scripts/40_freeze_shadow_strategy.py"

FREEZE_DATE = "2026-08-31"

# FROZEN historical candidate.
BUFFER_BP = 1.0
DIRECTION_THRESHOLD = 0.60
MAGNITUDE_THRESHOLD = 0.65

# Считаем день полным, если live observer дошёл почти
# до конца исследовательского торгового окна.
MIN_COMPLETE_MINUTES = 800
MIN_LAST_HOUR_UTC = 20


def load_freeze_module():
    spec = importlib.util.spec_from_file_location(
        "freeze40",
        SOURCE_40,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import script 40")

    m = importlib.util.module_from_spec(spec)
    sys.modules["freeze40"] = m
    spec.loader.exec_module(m)
    return m


def winsorized_mean(s, q=0.05):
    if len(s) == 0:
        return np.nan

    lo = s.quantile(q)
    hi = s.quantile(1 - q)

    return float(
        s.clip(lo, hi).mean()
    )


def main():
    m = load_freeze_module()

    # ---------------------------------------------------------
    # Hard guards against accidental strategy changes
    # ---------------------------------------------------------
    assert m.HORIZON == 15
    assert m.MAX_ENTRY_SPREAD_BP == 2.0
    assert m.FEE_PER_SIDE_BP == 0.5

    config = json.loads(
        FREEZE_CONFIG.read_text(
            encoding="utf-8"
        )
    )

    train_dates = list(
        config["train_dates"]
    )

    valid_dates = list(
        config.get("valid_dates", [])
    )

    print("=" * 110)
    print("MULTI-DAY TRUE FORWARD EVALUATION")
    print("=" * 110)

    print(
        "Historical train:",
        train_dates[0],
        "->",
        train_dates[-1],
    )

    if valid_dates:
        print(
            "Historical validation:",
            valid_dates[0],
            "->",
            valid_dates[-1],
        )

    print()
    print("FROZEN CANDIDATE")
    print("Horizon              :", m.HORIZON)
    print("Max entry spread     :", m.MAX_ENTRY_SPREAD_BP)
    print("Fee per side         :", m.FEE_PER_SIDE_BP)
    print("Buffer               :", BUFFER_BP)
    print("Direction threshold  :", DIRECTION_THRESHOLD)
    print("Magnitude threshold  :", MAGNITUDE_THRESHOLD)

    # ---------------------------------------------------------
    # Build SAME historical training set
    # ---------------------------------------------------------
    hist = pd.read_pickle(
        HIST_FILE
    ).copy()

    hist["time"] = pd.to_datetime(
        hist["time"],
        utc=True,
        errors="coerce",
    )

    hist = hist.dropna(
        subset=["time", "ticker"]
    )

    hist["date"] = (
        hist["time"]
        .dt.strftime("%Y-%m-%d")
    )

    train = hist[
        hist["date"].isin(train_dates)
    ].copy()

    train = m.add_exact_future(train)
    train = m.add_execution_fields(train)

    train = train[
        train["spread_bp"]
        <= m.MAX_ENTRY_SPREAD_BP
    ].copy()

    train = train.dropna(
        subset=(
            list(m.FEATURES_BASIC)
            + [
                "future_ret_bp",
                "expected_cost_bp",
            ]
        )
    )

    print()
    print("Training rows:", f"{len(train):,}")
    print(
        "Training tickers:",
        train["ticker"].nunique(),
    )

    direction_model, magnitude_model = (
        m.fit_models(
            train,
            BUFFER_BP,
        )
    )

    if (
        direction_model is None
        or magnitude_model is None
    ):
        raise SystemExit("Model fit failed")

    # ---------------------------------------------------------
    # Forward observations
    # ---------------------------------------------------------
    fwd = pd.read_csv(
        FORWARD_FILE
    )

    fwd["time"] = pd.to_datetime(
        fwd["time"],
        utc=True,
        errors="coerce",
    )

    fwd = fwd.dropna(
        subset=["time", "ticker"]
    )

    fwd["date"] = (
        fwd["time"]
        .dt.strftime("%Y-%m-%d")
    )

    fwd = fwd[
        fwd["date"] > FREEZE_DATE
    ].copy()

    if fwd.empty:
        raise SystemExit(
            "No post-freeze forward data"
        )

    dup = int(
        fwd.duplicated(
            ["time", "ticker"]
        ).sum()
    )

    if dup:
        raise SystemExit(
            f"Duplicate time+ticker: {dup}"
        )

    # ---------------------------------------------------------
    # Detect complete days WITHOUT using PnL
    # ---------------------------------------------------------
    coverage = (
        fwd.groupby("date")
        .agg(
            rows=("ticker", "size"),
            tickers=("ticker", "nunique"),
            minutes=("time", "nunique"),
            first_time=("time", "min"),
            last_time=("time", "max"),
        )
        .reset_index()
    )

    coverage["last_hour"] = (
        coverage["last_time"].dt.hour
    )

    coverage["complete"] = (
        (coverage["minutes"] >= MIN_COMPLETE_MINUTES)
        & (
            coverage["last_hour"]
            >= MIN_LAST_HOUR_UTC
        )
    )

    print()
    print("=" * 110)
    print("FORWARD COVERAGE")
    print("=" * 110)

    print(
        coverage[
            [
                "date",
                "rows",
                "tickers",
                "minutes",
                "first_time",
                "last_time",
                "complete",
            ]
        ].to_string(index=False)
    )

    complete_dates = (
        coverage.loc[
            coverage["complete"],
            "date",
        ]
        .sort_values()
        .tolist()
    )

    print()
    print(
        "Complete forward dates:",
        complete_dates,
    )

    if not complete_dates:
        raise SystemExit(
            "No complete forward days yet"
        )

    # Strict leakage guards.
    bad = (
        set(complete_dates)
        & (
            set(train_dates)
            | set(valid_dates)
        )
    )

    if bad:
        raise SystemExit(
            f"LEAKAGE: forward dates overlap history: {bad}"
        )

    # ---------------------------------------------------------
    # Evaluate each day independently
    # ---------------------------------------------------------
    daily_rows = []
    all_trades = []

    for date in complete_dates:
        raw = fwd[
            fwd["date"] == date
        ].copy()

        test = m.add_exact_future(raw)
        test = m.add_execution_fields(test)

        target_rows = len(test)

        test = test[
            test["spread_bp"]
            <= m.MAX_ENTRY_SPREAD_BP
        ].copy()

        eligible_rows = len(test)

        test = test.dropna(
            subset=(
                list(m.FEATURES_BASIC)
                + [
                    "future_mid",
                    "future_spread_bp",
                    "future_ret_bp",
                    "execution_cost_bp",
                ]
            )
        )

        pred = m.predict(
            test,
            direction_model,
            magnitude_model,
        )

        trades = m.non_overlapping(
            pred,
            DIRECTION_THRESHOLD,
            MAGNITUDE_THRESHOLD,
        )

        if trades is None:
            trades = pd.DataFrame()

        if trades.empty:
            daily_rows.append(
                {
                    "date": date,
                    "raw_rows": len(raw),
                    "target_rows": target_rows,
                    "eligible_rows": eligible_rows,
                    "trades": 0,
                    "longs": 0,
                    "shorts": 0,
                    "gross_mean": np.nan,
                    "cost_mean": np.nan,
                    "net_mean": np.nan,
                    "net_sum": 0.0,
                    "net_median": np.nan,
                    "winsor_mean": np.nan,
                    "hit_rate": np.nan,
                    "worst": np.nan,
                    "best": np.nan,
                }
            )
            continue

        trades = trades.copy()
        trades["forward_date"] = date

        all_trades.append(trades)

        daily_rows.append(
            {
                "date": date,
                "raw_rows": len(raw),
                "target_rows": target_rows,
                "eligible_rows": eligible_rows,
                "trades": len(trades),
                "longs": int(
                    (trades["side"] == 1).sum()
                ),
                "shorts": int(
                    (trades["side"] == -1).sum()
                ),
                "gross_mean": float(
                    trades["gross_bp"].mean()
                ),
                "cost_mean": float(
                    trades[
                        "execution_cost_bp"
                    ].mean()
                ),
                "net_mean": float(
                    trades["net_bp"].mean()
                ),
                "net_sum": float(
                    trades["net_bp"].sum()
                ),
                "net_median": float(
                    trades["net_bp"].median()
                ),
                "winsor_mean": winsorized_mean(
                    trades["net_bp"]
                ),
                "hit_rate": float(
                    (
                        trades["net_bp"] > 0
                    ).mean()
                ),
                "worst": float(
                    trades["net_bp"].min()
                ),
                "best": float(
                    trades["net_bp"].max()
                ),
            }
        )

    daily = pd.DataFrame(
        daily_rows
    )

    print()
    print("=" * 110)
    print("DAILY FORWARD RESULTS")
    print("=" * 110)

    show = daily.copy()

    for col in [
        "gross_mean",
        "cost_mean",
        "net_mean",
        "net_sum",
        "net_median",
        "winsor_mean",
        "worst",
        "best",
    ]:
        show[col] = show[col].map(
            lambda x:
            f"{x:+.3f}"
            if pd.notna(x)
            else "-"
        )

    show["hit_rate"] = (
        show["hit_rate"].map(
            lambda x:
            f"{100*x:.1f}%"
            if pd.notna(x)
            else "-"
        )
    )

    print(
        show[
            [
                "date",
                "trades",
                "longs",
                "shorts",
                "net_mean",
                "net_sum",
                "net_median",
                "winsor_mean",
                "hit_rate",
                "worst",
                "best",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Aggregate across ALL untouched forward days
    # ---------------------------------------------------------
    if not all_trades:
        print()
        print("NO FORWARD TRADES")
        return

    trades = pd.concat(
        all_trades,
        ignore_index=True,
    )

    print()
    print("=" * 110)
    print("AGGREGATE TRUE FORWARD RESULT")
    print("=" * 110)

    n = len(trades)

    print(
        "days:",
        len(complete_dates),
    )

    print(
        "trading days with trades:",
        trades["forward_date"].nunique(),
    )

    print(
        "trades:",
        n,
    )

    print(
        "longs:",
        int((trades["side"] == 1).sum()),
    )

    print(
        "shorts:",
        int((trades["side"] == -1).sum()),
    )

    print()

    print(
        "gross mean:",
        f"{trades['gross_bp'].mean():+.3f} bp",
    )

    print(
        "execution cost mean:",
        f"{trades['execution_cost_bp'].mean():.3f} bp",
    )

    print(
        "net mean:",
        f"{trades['net_bp'].mean():+.3f} bp",
    )

    print(
        "net sum:",
        f"{trades['net_bp'].sum():+.3f} bp",
    )

    print(
        "net median:",
        f"{trades['net_bp'].median():+.3f} bp",
    )

    print(
        "winsorized net mean:",
        f"{winsorized_mean(trades['net_bp']):+.3f} bp",
    )

    print(
        "hit rate:",
        f"{100*(trades['net_bp'] > 0).mean():.2f}%",
    )

    print()

    print(
        "worst trade:",
        f"{trades['net_bp'].min():+.3f} bp",
    )

    print(
        "5% quantile:",
        f"{trades['net_bp'].quantile(.05):+.3f} bp",
    )

    print(
        "95% quantile:",
        f"{trades['net_bp'].quantile(.95):+.3f} bp",
    )

    print(
        "best trade:",
        f"{trades['net_bp'].max():+.3f} bp",
    )

    # Daily stability.
    active_daily = daily[
        daily["trades"] > 0
    ].copy()

    print()
    print("=" * 110)
    print("DAY STABILITY")
    print("=" * 110)

    positive_days = int(
        (active_daily["net_sum"] > 0).sum()
    )

    print(
        "positive trading days:",
        f"{positive_days}/{len(active_daily)}",
    )

    print(
        "positive day share:",
        (
            f"{100*positive_days/len(active_daily):.1f}%"
            if len(active_daily)
            else "-"
        ),
    )

    print(
        "median daily net mean:",
        f"{active_daily['net_mean'].median():+.3f} bp",
    )

    print(
        "median daily net sum:",
        f"{active_daily['net_sum'].median():+.3f} bp",
    )

    print(
        "worst day net sum:",
        f"{active_daily['net_sum'].min():+.3f} bp",
    )

    print(
        "best day net sum:",
        f"{active_daily['net_sum'].max():+.3f} bp",
    )

    # ---------------------------------------------------------
    # Side diagnostic — descriptive only
    # ---------------------------------------------------------
    side = (
        trades.groupby("side")
        .agg(
            trades=("net_bp", "size"),
            gross_mean=("gross_bp", "mean"),
            cost_mean=(
                "execution_cost_bp",
                "mean",
            ),
            net_mean=("net_bp", "mean"),
            net_median=("net_bp", "median"),
            net_sum=("net_bp", "sum"),
            hit_rate=(
                "net_bp",
                lambda s:
                (s > 0).mean(),
            ),
        )
        .reset_index()
    )

    print()
    print("=" * 110)
    print("BY SIDE — DESCRIPTIVE ONLY")
    print("=" * 110)

    print(
        side.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ---------------------------------------------------------
    # Robustness to tails
    # ---------------------------------------------------------
    print()
    print("=" * 110)
    print("TAIL ROBUSTNESS")
    print("=" * 110)

    ordered = trades["net_bp"].sort_values(
        ascending=False
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
            f"{winsorized_mean(trades['net_bp'], q):+.3f} bp"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    daily_path = (
        STATE
        / "forward_eval_multiday_daily.csv"
    )

    trades_path = (
        STATE
        / "forward_eval_multiday_trades.csv"
    )

    summary_path = (
        STATE
        / "forward_eval_multiday_summary.json"
    )

    daily.to_csv(
        daily_path,
        index=False,
    )

    trades.to_csv(
        trades_path,
        index=False,
    )

    summary = {
        "freeze_date": FREEZE_DATE,
        "complete_dates": complete_dates,
        "buffer_bp": BUFFER_BP,
        "direction_threshold": (
            DIRECTION_THRESHOLD
        ),
        "magnitude_threshold": (
            MAGNITUDE_THRESHOLD
        ),
        "days": len(complete_dates),
        "trades": n,
        "positive_days": positive_days,
        "net_mean_bp": float(
            trades["net_bp"].mean()
        ),
        "net_median_bp": float(
            trades["net_bp"].median()
        ),
        "net_sum_bp": float(
            trades["net_bp"].sum()
        ),
        "winsorized_net_mean_bp": (
            winsorized_mean(
                trades["net_bp"]
            )
        ),
        "hit_rate": float(
            (trades["net_bp"] > 0).mean()
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)

    print(daily_path)
    print(trades_path)
    print(summary_path)

    print()
    print(
        "No parameters were tuned on forward data."
    )


if __name__ == "__main__":
    main()

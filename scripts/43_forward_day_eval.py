from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/rl-trading-tbank")
STATE = ROOT / "state"

HIST_FILE = STATE / "ofi_dataset_h5.pkl"
FORWARD_FILE = STATE / "shadow_observer_features.csv"
FREEZE_CONFIG = STATE / "shadow_strategy_v1.json"

SOURCE_40 = ROOT / "scripts/40_freeze_shadow_strategy.py"

TEST_DATE = "2026-09-01"

# ------------------------------------------------------------
# FROZEN RESEARCH CANDIDATE.
#
# Эти значения НЕ подбираются на 2026-09-01.
# Это лучший кандидат последнего исторического validation run,
# который при этом НЕ прошёл robustness gate.
# ------------------------------------------------------------

BUFFER_BP = 1.0
DIRECTION_THRESHOLD = 0.60
MAGNITUDE_THRESHOLD = 0.65


def load_freeze_module():
    spec = importlib.util.spec_from_file_location(
        "freeze40",
        SOURCE_40,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot import scripts/40_freeze_shadow_strategy.py"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["freeze40"] = module
    spec.loader.exec_module(module)

    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def print_frame(title: str, df: pd.DataFrame):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if df.empty:
        print("EMPTY")
    else:
        print(
            df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )


def main():
    m = load_freeze_module()

    print("=" * 100)
    print("FORWARD DAY EVALUATION")
    print("=" * 100)

    print("Test date:", TEST_DATE)
    print("Source logic:", SOURCE_40)
    print("40 sha256:", sha256(SOURCE_40))
    print()

    print("Frozen candidate:")
    print("  horizon:", m.HORIZON)
    print(
        "  max entry spread:",
        m.MAX_ENTRY_SPREAD_BP,
        "bp",
    )
    print(
        "  fee per side:",
        m.FEE_PER_SIDE_BP,
        "bp",
    )
    print("  buffer:", BUFFER_BP)
    print(
        "  direction threshold:",
        DIRECTION_THRESHOLD,
    )
    print(
        "  magnitude threshold:",
        MAGNITUDE_THRESHOLD,
    )

    # Hard fail if 40 changed unexpectedly.
    assert m.HORIZON == 15
    assert m.MAX_ENTRY_SPREAD_BP == 2.0
    assert m.FEE_PER_SIDE_BP == 0.5

    # ------------------------------------------------------------
    # Frozen train dates
    # ------------------------------------------------------------

    if not FREEZE_CONFIG.exists():
        raise SystemExit(
            f"Missing {FREEZE_CONFIG}"
        )

    config = json.loads(
        FREEZE_CONFIG.read_text(
            encoding="utf-8"
        )
    )

    train_dates = list(
        config.get(
            "train_dates",
            [],
        )
    )

    valid_dates = list(
        config.get(
            "valid_dates",
            [],
        )
    )

    if not train_dates:
        raise SystemExit(
            "No train_dates in shadow_strategy_v1.json"
        )

    print()
    print(
        "Historical train:",
        train_dates[0],
        "->",
        train_dates[-1],
        f"({len(train_dates)} days)",
    )

    if valid_dates:
        print(
            "Historical validation:",
            valid_dates[0],
            "->",
            valid_dates[-1],
            f"({len(valid_dates)} days)",
        )

    # Strict temporal guard.
    if TEST_DATE in train_dates:
        raise SystemExit(
            "ERROR: test day is in train_dates"
        )

    if TEST_DATE in valid_dates:
        raise SystemExit(
            "ERROR: test day is in valid_dates"
        )

    # ------------------------------------------------------------
    # Historical training data
    # ------------------------------------------------------------

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
        hist["date"].isin(
            train_dates
        )
    ].copy()

    # IMPORTANT:
    # reconstruct H=15 target using exact same function as script 40.
    train = m.add_exact_future(
        train
    )

    train = m.add_execution_fields(
        train
    )

    train = train[
        train["spread_bp"]
        <= m.MAX_ENTRY_SPREAD_BP
    ].copy()

    train["date"] = (
        train["time"]
        .dt.strftime("%Y-%m-%d")
    )

    required_train = (
        list(m.FEATURES_BASIC)
        + [
            "future_ret_bp",
            "expected_cost_bp",
        ]
    )

    train = train.dropna(
        subset=required_train
    )

    print()
    print("Training rows:", f"{len(train):,}")
    print(
        "Training tickers:",
        train["ticker"].nunique(),
    )
    print(
        "Training dates present:",
        train["date"].nunique(),
    )

    # ------------------------------------------------------------
    # Fit EXACT same two models as script 40.
    # Only buffer is fixed from historical candidate.
    # ------------------------------------------------------------

    (
        direction_model,
        magnitude_model,
    ) = m.fit_models(
        train,
        BUFFER_BP,
    )

    if (
        direction_model is None
        or magnitude_model is None
    ):
        raise SystemExit(
            "fit_models() returned None"
        )

    # ------------------------------------------------------------
    # Forward/live observations for 1 Sep
    # ------------------------------------------------------------

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

    test_raw = fwd[
        fwd["date"] == TEST_DATE
    ].copy()

    if test_raw.empty:
        raise SystemExit(
            f"No observer data for {TEST_DATE}"
        )

    dup = int(
        test_raw.duplicated(
            ["time", "ticker"]
        ).sum()
    )

    if dup:
        raise SystemExit(
            f"Forward data has {dup} duplicate time+ticker rows"
        )

    print()
    print("Forward raw rows:", f"{len(test_raw):,}")
    print(
        "Forward raw tickers:",
        test_raw["ticker"].nunique(),
    )
    print(
        "Forward range:",
        test_raw["time"].min(),
        "->",
        test_raw["time"].max(),
    )

    # ------------------------------------------------------------
    # Build the exact 15m future target on the COMPLETE forward day.
    #
    # This target is used ONLY for evaluation/PnL.
    # It is never used for model input or signal selection.
    # ------------------------------------------------------------

    test = m.add_exact_future(
        test_raw
    )

    test = m.add_execution_fields(
        test
    )

    before_spread = len(test)

    test = test[
        test["spread_bp"]
        <= m.MAX_ENTRY_SPREAD_BP
    ].copy()

    print(
        "Exact-target rows before spread gate:",
        f"{before_spread:,}",
    )

    print(
        "Eligible rows spread <= 2:",
        f"{len(test):,}",
    )

    print(
        "Eligible tickers:",
        test["ticker"].nunique(),
    )

    required_test = (
        list(m.FEATURES_BASIC)
        + [
            "future_mid",
            "future_spread_bp",
            "future_ret_bp",
            "execution_cost_bp",
        ]
    )

    test = test.dropna(
        subset=required_test
    )

    # ------------------------------------------------------------
    # Predictions.
    #
    # No fit / threshold selection uses TEST_DATE.
    # ------------------------------------------------------------

    pred = m.predict(
        test,
        direction_model,
        magnitude_model,
    )

    # ------------------------------------------------------------
    # EXACT same signal + non-overlap implementation from script 40.
    # ------------------------------------------------------------

    trades = m.non_overlapping(
        pred,
        DIRECTION_THRESHOLD,
        MAGNITUDE_THRESHOLD,
    )

    print()
    print("=" * 100)
    print("FORWARD RESULT")
    print("=" * 100)

    if trades is None or trades.empty:
        print("NO TRADES")
        metrics = {
            "test_date": TEST_DATE,
            "trades": 0,
        }

        trades_out = pd.DataFrame()

    else:
        trades = trades.copy()

        # Defensive date field for summaries.
        trades["date"] = (
            pd.to_datetime(
                trades["time"],
                utc=True,
            )
            .dt.strftime("%Y-%m-%d")
        )

        n = len(trades)

        longs = int(
            (trades["side"] == 1).sum()
        )

        shorts = int(
            (trades["side"] == -1).sum()
        )

        gross_mean = float(
            trades["gross_bp"].mean()
        )

        gross_sum = float(
            trades["gross_bp"].sum()
        )

        cost_mean = float(
            trades["execution_cost_bp"].mean()
        )

        net_mean = float(
            trades["net_bp"].mean()
        )

        net_sum = float(
            trades["net_bp"].sum()
        )

        net_median = float(
            trades["net_bp"].median()
        )

        hit_rate = float(
            (trades["net_bp"] > 0).mean()
        )

        winsor = float(
            m.winsorized_mean(
                trades["net_bp"]
            )
        )

        worst = float(
            trades["net_bp"].min()
        )

        best = float(
            trades["net_bp"].max()
        )

        q05 = float(
            trades["net_bp"].quantile(
                0.05
            )
        )

        q95 = float(
            trades["net_bp"].quantile(
                0.95
            )
        )

        print("trades:", n)
        print("longs:", longs)
        print("shorts:", shorts)

        print()
        print(
            "gross mean:",
            f"{gross_mean:+.3f} bp",
        )
        print(
            "gross sum:",
            f"{gross_sum:+.3f} bp",
        )
        print(
            "execution cost mean:",
            f"{cost_mean:.3f} bp",
        )

        print()
        print(
            "net mean:",
            f"{net_mean:+.3f} bp",
        )
        print(
            "net sum:",
            f"{net_sum:+.3f} bp",
        )
        print(
            "net median:",
            f"{net_median:+.3f} bp",
        )
        print(
            "winsorized net mean:",
            f"{winsor:+.3f} bp",
        )
        print(
            "net hit rate:",
            f"{100 * hit_rate:.2f}%",
        )

        print()
        print(
            "worst trade:",
            f"{worst:+.3f} bp",
        )
        print(
            "5% quantile:",
            f"{q05:+.3f} bp",
        )
        print(
            "95% quantile:",
            f"{q95:+.3f} bp",
        )
        print(
            "best trade:",
            f"{best:+.3f} bp",
        )

        metrics = {
            "evaluated_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "test_date": TEST_DATE,
            "candidate_status": (
                "historical_best_but_unarmed"
            ),
            "source_40_sha256": (
                sha256(SOURCE_40)
            ),
            "horizon_min": int(
                m.HORIZON
            ),
            "max_entry_spread_bp": float(
                m.MAX_ENTRY_SPREAD_BP
            ),
            "fee_per_side_bp": float(
                m.FEE_PER_SIDE_BP
            ),
            "buffer_bp": BUFFER_BP,
            "direction_threshold": (
                DIRECTION_THRESHOLD
            ),
            "magnitude_threshold": (
                MAGNITUDE_THRESHOLD
            ),
            "train_dates": train_dates,
            "validation_dates": valid_dates,
            "forward_raw_rows": int(
                len(test_raw)
            ),
            "eligible_rows": int(
                len(test)
            ),
            "trades": int(n),
            "longs": longs,
            "shorts": shorts,
            "gross_mean_bp": gross_mean,
            "gross_sum_bp": gross_sum,
            "execution_cost_mean_bp": (
                cost_mean
            ),
            "net_mean_bp": net_mean,
            "net_sum_bp": net_sum,
            "net_median_bp": net_median,
            "winsorized_net_mean_bp": (
                winsor
            ),
            "hit_rate": hit_rate,
            "worst_trade_bp": worst,
            "q05_net_bp": q05,
            "q95_net_bp": q95,
            "best_trade_bp": best,
        }

        trades_out = trades.copy()

        # --------------------------------------------------------
        # Additional descriptive summaries only.
        # DO NOT use these to change the strategy.
        # --------------------------------------------------------

        ticker_summary = (
            trades.groupby("ticker")
            .agg(
                trades=("net_bp", "size"),
                gross_mean=(
                    "gross_bp",
                    "mean",
                ),
                cost_mean=(
                    "execution_cost_bp",
                    "mean",
                ),
                net_mean=(
                    "net_bp",
                    "mean",
                ),
                net_sum=(
                    "net_bp",
                    "sum",
                ),
                hit_rate=(
                    "net_bp",
                    lambda s:
                    (s > 0).mean(),
                ),
            )
            .reset_index()
            .sort_values(
                "net_sum",
                ascending=False,
            )
        )

        print_frame(
            "DESCRIPTIVE BY TICKER — DO NOT SELECT TICKERS FROM THIS",
            ticker_summary,
        )

        side_summary = (
            trades.groupby("side")
            .agg(
                trades=("net_bp", "size"),
                gross_mean=(
                    "gross_bp",
                    "mean",
                ),
                cost_mean=(
                    "execution_cost_bp",
                    "mean",
                ),
                net_mean=(
                    "net_bp",
                    "mean",
                ),
                net_median=(
                    "net_bp",
                    "median",
                ),
                hit_rate=(
                    "net_bp",
                    lambda s:
                    (s > 0).mean(),
                ),
            )
            .reset_index()
        )

        print_frame(
            "BY SIDE",
            side_summary,
        )

    # ------------------------------------------------------------
    # Immutable-ish output names for this test date.
    # ------------------------------------------------------------

    metrics_path = (
        STATE
        / f"forward_eval_{TEST_DATE}.json"
    )

    trades_path = (
        STATE
        / f"forward_eval_{TEST_DATE}_trades.csv"
    )

    # Refuse silent overwrite.
    #
    # Re-running should produce the same model/result,
    # but we don't want accidental future code changes to silently
    # replace our first recorded forward result.
    if metrics_path.exists():
        old = json.loads(
            metrics_path.read_text(
                encoding="utf-8"
            )
        )

        print()
        print("=" * 100)
        print("WARNING")
        print("=" * 100)
        print(
            "Existing result already exists:",
            metrics_path,
        )
        print(
            "It will NOT be overwritten."
        )

        rerun_path = (
            STATE
            / (
                f"forward_eval_{TEST_DATE}"
                "_rerun.json"
            )
        )

        rerun_path.write_text(
            json.dumps(
                metrics,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            "Current rerun saved to:",
            rerun_path,
        )

    else:
        metrics_path.write_text(
            json.dumps(
                metrics,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "Saved metrics:",
            metrics_path,
        )

    if (
        isinstance(
            trades_out,
            pd.DataFrame,
        )
        and not trades_out.empty
        and not trades_path.exists()
    ):
        trades_out.to_csv(
            trades_path,
            index=False,
        )

        print(
            "Saved trades:",
            trades_path,
        )

    print()
    print("=" * 100)
    print("IMPORTANT")
    print("=" * 100)
    print(
        "This result is evaluation only."
    )
    print(
        "It does NOT arm the strategy."
    )
    print(
        "Do not tune parameters from this one forward day."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path("/root/rl-trading-tbank")
STATE = ROOT / "state"

HIST_FILE = STATE / "ofi_dataset_h5.pkl"
FORWARD_FILE = STATE / "shadow_observer_features.csv"
FREEZE_CONFIG = STATE / "shadow_strategy_v1.json"
SOURCE_40 = ROOT / "scripts/40_freeze_shadow_strategy.py"

FREEZE_DATE = "2026-08-31"

BUFFER_BP = 1.0
DIRECTION_THRESHOLD = 0.60
MAGNITUDE_THRESHOLD = 0.65

COMPLETE_DATES = [
    "2026-09-01",
    "2026-09-02",
    "2026-09-03",
    "2026-09-04",
]


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


def safe_auc(y, p):
    y = pd.Series(y)
    p = pd.Series(p)

    mask = (
        y.notna()
        & p.notna()
    )

    y = y[mask]
    p = p[mask]

    if len(y) == 0:
        return np.nan

    if y.nunique() < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            p,
        )
    )


def add_deciles(df, col):
    out = df.copy()

    try:
        out["decile"] = pd.qcut(
            out[col],
            10,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        out["decile"] = np.nan

    return out


def main():
    m = load_freeze_module()

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
        config.get(
            "valid_dates",
            [],
        )
    )

    print("=" * 110)
    print("FORWARD FAILURE DIAGNOSTICS")
    print("=" * 110)

    print("Train:", train_dates[0], "->", train_dates[-1])
    print(
        "Validation:",
        valid_dates[0] if valid_dates else "-",
        "->",
        valid_dates[-1] if valid_dates else "-",
    )
    print("Forward:", COMPLETE_DATES)

    # ------------------------------------------------------------------
    # Historical train
    # ------------------------------------------------------------------

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

    direction_model, magnitude_model = (
        m.fit_models(
            train,
            BUFFER_BP,
        )
    )

    # ------------------------------------------------------------------
    # Historical validation
    # ------------------------------------------------------------------

    valid = hist[
        hist["date"].isin(
            valid_dates
        )
    ].copy()

    valid = m.add_exact_future(valid)
    valid = m.add_execution_fields(valid)

    valid = valid[
        valid["spread_bp"]
        <= m.MAX_ENTRY_SPREAD_BP
    ].copy()

    valid = valid.dropna(
        subset=(
            list(m.FEATURES_BASIC)
            + [
                "future_ret_bp",
                "expected_cost_bp",
            ]
        )
    )

    valid_pred = m.predict(
        valid,
        direction_model,
        magnitude_model,
    )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        fwd["date"].isin(
            COMPLETE_DATES
        )
    ].copy()

    fwd = m.add_exact_future(fwd)
    fwd = m.add_execution_fields(fwd)

    fwd = fwd[
        fwd["spread_bp"]
        <= m.MAX_ENTRY_SPREAD_BP
    ].copy()

    fwd = fwd.dropna(
        subset=(
            list(m.FEATURES_BASIC)
            + [
                "future_mid",
                "future_spread_bp",
                "future_ret_bp",
                "expected_cost_bp",
                "execution_cost_bp",
            ]
        )
    )

    fwd_pred = m.predict(
        fwd,
        direction_model,
        magnitude_model,
    )

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    for frame in [train, valid_pred, fwd_pred]:
        frame["y_up"] = (
            frame["future_ret_bp"] > 0
        ).astype(int)

        frame["required_move"] = (
            frame["expected_cost_bp"]
            + BUFFER_BP
        )

        frame["y_large"] = (
            frame["future_ret_bp"].abs()
            >= frame["required_move"]
        ).astype(int)

    train_pred = m.predict(
        train,
        direction_model,
        magnitude_model,
    )

    train_pred["y_up"] = (
        train_pred["future_ret_bp"] > 0
    ).astype(int)

    train_pred["required_move"] = (
        train_pred["expected_cost_bp"]
        + BUFFER_BP
    )

    train_pred["y_large"] = (
        train_pred["future_ret_bp"].abs()
        >= train_pred["required_move"]
    ).astype(int)

    # ------------------------------------------------------------------
    # 1. Base rates / model distributions
    # ------------------------------------------------------------------

    rows = []

    for name, frame in [
        ("train", train_pred),
        ("validation", valid_pred),
        ("forward", fwd_pred),
    ]:
        rows.append(
            {
                "sample": name,
                "rows": len(frame),
                "up_rate": frame["y_up"].mean(),
                "large_rate": frame["y_large"].mean(),
                "p_up_mean": frame["p_up"].mean(),
                "p_up_median": frame["p_up"].median(),
                "p_up_lt_040": (
                    frame["p_up"] <= 0.40
                ).mean(),
                "p_up_gt_060": (
                    frame["p_up"] >= 0.60
                ).mean(),
                "p_large_mean": frame["p_large"].mean(),
                "p_large_ge_065": (
                    frame["p_large"] >= 0.65
                ).mean(),
                "direction_auc": safe_auc(
                    frame["y_up"],
                    frame["p_up"],
                ),
                "magnitude_auc": safe_auc(
                    frame["y_large"],
                    frame["p_large"],
                ),
            }
        )

    overview = pd.DataFrame(rows)

    print()
    print("=" * 110)
    print("MODEL / BASE RATE OVERVIEW")
    print("=" * 110)

    print(
        overview.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # 2. Forward day-by-day prediction behaviour
    # ------------------------------------------------------------------

    daily = (
        fwd_pred.groupby("date")
        .agg(
            rows=("ticker", "size"),
            up_rate=("y_up", "mean"),
            p_up_mean=("p_up", "mean"),
            p_up_median=("p_up", "median"),
            p_up_lt_040=(
                "p_up",
                lambda s:
                (s <= 0.40).mean(),
            ),
            p_up_gt_060=(
                "p_up",
                lambda s:
                (s >= 0.60).mean(),
            ),
            p_large_mean=("p_large", "mean"),
            p_large_ge_065=(
                "p_large",
                lambda s:
                (s >= 0.65).mean(),
            ),
        )
        .reset_index()
    )

    auc_rows = []

    for date, g in fwd_pred.groupby("date"):
        auc_rows.append(
            {
                "date": date,
                "direction_auc": safe_auc(
                    g["y_up"],
                    g["p_up"],
                ),
                "magnitude_auc": safe_auc(
                    g["y_large"],
                    g["p_large"],
                ),
            }
        )

    auc_daily = pd.DataFrame(
        auc_rows
    )

    daily = daily.merge(
        auc_daily,
        on="date",
        how="left",
    )

    print()
    print("=" * 110)
    print("FORWARD MODEL BEHAVIOUR BY DAY")
    print("=" * 110)

    print(
        daily.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # 3. Calibration-ish bins for p_up
    # ------------------------------------------------------------------

    for name, frame in [
        ("VALIDATION", valid_pred),
        ("FORWARD", fwd_pred),
    ]:
        z = frame.copy()

        bins = [
            0.0,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            1.0,
        ]

        labels = [
            "<.35",
            ".35-.40",
            ".40-.45",
            ".45-.50",
            ".50-.55",
            ".55-.60",
            ".60-.65",
            ">=.65",
        ]

        z["p_up_bin"] = pd.cut(
            z["p_up"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )

        cal = (
            z.groupby(
                "p_up_bin",
                observed=False,
            )
            .agg(
                rows=("ticker", "size"),
                p_up_mean=("p_up", "mean"),
                actual_up_rate=("y_up", "mean"),
                future_ret_mean=(
                    "future_ret_bp",
                    "mean",
                ),
            )
            .reset_index()
        )

        print()
        print("=" * 110)
        print(f"P_UP CALIBRATION — {name}")
        print("=" * 110)

        print(
            cal.to_string(
                index=False,
                float_format=lambda x:
                f"{x:.4f}",
            )
        )

    # ------------------------------------------------------------------
    # 4. Deciles: is p_up still ranking direction?
    # ------------------------------------------------------------------

    for name, frame in [
        ("VALIDATION", valid_pred),
        ("FORWARD", fwd_pred),
    ]:
        z = add_deciles(
            frame,
            "p_up",
        )

        dec = (
            z.groupby(
                "decile",
                dropna=False,
            )
            .agg(
                rows=("ticker", "size"),
                p_up_mean=("p_up", "mean"),
                actual_up_rate=("y_up", "mean"),
                future_ret_mean=(
                    "future_ret_bp",
                    "mean",
                ),
                future_ret_median=(
                    "future_ret_bp",
                    "median",
                ),
            )
            .reset_index()
        )

        print()
        print("=" * 110)
        print(f"P_UP DECILES — {name}")
        print("=" * 110)

        print(
            dec.to_string(
                index=False,
                float_format=lambda x:
                f"{x:.4f}",
            )
        )

    # ------------------------------------------------------------------
    # 5. Signal candidates BEFORE non-overlap
    # ------------------------------------------------------------------

    x = fwd_pred.copy()

    x["short_candidate"] = (
        (x["p_up"] <= 1 - DIRECTION_THRESHOLD)
        & (
            x["p_large"]
            >= MAGNITUDE_THRESHOLD
        )
    )

    x["long_candidate"] = (
        (x["p_up"] >= DIRECTION_THRESHOLD)
        & (
            x["p_large"]
            >= MAGNITUDE_THRESHOLD
        )
    )

    signal_daily = (
        x.groupby("date")
        .agg(
            rows=("ticker", "size"),
            short_candidates=(
                "short_candidate",
                "sum",
            ),
            long_candidates=(
                "long_candidate",
                "sum",
            ),
            up_rate=("y_up", "mean"),
        )
        .reset_index()
    )

    print()
    print("=" * 110)
    print("RAW SIGNAL CANDIDATES BEFORE NON-OVERLAP")
    print("=" * 110)

    print(
        signal_daily.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # 6. Actual trades with same frozen non-overlap
    # ------------------------------------------------------------------

    trades = m.non_overlapping(
        fwd_pred,
        DIRECTION_THRESHOLD,
        MAGNITUDE_THRESHOLD,
    )

    if trades is None:
        trades = pd.DataFrame()

    if not trades.empty:
        trades = trades.copy()

        trades["date"] = (
            pd.to_datetime(
                trades["time"],
                utc=True,
            )
            .dt.strftime("%Y-%m-%d")
        )

        trade_day = (
            trades.groupby(
                ["date", "side"]
            )
            .agg(
                trades=("net_bp", "size"),
                gross_mean=(
                    "gross_bp",
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

        print()
        print("=" * 110)
        print("TRADE PERFORMANCE BY DAY / SIDE")
        print("=" * 110)

        print(
            trade_day.to_string(
                index=False,
                float_format=lambda x:
                f"{x:.4f}",
            )
        )

    # ------------------------------------------------------------------
    # 7. Time-of-day diagnostics
    # ------------------------------------------------------------------

    fwd_pred["hour_utc"] = (
        fwd_pred["time"].dt.hour
    )

    tod = (
        fwd_pred.groupby("hour_utc")
        .agg(
            rows=("ticker", "size"),
            up_rate=("y_up", "mean"),
            p_up_mean=("p_up", "mean"),
            direction_auc=(
                "p_up",
                lambda s: np.nan,
            ),
            future_ret_mean=(
                "future_ret_bp",
                "mean",
            ),
        )
        .reset_index()
    )

    aucs = []

    for hour, g in fwd_pred.groupby(
        "hour_utc"
    ):
        aucs.append(
            {
                "hour_utc": hour,
                "direction_auc_real": safe_auc(
                    g["y_up"],
                    g["p_up"],
                ),
            }
        )

    tod = tod.drop(
        columns=["direction_auc"]
    ).merge(
        pd.DataFrame(aucs),
        on="hour_utc",
        how="left",
    )

    print()
    print("=" * 110)
    print("FORWARD BY HOUR UTC")
    print("=" * 110)

    print(
        tod.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # 8. Feature means conditional on model direction
    # ------------------------------------------------------------------

    x = fwd_pred.copy()

    x["model_region"] = np.select(
        [
            x["p_up"] <= 0.40,
            x["p_up"] >= 0.60,
        ],
        [
            "SHORT_REGION",
            "LONG_REGION",
        ],
        default="NEUTRAL",
    )

    feat = (
        x.groupby("model_region")
        .agg(
            rows=("ticker", "size"),
            ofi_mean=("ofi", "mean"),
            tfi_mean=("tfi", "mean"),
            imb1_mean=("imb1", "mean"),
            imb5_mean=("imb5", "mean"),
            micro_dev_mean=(
                "micro_dev",
                "mean",
            ),
            spread_mean=(
                "spread_bp",
                "mean",
            ),
            actual_up_rate=(
                "y_up",
                "mean",
            ),
            future_ret_mean=(
                "future_ret_bp",
                "mean",
            ),
        )
        .reset_index()
    )

    print()
    print("=" * 110)
    print("FEATURE PROFILE BY MODEL REGION")
    print("=" * 110)

    print(
        feat.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    overview.to_csv(
        STATE
        / "failure_diag_overview.csv",
        index=False,
    )

    daily.to_csv(
        STATE
        / "failure_diag_daily.csv",
        index=False,
    )

    if not trades.empty:
        trades.to_csv(
            STATE
            / "failure_diag_trades.csv",
            index=False,
        )

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)
    print(
        STATE
        / "failure_diag_overview.csv"
    )
    print(
        STATE
        / "failure_diag_daily.csv"
    )
    if not trades.empty:
        print(
            STATE
            / "failure_diag_trades.csv"
        )

    print()
    print(
        "This script diagnoses the frozen v1 candidate only."
    )
    print(
        "It does not change thresholds, features or model parameters."
    )


if __name__ == "__main__":
    main()

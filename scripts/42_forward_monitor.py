from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/rl-trading-tbank")

FORWARD_FILE = (
    ROOT / "state/shadow_observer_features.csv"
)

HIST_FILE = (
    ROOT / "state/ofi_dataset_h5.pkl"
)

FREEZE_DATE = "2026-08-31"

FEATURES = [
    "ofi",
    "tfi",
    "imb1",
    "imb5",
    "micro_dev",
    "spread_bp",
    "ret_1m",
]

MAX_SPREAD_BP = 2.0


def safe_iqr(s: pd.Series) -> float:
    q25 = s.quantile(0.25)
    q75 = s.quantile(0.75)
    return float(q75 - q25)


def drift_table(
    hist: pd.DataFrame,
    fwd: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for feature in FEATURES:
        h = (
            pd.to_numeric(
                hist[feature],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        x = (
            pd.to_numeric(
                fwd[feature],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if h.empty or x.empty:
            rows.append(
                {
                    "feature": feature,
                    "hist_median": np.nan,
                    "fwd_median": np.nan,
                    "median_shift": np.nan,
                    "hist_iqr": np.nan,
                    "fwd_iqr": np.nan,
                    "shift_in_hist_iqr": np.nan,
                }
            )
            continue

        hist_med = float(
            h.median()
        )

        fwd_med = float(
            x.median()
        )

        hist_iqr = safe_iqr(h)
        fwd_iqr = safe_iqr(x)

        if (
            np.isfinite(hist_iqr)
            and hist_iqr > 0
        ):
            normalized_shift = (
                fwd_med - hist_med
            ) / hist_iqr
        else:
            normalized_shift = np.nan

        rows.append(
            {
                "feature": feature,
                "hist_median": hist_med,
                "fwd_median": fwd_med,
                "median_shift": (
                    fwd_med - hist_med
                ),
                "hist_iqr": hist_iqr,
                "fwd_iqr": fwd_iqr,
                "shift_in_hist_iqr": (
                    normalized_shift
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 96)
    print("FORWARD DATA MONITOR")
    print("=" * 96)

    if not FORWARD_FILE.exists():
        raise SystemExit(
            f"Нет файла: {FORWARD_FILE}"
        )

    if not HIST_FILE.exists():
        raise SystemExit(
            f"Нет файла: {HIST_FILE}"
        )

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
    ).copy()

    fwd["date"] = (
        fwd["time"]
        .dt.strftime("%Y-%m-%d")
    )

    fwd = fwd[
        fwd["date"] > FREEZE_DATE
    ].copy()

    if fwd.empty:
        print(
            "Нет post-freeze данных."
        )
        return

    hist = pd.read_pickle(
        HIST_FILE
    )

    hist["time"] = pd.to_datetime(
        hist["time"],
        utc=True,
        errors="coerce",
    )

    hist = hist.dropna(
        subset=["time", "ticker"]
    ).copy()

    hist = hist[
        hist["time"]
        < pd.Timestamp(
            "2026-09-01",
            tz="UTC",
        )
    ].copy()

    print(
        "Forward dates:",
        fwd["date"].min(),
        "->",
        fwd["date"].max(),
    )

    print(
        "Forward rows:",
        f"{len(fwd):,}",
    )

    print(
        "Forward tickers:",
        fwd["ticker"].nunique(),
    )

    print(
        "Forward minutes:",
        fwd["time"].nunique(),
    )

    print()

    # --------------------------------------------------
    # DATA HEALTH
    # --------------------------------------------------

    print("=" * 96)
    print("DATA HEALTH")
    print("=" * 96)

    duplicates = int(
        fwd.duplicated(
            ["time", "ticker"]
        ).sum()
    )

    print(
        "duplicates time+ticker:",
        duplicates,
    )

    missing = {}

    for col in FEATURES:
        missing[col] = int(
            fwd[col].isna().sum()
        )

    print(
        "missing feature values:"
    )

    for col, n in missing.items():
        print(
            f"  {col:12s}: {n}"
        )

    print()

    # --------------------------------------------------
    # COVERAGE BY DAY
    # --------------------------------------------------

    print("=" * 96)
    print("COVERAGE BY DAY")
    print("=" * 96)

    daily = (
        fwd.groupby("date")
        .agg(
            rows=("ticker", "size"),
            tickers=(
                "ticker",
                "nunique",
            ),
            minutes=(
                "time",
                "nunique",
            ),
            first_time=(
                "time",
                "min",
            ),
            last_time=(
                "time",
                "max",
            ),
        )
    )

    print(
        daily.to_string()
    )

    print()

    # --------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------

    print("=" * 96)
    print("LIQUIDITY")
    print("=" * 96)

    eligible = fwd[
        fwd["spread_bp"]
        <= MAX_SPREAD_BP
    ].copy()

    eligible_pct = (
        len(eligible)
        / len(fwd)
        * 100
    )

    print(
        f"spread <= {MAX_SPREAD_BP:g} bp:"
    )

    print(
        "  rows:",
        f"{len(eligible):,}",
    )

    print(
        "  share:",
        f"{eligible_pct:.2f}%",
    )

    print(
        "  tickers:",
        eligible[
            "ticker"
        ].nunique(),
    )

    print()

    liq_daily = (
        fwd.assign(
            eligible=(
                fwd["spread_bp"]
                <= MAX_SPREAD_BP
            )
        )
        .groupby("date")
        .agg(
            rows=("ticker", "size"),
            eligible_rows=(
                "eligible",
                "sum",
            ),
            eligible_share=(
                "eligible",
                "mean",
            ),
        )
    )

    liq_daily[
        "eligible_share"
    ] *= 100

    print(
        liq_daily.to_string(
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    print()

    # --------------------------------------------------
    # PER-TICKER COVERAGE
    # --------------------------------------------------

    print("=" * 96)
    print("TICKER COVERAGE")
    print("=" * 96)

    by_ticker = (
        fwd.groupby("ticker")
        .agg(
            rows=("ticker", "size"),
            days=(
                "date",
                "nunique",
            ),
            first_time=(
                "time",
                "min",
            ),
            last_time=(
                "time",
                "max",
            ),
            spread_median=(
                "spread_bp",
                "median",
            ),
            liquid_share=(
                "spread_bp",
                lambda s:
                (s <= MAX_SPREAD_BP)
                .mean()
                * 100,
            ),
        )
        .sort_values(
            "rows",
            ascending=False,
        )
    )

    print(
        by_ticker.to_string(
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    print()

    # --------------------------------------------------
    # FEATURE DRIFT
    # --------------------------------------------------

    print("=" * 96)
    print("FEATURE DRIFT")
    print("=" * 96)

    drift = drift_table(
        hist,
        fwd,
    )

    print(
        drift.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}",
        )
    )

    print()

    print(
        "Interpretation:"
    )
    print(
        "  |shift_in_hist_iqr| < 0.25  -> small"
    )
    print(
        "  0.25 .. 0.50              -> noticeable"
    )
    print(
        "  > 0.50                    -> investigate"
    )

    print()

    # --------------------------------------------------
    # FRESHNESS
    # --------------------------------------------------

    print("=" * 96)
    print("LATEST DATA")
    print("=" * 96)

    latest = (
        fwd.groupby("ticker")["time"]
        .max()
        .sort_values()
    )

    print(
        latest.to_string()
    )

    # --------------------------------------------------
    # SAVE SNAPSHOT
    # --------------------------------------------------

    out_dir = ROOT / "state"

    daily.to_csv(
        out_dir
        / "forward_monitor_daily.csv"
    )

    by_ticker.to_csv(
        out_dir
        / "forward_monitor_tickers.csv"
    )

    drift.to_csv(
        out_dir
        / "forward_monitor_drift.csv",
        index=False,
    )

    print()
    print("=" * 96)
    print("SAVED")
    print("=" * 96)

    print(
        "state/forward_monitor_daily.csv"
    )
    print(
        "state/forward_monitor_tickers.csv"
    )
    print(
        "state/forward_monitor_drift.csv"
    )


if __name__ == "__main__":
    main()

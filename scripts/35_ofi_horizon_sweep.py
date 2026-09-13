from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lob.dataset import FEATURES_BASIC


CACHE = PROJECT_ROOT / "state" / "ofi_dataset_h5.pkl"

HORIZONS = (3, 5, 10, 15)
TRAIN_DAYS = 20
VALID_DAYS = 5
QUANTILES = (0.80, 0.90, 0.95, 0.97, 0.99)

MAX_SPREAD_BP = 2.0
FEE_PER_SIDE_BP = 0.5
MIN_VALID_TRADES = 50


def make_exact_target(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    parts = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("time").copy()

        mid = (
            g.set_index("time")["mid"]
            .astype(float)
        )

        future_mid = mid.reindex(
            g["time"]
            + pd.Timedelta(minutes=horizon)
        ).to_numpy()

        g["fwd_ret_h"] = (
            future_mid
            / g["mid"].to_numpy()
            - 1.0
        )

        parts.append(g)

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    return (
        out.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=["fwd_ret_h"]
        )
        .sort_values(
            ["time", "ticker"]
        )
        .reset_index(drop=True)
    )


def add_pnl(
    df: pd.DataFrame,
    side: int,
) -> pd.DataFrame:
    out = df.copy()

    out["gross_bp"] = (
        side
        * out["fwd_ret_h"]
        * 10_000
    )

    out["cost_bp"] = (
        out["spread_bp"]
        + 2 * FEE_PER_SIDE_BP
    )

    out["net_bp"] = (
        out["gross_bp"]
        - out["cost_bp"]
    )

    return out


def non_overlapping(
    frame: pd.DataFrame,
    side: int,
    threshold: float,
    horizon: int,
) -> pd.DataFrame:
    if side == 1:
        selected = frame[
            frame["p"] >= threshold
        ].copy()
    else:
        selected = frame[
            frame["p"] <= threshold
        ].copy()

    selected = selected.sort_values(
        ["time", "ticker"]
    )

    accepted = []
    busy_until = {}

    for idx, row in selected.iterrows():
        ticker = str(row["ticker"])
        now = pd.Timestamp(row["time"])

        if (
            ticker in busy_until
            and now < busy_until[ticker]
        ):
            continue

        busy_until[ticker] = (
            now
            + pd.Timedelta(minutes=horizon)
        )

        accepted.append(idx)

    trades = selected.loc[
        accepted
    ].copy()

    if trades.empty:
        return trades

    return add_pnl(
        trades,
        side,
    )


def select_rule(
    valid: pd.DataFrame,
    horizon: int,
) -> dict | None:
    candidates = []

    for side, side_name in (
        (1, "LONG"),
        (-1, "SHORT"),
    ):
        for q in QUANTILES:
            if side == 1:
                threshold = float(
                    valid["p"].quantile(q)
                )
            else:
                threshold = float(
                    valid["p"].quantile(
                        1 - q
                    )
                )

            trades = non_overlapping(
                valid,
                side,
                threshold,
                horizon,
            )

            if len(trades) < MIN_VALID_TRADES:
                continue

            daily = (
                trades.groupby("date")["net_bp"]
                .mean()
            )

            candidates.append(
                {
                    "side": side,
                    "side_name": side_name,
                    "q": q,
                    "threshold": threshold,
                    "trades": len(trades),
                    "gross_mean": float(
                        trades["gross_bp"].mean()
                    ),
                    "net_mean": float(
                        trades["net_bp"].mean()
                    ),
                    "positive_days": float(
                        (daily > 0).mean()
                    ),
                }
            )

    if not candidates:
        return None

    table = (
        pd.DataFrame(candidates)
        .sort_values(
            [
                "net_mean",
                "positive_days",
                "trades",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    best = table.iloc[0].to_dict()

    if best["net_mean"] <= 0:
        return None

    return best


def run_horizon(
    base: pd.DataFrame,
    horizon: int,
) -> dict:
    df = make_exact_target(
        base,
        horizon,
    )

    df = df[
        df["spread_bp"]
        <= MAX_SPREAD_BP
    ].copy()

    df["date"] = (
        df["time"].dt.date
    )

    dates = sorted(
        df["date"].unique()
    )

    summaries = []
    all_trades = []

    for test_pos in range(
        TRAIN_DAYS + VALID_DAYS,
        len(dates),
    ):
        train_dates = dates[
            test_pos
            - VALID_DAYS
            - TRAIN_DAYS:
            test_pos
            - VALID_DAYS
        ]

        valid_dates = dates[
            test_pos
            - VALID_DAYS:
            test_pos
        ]

        test_date = dates[
            test_pos
        ]

        train = df[
            df["date"].isin(
                train_dates
            )
        ].copy()

        valid = df[
            df["date"].isin(
                valid_dates
            )
        ].copy()

        test = df[
            df["date"] == test_date
        ].copy()

        if (
            train.empty
            or valid.empty
            or test.empty
        ):
            continue

        valid_start = (
            valid["time"].min()
        )

        train = train[
            train["time"]
            < (
                valid_start
                - pd.Timedelta(
                    minutes=horizon
                )
            )
        ].copy()

        test_start = (
            test["time"].min()
        )

        valid = valid[
            valid["time"]
            < (
                test_start
                - pd.Timedelta(
                    minutes=horizon
                )
            )
        ].copy()

        y_train = (
            train["fwd_ret_h"] > 0
        ).astype(int)

        if y_train.nunique() < 2:
            continue

        model = HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=200,
            learning_rate=0.05,
            random_state=42,
        )

        model.fit(
            train[FEATURES_BASIC],
            y_train,
        )

        valid["p"] = (
            model.predict_proba(
                valid[FEATURES_BASIC]
            )[:, 1]
        )

        test["p"] = (
            model.predict_proba(
                test[FEATURES_BASIC]
            )[:, 1]
        )

        valid_y = (
            valid["fwd_ret_h"] > 0
        ).astype(int)

        test_y = (
            test["fwd_ret_h"] > 0
        ).astype(int)

        if (
            valid_y.nunique() < 2
            or test_y.nunique() < 2
        ):
            continue

        valid_auc = roc_auc_score(
            valid_y,
            valid["p"],
        )

        test_auc = roc_auc_score(
            test_y,
            test["p"],
        )

        rule = select_rule(
            valid,
            horizon,
        )

        if rule is None:
            summaries.append(
                {
                    "date": str(test_date),
                    "valid_auc": valid_auc,
                    "test_auc": test_auc,
                    "trades": 0,
                }
            )
            continue

        trades = non_overlapping(
            test,
            int(rule["side"]),
            float(rule["threshold"]),
            horizon,
        )

        if not trades.empty:
            trades["test_date"] = str(
                test_date
            )
            all_trades.append(
                trades
            )

        summaries.append(
            {
                "date": str(test_date),
                "valid_auc": valid_auc,
                "test_auc": test_auc,
                "trades": len(trades),
            }
        )

    summary = pd.DataFrame(
        summaries
    )

    if all_trades:
        trades = pd.concat(
            all_trades,
            ignore_index=True,
        )

        return {
            "horizon": horizon,
            "test_days": len(summary),
            "trade_days": int(
                (summary["trades"] > 0).sum()
            ),
            "trades": len(trades),
            "gross_mean": float(
                trades["gross_bp"].mean()
            ),
            "net_mean": float(
                trades["net_bp"].mean()
            ),
            "hit_rate": float(
                (trades["net_bp"] > 0).mean()
            ),
            "test_auc": float(
                summary["test_auc"].mean()
            ),
        }

    return {
        "horizon": horizon,
        "test_days": len(summary),
        "trade_days": 0,
        "trades": 0,
        "gross_mean": np.nan,
        "net_mean": np.nan,
        "hit_rate": np.nan,
        "test_auc": float(
            summary["test_auc"].mean()
        ),
    }


def main() -> None:
    print("=" * 88)
    print("OFI HORIZON SWEEP")
    print("=" * 88)

    base = pd.read_pickle(
        CACHE
    )

    base["time"] = pd.to_datetime(
        base["time"],
        utc=True,
    )

    print(
        f"Кэш: {len(base):,} баров, "
        f"{base['ticker'].nunique()} тикеров"
    )

    rows = []

    for horizon in HORIZONS:
        print()
        print(
            f"--- HORIZON {horizon} MIN ---"
        )

        result = run_horizon(
            base,
            horizon,
        )

        rows.append(result)

        print(result)

    out = pd.DataFrame(rows)

    print()
    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)

    print(
        out.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    output = (
        PROJECT_ROOT
        / "state"
        / "ofi_horizon_sweep.csv"
    )

    out.to_csv(
        output,
        index=False,
    )

    print()
    print("Сохранено:", output)


if __name__ == "__main__":
    main()

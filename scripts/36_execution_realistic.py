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

HORIZON = 15
TRAIN_DAYS = 20
VALID_DAYS = 5
QUANTILES = (0.80, 0.90, 0.95, 0.97, 0.99)

MAX_SPREAD_BP = 2.0

# Пока берём ту же предпосылку, что и раньше:
# 0.5 bp на сторону.
FEE_PER_SIDE_BP = 0.5

MIN_VALID_TRADES = 50


def add_exact_future(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    parts = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("time").copy()

        indexed = g.set_index("time")

        future_index = (
            g["time"]
            + pd.Timedelta(minutes=horizon)
        )

        future_mid = (
            indexed["mid"]
            .reindex(future_index)
            .to_numpy()
        )

        future_spread = (
            indexed["spread_bp"]
            .reindex(future_index)
            .to_numpy()
        )

        g["future_mid"] = future_mid
        g["future_spread_bp"] = future_spread

        g["fwd_ret_h"] = (
            g["future_mid"]
            / g["mid"]
            - 1.0
        )

        parts.append(g)

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    out = (
        out.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "mid",
                "future_mid",
                "spread_bp",
                "future_spread_bp",
                "fwd_ret_h",
            ]
        )
        .sort_values(
            ["time", "ticker"]
        )
        .reset_index(drop=True)
    )

    return out


def add_realistic_taker_pnl(
    df: pd.DataFrame,
    side: int,
) -> pd.DataFrame:
    out = df.copy()

    out["gross_bp"] = (
        side
        * out["fwd_ret_h"]
        * 10_000
    )

    # Taker execution:
    # вход: половина текущего spread
    # выход: половина future spread
    # плюс комиссия на входе и выходе.
    out["entry_half_spread_bp"] = (
        out["spread_bp"] / 2.0
    )

    out["exit_half_spread_bp"] = (
        out["future_spread_bp"] / 2.0
    )

    out["fees_bp"] = (
        2 * FEE_PER_SIDE_BP
    )

    out["execution_cost_bp"] = (
        out["entry_half_spread_bp"]
        + out["exit_half_spread_bp"]
        + out["fees_bp"]
    )

    out["net_bp"] = (
        out["gross_bp"]
        - out["execution_cost_bp"]
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
    busy_until: dict[str, pd.Timestamp] = {}

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

    return add_realistic_taker_pnl(
        trades,
        side,
    )


def select_rule(
    valid: pd.DataFrame,
    horizon: int,
) -> tuple[dict | None, pd.DataFrame]:
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
                    "entry_spread_mean": float(
                        trades[
                            "entry_half_spread_bp"
                        ].mean()
                    ),
                    "exit_spread_mean": float(
                        trades[
                            "exit_half_spread_bp"
                        ].mean()
                    ),
                    "fees_mean": float(
                        trades["fees_bp"].mean()
                    ),
                    "cost_mean": float(
                        trades[
                            "execution_cost_bp"
                        ].mean()
                    ),
                    "net_mean": float(
                        trades["net_bp"].mean()
                    ),
                    "hit_rate": float(
                        (
                            trades["net_bp"] > 0
                        ).mean()
                    ),
                    "positive_days": float(
                        (daily > 0).mean()
                    ),
                }
            )

    if not candidates:
        return None, pd.DataFrame()

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
        return None, table

    return best, table


def main() -> None:
    print("=" * 92)
    print("OFI REALISTIC EXECUTION")
    print("=" * 92)
    print("Horizon:", HORIZON, "min")
    print("Max entry spread:", MAX_SPREAD_BP, "bp")

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

    df = add_exact_future(
        base,
        HORIZON,
    )

    before = len(df)

    df = df[
        df["spread_bp"]
        <= MAX_SPREAD_BP
    ].copy()

    df["date"] = (
        df["time"].dt.date
    )

    print(
        f"После exact future: {before:,}"
    )

    print(
        f"После spread <= {MAX_SPREAD_BP:g}: "
        f"{len(df):,}"
    )

    dates = sorted(
        df["date"].unique()
    )

    summaries = []
    all_test_trades = []
    all_candidates = []

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

        test_date = dates[test_pos]

        train = df[
            df["date"].isin(train_dates)
        ].copy()

        valid = df[
            df["date"].isin(valid_dates)
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

        valid_start = valid["time"].min()

        train = train[
            train["time"]
            < (
                valid_start
                - pd.Timedelta(
                    minutes=HORIZON
                )
            )
        ].copy()

        test_start = test["time"].min()

        valid = valid[
            valid["time"]
            < (
                test_start
                - pd.Timedelta(
                    minutes=HORIZON
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

        rule, candidate_table = select_rule(
            valid,
            HORIZON,
        )

        if not candidate_table.empty:
            candidate_table = (
                candidate_table.copy()
            )

            candidate_table.insert(
                0,
                "test_date",
                str(test_date),
            )

            all_candidates.append(
                candidate_table
            )

        if rule is None:
            if candidate_table.empty:
                diagnostic = (
                    "NO TRADE | no candidates"
                )
            else:
                best = (
                    candidate_table.iloc[0]
                )

                diagnostic = (
                    f"NO TRADE | "
                    f"best={best['side_name']} "
                    f"q={best['q']:.2f} | "
                    f"gross={best['gross_mean']:+.2f} | "
                    f"cost={best['cost_mean']:.2f} | "
                    f"net={best['net_mean']:+.2f}"
                )

            print(
                f"{test_date} | "
                f"valid AUC={valid_auc:.3f} | "
                f"test AUC={test_auc:.3f} | "
                f"{diagnostic}"
            )

            summaries.append(
                {
                    "date": str(test_date),
                    "valid_auc": valid_auc,
                    "test_auc": test_auc,
                    "side": "NONE",
                    "q": np.nan,
                    "trades": 0,
                    "gross_mean": np.nan,
                    "cost_mean": np.nan,
                    "net_mean": np.nan,
                }
            )

            continue

        trades = non_overlapping(
            test,
            int(rule["side"]),
            float(rule["threshold"]),
            HORIZON,
        )

        if trades.empty:
            gross_mean = np.nan
            cost_mean = np.nan
            net_mean = np.nan
        else:
            gross_mean = float(
                trades["gross_bp"].mean()
            )

            cost_mean = float(
                trades[
                    "execution_cost_bp"
                ].mean()
            )

            net_mean = float(
                trades["net_bp"].mean()
            )

            trades[
                "test_date"
            ] = str(test_date)

            trades[
                "side_name"
            ] = rule["side_name"]

            trades[
                "validation_q"
            ] = rule["q"]

            all_test_trades.append(
                trades
            )

        print(
            f"{test_date} | "
            f"valid AUC={valid_auc:.3f} | "
            f"test AUC={test_auc:.3f} | "
            f"{rule['side_name']} "
            f"q={rule['q']:.2f} | "
            f"n={len(trades):4d} | "
            f"gross={gross_mean:+.2f} | "
            f"cost={cost_mean:.2f} | "
            f"net={net_mean:+.2f}"
        )

        summaries.append(
            {
                "date": str(test_date),
                "valid_auc": valid_auc,
                "test_auc": test_auc,
                "side": rule["side_name"],
                "q": rule["q"],
                "trades": len(trades),
                "gross_mean": gross_mean,
                "cost_mean": cost_mean,
                "net_mean": net_mean,
            }
        )

    summary = pd.DataFrame(
        summaries
    )

    print()
    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    if all_test_trades:
        trades = pd.concat(
            all_test_trades,
            ignore_index=True,
        )

        print()
        print("TRADE SUMMARY")
        print("-" * 92)

        print(
            "Trades:",
            len(trades),
        )

        print(
            "Gross:",
            f"{trades['gross_bp'].mean():+.3f} bp",
        )

        print(
            "Entry half spread:",
            f"{trades['entry_half_spread_bp'].mean():.3f} bp",
        )

        print(
            "Exit half spread:",
            f"{trades['exit_half_spread_bp'].mean():.3f} bp",
        )

        print(
            "Fees:",
            f"{trades['fees_bp'].mean():.3f} bp",
        )

        print(
            "Total execution cost:",
            f"{trades['execution_cost_bp'].mean():.3f} bp",
        )

        print(
            "NET:",
            f"{trades['net_bp'].mean():+.3f} bp",
        )

        print(
            "Net hit:",
            f"{(trades['net_bp'] > 0).mean() * 100:.1f}%",
        )

        print()
        print("BY TICKER")

        by_ticker = (
            trades.groupby("ticker")
            .agg(
                trades=("ticker", "size"),
                gross=("gross_bp", "mean"),
                cost=(
                    "execution_cost_bp",
                    "mean",
                ),
                net=("net_bp", "mean"),
                hit=(
                    "net_bp",
                    lambda x:
                    (x > 0).mean(),
                ),
            )
            .sort_values(
                "net",
                ascending=False,
            )
        )

        print(
            by_ticker.to_string(
                float_format=lambda x: f"{x:.3f}",
            )
        )

    else:
        trades = pd.DataFrame()

        print()
        print(
            "Ни одного разрешённого "
            "test-сигнала."
        )

    state = (
        PROJECT_ROOT
        / "state"
    )

    summary.to_csv(
        state
        / "ofi_execution_realistic_daily.csv",
        index=False,
    )

    if not trades.empty:
        trades.to_csv(
            state
            / "ofi_execution_realistic_trades.csv",
            index=False,
        )

    if all_candidates:
        candidates = pd.concat(
            all_candidates,
            ignore_index=True,
        )

        candidates.to_csv(
            state
            / "ofi_execution_realistic_candidates.csv",
            index=False,
        )

    print()
    print("Сохранено в state/")
    print("=" * 92)


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lob.dataset import FEATURES_BASIC


CACHE = PROJECT_ROOT / "state" / "ofi_dataset_h5.pkl"

HORIZON = 15
TRAIN_DAYS = 20
VALID_DAYS = 5

MAX_ENTRY_SPREAD_BP = 2.0
FEE_PER_SIDE_BP = 0.5

# Запас сверх ожидаемых execution costs.
BUFFERS_BP = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
)

MIN_VALID_TRADES = 50


def add_exact_future(
    df: pd.DataFrame,
) -> pd.DataFrame:
    parts = []

    for ticker, g in df.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values("time").copy()

        indexed = g.set_index("time")

        future_index = (
            g["time"]
            + pd.Timedelta(
                minutes=HORIZON
            )
        )

        g["future_mid"] = (
            indexed["mid"]
            .reindex(future_index)
            .to_numpy()
        )

        g["future_spread_bp"] = (
            indexed["spread_bp"]
            .reindex(future_index)
            .to_numpy()
        )

        g["future_ret_bp"] = (
            (
                g["future_mid"]
                / g["mid"]
            )
            - 1.0
        ) * 10_000

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
            subset=(
                FEATURES_BASIC
                + [
                    "time",
                    "ticker",
                    "mid",
                    "spread_bp",
                    "future_mid",
                    "future_spread_bp",
                    "future_ret_bp",
                ]
            )
        )
        .sort_values(
            ["time", "ticker"]
        )
        .reset_index(drop=True)
    )

    return out


def add_execution_cost(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

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

    return out


def select_non_overlapping(
    frame: pd.DataFrame,
    buffer_bp: float,
) -> pd.DataFrame:
    x = frame.copy()

    # Мы используем известную в симуляции actual future
    # execution cost только для PnL.
    #
    # Сигнал не имеет права знать future spread.
    #
    # Для решения в момент входа используем proxy expected cost:
    # текущий полный spread + комиссии.
    x["expected_cost_bp"] = (
        x["spread_bp"]
        + 2 * FEE_PER_SIDE_BP
    )

    long_mask = (
        x["pred_ret_bp"]
        >= (
            x["expected_cost_bp"]
            + buffer_bp
        )
    )

    short_mask = (
        x["pred_ret_bp"]
        <= -(
            x["expected_cost_bp"]
            + buffer_bp
        )
    )

    x["side"] = 0
    x.loc[long_mask, "side"] = 1
    x.loc[short_mask, "side"] = -1

    x = x[
        x["side"] != 0
    ].copy()

    x = x.sort_values(
        ["time", "ticker"]
    )

    accepted = []
    busy_until: dict[
        str,
        pd.Timestamp,
    ] = {}

    for idx, row in x.iterrows():
        ticker = str(
            row["ticker"]
        )

        now = pd.Timestamp(
            row["time"]
        )

        if (
            ticker in busy_until
            and now < busy_until[ticker]
        ):
            continue

        busy_until[ticker] = (
            now
            + pd.Timedelta(
                minutes=HORIZON
            )
        )

        accepted.append(idx)

    trades = x.loc[
        accepted
    ].copy()

    if trades.empty:
        return trades

    trades["gross_bp"] = (
        trades["side"]
        * trades["future_ret_bp"]
    )

    trades["net_bp"] = (
        trades["gross_bp"]
        - trades["execution_cost_bp"]
    )

    return trades


def choose_buffer(
    valid: pd.DataFrame,
) -> tuple[
    dict | None,
    pd.DataFrame,
]:
    rows = []

    for buffer_bp in BUFFERS_BP:
        trades = select_non_overlapping(
            valid,
            buffer_bp,
        )

        if len(trades) < MIN_VALID_TRADES:
            continue

        daily = (
            trades.groupby("date")["net_bp"]
            .mean()
        )

        long_n = int(
            (trades["side"] == 1).sum()
        )

        short_n = int(
            (trades["side"] == -1).sum()
        )

        rows.append(
            {
                "buffer_bp": buffer_bp,
                "trades": len(trades),
                "long_trades": long_n,
                "short_trades": short_n,
                "pred_abs_mean": float(
                    trades[
                        "pred_ret_bp"
                    ].abs().mean()
                ),
                "gross_mean": float(
                    trades[
                        "gross_bp"
                    ].mean()
                ),
                "cost_mean": float(
                    trades[
                        "execution_cost_bp"
                    ].mean()
                ),
                "net_mean": float(
                    trades[
                        "net_bp"
                    ].mean()
                ),
                "net_median": float(
                    trades[
                        "net_bp"
                    ].median()
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

    if not rows:
        return None, pd.DataFrame()

    table = (
        pd.DataFrame(rows)
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
    print("=" * 94)
    print("OFI COST-AWARE ALPHA")
    print("=" * 94)

    print(
        "Horizon:",
        HORIZON,
        "min",
    )

    print(
        "Max entry spread:",
        MAX_ENTRY_SPREAD_BP,
        "bp",
    )

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
        base
    )

    df = add_execution_cost(
        df
    )

    before = len(df)

    df = df[
        df["spread_bp"]
        <= MAX_ENTRY_SPREAD_BP
    ].copy()

    df["date"] = (
        df["time"].dt.date
    )

    print(
        f"После exact future: "
        f"{before:,}"
    )

    print(
        f"После spread <= "
        f"{MAX_ENTRY_SPREAD_BP:g}: "
        f"{len(df):,}"
    )

    dates = sorted(
        df["date"].unique()
    )

    summaries = []
    all_test_trades = []
    all_candidates = []

    first_test_pos = (
        TRAIN_DAYS
        + VALID_DAYS
    )

    for test_pos in range(
        first_test_pos,
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
            df["date"]
            == test_date
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
                    minutes=HORIZON
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
                    minutes=HORIZON
                )
            )
        ].copy()

        if (
            train.empty
            or valid.empty
        ):
            continue

        model = (
            HistGradientBoostingRegressor(
                loss="absolute_error",
                max_depth=5,
                max_iter=250,
                learning_rate=0.05,
                l2_regularization=1.0,
                random_state=42,
            )
        )

        model.fit(
            train[FEATURES_BASIC],
            train["future_ret_bp"],
        )

        valid["pred_ret_bp"] = (
            model.predict(
                valid[FEATURES_BASIC]
            )
        )

        test["pred_ret_bp"] = (
            model.predict(
                test[FEATURES_BASIC]
            )
        )

        rule, candidates = (
            choose_buffer(
                valid
            )
        )

        if not candidates.empty:
            candidates = (
                candidates.copy()
            )

            candidates.insert(
                0,
                "test_date",
                str(test_date),
            )

            all_candidates.append(
                candidates
            )

        if rule is None:
            if candidates.empty:
                diagnostic = (
                    "NO TRADE | "
                    "no candidates"
                )
            else:
                best = (
                    candidates.iloc[0]
                )

                diagnostic = (
                    f"NO TRADE | "
                    f"best buffer="
                    f"{best['buffer_bp']:.1f} | "
                    f"n={int(best['trades'])} | "
                    f"gross="
                    f"{best['gross_mean']:+.2f} | "
                    f"cost="
                    f"{best['cost_mean']:.2f} | "
                    f"net="
                    f"{best['net_mean']:+.2f}"
                )

            print(
                f"{test_date} | "
                f"{diagnostic}"
            )

            summaries.append(
                {
                    "date": str(
                        test_date
                    ),
                    "buffer_bp": np.nan,
                    "trades": 0,
                    "gross_mean": np.nan,
                    "cost_mean": np.nan,
                    "net_mean": np.nan,
                    "hit_rate": np.nan,
                }
            )

            continue

        buffer_bp = float(
            rule["buffer_bp"]
        )

        trades = (
            select_non_overlapping(
                test,
                buffer_bp,
            )
        )

        if trades.empty:
            gross_mean = np.nan
            cost_mean = np.nan
            net_mean = np.nan
            hit_rate = np.nan
        else:
            gross_mean = float(
                trades[
                    "gross_bp"
                ].mean()
            )

            cost_mean = float(
                trades[
                    "execution_cost_bp"
                ].mean()
            )

            net_mean = float(
                trades[
                    "net_bp"
                ].mean()
            )

            hit_rate = float(
                (
                    trades[
                        "net_bp"
                    ] > 0
                ).mean()
            )

            trades[
                "test_date"
            ] = str(
                test_date
            )

            trades[
                "validation_buffer_bp"
            ] = buffer_bp

            all_test_trades.append(
                trades
            )

        print(
            f"{test_date} | "
            f"buffer={buffer_bp:.1f} | "
            f"n={len(trades):4d} | "
            f"gross="
            f"{gross_mean:+.2f} | "
            f"cost="
            f"{cost_mean:.2f} | "
            f"net="
            f"{net_mean:+.2f}"
        )

        summaries.append(
            {
                "date": str(
                    test_date
                ),
                "buffer_bp": (
                    buffer_bp
                ),
                "trades": len(
                    trades
                ),
                "gross_mean": (
                    gross_mean
                ),
                "cost_mean": (
                    cost_mean
                ),
                "net_mean": (
                    net_mean
                ),
                "hit_rate": (
                    hit_rate
                ),
            }
        )

    summary = pd.DataFrame(
        summaries
    )

    print()
    print("=" * 94)
    print("WALK-FORWARD SUMMARY")
    print("=" * 94)

    print(
        summary.to_string(
            index=False,
            float_format=(
                lambda x:
                f"{x:.3f}"
            ),
        )
    )

    if all_test_trades:
        trades = pd.concat(
            all_test_trades,
            ignore_index=True,
        )

        print()
        print("=" * 94)
        print("TRADE SUMMARY")
        print("=" * 94)

        print(
            "Trade days:",
            int(
                (
                    summary[
                        "trades"
                    ] > 0
                ).sum()
            ),
        )

        print(
            "Trades:",
            len(trades),
        )

        print(
            "Pred abs mean:",
            f"{trades['pred_ret_bp'].abs().mean():.3f} bp",
        )

        print(
            "Gross:",
            f"{trades['gross_bp'].mean():+.3f} bp",
        )

        print(
            "Execution cost:",
            f"{trades['execution_cost_bp'].mean():.3f} bp",
        )

        print(
            "NET:",
            f"{trades['net_bp'].mean():+.3f} bp",
        )

        print(
            "Net median:",
            f"{trades['net_bp'].median():+.3f} bp",
        )

        print(
            "Net hit:",
            f"{(trades['net_bp'] > 0).mean() * 100:.1f}%",
        )

        print()
        print("BY SIDE")

        by_side = (
            trades.groupby("side")
            .agg(
                trades=(
                    "side",
                    "size",
                ),
                pred=(
                    "pred_ret_bp",
                    lambda s:
                    s.abs().mean(),
                ),
                gross=(
                    "gross_bp",
                    "mean",
                ),
                cost=(
                    "execution_cost_bp",
                    "mean",
                ),
                net=(
                    "net_bp",
                    "mean",
                ),
                hit=(
                    "net_bp",
                    lambda s:
                    (s > 0).mean(),
                ),
            )
        )

        print(
            by_side.to_string(
                float_format=(
                    lambda x:
                    f"{x:.3f}"
                )
            )
        )

        print()
        print("BY TICKER")

        by_ticker = (
            trades.groupby("ticker")
            .agg(
                trades=(
                    "ticker",
                    "size",
                ),
                pred=(
                    "pred_ret_bp",
                    lambda s:
                    s.abs().mean(),
                ),
                gross=(
                    "gross_bp",
                    "mean",
                ),
                cost=(
                    "execution_cost_bp",
                    "mean",
                ),
                net=(
                    "net_bp",
                    "mean",
                ),
                hit=(
                    "net_bp",
                    lambda s:
                    (s > 0).mean(),
                ),
            )
            .sort_values(
                "net",
                ascending=False,
            )
        )

        print(
            by_ticker.to_string(
                float_format=(
                    lambda x:
                    f"{x:.3f}"
                )
            )
        )

    else:
        trades = pd.DataFrame()

        print()
        print(
            "Ни один validation-период "
            "не разрешил торговлю."
        )

    state = (
        PROJECT_ROOT
        / "state"
    )

    summary.to_csv(
        state
        / "ofi_cost_aware_daily.csv",
        index=False,
    )

    if not trades.empty:
        trades.to_csv(
            state
            / "ofi_cost_aware_trades.csv",
            index=False,
        )

    if all_candidates:
        candidates = pd.concat(
            all_candidates,
            ignore_index=True,
        )

        candidates.to_csv(
            state
            / "ofi_cost_aware_candidates.csv",
            index=False,
        )

    print()
    print("Сохранено в state/")
    print("=" * 94)


if __name__ == "__main__":
    main()

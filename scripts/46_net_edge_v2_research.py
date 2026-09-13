from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path("/root/rl-trading-tbank")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.lob.dataset import FEATURES_BASIC


ROOT = Path("/root/rl-trading-tbank")
STATE = ROOT / "state"

DATA = STATE / "ofi_dataset_h5.pkl"

HORIZON = 15
TRAIN_DAYS = 20
VALID_DAYS = 5
TEST_DAYS = 1

MAX_ENTRY_SPREAD_BP = 2.0
FEE_PER_SIDE_BP = 0.5

# Economic safety margins, chosen only inside historical WF.
MARGINS_BP = (
    0.0,
    0.5,
    1.0,
)

# Extreme direction-score tails.
# Meaning:
# q=0.10 => lowest 10% short candidates, highest 10% long candidates
TAIL_Q = (
    0.05,
    0.10,
    0.15,
)

# Economic model thresholds.
EDGE_THRESHOLDS = (
    0.55,
    0.60,
    0.65,
)

MIN_VALID_TRADES = 40
WINSOR_Q = 0.05


def safe_auc(y, p):
    y = pd.Series(y)
    p = pd.Series(p)

    mask = y.notna() & p.notna()
    y = y[mask]
    p = p[mask]

    if len(y) == 0 or y.nunique() < 2:
        return np.nan

    return float(
        roc_auc_score(y, p)
    )


def winsorized_mean(s, q=WINSOR_Q):
    if len(s) == 0:
        return np.nan

    lo = s.quantile(q)
    hi = s.quantile(1 - q)

    return float(
        s.clip(lo, hi).mean()
    )


def add_exact_future(df):
    rows = []

    for ticker, g in df.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values(
            "time"
        ).copy()

        indexed = (
            g.set_index("time")
            .sort_index()
        )

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
                / g["mid"].to_numpy()
            )
            - 1.0
        ) * 10000.0

        rows.append(g)

    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(
            rows,
            ignore_index=True,
        )
        .sort_values(
            ["time", "ticker"]
        )
        .reset_index(drop=True)
    )


def add_execution_fields(df):
    out = df.copy()

    out["expected_cost_bp"] = (
        out["spread_bp"]
        + 2 * FEE_PER_SIDE_BP
    )

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


def fit_direction(train):
    y = (
        train["future_ret_bp"] > 0
    ).astype(int)

    model = HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        train[FEATURES_BASIC],
        y,
    )

    return model


def fit_edge_models(
    train,
    margin_bp,
):
    long_target = (
        train["future_ret_bp"]
        > (
            train["expected_cost_bp"]
            + margin_bp
        )
    ).astype(int)

    short_target = (
        -train["future_ret_bp"]
        > (
            train["expected_cost_bp"]
            + margin_bp
        )
    ).astype(int)

    long_model = (
        HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=43,
        )
    )

    short_model = (
        HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=44,
        )
    )

    long_model.fit(
        train[FEATURES_BASIC],
        long_target,
    )

    short_model.fit(
        train[FEATURES_BASIC],
        short_target,
    )

    return (
        long_model,
        short_model,
        float(long_target.mean()),
        float(short_target.mean()),
    )


def predict_all(
    frame,
    direction_model,
    long_model,
    short_model,
):
    out = frame.copy()

    out["p_up"] = (
        direction_model.predict_proba(
            out[FEATURES_BASIC]
        )[:, 1]
    )

    out["p_long_edge"] = (
        long_model.predict_proba(
            out[FEATURES_BASIC]
        )[:, 1]
    )

    out["p_short_edge"] = (
        short_model.predict_proba(
            out[FEATURES_BASIC]
        )[:, 1]
    )

    return out


def select_trades(
    frame,
    short_cut,
    long_cut,
    edge_threshold,
):
    x = frame.copy()

    x["side"] = 0

    short_mask = (
        (x["p_up"] <= short_cut)
        & (
            x["p_short_edge"]
            >= edge_threshold
        )
    )

    long_mask = (
        (x["p_up"] >= long_cut)
        & (
            x["p_long_edge"]
            >= edge_threshold
        )
    )

    x.loc[
        short_mask,
        "side",
    ] = -1

    x.loc[
        long_mask,
        "side",
    ] = 1

    x = x[
        x["side"] != 0
    ].copy()

    if x.empty:
        return x

    # Same per-ticker non-overlap logic.
    x = x.sort_values(
        ["time", "ticker"]
    )

    accepted = []
    busy_until = {}

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

        accepted.append(idx)

        busy_until[ticker] = (
            now
            + pd.Timedelta(
                minutes=HORIZON
            )
        )

    trades = x.loc[
        accepted
    ].copy()

    trades["gross_bp"] = (
        trades["side"]
        * trades["future_ret_bp"]
    )

    trades["net_bp"] = (
        trades["gross_bp"]
        - trades["execution_cost_bp"]
    )

    return trades


def evaluate_trades(trades):
    if trades.empty:
        return {
            "trades": 0,
            "net_mean": np.nan,
            "net_median": np.nan,
            "net_sum": 0.0,
            "winsor_mean": np.nan,
            "hit_rate": np.nan,
            "positive_days": np.nan,
            "median_daily_net": np.nan,
        }

    temp = trades.copy()

    temp["date"] = (
        pd.to_datetime(
            temp["time"],
            utc=True,
        )
        .dt.strftime("%Y-%m-%d")
    )

    daily = (
        temp.groupby("date")
        ["net_bp"]
        .mean()
    )

    return {
        "trades": int(
            len(temp)
        ),
        "net_mean": float(
            temp["net_bp"].mean()
        ),
        "net_median": float(
            temp["net_bp"].median()
        ),
        "net_sum": float(
            temp["net_bp"].sum()
        ),
        "winsor_mean": (
            winsorized_mean(
                temp["net_bp"]
            )
        ),
        "hit_rate": float(
            (
                temp["net_bp"] > 0
            ).mean()
        ),
        "positive_days": float(
            (daily > 0).mean()
        ),
        "median_daily_net": float(
            daily.median()
        ),
    }


def candidate_passes(x):
    return (
        x["trades"]
        >= MIN_VALID_TRADES
        and x["net_mean"] > 0
        and x["net_median"] > 0
        and x["winsor_mean"] > 0
        and x["positive_days"] >= 0.60
        and x["median_daily_net"] > 0
    )


def main():
    print("=" * 110)
    print("NET EDGE V2 — HISTORICAL WALK-FORWARD ONLY")
    print("=" * 110)

    df = pd.read_pickle(
        DATA
    ).copy()

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["time", "ticker"]
    )

    df["date"] = (
        df["time"]
        .dt.strftime("%Y-%m-%d")
    )

    # Rebuild exact H=15 target.
    df = add_exact_future(df)
    df = add_execution_fields(df)

    df = df[
        df["spread_bp"]
        <= MAX_ENTRY_SPREAD_BP
    ].copy()

    required = (
        list(FEATURES_BASIC)
        + [
            "future_mid",
            "future_spread_bp",
            "future_ret_bp",
            "expected_cost_bp",
            "execution_cost_bp",
        ]
    )

    df = df.dropna(
        subset=required
    )

    dates = sorted(
        df["date"].unique()
    )

    print(
        "Historical dates:",
        dates[0],
        "->",
        dates[-1],
        f"({len(dates)} days)",
    )

    print(
        "Rows:",
        f"{len(df):,}",
    )

    print(
        "Tickers:",
        df["ticker"].nunique(),
    )

    needed = (
        TRAIN_DAYS
        + VALID_DAYS
        + TEST_DAYS
    )

    if len(dates) < needed:
        raise SystemExit(
            "Not enough historical days"
        )

    wf_rows = []
    all_test_trades = []

    # Each step:
    # 20 train / 5 valid / 1 test
    for end in range(
        TRAIN_DAYS + VALID_DAYS,
        len(dates),
    ):
        train_dates = dates[
            end
            - VALID_DAYS
            - TRAIN_DAYS:
            end
            - VALID_DAYS
        ]

        valid_dates = dates[
            end
            - VALID_DAYS:
            end
        ]

        test_date = dates[end]

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

        direction_model = (
            fit_direction(train)
        )

        # Direction scores needed to derive
        # validation-only quantile cutoffs.
        valid_dir = valid.copy()

        valid_dir["p_up"] = (
            direction_model.predict_proba(
                valid_dir[
                    FEATURES_BASIC
                ]
            )[:, 1]
        )

        best = None
        best_models = None

        for margin_bp in MARGINS_BP:
            (
                long_model,
                short_model,
                long_rate,
                short_rate,
            ) = fit_edge_models(
                train,
                margin_bp,
            )

            valid_pred = predict_all(
                valid,
                direction_model,
                long_model,
                short_model,
            )

            for q in TAIL_Q:
                # Cutoffs derived ONLY from validation scores.
                short_cut = float(
                    valid_pred[
                        "p_up"
                    ].quantile(q)
                )

                long_cut = float(
                    valid_pred[
                        "p_up"
                    ].quantile(1 - q)
                )

                for edge_thr in EDGE_THRESHOLDS:
                    trades = select_trades(
                        valid_pred,
                        short_cut,
                        long_cut,
                        edge_thr,
                    )

                    metrics = (
                        evaluate_trades(
                            trades
                        )
                    )

                    row = {
                        "margin_bp": (
                            margin_bp
                        ),
                        "tail_q": q,
                        "edge_threshold": (
                            edge_thr
                        ),
                        "short_cut": (
                            short_cut
                        ),
                        "long_cut": (
                            long_cut
                        ),
                        "long_target_rate": (
                            long_rate
                        ),
                        "short_target_rate": (
                            short_rate
                        ),
                        **metrics,
                    }

                    row["passes"] = (
                        candidate_passes(
                            row
                        )
                    )

                    # Selection entirely on validation.
                    score = (
                        int(row["passes"]),
                        row["winsor_mean"]
                        if pd.notna(
                            row["winsor_mean"]
                        )
                        else -1e9,
                        row["net_median"]
                        if pd.notna(
                            row["net_median"]
                        )
                        else -1e9,
                        row["net_mean"]
                        if pd.notna(
                            row["net_mean"]
                        )
                        else -1e9,
                    )

                    if (
                        best is None
                        or score
                        > best["_score"]
                    ):
                        best = {
                            **row,
                            "_score": score,
                        }

                        best_models = (
                            long_model,
                            short_model,
                        )

        if best is None:
            continue

        (
            long_model,
            short_model,
        ) = best_models

        test_pred = predict_all(
            test,
            direction_model,
            long_model,
            short_model,
        )

        test_trades = select_trades(
            test_pred,
            best["short_cut"],
            best["long_cut"],
            best["edge_threshold"],
        )

        if not test_trades.empty:
            test_trades = (
                test_trades.copy()
            )

            test_trades[
                "wf_test_date"
            ] = test_date

            all_test_trades.append(
                test_trades
            )

        test_metrics = (
            evaluate_trades(
                test_trades
            )
        )

        y_test = (
            test_pred[
                "future_ret_bp"
            ] > 0
        ).astype(int)

        direction_auc = safe_auc(
            y_test,
            test_pred["p_up"],
        )

        wf_rows.append(
            {
                "test_date": test_date,
                "train_start": (
                    train_dates[0]
                ),
                "train_end": (
                    train_dates[-1]
                ),
                "valid_start": (
                    valid_dates[0]
                ),
                "valid_end": (
                    valid_dates[-1]
                ),
                "selected_passed": bool(
                    best["passes"]
                ),
                "margin_bp": (
                    best["margin_bp"]
                ),
                "tail_q": (
                    best["tail_q"]
                ),
                "edge_threshold": (
                    best[
                        "edge_threshold"
                    ]
                ),
                "short_cut": (
                    best["short_cut"]
                ),
                "long_cut": (
                    best["long_cut"]
                ),
                "valid_trades": (
                    best["trades"]
                ),
                "valid_net_mean": (
                    best["net_mean"]
                ),
                "valid_net_median": (
                    best[
                        "net_median"
                    ]
                ),
                "valid_winsor": (
                    best["winsor_mean"]
                ),
                "valid_positive_days": (
                    best[
                        "positive_days"
                    ]
                ),
                "test_direction_auc": (
                    direction_auc
                ),
                "test_trades": (
                    test_metrics[
                        "trades"
                    ]
                ),
                "test_net_mean": (
                    test_metrics[
                        "net_mean"
                    ]
                ),
                "test_net_median": (
                    test_metrics[
                        "net_median"
                    ]
                ),
                "test_winsor": (
                    test_metrics[
                        "winsor_mean"
                    ]
                ),
                "test_net_sum": (
                    test_metrics[
                        "net_sum"
                    ]
                ),
                "test_hit_rate": (
                    test_metrics[
                        "hit_rate"
                    ]
                ),
            }
        )

        print(
            f"{test_date} | "
            f"pass={best['passes']} | "
            f"margin={best['margin_bp']:.1f} | "
            f"q={best['tail_q']:.2f} | "
            f"edge={best['edge_threshold']:.2f} | "
            f"valid n={best['trades']:4d} "
            f"net={best['net_mean']:+.3f} | "
            f"test n={test_metrics['trades']:4d} "
            f"net="
            f"{test_metrics['net_mean']:+.3f}"
            if pd.notna(
                test_metrics[
                    "net_mean"
                ]
            )
            else
            f"{test_date} | "
            "test NO TRADES"
        )

    wf = pd.DataFrame(
        wf_rows
    )

    print()
    print("=" * 110)
    print("WALK-FORWARD SUMMARY")
    print("=" * 110)

    if wf.empty:
        print("NO RESULTS")
        return

    print(
        wf[
            [
                "test_date",
                "selected_passed",
                "margin_bp",
                "tail_q",
                "edge_threshold",
                "valid_trades",
                "valid_net_mean",
                "test_direction_auc",
                "test_trades",
                "test_net_mean",
                "test_net_median",
                "test_winsor",
                "test_hit_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    if not all_test_trades:
        print()
        print("NO TEST TRADES")
        return

    trades = pd.concat(
        all_test_trades,
        ignore_index=True,
    )

    print()
    print("=" * 110)
    print("AGGREGATE TEST RESULT")
    print("=" * 110)

    print(
        "test days:",
        wf["test_date"].nunique(),
    )

    print(
        "days selected candidate passed:",
        int(
            wf[
                "selected_passed"
            ].sum()
        ),
    )

    print(
        "trading days:",
        trades[
            "wf_test_date"
        ].nunique(),
    )

    print(
        "trades:",
        len(trades),
    )

    print(
        "longs:",
        int(
            (
                trades["side"] == 1
            ).sum()
        ),
    )

    print(
        "shorts:",
        int(
            (
                trades["side"] == -1
            ).sum()
        ),
    )

    print()

    print(
        "gross mean:",
        f"{trades['gross_bp'].mean():+.3f} bp",
    )

    print(
        "cost mean:",
        f"{trades['execution_cost_bp'].mean():.3f} bp",
    )

    print(
        "net mean:",
        f"{trades['net_bp'].mean():+.3f} bp",
    )

    print(
        "net median:",
        f"{trades['net_bp'].median():+.3f} bp",
    )

    print(
        "winsorized mean:",
        f"{winsorized_mean(trades['net_bp']):+.3f} bp",
    )

    print(
        "net sum:",
        f"{trades['net_bp'].sum():+.3f} bp",
    )

    print(
        "hit rate:",
        (
            f"{100*(trades['net_bp'] > 0).mean():.2f}%"
        ),
    )

    print()

    daily = (
        trades.groupby(
            "wf_test_date"
        )
        .agg(
            trades=("net_bp", "size"),
            net_mean=("net_bp", "mean"),
            net_median=(
                "net_bp",
                "median",
            ),
            net_sum=("net_bp", "sum"),
            hit_rate=(
                "net_bp",
                lambda s:
                (s > 0).mean(),
            ),
        )
        .reset_index()
    )

    print("=" * 110)
    print("TEST DAY STABILITY")
    print("=" * 110)

    print(
        daily.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    print()

    positive_days = int(
        (daily["net_sum"] > 0).sum()
    )

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

    print()

    print("=" * 110)
    print("TAIL ROBUSTNESS")
    print("=" * 110)

    ordered = (
        trades["net_bp"]
        .sort_values(
            ascending=False
        )
    )

    for k in [
        0,
        1,
        2,
        3,
        5,
    ]:
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

    for q in [
        0.01,
        0.025,
        0.05,
        0.10,
    ]:
        print(
            f"winsor {100*q:4.1f}%: "
            f"{winsorized_mean(trades['net_bp'], q):+.3f} bp"
        )

    # Save
    wf_path = (
        STATE
        / "net_edge_v2_walkforward.csv"
    )

    trades_path = (
        STATE
        / "net_edge_v2_test_trades.csv"
    )

    summary_path = (
        STATE
        / "net_edge_v2_summary.json"
    )

    wf.to_csv(
        wf_path,
        index=False,
    )

    trades.to_csv(
        trades_path,
        index=False,
    )

    summary = {
        "architecture": (
            "direction_quantiles_plus_side_specific_edge"
        ),
        "horizon_min": HORIZON,
        "max_entry_spread_bp": (
            MAX_ENTRY_SPREAD_BP
        ),
        "fee_per_side_bp": (
            FEE_PER_SIDE_BP
        ),
        "historical_only": True,
        "test_days": int(
            wf["test_date"].nunique()
        ),
        "trades": int(
            len(trades)
        ),
        "net_mean_bp": float(
            trades["net_bp"].mean()
        ),
        "net_median_bp": float(
            trades["net_bp"].median()
        ),
        "winsorized_net_mean_bp": (
            winsorized_mean(
                trades["net_bp"]
            )
        ),
        "net_sum_bp": float(
            trades["net_bp"].sum()
        ),
        "hit_rate": float(
            (
                trades["net_bp"] > 0
            ).mean()
        ),
        "positive_days": (
            positive_days
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

    print(wf_path)
    print(trades_path)
    print(summary_path)

    print()
    print(
        "IMPORTANT: this script never reads shadow_observer_features.csv"
    )
    print(
        "Forward 2026-09-01..04 is not used anywhere in v2 selection."
    )


if __name__ == "__main__":
    main()

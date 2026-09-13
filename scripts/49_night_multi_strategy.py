from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/rl-trading-tbank")
STATE = ROOT / "state"
DATA = STATE / "ofi_dataset_h5.pkl"

HORIZONS = [5, 10, 15, 30, 45]

TRAIN_DAYS = 20
VALID_DAYS = 5

MAX_SPREAD_BP = 2.0
FEE_ROUNDTRIP_BP = 1.0

TAIL_QS = [0.05, 0.10]

MIN_VALID_TRADES = 30


def winsor_mean(s, q=0.05):
    if len(s) == 0:
        return np.nan

    lo = s.quantile(q)
    hi = s.quantile(1 - q)

    return float(
        s.clip(lo, hi).mean()
    )


def zscore_train_apply(
    train,
    other,
    cols,
):
    tr = train.copy()
    ot = other.copy()

    for col in cols:
        mu = tr[col].mean()
        sd = tr[col].std()

        if (
            not np.isfinite(sd)
            or sd == 0
        ):
            sd = 1.0

        tr[f"z_{col}"] = (
            tr[col] - mu
        ) / sd

        ot[f"z_{col}"] = (
            ot[col] - mu
        ) / sd

    return tr, ot


def add_future(df, horizon):
    parts = []

    for ticker, g in df.groupby(
        "ticker",
        sort=False,
    ):
        g = (
            g.sort_values("time")
            .copy()
        )

        idx = (
            g.set_index("time")
            .sort_index()
        )

        future_time = (
            g["time"]
            + pd.Timedelta(
                minutes=horizon
            )
        )

        g["future_mid"] = (
            idx["mid"]
            .reindex(future_time)
            .to_numpy()
        )

        g["future_spread_bp"] = (
            idx["spread_bp"]
            .reindex(future_time)
            .to_numpy()
        )

        g["future_ret_bp"] = (
            (
                g["future_mid"]
                / g["mid"]
            )
            - 1.0
        ) * 10000.0

        parts.append(g)

    return pd.concat(
        parts,
        ignore_index=True,
    )


def add_cost(df):
    x = df.copy()

    x["execution_cost_bp"] = (
        x["spread_bp"] / 2.0
        + x["future_spread_bp"] / 2.0
        + FEE_ROUNDTRIP_BP
    )

    return x


def make_scores(
    train,
    frame,
):
    cols = [
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
        for c in cols
        if c in train.columns
    ]

    tr, x = zscore_train_apply(
        train,
        frame,
        cols,
    )

    def z(name):
        col = f"z_{name}"

        if col in x.columns:
            return x[col]

        return pd.Series(
            0.0,
            index=x.index,
        )

    # 1. Pure order-flow pressure.
    x["score_OFI_FLOW"] = (
        0.40 * z("ofi")
        + 0.25 * z("tfi")
        + 0.15 * z("imb1")
        + 0.15 * z("imb5")
        + 0.05 * z("micro_dev")
    )

    # 2. Price continuation.
    x["score_MOMENTUM"] = (
        0.75 * z("ret_5m")
        + 0.25 * z("ret_1m")
    )

    # 3. Price reversal.
    x["score_MEAN_REVERSION"] = (
        -0.75 * z("ret_5m")
        -0.25 * z("ret_1m")
    )

    # 4. Price + flow confirmation.
    x["score_HYBRID"] = (
        0.30 * z("ret_5m")
        + 0.25 * z("ofi")
        + 0.15 * z("tfi")
        + 0.15 * z("imb5")
        + 0.10 * z("imb1")
        + 0.05 * z("micro_dev")
    )

    return x


STRATEGIES = [
    "OFI_FLOW",
    "MOMENTUM",
    "MEAN_REVERSION",
    "HYBRID",
]


def make_trades(
    frame,
    strategy,
    q,
    low_cut,
    high_cut,
    horizon,
):
    x = frame.copy()

    score_col = (
        f"score_{strategy}"
    )

    x["side"] = 0

    x.loc[
        x[score_col] <= low_cut,
        "side",
    ] = -1

    x.loc[
        x[score_col] >= high_cut,
        "side",
    ] = 1

    x = x[
        x["side"] != 0
    ].copy()

    if x.empty:
        return x

    # Avoid overlapping positions per ticker.
    x = x.sort_values(
        ["time", "ticker"]
    )

    keep = []
    busy_until = {}

    for idx, row in x.iterrows():
        ticker = row["ticker"]
        now = row["time"]

        if (
            ticker in busy_until
            and now < busy_until[ticker]
        ):
            continue

        keep.append(idx)

        busy_until[ticker] = (
            now
            + pd.Timedelta(
                minutes=horizon
            )
        )

    x = x.loc[keep].copy()

    x["gross_bp"] = (
        x["side"]
        * x["future_ret_bp"]
    )

    x["net_bp"] = (
        x["gross_bp"]
        - x["execution_cost_bp"]
    )

    return x


def metrics(trades):
    if trades.empty:
        return {
            "trades": 0,
            "gross_mean": np.nan,
            "cost_mean": np.nan,
            "net_mean": np.nan,
            "net_median": np.nan,
            "winsor": np.nan,
            "net_sum": 0.0,
            "hit": np.nan,
        }

    return {
        "trades": len(trades),
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
        "net_median": float(
            trades["net_bp"].median()
        ),
        "winsor": winsor_mean(
            trades["net_bp"]
        ),
        "net_sum": float(
            trades["net_bp"].sum()
        ),
        "hit": float(
            (
                trades["net_bp"] > 0
            ).mean()
        ),
    }


def validation_score(m):
    if (
        m["trades"]
        < MIN_VALID_TRADES
    ):
        return -1e12

    # Robust preference rather than raw sum.
    vals = [
        m["net_mean"],
        m["net_median"],
        m["winsor"],
    ]

    if any(
        pd.isna(v)
        for v in vals
    ):
        return -1e12

    return (
        m["winsor"]
        + 0.50 * m["net_median"]
        + 0.25 * m["net_mean"]
    )


def main():
    print("=" * 110)
    print("NIGHT MULTI-STRATEGY RESEARCH")
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

    # IMPORTANT:
    # historical cache only.
    # September observer is never read.
    dates = sorted(
        df["date"].unique()
    )

    results = []
    trade_parts = []

    for horizon in HORIZONS:
        print()
        print("=" * 110)
        print(
            f"HORIZON {horizon} MIN"
        )
        print("=" * 110)

        hdf = add_future(
            df,
            horizon,
        )

        hdf = add_cost(hdf)

        hdf = hdf[
            hdf["spread_bp"]
            <= MAX_SPREAD_BP
        ].copy()

        hdf = hdf.dropna(
            subset=[
                "future_mid",
                "future_spread_bp",
                "future_ret_bp",
                "execution_cost_bp",
            ]
        )

        for i in range(
            TRAIN_DAYS + VALID_DAYS,
            len(dates),
        ):
            train_dates = dates[
                i
                - TRAIN_DAYS
                - VALID_DAYS:
                i
                - VALID_DAYS
            ]

            valid_dates = dates[
                i - VALID_DAYS:
                i
            ]

            test_date = dates[i]

            train = hdf[
                hdf["date"].isin(
                    train_dates
                )
            ].copy()

            valid = hdf[
                hdf["date"].isin(
                    valid_dates
                )
            ].copy()

            test = hdf[
                hdf["date"]
                == test_date
            ].copy()

            valid_s = make_scores(
                train,
                valid,
            )

            test_s = make_scores(
                train,
                test,
            )

            for strategy in STRATEGIES:
                score_col = (
                    f"score_{strategy}"
                )

                best = None

                for q in TAIL_QS:
                    low_cut = float(
                        valid_s[
                            score_col
                        ].quantile(q)
                    )

                    high_cut = float(
                        valid_s[
                            score_col
                        ].quantile(
                            1 - q
                        )
                    )

                    vt = make_trades(
                        valid_s,
                        strategy,
                        q,
                        low_cut,
                        high_cut,
                        horizon,
                    )

                    vm = metrics(vt)

                    score = (
                        validation_score(
                            vm
                        )
                    )

                    candidate = {
                        "q": q,
                        "low_cut": low_cut,
                        "high_cut": high_cut,
                        "valid": vm,
                        "score": score,
                    }

                    if (
                        best is None
                        or score
                        > best["score"]
                    ):
                        best = candidate

                if best is None:
                    continue

                tt = make_trades(
                    test_s,
                    strategy,
                    best["q"],
                    best["low_cut"],
                    best["high_cut"],
                    horizon,
                )

                tm = metrics(tt)

                results.append(
                    {
                        "horizon": horizon,
                        "strategy": strategy,
                        "test_date": test_date,
                        "tail_q": (
                            best["q"]
                        ),
                        "valid_trades": (
                            best[
                                "valid"
                            ]["trades"]
                        ),
                        "valid_net": (
                            best[
                                "valid"
                            ]["net_mean"]
                        ),
                        "test_trades": (
                            tm["trades"]
                        ),
                        "gross_mean": (
                            tm["gross_mean"]
                        ),
                        "cost_mean": (
                            tm["cost_mean"]
                        ),
                        "net_mean": (
                            tm["net_mean"]
                        ),
                        "net_median": (
                            tm["net_median"]
                        ),
                        "winsor": (
                            tm["winsor"]
                        ),
                        "net_sum": (
                            tm["net_sum"]
                        ),
                        "hit": (
                            tm["hit"]
                        ),
                    }
                )

                if not tt.empty:
                    tt = tt.copy()

                    tt["strategy"] = (
                        strategy
                    )

                    tt["horizon"] = (
                        horizon
                    )

                    tt["wf_test_date"] = (
                        test_date
                    )

                    trade_parts.append(
                        tt
                    )

        print(
            "finished horizon",
            horizon,
        )

    res = pd.DataFrame(
        results
    )

    if res.empty:
        raise SystemExit(
            "NO RESULTS"
        )

    daily = (
        res[
            res["test_trades"] > 0
        ]
        .copy()
    )

    summary_rows = []

    for (
        strategy,
        horizon,
    ), g in daily.groupby(
        [
            "strategy",
            "horizon",
        ]
    ):
        if trade_parts:
            pass

        days = len(g)

        positive_days = int(
            (
                g["net_sum"] > 0
            ).sum()
        )

        total_trades = int(
            g["test_trades"].sum()
        )

        weighted_net_sum = (
            g["net_sum"].sum()
        )

        weighted_gross_sum = (
            (
                g["gross_mean"]
                * g["test_trades"]
            )
            .sum()
        )

        weighted_cost_sum = (
            (
                g["cost_mean"]
                * g["test_trades"]
            )
            .sum()
        )

        if total_trades:
            net_mean = (
                weighted_net_sum
                / total_trades
            )

            gross_mean = (
                weighted_gross_sum
                / total_trades
            )

            cost_mean = (
                weighted_cost_sum
                / total_trades
            )
        else:
            net_mean = np.nan
            gross_mean = np.nan
            cost_mean = np.nan

        summary_rows.append(
            {
                "strategy": strategy,
                "horizon": horizon,
                "test_days": days,
                "trades": total_trades,
                "gross_mean": gross_mean,
                "cost_mean": cost_mean,
                "net_mean": net_mean,
                "daily_net_median": float(
                    g["net_mean"].median()
                ),
                "positive_days": (
                    positive_days
                ),
                "positive_day_share": (
                    positive_days / days
                    if days
                    else np.nan
                ),
                "net_sum": float(
                    weighted_net_sum
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary.sort_values(
        [
            "net_mean",
            "positive_day_share",
        ],
        ascending=False,
    )

    print()
    print("=" * 110)
    print("NIGHT LEADERBOARD")
    print("=" * 110)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    res.to_csv(
        STATE
        / "night_strategy_daily.csv",
        index=False,
    )

    summary.to_csv(
        STATE
        / "night_strategy_leaderboard.csv",
        index=False,
    )

    if trade_parts:
        trades = pd.concat(
            trade_parts,
            ignore_index=True,
        )

        trades.to_csv(
            STATE
            / "night_strategy_trades.csv",
            index=False,
        )

    top = (
        summary.head(10)
        .to_dict(
            orient="records"
        )
    )

    (
        STATE
        / "night_strategy_top.json"
    ).write_text(
        json.dumps(
            top,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)

    print(
        "state/night_strategy_daily.csv"
    )
    print(
        "state/night_strategy_leaderboard.csv"
    )
    print(
        "state/night_strategy_trades.csv"
    )
    print(
        "state/night_strategy_top.json"
    )

    print()
    print(
        "Historical research only."
    )
    print(
        "No broker orders."
    )
    print(
        "September forward data was not used."
    )


if __name__ == "__main__":
    main()

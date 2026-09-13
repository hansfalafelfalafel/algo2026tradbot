from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path("/root/rl-trading-tbank")
STATE = ROOT / "state"

DATA_FILE = STATE / "ofi_dataset_h5.pkl"

HORIZON = 30
TRAIN_DAYS = 20
VALID_DAYS = 5

MAX_ENTRY_SPREAD_BP = 2.0
FEE_ROUNDTRIP_BP = 1.0

TAIL_Q = 0.10


def winsor_mean(s: pd.Series, q=0.05):
    if len(s) == 0:
        return np.nan

    lo = s.quantile(q)
    hi = s.quantile(1 - q)

    return float(
        s.clip(lo, hi).mean()
    )


def add_future(df: pd.DataFrame) -> pd.DataFrame:
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
            + pd.Timedelta(minutes=HORIZON)
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
            ) - 1.0
        ) * 10000.0

        parts.append(g)

    return pd.concat(
        parts,
        ignore_index=True,
    )


def add_execution(df):
    x = df.copy()

    x["execution_cost_bp"] = (
        x["spread_bp"] / 2.0
        + x["future_spread_bp"] / 2.0
        + FEE_ROUNDTRIP_BP
    )

    return x


def add_causal_features(df):
    rows = []

    for ticker, g in df.groupby(
        "ticker",
        sort=False,
    ):
        g = (
            g.sort_values("time")
            .copy()
        )

        if "ret_5m" not in g.columns:
            g["ret_5m"] = (
                g["mid"]
                / g["mid"].shift(5)
                - 1.0
            )

        g["ret_15m"] = (
            g["mid"]
            / g["mid"].shift(15)
            - 1.0
        )

        g["ret_30m"] = (
            g["mid"]
            / g["mid"].shift(30)
            - 1.0
        )

        r1 = g["mid"].pct_change()

        for w in [5, 15, 30]:
            g[f"vol_{w}m_bp"] = (
                r1
                .rolling(
                    w,
                    min_periods=max(3, w // 3)
                )
                .std()
                * 10000.0
            )

        g["spread_mean_30"] = (
            g["spread_bp"]
            .rolling(
                30,
                min_periods=10
            )
            .mean()
        )

        g["spread_ratio_30"] = (
            g["spread_bp"]
            / g["spread_mean_30"]
            .replace(0, np.nan)
        )

        g["price_sign_5"] = np.sign(
            g["ret_5m"]
        )

        g["ofi_confirms"] = (
            (
                np.sign(g["ofi"])
                == g["price_sign_5"]
            )
            & (g["price_sign_5"] != 0)
        ).astype(float)

        g["tfi_confirms"] = (
            (
                np.sign(g["tfi"])
                == g["price_sign_5"]
            )
            & (g["price_sign_5"] != 0)
        ).astype(float)

        rows.append(g)

    x = pd.concat(
        rows,
        ignore_index=True,
    )

    minute = (
        x.groupby("time")
        .agg(
            mkt_ret_5m=(
                "ret_5m",
                "median",
            ),
            breadth_up=(
                "ret_5m",
                lambda s:
                (s > 0).mean(),
            ),
            breadth_down=(
                "ret_5m",
                lambda s:
                (s < 0).mean(),
            ),
        )
        .reset_index()
    )

    minute["breadth_extreme"] = (
        minute[
            ["breadth_up", "breadth_down"]
        ]
        .max(axis=1)
    )

    x = x.merge(
        minute,
        on="time",
        how="left",
    )

    x["mkt_ret_5m_bp"] = (
        x["mkt_ret_5m"]
        * 10000.0
    )

    x["idio_ret_5m"] = (
        x["ret_5m"]
        - x["mkt_ret_5m"]
    )

    x["abs_idio_ret_5m_bp"] = (
        x["idio_ret_5m"]
        .abs()
        * 10000.0
    )

    x["confirm_count"] = (
        x["ofi_confirms"]
        + x["tfi_confirms"]
    )

    return x


def z_apply(train, frame, col):
    mu = train[col].mean()
    sd = train[col].std()

    if (
        not np.isfinite(sd)
        or sd == 0
    ):
        sd = 1.0

    return (
        frame[col] - mu
    ) / sd


def make_mr_score(
    train,
    frame,
):
    x = frame.copy()

    z5 = z_apply(
        train,
        x,
        "ret_5m",
    )

    z15 = z_apply(
        train,
        x,
        "ret_15m",
    )

    x["mr_score"] = (
        -0.70 * z5
        -0.30 * z15
    )

    return x


def make_base_signals(
    train,
    frame,
):
    train_scored = make_mr_score(
        train,
        train,
    )

    low_cut = float(
        train_scored[
            "mr_score"
        ].quantile(
            TAIL_Q
        )
    )

    high_cut = float(
        train_scored[
            "mr_score"
        ].quantile(
            1 - TAIL_Q
        )
    )

    x = make_mr_score(
        train,
        frame,
    )

    x["side"] = 0

    x.loc[
        x["mr_score"] <= low_cut,
        "side",
    ] = -1

    x.loc[
        x["mr_score"] >= high_cut,
        "side",
    ] = 1

    return (
        x[
            x["side"] != 0
        ].copy(),
        low_cut,
        high_cut,
    )


FILTERS = {
    "BASE": {},

    "NO_BROAD_TREND": {
        "breadth_max": 0.70,
    },

    "NO_STRONG_MARKET_MOVE": {
        "market_abs_max": 15.0,
    },

    "NO_HIGH_VOL": {
        "vol30_max": 15.0,
    },

    "IDIOSYNCRATIC": {
        "idio_min": 5.0,
    },

    "NO_FLOW_CONFIRM": {
        "max_confirm": 1.0,
    },

    "REGIME_B": {
        "breadth_max": 0.70,
        "vol30_max": 15.0,
    },

    "REGIME_C": {
        "breadth_max": 0.70,
        "market_abs_max": 15.0,
        "vol30_max": 15.0,
    },

    "REGIME_IDIO": {
        "breadth_max": 0.70,
        "idio_min": 5.0,
    },
}


def apply_filter(x, cfg):
    z = x.copy()

    if "breadth_max" in cfg:
        z = z[
            z["breadth_extreme"]
            <= cfg["breadth_max"]
        ]

    if "market_abs_max" in cfg:
        z = z[
            z["mkt_ret_5m_bp"]
            .abs()
            <= cfg["market_abs_max"]
        ]

    if "vol30_max" in cfg:
        z = z[
            z["vol_30m_bp"]
            <= cfg["vol30_max"]
        ]

    if "idio_min" in cfg:
        z = z[
            z["abs_idio_ret_5m_bp"]
            >= cfg["idio_min"]
        ]

    if "max_confirm" in cfg:
        z = z[
            z["confirm_count"]
            <= cfg["max_confirm"]
        ]

    return z


def non_overlap(x):
    if x.empty:
        return x

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
                minutes=HORIZON
            )
        )

    return x.loc[keep].copy()


def add_pnl(x):
    x = x.copy()

    x["gross_bp"] = (
        x["side"]
        * x["future_ret_bp"]
    )

    x["net_bp"] = (
        x["gross_bp"]
        - x["execution_cost_bp"]
    )

    return x


def metrics(x):
    if x.empty:
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
        "trades": len(x),
        "gross_mean": float(
            x["gross_bp"].mean()
        ),
        "cost_mean": float(
            x["execution_cost_bp"].mean()
        ),
        "net_mean": float(
            x["net_bp"].mean()
        ),
        "net_median": float(
            x["net_bp"].median()
        ),
        "winsor": winsor_mean(
            x["net_bp"]
        ),
        "net_sum": float(
            x["net_bp"].sum()
        ),
        "hit": float(
            (
                x["net_bp"] > 0
            ).mean()
        ),
    }


def summarize_strategy(
    name,
    trades,
):
    m = metrics(trades)

    daily = (
        trades.groupby(
            "wf_test_date"
        )
        .agg(
            trades=(
                "net_bp",
                "size",
            ),
            net_mean=(
                "net_bp",
                "mean",
            ),
            net_median=(
                "net_bp",
                "median",
            ),
            net_sum=(
                "net_bp",
                "sum",
            ),
            hit=(
                "net_bp",
                lambda s:
                (s > 0).mean(),
            ),
        )
        .reset_index()
    )

    positive_days = int(
        (
            daily["net_sum"] > 0
        ).sum()
    )

    ordered = daily.sort_values(
        "net_sum",
        ascending=False,
    )

    robust = {}

    for k in [1, 2, 3]:
        kept = ordered.iloc[k:]

        dates = set(
            kept[
                "wf_test_date"
            ].astype(str)
        )

        z = trades[
            trades[
                "wf_test_date"
            ]
            .astype(str)
            .isin(dates)
        ]

        robust[
            f"drop_best_{k}_days_mean"
        ] = (
            float(
                z["net_bp"].mean()
            )
            if not z.empty
            else np.nan
        )

    return {
        "strategy": name,
        "days": int(
            daily[
                "wf_test_date"
            ].nunique()
        ),
        "trades": m["trades"],
        "gross_mean": (
            m["gross_mean"]
        ),
        "cost_mean": (
            m["cost_mean"]
        ),
        "net_mean": (
            m["net_mean"]
        ),
        "net_median": (
            m["net_median"]
        ),
        "winsor": (
            m["winsor"]
        ),
        "net_sum": (
            m["net_sum"]
        ),
        "hit": (
            m["hit"]
        ),
        "positive_days": (
            positive_days
        ),
        "positive_day_share": (
            positive_days
            / len(daily)
            if len(daily)
            else np.nan
        ),
        "median_daily_net": (
            float(
                daily[
                    "net_mean"
                ].median()
            )
        ),
        **robust,
    }


def main():
    print("=" * 110)
    print("MR30 FIXED REGIME COMPARISON")
    print("=" * 110)
    print("Historical OOS only.")
    print("No September data.")
    print("No daily regime switching.")

    df = pd.read_pickle(
        DATA_FILE
    ).copy()

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "time",
            "ticker",
            "mid",
            "spread_bp",
        ]
    )

    df["date"] = (
        df["time"]
        .dt.strftime("%Y-%m-%d")
    )

    df = add_causal_features(df)
    df = add_future(df)
    df = add_execution(df)

    df = df[
        df["spread_bp"]
        <= MAX_ENTRY_SPREAD_BP
    ].copy()

    df = df.dropna(
        subset=[
            "ret_5m",
            "ret_15m",
            "future_ret_bp",
            "future_spread_bp",
            "execution_cost_bp",
            "vol_30m_bp",
            "breadth_extreme",
            "mkt_ret_5m_bp",
            "abs_idio_ret_5m_bp",
        ]
    )

    dates = sorted(
        df["date"].unique()
    )

    all_trades = {
        name: []
        for name in FILTERS
    }

    daily_rows = []

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

        test_date = dates[i]

        train = df[
            df["date"].isin(
                train_dates
            )
        ].copy()

        test = df[
            df["date"] == test_date
        ].copy()

        if (
            train.empty
            or test.empty
        ):
            continue

        base, _, _ = (
            make_base_signals(
                train,
                test,
            )
        )

        for name, cfg in FILTERS.items():
            z = apply_filter(
                base,
                cfg,
            )

            z = non_overlap(z)
            z = add_pnl(z)

            if z.empty:
                continue

            z = z.copy()

            z["wf_test_date"] = (
                test_date
            )

            z["fixed_strategy"] = (
                name
            )

            all_trades[
                name
            ].append(z)

            m = metrics(z)

            daily_rows.append(
                {
                    "test_date": (
                        test_date
                    ),
                    "strategy": (
                        name
                    ),
                    "trades": (
                        m["trades"]
                    ),
                    "gross_mean": (
                        m["gross_mean"]
                    ),
                    "cost_mean": (
                        m["cost_mean"]
                    ),
                    "net_mean": (
                        m["net_mean"]
                    ),
                    "net_median": (
                        m["net_median"]
                    ),
                    "winsor": (
                        m["winsor"]
                    ),
                    "net_sum": (
                        m["net_sum"]
                    ),
                    "hit": (
                        m["hit"]
                    ),
                }
            )

    summary_rows = []
    trade_parts = []

    for name, parts in all_trades.items():
        if not parts:
            continue

        t = pd.concat(
            parts,
            ignore_index=True,
        )

        trade_parts.append(t)

        summary_rows.append(
            summarize_strategy(
                name,
                t,
            )
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary["robust_score"] = (
        summary[
            "drop_best_2_days_mean"
        ]
        + 0.5 * summary[
            "winsor"
        ]
        + 0.25 * summary[
            "median_daily_net"
        ]
    )

    summary = summary.sort_values(
        [
            "robust_score",
            "net_mean",
        ],
        ascending=False,
    )

    print()
    print("=" * 110)
    print("FIXED STRATEGY LEADERBOARD")
    print("=" * 110)

    cols = [
        "strategy",
        "days",
        "trades",
        "gross_mean",
        "cost_mean",
        "net_mean",
        "net_median",
        "winsor",
        "positive_days",
        "positive_day_share",
        "median_daily_net",
        "drop_best_1_days_mean",
        "drop_best_2_days_mean",
        "drop_best_3_days_mean",
        "robust_score",
    ]

    print(
        summary[cols]
        .to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    print()
    print("=" * 110)
    print("TOP CANDIDATE DAILY")
    print("=" * 110)

    best_name = (
        summary.iloc[0][
            "strategy"
        ]
    )

    daily = pd.DataFrame(
        daily_rows
    )

    best_daily = daily[
        daily["strategy"]
        == best_name
    ].copy()

    print(
        best_daily.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    print()
    print(
        "TOP CANDIDATE:",
        best_name
    )

    all_t = pd.concat(
        trade_parts,
        ignore_index=True,
    )

    best_t = all_t[
        all_t[
            "fixed_strategy"
        ] == best_name
    ].copy()

    print()
    print("=" * 110)
    print("TOP CANDIDATE BY SIDE")
    print("=" * 110)

    side = (
        best_t.groupby("side")
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
            net_sum=(
                "net_bp",
                "sum",
            ),
            hit=(
                "net_bp",
                lambda s:
                (s > 0).mean(),
            ),
            worst=(
                "net_bp",
                "min",
            ),
            best=(
                "net_bp",
                "max",
            ),
        )
    )

    print(
        side.to_string(
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    print()
    print("=" * 110)
    print("TOP CANDIDATE TAIL ROBUSTNESS")
    print("=" * 110)

    for q in [
        0.01,
        0.025,
        0.05,
        0.10,
    ]:
        print(
            f"winsor {q*100:4.1f}%:"
            f" {winsor_mean(best_t['net_bp'], q):+.4f}"
        )

    ordered = (
        best_t["net_bp"]
        .sort_values(
            ascending=False
        )
    )

    for k in [
        1,
        3,
        5,
        10,
        20,
    ]:
        if len(ordered) > k:
            z = ordered.iloc[k:]

            print(
                f"drop best {k:2d} trades:"
                f" mean={z.mean():+.4f}"
                f" sum={z.sum():+.1f}"
            )

    daily.to_csv(
        STATE
        / "mr30_fixed_regime_daily.csv",
        index=False,
    )

    summary.to_csv(
        STATE
        / "mr30_fixed_regime_leaderboard.csv",
        index=False,
    )

    all_t.to_csv(
        STATE
        / "mr30_fixed_regime_trades.csv",
        index=False,
    )

    (
        STATE
        / "mr30_fixed_regime_top.json"
    ).write_text(
        json.dumps(
            summary.head(5)
            .to_dict(
                orient="records"
            ),
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
        "state/mr30_fixed_regime_daily.csv"
    )

    print(
        "state/mr30_fixed_regime_leaderboard.csv"
    )

    print(
        "state/mr30_fixed_regime_trades.csv"
    )

    print(
        "state/mr30_fixed_regime_top.json"
    )


if __name__ == "__main__":
    main()

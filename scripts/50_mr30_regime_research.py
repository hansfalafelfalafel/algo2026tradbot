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
MIN_VALID_TRADES = 80


def winsor_mean(s: pd.Series, q=0.05):
    if len(s) == 0:
        return np.nan
    lo = s.quantile(q)
    hi = s.quantile(1 - q)
    return float(s.clip(lo, hi).mean())


def add_future(df: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("time").copy()

        idx = g.set_index("time").sort_index()

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

    return pd.concat(parts, ignore_index=True)


def add_execution(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["execution_cost_bp"] = (
        x["spread_bp"] / 2.0
        + x["future_spread_bp"] / 2.0
        + FEE_ROUNDTRIP_BP
    )

    return x


def add_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("time").copy()

        # Past/current returns only.
        if "ret_5m" not in g.columns:
            g["ret_5m"] = (
                g["mid"] / g["mid"].shift(5) - 1.0
            )

        g["ret_15m"] = (
            g["mid"] / g["mid"].shift(15) - 1.0
        )

        g["ret_30m"] = (
            g["mid"] / g["mid"].shift(30) - 1.0
        )

        g["abs_ret_5m_bp"] = (
            g["ret_5m"].abs() * 10000.0
        )

        g["abs_ret_15m_bp"] = (
            g["ret_15m"].abs() * 10000.0
        )

        g["abs_ret_30m_bp"] = (
            g["ret_30m"].abs() * 10000.0
        )

        r1 = (
            g["mid"].pct_change()
        )

        for w in [5, 15, 30]:
            g[f"vol_{w}m_bp"] = (
                r1.rolling(
                    w,
                    min_periods=max(3, w // 3)
                ).std()
                * 10000.0
            )

        g["spread_mean_30"] = (
            g["spread_bp"]
            .rolling(30, min_periods=10)
            .mean()
        )

        g["spread_ratio_30"] = (
            g["spread_bp"]
            / g["spread_mean_30"].replace(0, np.nan)
        )

        # Is flow confirming current move?
        g["price_sign_5"] = np.sign(g["ret_5m"])

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

    x = pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------
    # Cross-sectional / market state at each minute
    # ------------------------------------------------------

    def minute_stats(g):
        ret = g["ret_5m"]

        up_share = float((ret > 0).mean())
        down_share = float((ret < 0).mean())

        return pd.Series(
            {
                "mkt_ret_5m": ret.median(),
                "breadth_up": up_share,
                "breadth_down": down_share,
                "breadth_extreme": max(
                    up_share,
                    down_share,
                ),
                "cross_abs_ret_5m_bp": (
                    ret.abs().median()
                    * 10000.0
                ),
            }
        )

    market = (
        x.groupby("time", sort=False)
        .apply(minute_stats)
        .reset_index()
    )

    x = x.merge(
        market,
        on="time",
        how="left",
    )

    x["mkt_ret_5m_bp"] = (
        x["mkt_ret_5m"]
        * 10000.0
    )

    # Relative move = stock move minus broad market move.
    x["idio_ret_5m"] = (
        x["ret_5m"]
        - x["mkt_ret_5m"]
    )

    x["abs_idio_ret_5m_bp"] = (
        x["idio_ret_5m"].abs()
        * 10000.0
    )

    # Trend continuation danger:
    # stock and broad market moving same way.
    x["same_as_market"] = (
        (
            np.sign(x["ret_5m"])
            == np.sign(x["mkt_ret_5m"])
        )
        & (x["ret_5m"] != 0)
        & (x["mkt_ret_5m"] != 0)
    ).astype(float)

    return x


def train_zscore_apply(train, frame, col):
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
    train: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    x = frame.copy()

    # Base 30m mean-reversion signal from recent movement.
    # Use both 5m and 15m recent move, but no future data.
    z5 = train_zscore_apply(
        train,
        x,
        "ret_5m",
    )

    z15 = train_zscore_apply(
        train,
        x,
        "ret_15m",
    )

    x["mr_score"] = (
        -0.70 * z5
        -0.30 * z15
    )

    return x


def build_base_trades(
    train,
    frame,
):
    x = make_mr_score(
        train,
        frame,
    )

    low_cut = float(
        make_mr_score(
            train,
            train,
        )["mr_score"]
        .quantile(TAIL_Q)
    )

    high_cut = float(
        make_mr_score(
            train,
            train,
        )["mr_score"]
        .quantile(1 - TAIL_Q)
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

    x = x[
        x["side"] != 0
    ].copy()

    return x, low_cut, high_cut


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


def pnl_fields(x):
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
            "net_mean": np.nan,
            "net_median": np.nan,
            "winsor": np.nan,
            "net_sum": 0.0,
            "gross_mean": np.nan,
            "cost_mean": np.nan,
            "hit": np.nan,
        }

    return {
        "trades": len(x),
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
        "gross_mean": float(
            x["gross_bp"].mean()
        ),
        "cost_mean": float(
            x[
                "execution_cost_bp"
            ].mean()
        ),
        "hit": float(
            (
                x["net_bp"] > 0
            ).mean()
        ),
    }


def candidate_filters():
    """
    IMPORTANT:
    No ticker-specific rules.
    No test-derived thresholds.

    All thresholds below are generic regime definitions.
    """
    return [
        {
            "name": "BASE",
        },

        {
            "name": "NO_BROAD_TREND",
            "breadth_max": 0.70,
        },

        {
            "name": "NO_STRONG_MARKET_MOVE",
            "market_abs_max": 15.0,
        },

        {
            "name": "NO_HIGH_VOL",
            "vol30_max": 15.0,
        },

        {
            "name": "IDIOSYNCRATIC",
            "idio_min": 5.0,
        },

        {
            "name": "NO_FLOW_CONFIRM",
            "max_confirm": 1.0,
        },

        {
            "name": "REGIME_A",
            "breadth_max": 0.70,
            "market_abs_max": 15.0,
        },

        {
            "name": "REGIME_B",
            "breadth_max": 0.70,
            "vol30_max": 15.0,
        },

        {
            "name": "REGIME_C",
            "breadth_max": 0.70,
            "market_abs_max": 15.0,
            "vol30_max": 15.0,
        },

        {
            "name": "REGIME_IDIO",
            "breadth_max": 0.70,
            "idio_min": 5.0,
        },
    ]


def apply_filter(x, cfg):
    z = x.copy()

    if "breadth_max" in cfg:
        z = z[
            z["breadth_extreme"]
            <= cfg["breadth_max"]
        ]

    if "market_abs_max" in cfg:
        z = z[
            z["mkt_ret_5m_bp"].abs()
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
        confirm = (
            z["ofi_confirms"]
            + z["tfi_confirms"]
        )

        z = z[
            confirm
            <= cfg["max_confirm"]
        ]

    return z


def validation_rank(m):
    if m["trades"] < MIN_VALID_TRADES:
        return -1e12

    if any(
        pd.isna(
            m[k]
        )
        for k in [
            "net_mean",
            "net_median",
            "winsor",
        ]
    ):
        return -1e12

    # Preference for robust central tendency.
    return (
        1.00 * m["winsor"]
        + 0.75 * m["net_median"]
        + 0.25 * m["net_mean"]
    )


def main():
    print("=" * 110)
    print("MR30 REGIME RESEARCH — HISTORICAL WALK-FORWARD")
    print("=" * 110)

    print(
        "September observer data is NOT read."
    )

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

    required = [
        "ret_5m",
        "ret_15m",
        "ret_30m",
        "vol_30m_bp",
        "breadth_extreme",
        "mkt_ret_5m_bp",
        "abs_idio_ret_5m_bp",
        "future_ret_bp",
        "future_spread_bp",
        "execution_cost_bp",
    ]

    df = df.dropna(
        subset=required
    )

    dates = sorted(
        df["date"].unique()
    )

    cfgs = candidate_filters()

    result_rows = []
    trade_parts = []

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

        valid_base, low_cut, high_cut = (
            build_base_trades(
                train,
                valid,
            )
        )

        test_base = make_mr_score(
            train,
            test,
        )

        test_base["side"] = 0

        test_base.loc[
            test_base["mr_score"]
            <= low_cut,
            "side",
        ] = -1

        test_base.loc[
            test_base["mr_score"]
            >= high_cut,
            "side",
        ] = 1

        test_base = test_base[
            test_base["side"] != 0
        ].copy()

        best = None

        print()
        print(
            f"TEST {test_date}"
        )

        for cfg in cfgs:
            vv = apply_filter(
                valid_base,
                cfg,
            )

            vv = non_overlap(vv)
            vv = pnl_fields(vv)

            vm = metrics(vv)

            score = validation_rank(
                vm
            )

            candidate = {
                "cfg": cfg,
                "metrics": vm,
                "score": score,
            }

            print(
                f"  {cfg['name']:22s}"
                f" n={vm['trades']:4d}"
                f" net={vm['net_mean']:+7.3f}"
                f" med={vm['net_median']:+7.3f}"
                f" win={vm['winsor']:+7.3f}"
            )

            if (
                best is None
                or score > best["score"]
            ):
                best = candidate

        if best is None:
            continue

        selected = best["cfg"]

        tt = apply_filter(
            test_base,
            selected,
        )

        tt = non_overlap(tt)
        tt = pnl_fields(tt)

        tm = metrics(tt)

        result_rows.append(
            {
                "test_date": test_date,
                "selected_regime": (
                    selected["name"]
                ),
                "valid_trades": (
                    best["metrics"]["trades"]
                ),
                "valid_net_mean": (
                    best["metrics"]["net_mean"]
                ),
                "valid_net_median": (
                    best["metrics"]["net_median"]
                ),
                "valid_winsor": (
                    best["metrics"]["winsor"]
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
            tt["wf_test_date"] = (
                test_date
            )
            tt["selected_regime"] = (
                selected["name"]
            )
            trade_parts.append(tt)

        print(
            "  SELECTED:",
            selected["name"],
            "| test n=",
            tm["trades"],
            "| net=",
            f"{tm['net_mean']:+.3f}",
        )

    results = pd.DataFrame(
        result_rows
    )

    if results.empty:
        raise SystemExit(
            "NO WALK-FORWARD RESULTS"
        )

    print()
    print("=" * 110)
    print("WALK-FORWARD RESULTS")
    print("=" * 110)

    print(
        results.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    if trade_parts:
        trades = pd.concat(
            trade_parts,
            ignore_index=True,
        )
    else:
        trades = pd.DataFrame()

    print()
    print("=" * 110)
    print("AGGREGATE TEST RESULT")
    print("=" * 110)

    if trades.empty:
        print("NO TEST TRADES")
    else:
        agg = metrics(trades)

        print(
            "days:",
            trades[
                "wf_test_date"
            ].nunique(),
        )

        print(
            "trades:",
            agg["trades"],
        )

        print(
            "gross mean:",
            f"{agg['gross_mean']:+.3f} bp",
        )

        print(
            "cost mean:",
            f"{agg['cost_mean']:.3f} bp",
        )

        print(
            "net mean:",
            f"{agg['net_mean']:+.3f} bp",
        )

        print(
            "net median:",
            f"{agg['net_median']:+.3f} bp",
        )

        print(
            "winsor 5%:",
            f"{agg['winsor']:+.3f} bp",
        )

        print(
            "net sum:",
            f"{agg['net_sum']:+.1f} bp",
        )

        print(
            "hit rate:",
            f"{100*agg['hit']:.2f}%",
        )

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
                worst=(
                    "net_bp",
                    "min",
                ),
                best=(
                    "net_bp",
                    "max",
                ),
            )
            .reset_index()
        )

        print()
        print("=" * 110)
        print("DAILY")
        print("=" * 110)

        print(
            daily.to_string(
                index=False,
                float_format=lambda x:
                f"{x:.3f}",
            )
        )

        positive_days = int(
            (
                daily["net_sum"] > 0
            ).sum()
        )

        print()
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
        print("REMOVE BEST DAYS")
        print("=" * 110)

        ordered = daily.sort_values(
            "net_sum",
            ascending=False,
        )

        for k in [0, 1, 2, 3]:
            kept = ordered.iloc[k:]

            keep_dates = set(
                kept[
                    "wf_test_date"
                ].astype(str)
            )

            z = trades[
                trades[
                    "wf_test_date"
                ]
                .astype(str)
                .isin(
                    keep_dates
                )
            ]

            print(
                f"drop best {k}:"
                f" days={len(kept)}"
                f" trades={len(z)}"
                f" mean={z['net_bp'].mean():+.3f}"
                f" sum={z['net_bp'].sum():+.1f}"
            )

        print()
        print("=" * 110)
        print("BY SIDE")
        print("=" * 110)

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
                net_median=(
                    "net_bp",
                    "median",
                ),
                net_sum=("net_bp", "sum"),
                hit=(
                    "net_bp",
                    lambda s:
                    (s > 0).mean(),
                ),
                worst=("net_bp", "min"),
                best=("net_bp", "max"),
            )
        )

        print(
            side.to_string(
                float_format=lambda x:
                f"{x:.3f}",
            )
        )

        print()
        print("=" * 110)
        print("REGIME SELECTION FREQUENCY")
        print("=" * 110)

        print(
            results[
                "selected_regime"
            ]
            .value_counts()
            .to_string()
        )

    results.to_csv(
        STATE
        / "mr30_regime_walkforward.csv",
        index=False,
    )

    if not trades.empty:
        trades.to_csv(
            STATE
            / "mr30_regime_trades.csv",
            index=False,
        )

    summary = {
        "architecture": (
            "mr30_plus_validation_selected_regime"
        ),
        "historical_only": True,
        "horizon_min": HORIZON,
        "max_entry_spread_bp": (
            MAX_ENTRY_SPREAD_BP
        ),
        "test_days": int(
            len(results)
        ),
    }

    if not trades.empty:
        summary.update(
            {
                "trades": int(
                    len(trades)
                ),
                "net_mean_bp": float(
                    trades[
                        "net_bp"
                    ].mean()
                ),
                "net_median_bp": float(
                    trades[
                        "net_bp"
                    ].median()
                ),
                "winsorized_net_mean_bp": (
                    winsor_mean(
                        trades["net_bp"]
                    )
                ),
                "net_sum_bp": float(
                    trades[
                        "net_bp"
                    ].sum()
                ),
                "hit_rate": float(
                    (
                        trades[
                            "net_bp"
                        ] > 0
                    ).mean()
                ),
            }
        )

    (
        STATE
        / "mr30_regime_summary.json"
    ).write_text(
        json.dumps(
            summary,
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
        "state/mr30_regime_walkforward.csv"
    )
    print(
        "state/mr30_regime_trades.csv"
    )
    print(
        "state/mr30_regime_summary.json"
    )

    print()
    print(
        "No September forward data was used."
    )
    print(
        "No ticker blacklist was used."
    )
    print(
        "Regime choice was made on validation only."
    )


if __name__ == "__main__":
    main()

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/rl-trading-tbank")
STATE = ROOT / "state"
DATA = STATE / "ofi_dataset_h5.pkl"

H = 30
TRAIN_DAYS = 20
VALID_DAYS = 5
MAX_SPREAD = 2.0
FEE_RT = 1.0
TAIL_QS = [0.05, 0.10]
MIN_VALID_TRADES = 30


def winsor_mean(s, q=.05):
    if len(s) == 0:
        return np.nan
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return float(s.clip(lo, hi).mean())


def add_future(df):
    out = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("time").copy()
        idx = g.set_index("time").sort_index()

        ft = g["time"] + pd.Timedelta(minutes=H)

        g["future_mid"] = idx["mid"].reindex(ft).to_numpy()
        g["future_spread_bp"] = idx["spread_bp"].reindex(ft).to_numpy()

        g["future_ret_bp"] = (
            (g["future_mid"] / g["mid"] - 1.0) * 10000.0
        )

        out.append(g)

    return pd.concat(out, ignore_index=True)


def add_features(df):
    parts = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("time").copy()

        if "ret_5m" not in g.columns:
            g["ret_5m"] = g["mid"] / g["mid"].shift(5) - 1.0

        if "ret_1m" not in g.columns:
            g["ret_1m"] = g["mid"] / g["mid"].shift(1) - 1.0

        r1 = g["mid"].pct_change()

        g["vol_30m_bp"] = (
            r1.rolling(30, min_periods=10).std() * 10000.0
        )

        sign5 = np.sign(g["ret_5m"])

        g["ofi_confirms"] = (
            (np.sign(g["ofi"]) == sign5)
            & (sign5 != 0)
        ).astype(int)

        g["tfi_confirms"] = (
            (np.sign(g["tfi"]) == sign5)
            & (sign5 != 0)
        ).astype(int)

        g["confirm_count"] = (
            g["ofi_confirms"] + g["tfi_confirms"]
        )

        parts.append(g)

    x = pd.concat(parts, ignore_index=True)

    market = (
        x.groupby("time")
        .agg(
            mkt_ret_5m=("ret_5m", "median"),
            breadth_up=("ret_5m", lambda s: (s > 0).mean()),
            breadth_down=("ret_5m", lambda s: (s < 0).mean()),
        )
        .reset_index()
    )

    market["breadth_extreme"] = (
        market[["breadth_up", "breadth_down"]].max(axis=1)
    )

    x = x.merge(market, on="time", how="left")

    x["mkt_ret_5m_bp"] = x["mkt_ret_5m"] * 10000.0

    x["abs_idio_ret_5m_bp"] = (
        (x["ret_5m"] - x["mkt_ret_5m"]).abs() * 10000.0
    )

    return x


def z_apply(train, frame, col):
    mu = train[col].mean()
    sd = train[col].std()

    if not np.isfinite(sd) or sd == 0:
        sd = 1.0

    return (frame[col] - mu) / sd


def score(train, frame):
    x = frame.copy()

    # Exact NIGHT-49 mean-reversion signal.
    x["mr_score"] = (
        -0.75 * z_apply(train, x, "ret_5m")
        -0.25 * z_apply(train, x, "ret_1m")
    )

    return x


FILTERS = {
    "BASE": {},

    "NO_FLOW_CONFIRM": {
        "max_confirm": 1,
    },

    "NO_BROAD_TREND": {
        "breadth_max": 0.70,
    },

    "NO_HIGH_VOL": {
        "vol_max": 15.0,
    },

    "NO_STRONG_MARKET_MOVE": {
        "market_abs_max": 15.0,
    },

    "IDIOSYNCRATIC": {
        "idio_min": 5.0,
    },
}


def apply_filter(x, cfg):
    z = x.copy()

    if "max_confirm" in cfg:
        z = z[
            z["confirm_count"] <= cfg["max_confirm"]
        ]

    if "breadth_max" in cfg:
        z = z[
            z["breadth_extreme"] <= cfg["breadth_max"]
        ]

    if "vol_max" in cfg:
        z = z[
            z["vol_30m_bp"] <= cfg["vol_max"]
        ]

    if "market_abs_max" in cfg:
        z = z[
            z["mkt_ret_5m_bp"].abs() <= cfg["market_abs_max"]
        ]

    if "idio_min" in cfg:
        z = z[
            z["abs_idio_ret_5m_bp"] >= cfg["idio_min"]
        ]

    return z


def non_overlap(x):
    if x.empty:
        return x

    x = x.sort_values(["time", "ticker"])

    keep = []
    busy = {}

    for idx, row in x.iterrows():
        ticker = row["ticker"]
        now = row["time"]

        if ticker in busy and now < busy[ticker]:
            continue

        keep.append(idx)
        busy[ticker] = now + pd.Timedelta(minutes=H)

    return x.loc[keep].copy()


def pnl(x):
    x = x.copy()

    x["gross_bp"] = (
        x["side"] * x["future_ret_bp"]
    )

    x["execution_cost_bp"] = (
        x["spread_bp"] / 2.0
        + x["future_spread_bp"] / 2.0
        + FEE_RT
    )

    x["net_bp"] = (
        x["gross_bp"] - x["execution_cost_bp"]
    )

    return x


def make_trades(scored, low_cut, high_cut, cfg):
    x = scored.copy()
    x["side"] = 0

    x.loc[x["mr_score"] <= low_cut, "side"] = -1
    x.loc[x["mr_score"] >= high_cut, "side"] = 1

    x = x[x["side"] != 0].copy()

    x = apply_filter(x, cfg)
    x = non_overlap(x)
    x = pnl(x)

    return x


def metrics(x):
    if x.empty:
        return {
            "n": 0,
            "net": np.nan,
            "med": np.nan,
            "winsor": np.nan,
            "sum": 0.0,
            "hit": np.nan,
            "gross": np.nan,
            "cost": np.nan,
        }

    return {
        "n": len(x),
        "net": float(x["net_bp"].mean()),
        "med": float(x["net_bp"].median()),
        "winsor": winsor_mean(x["net_bp"]),
        "sum": float(x["net_bp"].sum()),
        "hit": float((x["net_bp"] > 0).mean()),
        "gross": float(x["gross_bp"].mean()),
        "cost": float(x["execution_cost_bp"].mean()),
    }


def validation_score(m):
    if m["n"] < MIN_VALID_TRADES:
        return -1e12

    if not all(
        np.isfinite(m[k])
        for k in ["net", "med", "winsor"]
    ):
        return -1e12

    return (
        m["winsor"]
        + 0.50 * m["med"]
        + 0.25 * m["net"]
    )


def main():
    print("=" * 110)
    print("MR30 EXACT FILTER ABLATION")
    print("=" * 110)
    print("Exact NIGHT-49 alpha.")
    print("Validation chooses q only.")
    print("Same q/cuts then used for every filter.")
    print("No September data.")

    df = pd.read_pickle(DATA).copy()

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
        df["time"].dt.strftime("%Y-%m-%d")
    )

    df = add_features(df)
    df = add_future(df)

    df = df[
        df["spread_bp"] <= MAX_SPREAD
    ].copy()

    df = df.dropna(
        subset=[
            "ret_1m",
            "ret_5m",
            "future_ret_bp",
            "future_spread_bp",
            "confirm_count",
            "breadth_extreme",
            "vol_30m_bp",
            "mkt_ret_5m_bp",
            "abs_idio_ret_5m_bp",
        ]
    )

    dates = sorted(df["date"].unique())

    all_trades = {
        k: [] for k in FILTERS
    }

    daily_rows = []

    for i in range(
        TRAIN_DAYS + VALID_DAYS,
        len(dates),
    ):
        train_dates = dates[
            i - TRAIN_DAYS - VALID_DAYS:
            i - VALID_DAYS
        ]

        valid_dates = dates[
            i - VALID_DAYS:i
        ]

        test_date = dates[i]

        train = df[
            df["date"].isin(train_dates)
        ].copy()

        valid = df[
            df["date"].isin(valid_dates)
        ].copy()

        test = df[
            df["date"] == test_date
        ].copy()

        if train.empty or valid.empty or test.empty:
            continue

        vs = score(train, valid)
        ts = score(train, test)

        best = None

        # q is chosen only from validation BASE.
        for q in TAIL_QS:
            lo = float(
                vs["mr_score"].quantile(q)
            )
            hi = float(
                vs["mr_score"].quantile(1 - q)
            )

            vt = make_trades(
                vs,
                lo,
                hi,
                FILTERS["BASE"],
            )

            vm = metrics(vt)
            rank = validation_score(vm)

            if best is None or rank > best["rank"]:
                best = {
                    "q": q,
                    "lo": lo,
                    "hi": hi,
                    "rank": rank,
                }

        if best is None:
            continue

        print(
            f"{test_date} q={best['q']:.2f}"
        )

        for name, cfg in FILTERS.items():
            tt = make_trades(
                ts,
                best["lo"],
                best["hi"],
                cfg,
            )

            if tt.empty:
                continue

            tt["wf_test_date"] = test_date
            tt["strategy"] = name

            all_trades[name].append(tt)

            m = metrics(tt)

            daily_rows.append({
                "test_date": test_date,
                "strategy": name,
                "trades": m["n"],
                "gross_mean": m["gross"],
                "cost_mean": m["cost"],
                "net_mean": m["net"],
                "net_median": m["med"],
                "winsor": m["winsor"],
                "net_sum": m["sum"],
                "hit": m["hit"],
            })

    summaries = []
    trade_parts = []

    for name, parts in all_trades.items():
        if not parts:
            continue

        t = pd.concat(parts, ignore_index=True)
        trade_parts.append(t)

        m = metrics(t)

        daily = (
            t.groupby("wf_test_date")
            .agg(
                trades=("net_bp", "size"),
                net_mean=("net_bp", "mean"),
                net_sum=("net_bp", "sum"),
            )
            .reset_index()
        )

        positive = int(
            (daily["net_sum"] > 0).sum()
        )

        ordered = daily.sort_values(
            "net_sum",
            ascending=False,
        )

        drops = {}

        for k in [1, 2, 3]:
            kept = ordered.iloc[k:]

            keep_dates = set(
                kept["wf_test_date"].astype(str)
            )

            z = t[
                t["wf_test_date"]
                .astype(str)
                .isin(keep_dates)
            ]

            drops[f"drop{k}"] = (
                float(z["net_bp"].mean())
                if len(z)
                else np.nan
            )

        summaries.append({
            "strategy": name,
            "days": len(daily),
            "trades": m["n"],
            "gross_mean": m["gross"],
            "cost_mean": m["cost"],
            "net_mean": m["net"],
            "net_median": m["med"],
            "winsor": m["winsor"],
            "net_sum": m["sum"],
            "hit": m["hit"],
            "positive_days": positive,
            "positive_day_share": positive / len(daily),
            "median_daily_net": float(
                daily["net_mean"].median()
            ),
            "drop_best_1_day": drops["drop1"],
            "drop_best_2_days": drops["drop2"],
            "drop_best_3_days": drops["drop3"],
        })

    summary = pd.DataFrame(summaries)

    summary["robust_score"] = (
        summary["drop_best_2_days"]
        + 0.5 * summary["winsor"]
        + 0.25 * summary["median_daily_net"]
    )

    summary = summary.sort_values(
        "robust_score",
        ascending=False,
    )

    print()
    print("=" * 110)
    print("EXACT ABLATION LEADERBOARD")
    print("=" * 110)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    daily_df = pd.DataFrame(daily_rows)

    print()
    print("=" * 110)
    print("NO_FLOW_CONFIRM DAILY")
    print("=" * 110)

    print(
        daily_df[
            daily_df["strategy"] == "NO_FLOW_CONFIRM"
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    all_t = pd.concat(
        trade_parts,
        ignore_index=True,
    )

    nf = all_t[
        all_t["strategy"] == "NO_FLOW_CONFIRM"
    ]

    print()
    print("=" * 110)
    print("NO_FLOW_CONFIRM BY SIDE")
    print("=" * 110)

    print(
        nf.groupby("side")
        .agg(
            trades=("net_bp", "size"),
            gross_mean=("gross_bp", "mean"),
            cost_mean=("execution_cost_bp", "mean"),
            net_mean=("net_bp", "mean"),
            net_median=("net_bp", "median"),
            net_sum=("net_bp", "sum"),
            hit=("net_bp", lambda s: (s > 0).mean()),
            worst=("net_bp", "min"),
            best=("net_bp", "max"),
        )
        .to_string(
            float_format=lambda x: f"{x:.4f}",
        )
    )

    summary.to_csv(
        STATE / "mr30_exact_ablation_leaderboard.csv",
        index=False,
    )

    daily_df.to_csv(
        STATE / "mr30_exact_ablation_daily.csv",
        index=False,
    )

    all_t.to_csv(
        STATE / "mr30_exact_ablation_trades.csv",
        index=False,
    )

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)
    print("state/mr30_exact_ablation_leaderboard.csv")
    print("state/mr30_exact_ablation_daily.csv")
    print("state/mr30_exact_ablation_trades.csv")


if __name__ == "__main__":
    main()

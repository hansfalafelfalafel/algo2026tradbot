from pathlib import Path
import gc

import numpy as np
import pandas as pd

ROOT = Path("/root/rl-trading-tbank")

HIST_FILE = ROOT / "state" / "ofi_dataset_h5.pkl"
LIVE_FILE = ROOT / "state" / "shadow_observer_features.csv"

RET5_MEAN = -0.0000156187
RET5_STD  =  0.0020025922

RET1_MEAN = -0.0000033215
RET1_STD  =  0.0008796108

VALID_FROM = pd.Timestamp("2026-08-25", tz="UTC")
VALID_TO   = pd.Timestamp("2026-09-01", tz="UTC")
FORWARD_FROM = pd.Timestamp("2026-09-06", tz="UTC")

HORIZON = 120
FEE_BP = 10.0

CANDIDATES = [
    {
        "name": "MOM120_LONG_B070",
        "score_col": "score_mom",
        "threshold": 1.30,
        "breadth_limit": 0.70,
        "top_n": 3,
    },
    {
        "name": "MR120_LONG_B060",
        "score_col": "score_mr",
        "threshold": 1.30,
        "breadth_limit": 0.60,
        "top_n": 1,
    },
]


def load_data():
    print("Loading historical...")

    hist = pd.read_pickle(HIST_FILE)

    hist = hist[
        [
            "time",
            "ticker",
            "mid",
            "spread_bp",
            "ret_1m",
            "ret_5m",
        ]
    ].copy()

    hist["time"] = pd.to_datetime(
        hist["time"],
        utc=True,
        errors="coerce",
    )

    print("historical:", f"{len(hist):,}")

    print("Loading live...")

    live = pd.read_csv(
        LIVE_FILE,
        usecols=[
            "time",
            "ticker",
            "mid",
            "spread_bp",
            "ret_1m",
        ],
    )

    live["time"] = pd.to_datetime(
        live["time"],
        utc=True,
        errors="coerce",
    )

    for c in [
        "mid",
        "spread_bp",
        "ret_1m",
    ]:
        live[c] = pd.to_numeric(
            live[c],
            errors="coerce",
        )

    live = (
        live.dropna(
            subset=[
                "time",
                "ticker",
                "mid",
                "ret_1m",
            ]
        )
        .sort_values(["ticker", "time"])
        .reset_index(drop=True)
    )

    print("live:", f"{len(live):,}")

    # ========================================================
    # ret_5m только из прошлых данных
    # ========================================================

    parts = []

    delta_ns = pd.Timedelta(
        minutes=5
    ).value

    tolerance_ns = pd.Timedelta(
        minutes=2
    ).value

    for ticker, g in live.groupby(
        "ticker",
        sort=False,
    ):
        g = g.copy()

        ts = (
            g["time"]
            .dt.tz_convert(None)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )

        mid = g["mid"].to_numpy(
            dtype=float
        )

        target = ts - delta_ns

        j = np.searchsorted(
            ts,
            target,
            side="right",
        ) - 1

        ret5 = np.full(
            len(g),
            np.nan,
            dtype=float,
        )

        valid = j >= 0

        rows = np.where(valid)[0]
        jj = j[valid]

        close = (
            target[rows] - ts[jj]
            <= tolerance_ns
        )

        rows = rows[close]
        jj = jj[close]

        good = (
            np.isfinite(mid[rows])
            & np.isfinite(mid[jj])
            & (mid[jj] > 0)
        )

        rows = rows[good]
        jj = jj[good]

        ret5[rows] = (
            mid[rows] / mid[jj] - 1
        )

        g["ret_5m"] = ret5
        parts.append(g)

    live = pd.concat(
        parts,
        ignore_index=True,
    )

    print(
        "live ret_5m:",
        f"{live['ret_5m'].notna().sum():,}",
        "/",
        f"{len(live):,}",
    )

    df = pd.concat(
        [hist, live],
        ignore_index=True,
        sort=False,
    )

    del hist, live, parts
    gc.collect()

    for c in [
        "mid",
        "spread_bp",
        "ret_1m",
        "ret_5m",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "time",
            "ticker",
            "mid",
            "ret_1m",
            "ret_5m",
            "spread_bp",
        ]
    )

    df = df[
        df["mid"] > 0
    ]

    df = (
        df.sort_values(
            ["ticker", "time"]
        )
        .drop_duplicates(
            ["ticker", "time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    print(
        "range:",
        df["time"].min(),
        "->",
        df["time"].max(),
    )

    # scores
    z5 = (
        df["ret_5m"] - RET5_MEAN
    ) / RET5_STD

    z1 = (
        df["ret_1m"] - RET1_MEAN
    ) / RET1_STD

    df["score_mr"] = (
        -0.75 * z5
        -0.25 * z1
    ).astype("float32")

    df["score_mom"] = (
        -df["score_mr"]
    ).astype("float32")

    # breadth proxy
    up = (
        df["ret_5m"] > 0
    ).astype("float32")

    breadth = (
        pd.DataFrame({
            "time": df["time"],
            "up": up,
        })
        .groupby("time")["up"]
        .mean()
    )

    breadth = np.maximum(
        breadth,
        1.0 - breadth,
    )

    df["breadth"] = (
        df["time"]
        .map(breadth)
        .astype("float32")
    )

    return df


def add_120m_exit(df):
    pieces = []

    delta_ns = pd.Timedelta(
        minutes=HORIZON
    ).value

    tolerance_ns = pd.Timedelta(
        minutes=3
    ).value

    keep = [
        "time",
        "ticker",
        "mid",
        "spread_bp",
        "score_mr",
        "score_mom",
        "breadth",
    ]

    base = df[keep]

    for ticker, g in base.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values("time").copy()

        ts = (
            g["time"]
            .dt.tz_convert(None)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )

        mid = g["mid"].to_numpy(
            dtype=float
        )

        spr = g["spread_bp"].to_numpy(
            dtype=float
        )

        target = ts + delta_ns

        j = np.searchsorted(
            ts,
            target,
            side="left",
        )

        exit_mid = np.full(
            len(g),
            np.nan,
            dtype="float32",
        )

        exit_spread = np.full(
            len(g),
            np.nan,
            dtype="float32",
        )

        exit_ns = np.full(
            len(g),
            np.iinfo(np.int64).min,
            dtype=np.int64,
        )

        valid = j < len(g)

        rows = np.where(valid)[0]
        jj = j[valid]

        close = (
            ts[jj] - target[rows]
            <= tolerance_ns
        )

        rows = rows[close]
        jj = jj[close]

        exit_mid[rows] = mid[jj]
        exit_spread[rows] = spr[jj]
        exit_ns[rows] = ts[jj]

        g["exit_mid"] = exit_mid
        g["exit_spread"] = exit_spread

        exit_time = pd.Series(
            pd.NaT,
            index=g.index,
            dtype="datetime64[ns, UTC]",
        )

        if len(rows):
            exit_time.iloc[rows] = pd.to_datetime(
                exit_ns[rows],
                utc=True,
            )

        g["exit_time"] = exit_time

        pieces.append(g)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def select_trades(df, cfg):
    c = df[
        df[cfg["score_col"]] >= cfg["threshold"]
    ].copy()

    # LONG only
    c["side"] = 1

    c = c[
        c["breadth"] <= cfg["breadth_limit"]
    ]

    c = c[
        c["spread_bp"] <= 2.0
    ]

    c = c[
        c["exit_mid"].notna()
        & c["exit_time"].notna()
    ]

    if c.empty:
        return c

    if cfg["top_n"] < 999:
        c["_strength"] = c[
            cfg["score_col"]
        ].abs()

        c = (
            c.sort_values(
                ["time", "_strength"],
                ascending=[True, False],
            )
            .groupby(
                "time",
                group_keys=False,
            )
            .head(cfg["top_n"])
        )

    # non-overlap per ticker
    keep = []

    for ticker, g in c.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values("time")

        busy_until = None

        for idx, r in g.iterrows():
            t = r["time"]

            if (
                busy_until is not None
                and t < busy_until
            ):
                continue

            keep.append(idx)
            busy_until = r["exit_time"]

    c = c.loc[keep].copy()

    c["gross_bp"] = (
        (
            c["exit_mid"] / c["mid"]
        ) - 1
    ) * 10000

    c["cost_bp"] = (
        FEE_BP
        + 0.5 * c["spread_bp"]
        + 0.5 * c["exit_spread"]
    )

    c["net_bp"] = (
        c["gross_bp"]
        - c["cost_bp"]
    )

    c["date_msk"] = (
        c["time"]
        .dt.tz_convert(
            "Europe/Moscow"
        )
        .dt.date
    )

    return c


def analyze(name, c):
    print()
    print("-" * 100)
    print(name)
    print("-" * 100)

    if c.empty:
        print("NO TRADES")
        return

    total = c["net_bp"].sum()

    print(
        "trades:",
        len(c),
        "| net:",
        f"{total:+.2f} bp",
        "| mean:",
        f"{c['net_bp'].mean():+.2f} bp",
        "| median:",
        f"{c['net_bp'].median():+.2f} bp",
        "| WR:",
        f"{100*(c['net_bp'] > 0).mean():.1f}%"
    )

    daily = (
        c.groupby("date_msk")
        .agg(
            trades=("net_bp", "size"),
            net_bp=("net_bp", "sum"),
            mean_bp=("net_bp", "mean"),
            wins=("net_bp", lambda s: (s > 0).sum()),
        )
        .reset_index()
    )

    daily["wr"] = (
        100
        * daily["wins"]
        / daily["trades"]
    )

    print()
    print("BY DAY")

    print(
        daily.to_string(
            index=False,
            formatters={
                "net_bp": "{:+.2f}".format,
                "mean_bp": "{:+.2f}".format,
                "wr": "{:.1f}".format,
            }
        )
    )

    by_ticker = (
        c.groupby("ticker")
        .agg(
            trades=("net_bp", "size"),
            net_bp=("net_bp", "sum"),
            mean_bp=("net_bp", "mean"),
            wins=("net_bp", lambda s: (s > 0).sum()),
        )
        .reset_index()
    )

    by_ticker["wr"] = (
        100
        * by_ticker["wins"]
        / by_ticker["trades"]
    )

    by_ticker = by_ticker.sort_values(
        "net_bp",
        ascending=False,
    )

    print()
    print("BY TICKER")

    print(
        by_ticker.to_string(
            index=False,
            formatters={
                "net_bp": "{:+.2f}".format,
                "mean_bp": "{:+.2f}".format,
                "wr": "{:.1f}".format,
            }
        )
    )

    best_day = daily.loc[
        daily["net_bp"].idxmax()
    ]

    best_ticker = by_ticker.iloc[0]

    print()
    print("ROBUSTNESS")

    print(
        "Total:",
        f"{total:+.2f} bp"
    )

    print(
        "Best day:",
        best_day["date_msk"],
        f"{best_day['net_bp']:+.2f} bp"
    )

    print(
        "Without best day:",
        f"{total - best_day['net_bp']:+.2f} bp"
    )

    print(
        "Best ticker:",
        best_ticker["ticker"],
        f"{best_ticker['net_bp']:+.2f} bp"
    )

    print(
        "Without best ticker:",
        f"{total - best_ticker['net_bp']:+.2f} bp"
    )

    print()
    print("COST STRESS")

    for fee in [
        10.0,
        11.0,
        12.5,
        15.0,
        20.0,
    ]:
        pnl = (
            c["net_bp"]
            - (fee - 10.0)
        )

        print(
            f"fee {fee:4.1f} bp -> "
            f"{pnl.sum():+8.2f} bp "
            f"| mean {pnl.mean():+6.2f}"
        )


print("=" * 110)
print("ROBUSTNESS TEST")
print("=" * 110)

df = load_data()

print()
print("Building 120m exits...")

hdf = add_120m_exit(df)

del df
gc.collect()

validation = hdf[
    (hdf["time"] >= VALID_FROM)
    & (hdf["time"] < VALID_TO)
].copy()

forward = hdf[
    hdf["time"] >= FORWARD_FROM
].copy()

print(
    "validation:",
    len(validation),
    "| forward:",
    len(forward)
)

for cfg in CANDIDATES:
    cv = select_trades(
        validation,
        cfg,
    )

    cf = select_trades(
        forward,
        cfg,
    )

    print()
    print("=" * 110)
    print(cfg["name"])
    print("=" * 110)

    analyze(
        "VALIDATION",
        cv,
    )

    analyze(
        "FORWARD",
        cf,
    )

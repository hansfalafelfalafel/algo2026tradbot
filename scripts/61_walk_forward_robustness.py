from pathlib import Path
import gc
import numpy as np
import pandas as pd

ROOT = Path("/root/rl-trading-tbank")

HIST_FILE = ROOT / "state" / "ofi_dataset_h5.pkl"
LIVE_FILE = ROOT / "state" / "shadow_observer_features.csv"
LAB_FILE  = ROOT / "state" / "strategy_lab_all.csv"

OUT = ROOT / "state" / "walk_forward_robustness.csv"

RET5_MEAN = -0.0000156187
RET5_STD  =  0.0020025922

RET1_MEAN = -0.0000033215
RET1_STD  =  0.0008796108

FEE_BP = 10.0

WINDOWS = [
    (
        "W1",
        pd.Timestamp("2026-08-11", tz="UTC"),
        pd.Timestamp("2026-08-18", tz="UTC"),
    ),
    (
        "W2",
        pd.Timestamp("2026-08-18", tz="UTC"),
        pd.Timestamp("2026-08-25", tz="UTC"),
    ),
    (
        "W3",
        pd.Timestamp("2026-08-25", tz="UTC"),
        pd.Timestamp("2026-09-01", tz="UTC"),
    ),
]

PRE_FROM = WINDOWS[0][1]
PRE_TO   = WINDOWS[-1][2]

FWD_FROM = pd.Timestamp("2026-09-06", tz="UTC")


def load_minimal():
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

    # --------------------------------------------------------
    # Rebuild live ret_5m using PAST ONLY
    # --------------------------------------------------------

    five_ns = pd.Timedelta(minutes=5).value
    tol_ns = pd.Timedelta(minutes=2).value

    parts = []

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

        mid = g["mid"].to_numpy(dtype=float)

        target = ts - five_ns

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

        ok = j >= 0

        rows = np.where(ok)[0]
        jj = j[ok]

        close = (
            target[rows] - ts[jj]
            <= tol_ns
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
            mid[rows] / mid[jj] - 1.0
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
            "spread_bp",
            "ret_1m",
            "ret_5m",
        ]
    )

    df = df[df["mid"] > 0]

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

    print(
        "range:",
        df["time"].min(),
        "->",
        df["time"].max(),
    )

    return df


def add_exit(df, horizon):
    pieces = []

    delta_ns = pd.Timedelta(
        minutes=int(horizon)
    ).value

    tol_ns = pd.Timedelta(
        minutes=3
    ).value

    cols = [
        "time",
        "ticker",
        "mid",
        "spread_bp",
        "score_mr",
        "score_mom",
        "breadth",
    ]

    base = df[cols]

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

        mid = g["mid"].to_numpy(dtype=float)
        spr = g["spread_bp"].to_numpy(dtype=float)

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
            <= tol_ns
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


def make_trades(df, cfg):
    score_col = (
        "score_mr"
        if cfg["strategy"] == "MR"
        else "score_mom"
    )

    score = df[score_col]

    threshold = float(
        cfg["threshold"]
    )

    c = df[
        score.abs() >= threshold
    ].copy()

    if c.empty:
        return c

    c["side"] = np.where(
        c[score_col] > 0,
        1,
        -1,
    )

    direction = str(
        cfg["direction"]
    )

    if direction == "LONG":
        c = c[c["side"] == 1]

    elif direction == "SHORT":
        c = c[c["side"] == -1]

    breadth_name = str(
        cfg["breadth"]
    )

    if breadth_name == "ALL":
        breadth_limit = 1.01

    elif breadth_name == "NO_BROAD_070":
        breadth_limit = 0.70

    elif breadth_name == "NO_BROAD_060":
        breadth_limit = 0.60

    else:
        raise ValueError(
            f"Unknown breadth {breadth_name}"
        )

    c = c[
        c["breadth"] <= breadth_limit
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

    top_n = int(cfg["top_n"])

    if top_n < 999:
        c["_strength"] = (
            c[score_col].abs()
        )

        c = (
            c.sort_values(
                ["time", "_strength"],
                ascending=[True, False],
            )
            .groupby(
                "time",
                group_keys=False,
            )
            .head(top_n)
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

    if not keep:
        return c.iloc[:0]

    c = c.loc[keep].copy()

    c["gross_bp"] = (
        c["side"]
        * (
            c["exit_mid"] / c["mid"]
            - 1
        )
        * 10000
    )

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
        .dt.tz_convert("Europe/Moscow")
        .dt.date
    )

    return c


def metrics(c):
    if c.empty:
        return {
            "trades": 0,
            "net": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "wr": np.nan,
            "days": 0,
            "positive_days": 0,
            "without_best_day": np.nan,
            "without_best_ticker": np.nan,
        }

    pnl = c["net_bp"]

    daily = (
        c.groupby("date_msk")
        ["net_bp"]
        .sum()
    )

    ticker = (
        c.groupby("ticker")
        ["net_bp"]
        .sum()
    )

    total = pnl.sum()

    return {
        "trades": int(len(c)),
        "net": float(total),
        "mean": float(pnl.mean()),
        "median": float(pnl.median()),
        "wr": float(
            100 * (pnl > 0).mean()
        ),
        "days": int(len(daily)),
        "positive_days": int(
            (daily > 0).sum()
        ),
        "without_best_day": float(
            total - daily.max()
        ),
        "without_best_ticker": float(
            total - ticker.max()
        ),
    }


print("=" * 120)
print("WALK-FORWARD ROBUSTNESS SEARCH")
print("=" * 120)

lab = pd.read_csv(LAB_FILE)

# ------------------------------------------------------------
# Только предварительный shortlist.
# Forward НЕ используется.
# Берём хорошие validation-конфигурации,
# затем независимо проверим их на нескольких прошлых окнах.
# ------------------------------------------------------------

lab["val_net_bp"] = pd.to_numeric(
    lab["val_net_bp"],
    errors="coerce",
)

lab["val_mean_bp"] = pd.to_numeric(
    lab["val_mean_bp"],
    errors="coerce",
)

lab["val_trades"] = pd.to_numeric(
    lab["val_trades"],
    errors="coerce",
)

pool = lab[
    lab["val_trades"] >= 15
].copy()

pool["pre_score"] = (
    pool["val_mean_bp"]
    + 0.01 * pool["val_net_bp"]
)

pool = (
    pool.sort_values(
        "pre_score",
        ascending=False,
    )
    .head(100)
    .copy()
)

print(
    "candidate configs:",
    len(pool)
)

print(
    "horizons:",
    sorted(pool["horizon"].unique())
)

df = load_minimal()

results = []

for horizon in sorted(
    pool["horizon"].unique()
):
    horizon = int(horizon)

    print()
    print("=" * 120)
    print(
        f"HORIZON {horizon}"
    )
    print("=" * 120)

    hdf = add_exit(
        df,
        horizon,
    )

    configs = pool[
        pool["horizon"] == horizon
    ]

    for n, (_, cfg) in enumerate(
        configs.iterrows(),
        start=1,
    ):
        trades = make_trades(
            hdf,
            cfg,
        )

        row = {
            "strategy": cfg["strategy"],
            "horizon": horizon,
            "threshold": cfg["threshold"],
            "breadth": cfg["breadth"],
            "direction": cfg["direction"],
            "top_n": int(cfg["top_n"]),
        }

        positive_windows = 0
        usable_windows = 0

        pre_parts = []

        for name, start, end in WINDOWS:
            w = trades[
                (trades["time"] >= start)
                & (trades["time"] < end)
            ].copy()

            m = metrics(w)

            row[f"{name}_trades"] = m["trades"]
            row[f"{name}_net"] = m["net"]
            row[f"{name}_mean"] = m["mean"]

            if m["trades"] > 0:
                usable_windows += 1

                if m["net"] > 0:
                    positive_windows += 1

                pre_parts.append(w)

        if pre_parts:
            pre = pd.concat(
                pre_parts,
                ignore_index=True,
            )
        else:
            pre = trades.iloc[:0].copy()

        pm = metrics(pre)

        fwd = trades[
            trades["time"] >= FWD_FROM
        ].copy()

        fm = metrics(fwd)

        row.update({
            "pre_trades": pm["trades"],
            "pre_net": pm["net"],
            "pre_mean": pm["mean"],
            "pre_median": pm["median"],
            "pre_wr": pm["wr"],
            "pre_windows": usable_windows,
            "pre_positive_windows": positive_windows,
            "pre_without_best_day":
                pm["without_best_day"],
            "pre_without_best_ticker":
                pm["without_best_ticker"],

            "fwd_trades": fm["trades"],
            "fwd_net": fm["net"],
            "fwd_mean": fm["mean"],
            "fwd_median": fm["median"],
            "fwd_wr": fm["wr"],
            "fwd_days": fm["days"],
            "fwd_positive_days":
                fm["positive_days"],
            "fwd_without_best_day":
                fm["without_best_day"],
            "fwd_without_best_ticker":
                fm["without_best_ticker"],
        })

        results.append(row)

        if n % 10 == 0:
            print(
                f"{n}/{len(configs)}"
            )

    del hdf
    gc.collect()

res = pd.DataFrame(results)

# ============================================================
# ROBUST SELECTION — PRE-FORWARD ONLY
# ============================================================

eligible = res[
    (res["pre_trades"] >= 30)
    & (res["pre_windows"] == 3)
    & (res["pre_positive_windows"] >= 2)
    & (res["pre_net"] > 0)
    & (res["pre_without_best_day"] > 0)
    & (res["pre_without_best_ticker"] > 0)
].copy()

# Более высокий балл за устойчивость,
# а не просто максимальный total.
eligible["robust_score"] = (
    eligible["pre_mean"]
    + 0.01 * eligible["pre_net"]
    + 0.01 * eligible["pre_without_best_day"]
    + 0.01 * eligible["pre_without_best_ticker"]
    + 2.0 * eligible["pre_positive_windows"]
)

eligible = eligible.sort_values(
    "robust_score",
    ascending=False,
)

res.to_csv(
    OUT,
    index=False,
)

cols = [
    "strategy",
    "horizon",
    "threshold",
    "breadth",
    "direction",
    "top_n",

    "W1_trades",
    "W1_net",
    "W2_trades",
    "W2_net",
    "W3_trades",
    "W3_net",

    "pre_trades",
    "pre_net",
    "pre_mean",
    "pre_positive_windows",
    "pre_without_best_day",
    "pre_without_best_ticker",

    "fwd_trades",
    "fwd_net",
    "fwd_mean",
    "fwd_wr",
    "fwd_positive_days",
    "fwd_days",
    "fwd_without_best_day",
    "fwd_without_best_ticker",
]

print()
print("=" * 120)
print("ROBUST PRE-FORWARD WINNERS")
print("=" * 120)

if eligible.empty:
    print(
        "Нет конфигураций, прошедших все "
        "robustness-критерии."
    )
else:
    print(
        eligible[
            cols
        ].head(20).to_string(
            index=False,
            formatters={
                c: "{:+.1f}".format
                for c in [
                    "W1_net",
                    "W2_net",
                    "W3_net",
                    "pre_net",
                    "pre_mean",
                    "pre_without_best_day",
                    "pre_without_best_ticker",
                    "fwd_net",
                    "fwd_mean",
                    "fwd_without_best_day",
                    "fwd_without_best_ticker",
                ]
            },
        )
    )

print()
print("=" * 120)
print("ROBUST + FORWARD POSITIVE")
print("=" * 120)

survivors = eligible[
    (eligible["fwd_trades"] >= 10)
    & (eligible["fwd_net"] > 0)
].copy()

if survivors.empty:
    print("NO SURVIVORS")
else:
    print(
        survivors[
            cols
        ].head(20).to_string(
            index=False,
            formatters={
                c: "{:+.1f}".format
                for c in [
                    "W1_net",
                    "W2_net",
                    "W3_net",
                    "pre_net",
                    "pre_mean",
                    "pre_without_best_day",
                    "pre_without_best_ticker",
                    "fwd_net",
                    "fwd_mean",
                    "fwd_without_best_day",
                    "fwd_without_best_ticker",
                ]
            },
        )
    )

print()
print("Saved:", OUT)

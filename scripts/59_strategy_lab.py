from __future__ import annotations

from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "state" / "ofi_dataset_h5.pkl"
LIVE_DATA = ROOT / "state" / "shadow_observer_features.csv"

OUT_ALL = ROOT / "state" / "strategy_lab_all.csv"
OUT_TOP = ROOT / "state" / "strategy_lab_top_forward.csv"

# ============================================================
# FROZEN NORMALIZATION — НЕ ПОДГОНЯЕМ ПО FORWARD
# ============================================================

RET5_MEAN = -0.0000156187
RET5_STD  =  0.0020025922

RET1_MEAN = -0.0000033215
RET1_STD  =  0.0008796108

# Наблюдаемая комиссия T-Bank Sandbox:
# примерно 10 bp за полный round-trip.
ROUND_TRIP_FEE_BP = 10.0

VALID_FROM = pd.Timestamp("2026-08-25", tz="UTC")
VALID_TO   = pd.Timestamp("2026-09-01", tz="UTC")

FORWARD_FROM = pd.Timestamp("2026-09-06", tz="UTC")

HORIZONS = [30, 60, 90, 120]
THRESHOLDS = [1.30, 1.50, 1.75, 2.00, 2.50]

BREADTH_LIMITS = [
    ("ALL", 1.01),
    ("NO_BROAD_070", 0.70),
    ("NO_BROAD_060", 0.60),
]

DIRECTIONS = [
    "BOTH",
    "LONG",
    "SHORT",
]

TOP_NS = [
    1,
    3,
    999,
]

STRATEGIES = [
    "MR",
    "MOMENTUM",
]


def detect_time_col(df):
    for c in ["time", "timestamp", "datetime", "dt"]:
        if c in df.columns:
            return c
    raise RuntimeError(
        "Не нашла временной столбец. "
        f"columns={list(df.columns)}"
    )


def prepare(df):
    time_col = detect_time_col(df)

    required = [
        "ticker",
        "mid",
        "ret_1m",
        "ret_5m",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Не хватает колонок: {missing}\n"
            f"Есть: {list(df.columns)}"
        )

    x = df.copy()

    x["time"] = pd.to_datetime(
        x[time_col],
        utc=True,
        errors="coerce",
    )

    # shadow_observer_features.csv хранит фактическое
    # время наблюдения в observed_at_utc.
    # После concat исторические строки имеют time,
    # а live-строки могут иметь time пустым.
    if "observed_at_utc" in x.columns:
        obs = pd.to_datetime(
            x["observed_at_utc"],
            utc=True,
            errors="coerce",
        )

        x["time"] = x["time"].fillna(obs)

    for c in [
        "mid",
        "ret_1m",
        "ret_5m",
        "spread_bp",
    ]:
        if c in x.columns:
            x[c] = pd.to_numeric(
                x[c],
                errors="coerce",
            )

    # ========================================================
    # LIVE может не содержать готовый ret_5m.
    # Восстанавливаем только пропуски из ПРОШЛОЙ цены
    # примерно 5 минут назад. Future data здесь не используется.
    # ========================================================

    if "ret_5m" not in x.columns:
        x["ret_5m"] = np.nan

    need_ret5 = (
        x["ret_5m"].isna()
        & x["time"].notna()
        & x["mid"].notna()
        & x["ticker"].notna()
    )

    print(
        "ret_5m missing before rebuild:",
        f"{int(need_ret5.sum()):,}"
    )

    if need_ret5.any():
        rebuilt = pd.Series(
            np.nan,
            index=x.index,
            dtype=float,
        )

        five_ns = pd.Timedelta(
            minutes=5
        ).value

        tolerance_ns = pd.Timedelta(
            minutes=2
        ).value

        for ticker, g in x[
            x["time"].notna()
            & x["mid"].notna()
            & x["ticker"].notna()
        ].groupby("ticker", sort=False):

            g = g.sort_values("time")

            idx_orig = g.index.to_numpy()

            ts = (
                g["time"]
                .dt.tz_convert(None)
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            )

            mid = pd.to_numeric(
                g["mid"],
                errors="coerce",
            ).to_numpy(dtype=float)

            target = ts - five_ns

            j = np.searchsorted(
                ts,
                target,
                side="right",
            ) - 1

            good = j >= 0

            rows = np.where(good)[0]
            jj = j[good]

            # Не используем слишком далёкое наблюдение:
            # прошлое значение должно быть максимум
            # на 2 минуты раньше целевой точки t-5m.
            ok = (
                target[rows] - ts[jj]
                <= tolerance_ns
            )

            rows = rows[ok]
            jj = jj[ok]

            valid_price = (
                np.isfinite(mid[rows])
                & np.isfinite(mid[jj])
                & (mid[jj] > 0)
            )

            rows = rows[valid_price]
            jj = jj[valid_price]

            vals = (
                mid[rows] / mid[jj] - 1.0
            )

            rebuilt.loc[
                idx_orig[rows]
            ] = vals

        fill_mask = (
            x["ret_5m"].isna()
            & rebuilt.notna()
        )

        x.loc[
            fill_mask,
            "ret_5m"
        ] = rebuilt.loc[fill_mask]

        print(
            "ret_5m rebuilt:",
            f"{int(fill_mask.sum()):,}"
        )

    print(
        "ret_5m still missing:",
        f"{int(x['ret_5m'].isna().sum()):,}"
    )

    if "spread_bp" not in x.columns:
        print(
            "WARNING: spread_bp отсутствует; "
            "использую 0 bp spread."
        )
        x["spread_bp"] = 0.0

    x = x.dropna(
        subset=[
            "time",
            "ticker",
            "mid",
            "ret_1m",
            "ret_5m",
        ]
    ).copy()

    x = x[
        (x["mid"] > 0)
        & (x["spread_bp"] >= 0)
    ].copy()

    x = x.sort_values(
        ["ticker", "time"]
    ).reset_index(drop=True)

    # Замороженная нормализация MR30.
    z5 = (
        x["ret_5m"] - RET5_MEAN
    ) / RET5_STD

    z1 = (
        x["ret_1m"] - RET1_MEAN
    ) / RET1_STD

    # Положительный score => LONG.
    x["score_mr"] = (
        -0.75 * z5
        -0.25 * z1
    )

    x["score_mom"] = -x["score_mr"]

    return x


def add_breadth(df):
    """
    Если breadth_extreme уже есть — используем его.
    Иначе строим прокси:
    какая доля рынка в один момент движется
    в доминирующем направлении по ret_5m.
    """

    if "breadth_extreme" in df.columns:
        b = pd.to_numeric(
            df["breadth_extreme"],
            errors="coerce",
        )

        if b.notna().sum() > 0:
            df["breadth"] = b
            return df

    tmp = df[
        ["time", "ret_5m"]
    ].copy()

    tmp["up"] = (
        tmp["ret_5m"] > 0
    ).astype(float)

    breadth = (
        tmp.groupby("time")["up"]
        .mean()
    )

    breadth = np.maximum(
        breadth,
        1.0 - breadth,
    )

    df["breadth"] = (
        df["time"]
        .map(breadth)
        .astype(float)
    )

    return df


def add_future_prices(df, horizon):
    """
    Ищем первую доступную запись инструмента
    не раньше time+horizon.
    Допускаем максимум +3 минуты,
    чтобы не использовать далёкую цену после разрыва данных.
    """

    result_parts = []

    delta_ns = pd.Timedelta(
        minutes=horizon
    ).value

    tolerance_ns = pd.Timedelta(
        minutes=3
    ).value

    for ticker, g in df.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values("time").copy()

        # Явно приводим к ns.
        # В новых pandas datetime может иметь us-resolution,
        # тогда astype(int64) и Timedelta.value оказываются
        # в разных единицах.
        ts = (
            g["time"]
            .dt.tz_convert(None)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        mid = g["mid"].to_numpy(dtype=float)
        spr = g["spread_bp"].to_numpy(dtype=float)

        target = ts + delta_ns

        idx = np.searchsorted(
            ts,
            target,
            side="left",
        )

        valid = idx < len(g)

        exit_mid = np.full(
            len(g),
            np.nan,
            dtype=float,
        )

        exit_spread = np.full(
            len(g),
            np.nan,
            dtype=float,
        )

        exit_time_ns = np.full(
            len(g),
            np.iinfo(np.int64).min,
            dtype=np.int64,
        )

        rows = np.where(valid)[0]
        ii = idx[valid]

        good = (
            ts[ii] - target[valid]
            <= tolerance_ns
        )

        rows = rows[good]
        ii = ii[good]

        exit_mid[rows] = mid[ii]
        exit_spread[rows] = spr[ii]
        exit_time_ns[rows] = ts[ii]

        g[f"exit_mid_{horizon}"] = exit_mid
        g[f"exit_spread_{horizon}"] = exit_spread

        exit_time = pd.Series(
            pd.NaT,
            index=g.index,
            dtype="datetime64[ns, UTC]",
        )

        if len(rows):
            vals = pd.to_datetime(
                exit_time_ns[rows],
                utc=True,
            )
            exit_time.iloc[rows] = vals

        g[f"exit_time_{horizon}"] = pd.to_datetime(
            exit_time,
            utc=True,
            errors="coerce",
        )

        result_parts.append(g)

    return pd.concat(
        result_parts,
        ignore_index=True,
    ).sort_values(
        ["time", "ticker"]
    ).reset_index(drop=True)


def select_candidates(
    df,
    score_col,
    threshold,
    direction,
    breadth_limit,
    top_n,
    horizon,
):
    c = df[
        df[score_col].abs() >= threshold
    ].copy()

    if c.empty:
        return c

    c["side"] = np.where(
        c[score_col] > 0,
        1,
        -1,
    )

    if direction == "LONG":
        c = c[c["side"] == 1]
    elif direction == "SHORT":
        c = c[c["side"] == -1]

    c = c[
        c["breadth"] <= breadth_limit
    ]

    c = c[
        c[f"exit_mid_{horizon}"].notna()
    ]

    # Аналогично текущему MR30 — не берём
    # совсем широкий spread.
    c = c[
        c["spread_bp"] <= 2.0
    ]

    if c.empty:
        return c

    # Если в одну минуту много сигналов —
    # оставляем самые сильные.
    if top_n < 999:
        c["_strength"] = c[
            score_col
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
            .head(top_n)
        )

    # Non-overlap по тикеру.
    kept = []

    for ticker, g in c.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values("time")

        busy_until = None

        for idx, r in g.iterrows():
            t = pd.Timestamp(r["time"])

            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            else:
                t = t.tz_convert("UTC")

            if busy_until is not None:
                busy_until = pd.Timestamp(busy_until)

                if busy_until.tzinfo is None:
                    busy_until = busy_until.tz_localize("UTC")
                else:
                    busy_until = busy_until.tz_convert("UTC")

                if t < busy_until:
                    continue

            kept.append(idx)

            busy_until = r[
                f"exit_time_{horizon}"
            ]

    if not kept:
        return c.iloc[0:0]

    return c.loc[kept].copy()


def evaluate(c, horizon, fee_bp):
    if c.empty:
        return {
            "trades": 0,
            "gross_bp": np.nan,
            "net_bp": np.nan,
            "mean_bp": np.nan,
            "median_bp": np.nan,
            "win_rate": np.nan,
            "positive_days": 0,
            "days": 0,
            "worst_day_bp": np.nan,
        }

    exit_mid = c[
        f"exit_mid_{horizon}"
    ]

    exit_spread = c[
        f"exit_spread_{horizon}"
    ].fillna(0)

    c = c.copy()

    c["gross_trade_bp"] = (
        c["side"]
        * (
            exit_mid / c["mid"] - 1.0
        )
        * 10000.0
    )

    # Market-order approximation:
    # half-spread entry + half-spread exit
    # + observed round-trip broker commission.
    c["cost_trade_bp"] = (
        fee_bp
        + 0.5 * c["spread_bp"]
        + 0.5 * exit_spread
    )

    c["net_trade_bp"] = (
        c["gross_trade_bp"]
        - c["cost_trade_bp"]
    )

    pnl = c["net_trade_bp"].dropna()

    if pnl.empty:
        return {
            "trades": 0,
            "gross_bp": np.nan,
            "net_bp": np.nan,
            "mean_bp": np.nan,
            "median_bp": np.nan,
            "win_rate": np.nan,
            "positive_days": 0,
            "days": 0,
            "worst_day_bp": np.nan,
        }

    c["date_msk"] = (
        c["time"]
        .dt.tz_convert("Europe/Moscow")
        .dt.date
    )

    daily = c.groupby(
        "date_msk"
    )["net_trade_bp"].sum()

    return {
        "trades": int(len(pnl)),
        "gross_bp": float(
            c["gross_trade_bp"].sum()
        ),
        "net_bp": float(pnl.sum()),
        "mean_bp": float(pnl.mean()),
        "median_bp": float(pnl.median()),
        "win_rate": float(
            100.0 * (pnl > 0).mean()
        ),
        "positive_days": int(
            (daily > 0).sum()
        ),
        "days": int(len(daily)),
        "worst_day_bp": float(
            daily.min()
        ),
    }


def main():
    print("=" * 120)
    print("MR30 / MOMENTUM STRATEGY LAB")
    print("=" * 120)

    print("Loading:", DATA)

    hist = pd.read_pickle(DATA)

    print(
        "historical rows:",
        f"{len(hist):,}"
    )

    parts = [hist]

    if LIVE_DATA.exists():
        live = pd.read_csv(LIVE_DATA)

        print(
            "live rows:",
            f"{len(live):,}"
        )

        parts.append(live)
    else:
        print(
            "WARNING: live feature file not found:",
            LIVE_DATA
        )

    raw = pd.concat(
        parts,
        ignore_index=True,
        sort=False,
    )

    print(
        "combined rows:",
        f"{len(raw):,}"
    )
    print(
        "columns:",
        list(raw.columns)
    )

    df = prepare(raw)

    before = len(df)

    df = (
        df.sort_values(["time", "ticker"])
        .drop_duplicates(
            subset=["time", "ticker"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    print(
        "deduplicated:",
        f"{before:,} -> {len(df):,}"
    )

    df = add_breadth(df)

    print()
    print(
        "range:",
        df["time"].min(),
        "->",
        df["time"].max(),
    )
    print(
        "tickers:",
        df["ticker"].nunique()
    )

    print("STRATEGY LAB — MEMORY SAFE MODE")


    rows = []

    for horizon in HORIZONS:
        print()
        print("=" * 120)
        print(f"HORIZON {horizon} MIN")
        print("=" * 120)

        hdf = add_future_prices(
            df,
            horizon,
        )

        valid = hdf[
            (hdf["time"] >= VALID_FROM)
            & (hdf["time"] < VALID_TO)
        ].copy()

        forward = hdf[
            hdf["time"] >= FORWARD_FROM
        ].copy()

        print(
            "VALID rows:",
            f"{len(valid):,}",
            "| future:",
            f"{valid[f'exit_mid_{horizon}'].notna().sum():,}",
        )

        print(
            "FORWARD rows:",
            f"{len(forward):,}",
            "| future:",
            f"{forward[f'exit_mid_{horizon}'].notna().sum():,}",
        )

        configs = list(
            itertools.product(
                STRATEGIES,
                [horizon],
                THRESHOLDS,
                BREADTH_LIMITS,
                DIRECTIONS,
                TOP_NS,
            )
        )

        for n, (
            strategy,
            _h,
            threshold,
            breadth_cfg,
            direction,
            top_n,
        ) in enumerate(configs, start=1):

            breadth_name, breadth_limit = breadth_cfg

            score_col = (
                "score_mr"
                if strategy == "MR"
                else "score_mom"
            )

            cv = select_candidates(
                valid,
                score_col,
                threshold,
                direction,
                breadth_limit,
                top_n,
                horizon,
            )

            ev = evaluate(
                cv,
                horizon,
                ROUND_TRIP_FEE_BP,
            )

            cf = select_candidates(
                forward,
                score_col,
                threshold,
                direction,
                breadth_limit,
                top_n,
                horizon,
            )

            ef10 = evaluate(
                cf,
                horizon,
                10.0,
            )

            ef125 = evaluate(
                cf,
                horizon,
                12.5,
            )

            ef15 = evaluate(
                cf,
                horizon,
                15.0,
            )

            rows.append({
                "strategy": strategy,
                "horizon": horizon,
                "threshold": threshold,
                "breadth": breadth_name,
                "direction": direction,
                "top_n": top_n,

                "val_trades": ev["trades"],
                "val_net_bp": ev["net_bp"],
                "val_mean_bp": ev["mean_bp"],
                "val_median_bp": ev["median_bp"],
                "val_win_rate": ev["win_rate"],
                "val_pos_days": ev["positive_days"],
                "val_days": ev["days"],
                "val_worst_day_bp": ev["worst_day_bp"],

                "fwd_trades": ef10["trades"],
                "fwd_net_10bp": ef10["net_bp"],
                "fwd_mean_10bp": ef10["mean_bp"],
                "fwd_median_10bp": ef10["median_bp"],
                "fwd_win_rate_10bp": ef10["win_rate"],
                "fwd_pos_days_10bp": ef10["positive_days"],
                "fwd_days": ef10["days"],
                "fwd_worst_day_10bp": ef10["worst_day_bp"],

                "fwd_net_12_5bp": ef125["net_bp"],
                "fwd_net_15bp": ef15["net_bp"],
            })

            if n % 100 == 0 or n == len(configs):
                print(
                    f"horizon {horizon}: "
                    f"{n}/{len(configs)}"
                )

        del hdf
        del valid
        del forward

        import gc
        gc.collect()

        print(
            f"FINISHED HORIZON {horizon}"
        )

    res = pd.DataFrame(rows)

    res.to_csv(
        OUT_ALL,
        index=False,
    )

    # ========================================================
    # SELECTION ТОЛЬКО ПО VALIDATION
    # ========================================================

    eligible = res[
        (res["val_trades"] >= 20)
        & (res["val_days"] >= 4)
    ].copy()

    if eligible.empty:
        eligible = res[
            res["val_trades"] >= 10
        ].copy()

    # Не используем forward для выбора.
    eligible["val_score"] = (
        eligible["val_mean_bp"]
        + 0.02 * eligible["val_net_bp"]
        + 0.50 * (
            eligible["val_pos_days"]
            / eligible["val_days"].replace(
                0,
                np.nan,
            )
        )
    )

    top = (
        eligible.sort_values(
            [
                "val_score",
                "val_net_bp",
            ],
            ascending=False,
        )
        .head(20)
        .copy()
    )

    top.to_csv(
        OUT_TOP,
        index=False,
    )

    print()
    print("=" * 120)
    print(
        "TOP-20 — ВЫБРАНЫ ТОЛЬКО ПО VALIDATION"
    )
    print("=" * 120)

    show_cols = [
        "strategy",
        "horizon",
        "threshold",
        "breadth",
        "direction",
        "top_n",

        "val_trades",
        "val_net_bp",
        "val_mean_bp",
        "val_pos_days",

        "fwd_trades",
        "fwd_net_10bp",
        "fwd_mean_10bp",
        "fwd_win_rate_10bp",
        "fwd_pos_days_10bp",
        "fwd_days",

        "fwd_net_12_5bp",
        "fwd_net_15bp",
    ]

    print(
        top[show_cols].to_string(
            index=False,
            formatters={
                "val_net_bp":
                    "{:+.1f}".format,
                "val_mean_bp":
                    "{:+.2f}".format,

                "fwd_net_10bp":
                    "{:+.1f}".format,
                "fwd_mean_10bp":
                    "{:+.2f}".format,
                "fwd_win_rate_10bp":
                    "{:.1f}".format,

                "fwd_net_12_5bp":
                    "{:+.1f}".format,
                "fwd_net_15bp":
                    "{:+.1f}".format,
            }
        )
    )

    print()
    print("=" * 120)
    print("FORWARD SURVIVORS @ 10 BP")
    print("=" * 120)

    survivors = top[
        (top["fwd_trades"] >= 10)
        & (top["fwd_net_10bp"] > 0)
    ].sort_values(
        "fwd_net_10bp",
        ascending=False,
    )

    if survivors.empty:
        print(
            "Ни одна из validation-selected "
            "конфигураций не выжила на forward @10bp."
        )
    else:
        print(
            survivors[
                show_cols
            ].to_string(
                index=False,
                formatters={
                    "val_net_bp":
                        "{:+.1f}".format,
                    "val_mean_bp":
                        "{:+.2f}".format,
                    "fwd_net_10bp":
                        "{:+.1f}".format,
                    "fwd_mean_10bp":
                        "{:+.2f}".format,
                    "fwd_win_rate_10bp":
                        "{:.1f}".format,
                    "fwd_net_12_5bp":
                        "{:+.1f}".format,
                    "fwd_net_15bp":
                        "{:+.1f}".format,
                }
            )
        )

    print()
    print("Saved:")
    print(OUT_ALL)
    print(OUT_TOP)


if __name__ == "__main__":
    main()

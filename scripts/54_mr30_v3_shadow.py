from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd


ROOT = Path("/root/rl-trading-tbank")
STATE = ROOT / "state"

CONFIG = STATE / "mr30_v3_frozen.json"
FEATURES = STATE / "shadow_observer_features.csv"

TRADES_FILE = STATE / "mr30_v3_shadow_trades.csv"
SIGNALS_FILE = STATE / "mr30_v3_shadow_signals.csv"
STATUS_FILE = STATE / "mr30_v3_shadow_status.json"

POLL_SECONDS = 60


def load_config():
    return json.loads(
        CONFIG.read_text(encoding="utf-8")
    )


def load_features():
    if not FEATURES.exists():
        return pd.DataFrame()

    df = pd.read_csv(FEATURES)

    if df.empty:
        return df

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
            "ofi",
            "tfi",
        ]
    )

    df = (
        df.sort_values(["ticker", "time"])
        .drop_duplicates(
            ["ticker", "time"],
            keep="last",
        )
    )

    return df


def exact_return(g, minutes):
    idx = (
        g.set_index("time")["mid"]
        .sort_index()
    )

    past_times = (
        g["time"]
        - pd.Timedelta(minutes=minutes)
    )

    past = idx.reindex(
        past_times
    ).to_numpy()

    return (
        g["mid"].to_numpy() / past - 1.0
    )


def add_live_features(df):
    parts = []

    for ticker, g in df.groupby(
        "ticker",
        sort=False,
    ):
        g = (
            g.sort_values("time")
            .copy()
        )

        g["ret_1m_live"] = exact_return(
            g,
            1,
        )

        g["ret_5m_live"] = exact_return(
            g,
            5,
        )

        parts.append(g)

    x = pd.concat(
        parts,
        ignore_index=True,
    )

    # Cross-sectional breadth at each completed minute.
    market = (
        x.groupby("time")
        .agg(
            breadth_up=(
                "ret_5m_live",
                lambda s:
                (s.dropna() > 0).mean()
                if s.notna().any()
                else np.nan
            ),
            breadth_down=(
                "ret_5m_live",
                lambda s:
                (s.dropna() < 0).mean()
                if s.notna().any()
                else np.nan
            ),
            n_market=(
                "ret_5m_live",
                lambda s:
                s.notna().sum()
            ),
        )
        .reset_index()
    )

    market["breadth_extreme"] = (
        market[
            ["breadth_up", "breadth_down"]
        ].max(axis=1)
    )

    x = x.merge(
        market,
        on="time",
        how="left",
    )

    sign5 = np.sign(
        x["ret_5m_live"]
    )

    x["ofi_confirm"] = (
        (
            np.sign(x["ofi"])
            == sign5
        )
        & (sign5 != 0)
    ).astype(int)

    x["tfi_confirm"] = (
        (
            np.sign(x["tfi"])
            == sign5
        )
        & (sign5 != 0)
    ).astype(int)

    x["confirm_count"] = (
        x["ofi_confirm"]
        + x["tfi_confirm"]
    )

    return x


def add_score(x, cfg):
    z = x.copy()

    norm = cfg["normalization"]

    z5 = (
        z["ret_5m_live"]
        - norm["ret_5m_mean"]
    ) / norm["ret_5m_std"]

    z1 = (
        z["ret_1m_live"]
        - norm["ret_1m_mean"]
    ) / norm["ret_1m_std"]

    z["mr_score"] = (
        -0.75 * z5
        -0.25 * z1
    )

    return z


def arm_allowed(row, arm, cfg):
    if arm == "BASE":
        return True

    if arm == "NO_BROAD_TREND":
        lim = (
            cfg["arms"][arm]
            ["breadth_max"]
        )

        return (
            np.isfinite(
                row["breadth_extreme"]
            )
            and row["breadth_extreme"] <= lim
        )

    if arm == "NO_FLOW_CONFIRM":
        lim = (
            cfg["arms"][arm]
            ["max_flow_confirmations"]
        )

        return (
            row["confirm_count"] <= lim
        )

    return False


def read_existing(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        x = pd.read_csv(path)

        if "entry_time" in x.columns:
            x["entry_time"] = pd.to_datetime(
                x["entry_time"],
                utc=True,
                errors="coerce",
            )

        if "exit_time" in x.columns:
            x["exit_time"] = pd.to_datetime(
                x["exit_time"],
                utc=True,
                errors="coerce",
            )

        return x

    except Exception:
        return pd.DataFrame()


def append_csv(path, rows):
    if not rows:
        return

    df = pd.DataFrame(rows)

    df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def existing_signal_keys():
    x = read_existing(SIGNALS_FILE)

    if x.empty:
        return set()

    return set(
        zip(
            x["arm"].astype(str),
            x["ticker"].astype(str),
            x["entry_time"]
            .astype(str),
        )
    )


def existing_trade_keys():
    x = read_existing(TRADES_FILE)

    if x.empty:
        return set()

    return set(
        zip(
            x["arm"].astype(str),
            x["ticker"].astype(str),
            x["entry_time"]
            .astype(str),
        )
    )


def busy_until_by_arm():
    x = read_existing(SIGNALS_FILE)

    busy = {}

    if x.empty:
        return busy

    for _, r in x.iterrows():
        if pd.isna(r["entry_time"]):
            continue

        k = (
            str(r["arm"]),
            str(r["ticker"]),
        )

        end = (
            r["entry_time"]
            + pd.Timedelta(minutes=30)
        )

        if (
            k not in busy
            or end > busy[k]
        ):
            busy[k] = end

    return busy


def generate_new_signals(x, cfg):
    fresh_start = pd.Timestamp(
        cfg["fresh_forward_start"],
        tz="UTC",
    )

    low = cfg["alpha"][
        "short_if_score_le"
    ]

    high = cfg["alpha"][
        "long_if_score_ge"
    ]

    max_spread = cfg["alpha"][
        "max_entry_spread_bp"
    ]

    arms = [
        name
        for name, d
        in cfg["arms"].items()
        if d.get("enabled", False)
    ]

    existing = existing_signal_keys()
    busy = busy_until_by_arm()

    rows = []

    z = x[
        x["time"] >= fresh_start
    ].copy()

    z = z.dropna(
        subset=[
            "ret_1m_live",
            "ret_5m_live",
            "mr_score",
        ]
    )

    z = z[
        z["spread_bp"]
        <= max_spread
    ]

    z = z.sort_values(
        ["time", "ticker"]
    )

    for _, r in z.iterrows():

        score = r["mr_score"]

        if score <= low:
            side = -1

        elif score >= high:
            side = 1

        else:
            continue

        for arm in arms:

            if not arm_allowed(
                r,
                arm,
                cfg,
            ):
                continue

            key = (
                arm,
                str(r["ticker"]),
                str(r["time"]),
            )

            if key in existing:
                continue

            busy_key = (
                arm,
                str(r["ticker"]),
            )

            if (
                busy_key in busy
                and r["time"] < busy[busy_key]
            ):
                continue

            exit_due = (
                r["time"]
                + pd.Timedelta(minutes=30)
            )

            rows.append(
                {
                    "arm": arm,
                    "ticker": r["ticker"],
                    "entry_time": r["time"],
                    "exit_due": exit_due,
                    "side": side,
                    "score": score,
                    "entry_mid": r["mid"],
                    "entry_spread_bp": r["spread_bp"],
                    "ret_1m": r["ret_1m_live"],
                    "ret_5m": r["ret_5m_live"],
                    "ofi": r["ofi"],
                    "tfi": r["tfi"],
                    "breadth_extreme": r["breadth_extreme"],
                    "confirm_count": r["confirm_count"],
                    "status": "OPEN",
                }
            )

            existing.add(key)
            busy[busy_key] = exit_due

    append_csv(
        SIGNALS_FILE,
        rows,
    )

    return len(rows)


def close_due_signals(x, cfg):
    sig = read_existing(
        SIGNALS_FILE
    )

    if sig.empty:
        return 0

    already = existing_trade_keys()

    lookup = (
        x.set_index(
            ["ticker", "time"]
        )
        .sort_index()
    )

    rows = []

    for _, r in sig.iterrows():

        key = (
            str(r["arm"]),
            str(r["ticker"]),
            str(r["entry_time"]),
        )

        if key in already:
            continue

        exit_time = (
            r["entry_time"]
            + pd.Timedelta(minutes=30)
        )

        lk = (
            str(r["ticker"]),
            exit_time,
        )

        if lk not in lookup.index:
            continue

        q = lookup.loc[lk]

        # Safety in case duplicate index somehow survives.
        if isinstance(q, pd.DataFrame):
            q = q.iloc[-1]

        exit_mid = float(
            q["mid"]
        )

        exit_spread = float(
            q["spread_bp"]
        )

        entry_mid = float(
            r["entry_mid"]
        )

        side = int(
            r["side"]
        )

        gross_bp = (
            side
            * (
                exit_mid / entry_mid
                - 1.0
            )
            * 10000.0
        )

        fee = cfg[
            "execution_assumption"
        ]["roundtrip_fee_bp"]

        cost_bp = (
            float(r["entry_spread_bp"]) / 2.0
            + exit_spread / 2.0
            + fee
        )

        net_bp = (
            gross_bp
            - cost_bp
        )

        rows.append(
            {
                "arm": r["arm"],
                "ticker": r["ticker"],
                "entry_time": r["entry_time"],
                "exit_time": exit_time,
                "side": side,
                "score": r["score"],
                "entry_mid": entry_mid,
                "exit_mid": exit_mid,
                "entry_spread_bp": r["entry_spread_bp"],
                "exit_spread_bp": exit_spread,
                "gross_bp": gross_bp,
                "cost_bp": cost_bp,
                "net_bp": net_bp,
                "breadth_extreme": r["breadth_extreme"],
                "confirm_count": r["confirm_count"],
            }
        )

        already.add(key)

    append_csv(
        TRADES_FILE,
        rows,
    )

    return len(rows)


def status(cfg, x):
    sig = read_existing(
        SIGNALS_FILE
    )

    tr = read_existing(
        TRADES_FILE
    )

    out = {
        "version": cfg["version"],
        "mode": "SHADOW_ONLY",
        "fresh_forward_start": cfg[
            "fresh_forward_start"
        ],
        "latest_feature_time": (
            str(x["time"].max())
            if not x.empty
            else None
        ),
        "signals": int(len(sig)),
        "closed_trades": int(len(tr)),
        "arms": {},
    }

    if not tr.empty:
        for arm, g in tr.groupby("arm"):
            out["arms"][str(arm)] = {
                "trades": int(len(g)),
                "net_mean_bp": float(
                    g["net_bp"].mean()
                ),
                "net_sum_bp": float(
                    g["net_bp"].sum()
                ),
                "hit_rate": float(
                    (g["net_bp"] > 0).mean()
                ),
            }

    STATUS_FILE.write_text(
        json.dumps(
            out,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return out


def run_once():
    cfg = load_config()

    df = load_features()

    if df.empty:
        print(
            "NO OBSERVER FEATURES YET",
            flush=True,
        )
        return

    x = add_live_features(df)
    x = add_score(x, cfg)

    new_signals = generate_new_signals(
        x,
        cfg,
    )

    closed = close_due_signals(
        x,
        cfg,
    )

    st = status(
        cfg,
        x,
    )

    print(
        pd.Timestamp.now(tz="UTC"),
        "latest=",
        st["latest_feature_time"],
        "new_signals=",
        new_signals,
        "new_closed=",
        closed,
        "signals_total=",
        st["signals"],
        "closed_total=",
        st["closed_trades"],
        flush=True,
    )


def main():
    print("=" * 110)
    print("MR30 V3 LIVE SHADOW")
    print("=" * 110)
    print("NO BROKER ORDERS")
    print("NO RETRAINING")
    print("FROZEN CONFIG:", CONFIG)
    print()

    while True:
        try:
            run_once()

        except KeyboardInterrupt:
            raise

        except Exception as e:
            print(
                "ERROR:",
                repr(e),
                flush=True,
            )

        time.sleep(
            POLL_SECONDS
        )


if __name__ == "__main__":
    main()

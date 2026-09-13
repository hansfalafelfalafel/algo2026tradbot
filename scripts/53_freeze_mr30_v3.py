from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path("/root/rl-trading-tbank")
STATE = ROOT / "state"
DATA = STATE / "ofi_dataset_h5.pkl"

HORIZON = 30
Q = 0.05

# Freeze strictly before September.
CUTOFF = pd.Timestamp(
    "2026-09-01 00:00:00",
    tz="UTC"
)

TRAIN_DAYS = 20
VALID_DAYS = 5


def ensure_returns(df):
    parts = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("time").copy()

        if "ret_1m" not in g:
            g["ret_1m"] = (
                g["mid"] /
                g["mid"].shift(1)
                - 1.0
            )

        if "ret_5m" not in g:
            g["ret_5m"] = (
                g["mid"] /
                g["mid"].shift(5)
                - 1.0
            )

        parts.append(g)

    return pd.concat(
        parts,
        ignore_index=True
    )


def score(frame, stats):
    z5 = (
        frame["ret_5m"]
        - stats["ret_5m_mean"]
    ) / stats["ret_5m_std"]

    z1 = (
        frame["ret_1m"]
        - stats["ret_1m_mean"]
    ) / stats["ret_1m_std"]

    return (
        -0.75 * z5
        -0.25 * z1
    )


df = pd.read_pickle(DATA).copy()

df["time"] = pd.to_datetime(
    df["time"],
    utc=True
)

df = df[
    df["time"] < CUTOFF
].copy()

df = ensure_returns(df)

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

df["date"] = (
    df["time"]
    .dt.strftime("%Y-%m-%d")
)

dates = sorted(
    df["date"].unique()
)

if len(dates) < TRAIN_DAYS + VALID_DAYS:
    raise RuntimeError(
        "Not enough historical dates"
    )

# Last completely historical 20/5 window.
train_dates = dates[
    -(TRAIN_DAYS + VALID_DAYS):
    -VALID_DAYS
]

valid_dates = dates[
    -VALID_DAYS:
]

train = df[
    df["date"].isin(train_dates)
].copy()

valid = df[
    df["date"].isin(valid_dates)
].copy()

stats = {
    "ret_5m_mean": float(
        train["ret_5m"].mean()
    ),
    "ret_5m_std": float(
        train["ret_5m"].std()
    ),
    "ret_1m_mean": float(
        train["ret_1m"].mean()
    ),
    "ret_1m_std": float(
        train["ret_1m"].std()
    ),
}

for k in [
    "ret_5m_std",
    "ret_1m_std",
]:
    if (
        not np.isfinite(stats[k])
        or stats[k] <= 0
    ):
        raise RuntimeError(
            f"Invalid std: {k}"
        )

valid["mr_score"] = score(
    valid,
    stats
)

low_cut = float(
    valid["mr_score"].quantile(Q)
)

high_cut = float(
    valid["mr_score"].quantile(1 - Q)
)

cfg = {
    "version": "MR30_V3_FROZEN_20260906",

    "status": "SHADOW_ONLY",

    "frozen_at": "2026-09-06",

    "historical_cutoff": "2026-09-01T00:00:00Z",

    "fresh_forward_start": "2026-09-07",

    "alpha": {
        "horizon_minutes": 30,

        "score": (
            "-0.75*z(ret_5m)"
            "-0.25*z(ret_1m)"
        ),

        "tail_q": Q,

        "short_if_score_le": low_cut,
        "long_if_score_ge": high_cut,

        "max_entry_spread_bp": 2.0,

        "non_overlap_minutes_per_ticker": 30,
    },

    "normalization": stats,

    "training_dates": list(
        map(str, train_dates)
    ),

    "validation_dates": list(
        map(str, valid_dates)
    ),

    "arms": {
        "BASE": {
            "enabled": True,
            "rules": [],
        },

        "NO_BROAD_TREND": {
            "enabled": True,
            "rules": [
                "breadth_extreme <= 0.70"
            ],
            "breadth_max": 0.70,
        },

        "NO_FLOW_CONFIRM": {
            "enabled": True,
            "rules": [
                "ofi_confirm + tfi_confirm <= 1"
            ],
            "max_flow_confirmations": 1,
        },
    },

    "execution_assumption": {
        "roundtrip_fee_bp": 1.0,
        "cost_formula": (
            "entry_half_spread"
            "+exit_half_spread"
            "+1bp_fee"
        ),
    },

    "research_reference": {
        "base": {
            "trades": 1809,
            "net_mean_bp": 2.0166,
            "positive_days": 5,
            "test_days": 11,
        },

        "no_broad_trend": {
            "trades": 514,
            "net_mean_bp": 2.5999,
            "positive_days": 9,
            "test_days": 11,
        },

        "no_flow_confirm": {
            "trades": 987,
            "net_mean_bp": 1.8221,
            "positive_days": 7,
            "test_days": 11,
        },
    },

    "forward_rules": {
        "do_not_retrain": True,
        "do_not_change_thresholds": True,
        "do_not_blacklist_tickers": True,
        "do_not_switch_long_short": True,
        "no_real_orders": True,
    },
}

out = (
    STATE /
    "mr30_v3_frozen.json"
)

out.write_text(
    json.dumps(
        cfg,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("=" * 100)
print("MR30 V3 FROZEN")
print("=" * 100)

print()
print("train:")
print(
    train_dates[0],
    "->",
    train_dates[-1]
)

print()
print("validation:")
print(
    valid_dates[0],
    "->",
    valid_dates[-1]
)

print()
print("normalization:")
for k, v in stats.items():
    print(
        f"  {k:16s} = {v:.10f}"
    )

print()
print("frozen cuts:")
print(
    "  short <=",
    f"{low_cut:.6f}"
)
print(
    "  long  >=",
    f"{high_cut:.6f}"
)

print()
print("arms:")
print("  BASE")
print("  NO_BROAD_TREND")
print("  NO_FLOW_CONFIRM")

print()
print("fresh forward starts: 2026-09-07")
print("status: SHADOW ONLY")
print()
print("saved:", out)

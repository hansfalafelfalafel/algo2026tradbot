from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/rl-trading-tbank")

SRC = ROOT / "state" / "triple_barrier_all.csv"
OUT = ROOT / "state" / "triple_barrier_cost_frontier.csv"

df = pd.read_csv(SRC)

WINDOWS = ["W0", "W1", "W2", "W3", "W4"]
COSTS = [0, 2, 4, 6, 8, 10, 12]

# ------------------------------------------------------------
# В triple_barrier_all результаты уже посчитаны
# при COMMISSION_BP = 10.
#
# Поэтому для другой комиссии:
#
# net(new) = net(10bp) + (10 - new_cost) * trades
#
# Спред при этом остаётся тем же.
# ------------------------------------------------------------

for w in WINDOWS:
    df[f"{w}_trades"] = pd.to_numeric(
        df[f"{w}_trades"],
        errors="coerce",
    ).fillna(0)

    df[f"{w}_net"] = pd.to_numeric(
        df[f"{w}_net"],
        errors="coerce",
    )

df["pre_trades"] = pd.to_numeric(
    df["pre_trades"],
    errors="coerce",
).fillna(0)

df["pre_net"] = pd.to_numeric(
    df["pre_net"],
    errors="coerce",
)

df["pre_without_best_day"] = pd.to_numeric(
    df["pre_without_best_day"],
    errors="coerce",
)

df["pre_without_best_ticker"] = pd.to_numeric(
    df["pre_without_best_ticker"],
    errors="coerce",
)

# ============================================================
# COST FRONTIER
# ============================================================

rows = []

for _, r in df.iterrows():

    base = {
        "score_cut": r["score_cut"],
        "breadth_cut": r["breadth_cut"],
        "spread_cut": r["spread_cut"],
        "hold_min": r["hold_min"],
        "tp_bp": r["tp_bp"],
        "sl_bp": r["sl_bp"],
        "pre_trades": int(r["pre_trades"]),
    }

    for cost in COSTS:

        delta = 10.0 - cost

        window_nets = []

        for w in WINDOWS:
            n = float(r[f"{w}_trades"])

            old_net = r[f"{w}_net"]

            if pd.isna(old_net):
                new_net = np.nan
            else:
                new_net = (
                    float(old_net)
                    + delta * n
                )

            window_nets.append(new_net)

        pre_net = (
            float(r["pre_net"])
            + delta * float(r["pre_trades"])
        )

        positive_windows = sum(
            1
            for x in window_nets
            if pd.notna(x) and x > 0
        )

        usable_windows = sum(
            1
            for x in window_nets
            if pd.notna(x)
        )

        # Для robustness без лучшего дня/тикера:
        # добавление комиссии происходит на каждой сделке.
        #
        # Точно пересчитать leave-one-out без trade journal
        # нельзя, поэтому здесь сохраняем исходные 10bp
        # robustness metrics как reference.
        row = {
            **base,
            "commission_bp": cost,

            "W0_net": window_nets[0],
            "W1_net": window_nets[1],
            "W2_net": window_nets[2],
            "W3_net": window_nets[3],
            "W4_net": window_nets[4],

            "positive_windows": positive_windows,
            "usable_windows": usable_windows,

            "pre_net": pre_net,
            "pre_mean": (
                pre_net / r["pre_trades"]
                if r["pre_trades"] > 0
                else np.nan
            ),

            "robust_day_10bp":
                r["pre_without_best_day"],

            "robust_ticker_10bp":
                r["pre_without_best_ticker"],
        }

        rows.append(row)

out = pd.DataFrame(rows)

out.to_csv(
    OUT,
    index=False,
)

# ============================================================
# BREAK-EVEN COMMISSION
#
# pre_net at 10bp:
#
# 0 = net10 + (10 - c) * N
#
# c = 10 + net10 / N
# ============================================================

base = df[
    df["pre_trades"] >= 40
].copy()

base["break_even_commission_bp"] = (
    10.0
    + base["pre_net"]
    / base["pre_trades"]
)

base["grossish_edge_bp"] = (
    base["pre_net"]
    / base["pre_trades"]
    + 10.0
)

print("=" * 120)
print("BEST BREAK-EVEN COST")
print("=" * 120)

show = [
    "score_cut",
    "breadth_cut",
    "spread_cut",
    "hold_min",
    "tp_bp",
    "sl_bp",
    "pre_trades",
    "pre_net",
    "grossish_edge_bp",
    "break_even_commission_bp",
    "pre_positive_windows",
    "pre_without_best_day",
    "pre_without_best_ticker",
]

print(
    base.sort_values(
        "break_even_commission_bp",
        ascending=False,
    )[show]
    .head(25)
    .to_string(
        index=False,
        formatters={
            "pre_net": "{:+.1f}".format,
            "grossish_edge_bp": "{:+.2f}".format,
            "break_even_commission_bp": "{:.2f}".format,
            "pre_without_best_day": "{:+.1f}".format,
            "pre_without_best_ticker": "{:+.1f}".format,
        },
    )
)

print()
print("=" * 120)
print("ROBUST BY COST")
print("=" * 120)

for cost in COSTS:
    z = out[
        (out["commission_bp"] == cost)
        & (out["pre_trades"] >= 40)
        & (out["usable_windows"] == 5)
        & (out["positive_windows"] >= 4)
        & (out["pre_net"] > 0)
        & (out["pre_mean"] > 0)
    ].copy()

    print(
        f"commission {cost:>2} bp:"
        f" {len(z):>3} configs"
    )

    if not z.empty:
        best = z.sort_values(
            "pre_net",
            ascending=False,
        ).iloc[0]

        print(
            "   BEST:",
            f"score={best.score_cut}",
            f"breadth={best.breadth_cut}",
            f"spread={best.spread_cut}",
            f"hold={int(best.hold_min)}",
            f"TP={best.tp_bp}",
            f"SL={best.sl_bp}",
            f"trades={int(best.pre_trades)}",
            f"net={best.pre_net:+.1f}bp",
            f"mean={best.pre_mean:+.2f}bp",
            f"windows={int(best.positive_windows)}/5",
        )

print()
print("=" * 120)
print("TOP @ 10 BP — EVEN IF NOT STRICT")
print("=" * 120)

z10 = out[
    (out["commission_bp"] == 10)
    & (out["pre_trades"] >= 40)
].copy()

print(
    z10.sort_values(
        ["positive_windows", "pre_net"],
        ascending=False,
    )
    .head(20)
    .to_string(
        index=False,
        formatters={
            "pre_net": "{:+.1f}".format,
            "pre_mean": "{:+.2f}".format,
        },
    )
)

print()
print("Saved:", OUT)

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


ROOT = Path("/root/rl-trading-tbank")
STATE = ROOT / "state"

sys.path.insert(0, str(ROOT))

DATA_FILE = STATE / "ofi_dataset_h5.pkl"
WF_FILE = STATE / "net_edge_v2_walkforward.csv"
SOURCE_46 = ROOT / "scripts/46_net_edge_v2_research.py"

TAIL_LOSS_BP = -25.0

OUT_TRADES = STATE / "tail_risk_oos_trades.csv"
OUT_SUMMARY = STATE / "tail_risk_oos_summary.csv"
OUT_DECILES = STATE / "tail_risk_oos_deciles.csv"


def load_v2():
    spec = importlib.util.spec_from_file_location(
        "v2_46",
        SOURCE_46,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot import script 46"
        )

    m = importlib.util.module_from_spec(spec)
    sys.modules["v2_46"] = m
    spec.loader.exec_module(m)

    return m


def safe_auc(y, p):
    y = pd.Series(y)
    p = pd.Series(p)

    mask = (
        y.notna()
        & p.notna()
    )

    y = y[mask]
    p = p[mask]

    if (
        len(y) == 0
        or y.nunique() < 2
    ):
        return np.nan

    return float(
        roc_auc_score(y, p)
    )


def add_risk_features(df):
    """
    Только causal features:
    текущая минута + прошлое.
    Никаких future_*.
    """

    rows = []

    for ticker, g in df.groupby(
        "ticker",
        sort=False,
    ):
        g = (
            g.sort_values("time")
            .copy()
        )

        # Current return magnitude.
        g["abs_ret_1m_bp"] = (
            g["ret_1m"].abs()
            * 10000.0
        )

        # Realized volatility from past/current returns.
        for w in [5, 15, 30]:
            g[f"vol_{w}m_bp"] = (
                g["ret_1m"]
                .rolling(
                    w,
                    min_periods=max(
                        3,
                        w // 3,
                    ),
                )
                .std()
                * 10000.0
            )

        # 5m momentum magnitude.
        if "ret_5m" in g.columns:
            g["abs_ret_5m_bp"] = (
                g["ret_5m"].abs()
                * 10000.0
            )
        else:
            g["abs_ret_5m_bp"] = (
                g["ret_1m"]
                .rolling(
                    5,
                    min_periods=3,
                )
                .sum()
                .abs()
                * 10000.0
            )

        # Spread state.
        g["spread_change_1m"] = (
            g["spread_bp"].diff()
        )

        g["spread_mean_15"] = (
            g["spread_bp"]
            .rolling(
                15,
                min_periods=5,
            )
            .mean()
        )

        g["spread_ratio_15"] = (
            g["spread_bp"]
            / g["spread_mean_15"]
            .replace(0, np.nan)
        )

        g["spread_std_15"] = (
            g["spread_bp"]
            .rolling(
                15,
                min_periods=5,
            )
            .std()
        )

        # Flow instability.
        g["ofi_abs"] = (
            g["ofi"].abs()
        )

        g["tfi_abs"] = (
            g["tfi"].abs()
        )

        g["ofi_std_15"] = (
            g["ofi"]
            .rolling(
                15,
                min_periods=5,
            )
            .std()
        )

        g["tfi_std_15"] = (
            g["tfi"]
            .rolling(
                15,
                min_periods=5,
            )
            .std()
        )

        # Disagreement.
        g["imb_disagreement"] = (
            g["imb1"]
            - g["imb5"]
        ).abs()

        g["flow_disagreement"] = (
            (
                np.sign(g["ofi"])
                != np.sign(g["tfi"])
            )
            & (g["ofi"] != 0)
            & (g["tfi"] != 0)
        ).astype(float)

        g["micro_abs"] = (
            g["micro_dev"].abs()
        )

        # Time of day encoded cyclically.
        minute_of_day = (
            g["time"].dt.hour * 60
            + g["time"].dt.minute
        )

        g["tod_sin"] = np.sin(
            2 * np.pi
            * minute_of_day
            / 1440.0
        )

        g["tod_cos"] = np.cos(
            2 * np.pi
            * minute_of_day
            / 1440.0
        )

        rows.append(g)

    return pd.concat(
        rows,
        ignore_index=True,
    )


RISK_FEATURES = [
    "spread_bp",
    "spread_change_1m",
    "spread_ratio_15",
    "spread_std_15",

    "abs_ret_1m_bp",
    "abs_ret_5m_bp",
    "vol_5m_bp",
    "vol_15m_bp",
    "vol_30m_bp",

    "ofi",
    "tfi",
    "ofi_abs",
    "tfi_abs",
    "ofi_std_15",
    "tfi_std_15",

    "imb1",
    "imb5",
    "imb_disagreement",

    "micro_dev",
    "micro_abs",

    "flow_disagreement",

    "tod_sin",
    "tod_cos",
]


def fit_tail_models(train):
    """
    Catastrophic outcome if realistic net <= -25 bp
    for the hypothetical LONG / SHORT side.
    """

    x = train.dropna(
        subset=RISK_FEATURES
    ).copy()

    long_net = (
        x["future_ret_bp"]
        - x["execution_cost_bp"]
    )

    short_net = (
        -x["future_ret_bp"]
        - x["execution_cost_bp"]
    )

    y_long = (
        long_net
        <= TAIL_LOSS_BP
    ).astype(int)

    y_short = (
        short_net
        <= TAIL_LOSS_BP
    ).astype(int)

    if (
        y_long.nunique() < 2
        or y_short.nunique() < 2
    ):
        return None

    long_model = (
        HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=101,
        )
    )

    short_model = (
        HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=102,
        )
    )

    long_model.fit(
        x[RISK_FEATURES],
        y_long,
    )

    short_model.fit(
        x[RISK_FEATURES],
        y_short,
    )

    return (
        long_model,
        short_model,
        float(y_long.mean()),
        float(y_short.mean()),
    )


def main():
    m = load_v2()

    print("=" * 110)
    print("TAIL RISK RESEARCH — HISTORICAL OOS ONLY")
    print("=" * 110)

    print(
        "Catastrophic definition:",
        f"net <= {TAIL_LOSS_BP:.1f} bp",
    )

    wf = pd.read_csv(
        WF_FILE
    )

    # Only windows that historical validation
    # actually allowed.
    wf = wf[
        wf["selected_passed"] == True
    ].copy()

    if wf.empty:
        raise SystemExit(
            "No selected_passed=True windows"
        )

    raw = pd.read_pickle(
        DATA_FILE
    ).copy()

    raw["time"] = pd.to_datetime(
        raw["time"],
        utc=True,
        errors="coerce",
    )

    raw = raw.dropna(
        subset=[
            "time",
            "ticker",
        ]
    )

    raw["date"] = (
        raw["time"]
        .dt.strftime("%Y-%m-%d")
    )

    # Add only causal risk features first.
    raw = add_risk_features(
        raw
    )

    # Then exact future and execution labels.
    df = m.add_exact_future(
        raw
    )

    df = m.add_execution_fields(
        df
    )

    df = df[
        df["spread_bp"]
        <= m.MAX_ENTRY_SPREAD_BP
    ].copy()

    required = (
        list(m.FEATURES_BASIC)
        + RISK_FEATURES
        + [
            "future_mid",
            "future_spread_bp",
            "future_ret_bp",
            "expected_cost_bp",
            "execution_cost_bp",
        ]
    )

    df = df.dropna(
        subset=required
    )

    all_test = []
    window_rows = []

    for _, cfg in wf.iterrows():
        test_date = str(
            cfg["test_date"]
        )

        train_start = str(
            cfg["train_start"]
        )

        train_end = str(
            cfg["train_end"]
        )

        # Reconstruct dates using the saved
        # historical window boundaries.
        unique_dates = sorted(
            df["date"].unique()
        )

        train_dates = [
            d
            for d in unique_dates
            if (
                d >= train_start
                and d <= train_end
            )
        ]

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

        # ------------------------------------------------------
        # Recreate frozen v2 candidate for THIS historical window.
        # ------------------------------------------------------

        direction_model = (
            m.fit_direction(
                train
            )
        )

        (
            long_edge_model,
            short_edge_model,
            _,
            _,
        ) = m.fit_edge_models(
            train,
            float(
                cfg["margin_bp"]
            ),
        )

        test_pred = m.predict_all(
            test,
            direction_model,
            long_edge_model,
            short_edge_model,
        )

        test_trades = (
            m.select_trades(
                test_pred,
                float(
                    cfg["short_cut"]
                ),
                float(
                    cfg["long_cut"]
                ),
                float(
                    cfg["edge_threshold"]
                ),
            )
        )

        if test_trades.empty:
            continue

        # ------------------------------------------------------
        # Fit tail models ONLY on training rows.
        # ------------------------------------------------------

        tail_fit = fit_tail_models(
            train
        )

        if tail_fit is None:
            continue

        (
            long_tail_model,
            short_tail_model,
            long_tail_rate,
            short_tail_rate,
        ) = tail_fit

        z = test_trades.dropna(
            subset=RISK_FEATURES
        ).copy()

        if z.empty:
            continue

        z["p_tail_long"] = (
            long_tail_model
            .predict_proba(
                z[RISK_FEATURES]
            )[:, 1]
        )

        z["p_tail_short"] = (
            short_tail_model
            .predict_proba(
                z[RISK_FEATURES]
            )[:, 1]
        )

        z["p_tail"] = np.where(
            z["side"] == 1,
            z["p_tail_long"],
            z["p_tail_short"],
        )

        z["actual_tail"] = (
            z["net_bp"]
            <= TAIL_LOSS_BP
        ).astype(int)

        z["wf_test_date"] = (
            test_date
        )

        all_test.append(z)

        window_rows.append(
            {
                "test_date": (
                    test_date
                ),
                "trades": len(z),
                "actual_tails": int(
                    z[
                        "actual_tail"
                    ].sum()
                ),
                "actual_tail_rate": float(
                    z[
                        "actual_tail"
                    ].mean()
                ),
                "train_long_tail_rate": (
                    long_tail_rate
                ),
                "train_short_tail_rate": (
                    short_tail_rate
                ),
                "tail_auc": safe_auc(
                    z["actual_tail"],
                    z["p_tail"],
                ),
                "p_tail_mean": float(
                    z["p_tail"].mean()
                ),
                "p_tail_tail_mean": float(
                    z.loc[
                        z["actual_tail"] == 1,
                        "p_tail",
                    ].mean()
                )
                if z["actual_tail"].sum()
                else np.nan,
                "p_tail_safe_mean": float(
                    z.loc[
                        z["actual_tail"] == 0,
                        "p_tail",
                    ].mean()
                ),
            }
        )

    if not all_test:
        raise SystemExit(
            "No OOS trades reconstructed"
        )

    trades = pd.concat(
        all_test,
        ignore_index=True,
    )

    window = pd.DataFrame(
        window_rows
    )

    print()
    print("=" * 110)
    print("WINDOW SUMMARY")
    print("=" * 110)

    print(
        window.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ----------------------------------------------------------
    # Aggregate OOS separation
    # ----------------------------------------------------------

    auc = safe_auc(
        trades["actual_tail"],
        trades["p_tail"],
    )

    print()
    print("=" * 110)
    print("AGGREGATE OOS TAIL RISK")
    print("=" * 110)

    print(
        "windows:",
        trades[
            "wf_test_date"
        ].nunique(),
    )

    print(
        "trades:",
        len(trades),
    )

    print(
        "actual catastrophic:",
        int(
            trades[
                "actual_tail"
            ].sum()
        ),
    )

    print(
        "actual catastrophic rate:",
        f"{100*trades['actual_tail'].mean():.2f}%",
    )

    print(
        "tail-risk ROC AUC:",
        (
            f"{auc:.4f}"
            if pd.notna(auc)
            else "NA"
        ),
    )

    tail_mean = (
        trades.loc[
            trades[
                "actual_tail"
            ] == 1,
            "p_tail",
        ].mean()
    )

    safe_mean = (
        trades.loc[
            trades[
                "actual_tail"
            ] == 0,
            "p_tail",
        ].mean()
    )

    print(
        "mean p_tail catastrophic:",
        f"{tail_mean:.4f}",
    )

    print(
        "mean p_tail non-catastrophic:",
        f"{safe_mean:.4f}",
    )

    # ----------------------------------------------------------
    # Risk quintiles
    # ----------------------------------------------------------

    try:
        trades["risk_quintile"] = (
            pd.qcut(
                trades["p_tail"],
                5,
                labels=False,
                duplicates="drop",
            )
        )
    except Exception:
        trades["risk_quintile"] = (
            np.nan
        )

    quint = (
        trades.groupby(
            "risk_quintile",
            dropna=False,
        )
        .agg(
            trades=("net_bp", "size"),
            p_tail_mean=(
                "p_tail",
                "mean",
            ),
            catastrophic_rate=(
                "actual_tail",
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
            hit_rate=(
                "net_bp",
                lambda s:
                (s > 0).mean(),
            ),
        )
        .reset_index()
    )

    print()
    print("=" * 110)
    print("OOS RISK QUINTILES")
    print("=" * 110)

    print(
        quint.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ----------------------------------------------------------
    # Extreme p-tail observations
    # ----------------------------------------------------------

    cols = [
        "wf_test_date",
        "time",
        "ticker",
        "side",
        "p_tail",
        "actual_tail",
        "gross_bp",
        "execution_cost_bp",
        "net_bp",
        "spread_bp",
        "abs_ret_1m_bp",
        "abs_ret_5m_bp",
        "vol_5m_bp",
        "vol_15m_bp",
        "vol_30m_bp",
        "ofi",
        "tfi",
        "imb1",
        "imb5",
    ]

    cols = [
        c
        for c in cols
        if c in trades.columns
    ]

    print()
    print("=" * 110)
    print("HIGHEST PREDICTED TAIL RISK")
    print("=" * 110)

    print(
        trades.sort_values(
            "p_tail",
            ascending=False,
        )
        .head(20)[cols]
        .to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    print()
    print("=" * 110)
    print("ACTUAL CATASTROPHIC TRADES")
    print("=" * 110)

    actual_bad = (
        trades[
            trades["actual_tail"] == 1
        ]
        .sort_values(
            "net_bp"
        )
    )

    print(
        actual_bad[cols]
        .to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ----------------------------------------------------------
    # Descriptive by side
    # ----------------------------------------------------------

    side = (
        trades.groupby("side")
        .agg(
            trades=("net_bp", "size"),
            tails=(
                "actual_tail",
                "sum",
            ),
            tail_rate=(
                "actual_tail",
                "mean",
            ),
            p_tail_mean=(
                "p_tail",
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
        )
        .reset_index()
    )

    print()
    print("=" * 110)
    print("BY SIDE")
    print("=" * 110)

    print(
        side.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    trades.to_csv(
        OUT_TRADES,
        index=False,
    )

    window.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    quint.to_csv(
        OUT_DECILES,
        index=False,
    )

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)

    print(OUT_TRADES)
    print(OUT_SUMMARY)
    print(OUT_DECILES)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No tail-risk threshold was selected."
    )

    print(
        "This experiment only asks whether catastrophic"
    )

    print(
        "losses are rankable ex ante on historical OOS trades."
    )

    print(
        "September forward data is never read."
    )


if __name__ == "__main__":
    main()

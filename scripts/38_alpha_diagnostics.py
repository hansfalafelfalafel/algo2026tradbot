from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lob.dataset import FEATURES_BASIC


CACHE = PROJECT_ROOT / "state" / "ofi_dataset_h5.pkl"

HORIZON = 15
TRAIN_DAYS = 20
VALID_DAYS = 5
MAX_SPREAD_BP = 2.0


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("time").copy()
        indexed = g.set_index("time")

        future_index = (
            g["time"]
            + pd.Timedelta(minutes=HORIZON)
        )

        future_mid = (
            indexed["mid"]
            .reindex(future_index)
            .to_numpy()
        )

        g["future_ret_bp"] = (
            future_mid / g["mid"].to_numpy() - 1
        ) * 10_000

        parts.append(g)

    return (
        pd.concat(parts, ignore_index=True)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(
            subset=FEATURES_BASIC
            + [
                "future_ret_bp",
                "time",
                "ticker",
                "spread_bp",
            ]
        )
        .sort_values(["time", "ticker"])
        .reset_index(drop=True)
    )


def main() -> None:
    base = pd.read_pickle(CACHE)
    base["time"] = pd.to_datetime(base["time"], utc=True)

    df = add_target(base)

    df = df[
        df["spread_bp"] <= MAX_SPREAD_BP
    ].copy()

    df["date"] = df["time"].dt.date
    dates = sorted(df["date"].unique())

    predictions = []

    for test_pos in range(
        TRAIN_DAYS + VALID_DAYS,
        len(dates),
    ):
        train_dates = dates[
            test_pos - VALID_DAYS - TRAIN_DAYS:
            test_pos - VALID_DAYS
        ]

        valid_dates = dates[
            test_pos - VALID_DAYS:
            test_pos
        ]

        test_date = dates[test_pos]

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

        valid_start = valid["time"].min()

        train = train[
            train["time"]
            < valid_start - pd.Timedelta(minutes=HORIZON)
        ].copy()

        test_start = test["time"].min()

        valid = valid[
            valid["time"]
            < test_start - pd.Timedelta(minutes=HORIZON)
        ].copy()

        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=5,
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        )

        model.fit(
            train[FEATURES_BASIC],
            train["future_ret_bp"],
        )

        test["pred_ret_bp"] = model.predict(
            test[FEATURES_BASIC]
        )

        test["test_date"] = str(test_date)

        predictions.append(
            test[
                [
                    "time",
                    "test_date",
                    "ticker",
                    "spread_bp",
                    "future_ret_bp",
                    "pred_ret_bp",
                ]
            ]
        )

    pred = pd.concat(
        predictions,
        ignore_index=True,
    )

    print("=" * 90)
    print("GLOBAL")
    print("=" * 90)

    pearson = pred[
        ["pred_ret_bp", "future_ret_bp"]
    ].corr().iloc[0, 1]

    spearman = pred[
        ["pred_ret_bp", "future_ret_bp"]
    ].corr(method="spearman").iloc[0, 1]

    sign_acc = (
        np.sign(pred["pred_ret_bp"])
        == np.sign(pred["future_ret_bp"])
    ).mean()

    print("rows:", f"{len(pred):,}")
    print("pearson:", f"{pearson:.4f}")
    print("spearman:", f"{spearman:.4f}")
    print("sign accuracy:", f"{sign_acc * 100:.2f}%")
    print(
        "pred mean/std:",
        f"{pred['pred_ret_bp'].mean():+.3f}",
        f"{pred['pred_ret_bp'].std():.3f}",
    )
    print(
        "actual mean/std:",
        f"{pred['future_ret_bp'].mean():+.3f}",
        f"{pred['future_ret_bp'].std():.3f}",
    )

    # Decile по самому прогнозу.
    pred["pred_decile"] = pd.qcut(
        pred["pred_ret_bp"],
        10,
        labels=False,
        duplicates="drop",
    )

    deciles = (
        pred.groupby("pred_decile")
        .agg(
            n=("ticker", "size"),
            pred_mean=("pred_ret_bp", "mean"),
            actual_mean=("future_ret_bp", "mean"),
            actual_median=("future_ret_bp", "median"),
            up_rate=(
                "future_ret_bp",
                lambda x: (x > 0).mean(),
            ),
        )
    )

    print()
    print("=" * 90)
    print("PREDICTION DECILES")
    print("=" * 90)
    print(
        deciles.to_string(
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # Нас особенно интересует confidence.
    pred["abs_pred_decile"] = pd.qcut(
        pred["pred_ret_bp"].abs(),
        10,
        labels=False,
        duplicates="drop",
    )

    conf = (
        pred.groupby("abs_pred_decile")
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "abs_pred": g["pred_ret_bp"].abs().mean(),
                    "signed_gross": (
                        np.sign(g["pred_ret_bp"])
                        * g["future_ret_bp"]
                    ).mean(),
                    "sign_acc": (
                        np.sign(g["pred_ret_bp"])
                        == np.sign(g["future_ret_bp"])
                    ).mean(),
                }
            ),
            include_groups=False,
        )
    )

    print()
    print("=" * 90)
    print("CONFIDENCE DECILES")
    print("=" * 90)
    print(
        conf.to_string(
            float_format=lambda x: f"{x:.3f}",
        )
    )

    by_ticker = (
        pred.groupby("ticker")
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "corr": (
                        g["pred_ret_bp"]
                        .corr(g["future_ret_bp"])
                    ),
                    "sign_acc": (
                        np.sign(g["pred_ret_bp"])
                        == np.sign(g["future_ret_bp"])
                    ).mean(),
                    "signed_gross": (
                        np.sign(g["pred_ret_bp"])
                        * g["future_ret_bp"]
                    ).mean(),
                    "actual_vol_bp": g[
                        "future_ret_bp"
                    ].std(),
                }
            ),
            include_groups=False,
        )
        .sort_values("corr", ascending=False)
    )

    print()
    print("=" * 90)
    print("BY TICKER")
    print("=" * 90)

    print(
        by_ticker.to_string(
            float_format=lambda x: f"{x:.3f}",
        )
    )

    output = (
        PROJECT_ROOT
        / "state"
        / "ofi_alpha_diagnostics_predictions.csv"
    )

    pred.to_csv(
        output,
        index=False,
    )

    print()
    print("Сохранено:", output)


if __name__ == "__main__":
    main()

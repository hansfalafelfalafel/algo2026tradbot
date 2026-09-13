from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lob.dataset import FEATURES_BASIC


CACHE = PROJECT_ROOT / "state" / "ofi_dataset_h5.pkl"
STATE = PROJECT_ROOT / "state"

HORIZON = 15
TRAIN_DAYS = 20
VALID_DAYS = 5

MAX_ENTRY_SPREAD_BP = 2.0
FEE_PER_SIDE_BP = 0.5

BUFFERS_BP = (
    0.0,
    0.5,
    1.0,
)

DIRECTION_THRESHOLDS = (
    0.55,
    0.60,
    0.65,
)

MAGNITUDE_THRESHOLDS = (
    0.55,
    0.60,
    0.65,
)

MIN_VALID_TRADES = 50

# Robustness gate.
MIN_POSITIVE_DAYS = 0.60
MIN_WINSORIZED_NET_BP = 0.0
MIN_MEDIAN_DAILY_NET_BP = 0.0

# Не даём validation выбрать стратегию,
# живущую только за счёт одного огромного хвоста.
WINSOR_Q = 0.05


def add_exact_future(
    df: pd.DataFrame,
) -> pd.DataFrame:
    parts = []

    for ticker, g in df.groupby(
        "ticker",
        sort=False,
    ):
        g = g.sort_values("time").copy()
        indexed = g.set_index("time")

        future_index = (
            g["time"]
            + pd.Timedelta(minutes=HORIZON)
        )

        g["future_mid"] = (
            indexed["mid"]
            .reindex(future_index)
            .to_numpy()
        )

        g["future_spread_bp"] = (
            indexed["spread_bp"]
            .reindex(future_index)
            .to_numpy()
        )

        g["future_ret_bp"] = (
            g["future_mid"]
            / g["mid"].to_numpy()
            - 1.0
        ) * 10_000

        parts.append(g)

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    out = (
        out.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=FEATURES_BASIC
            + [
                "time",
                "ticker",
                "mid",
                "spread_bp",
                "future_mid",
                "future_spread_bp",
                "future_ret_bp",
            ]
        )
        .sort_values(
            ["time", "ticker"]
        )
        .reset_index(drop=True)
    )

    return out


def add_execution_fields(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    out["expected_cost_bp"] = (
        out["spread_bp"]
        + 2 * FEE_PER_SIDE_BP
    )

    out["entry_half_spread_bp"] = (
        out["spread_bp"] / 2.0
    )

    out["exit_half_spread_bp"] = (
        out["future_spread_bp"] / 2.0
    )

    out["fees_bp"] = (
        2 * FEE_PER_SIDE_BP
    )

    out["execution_cost_bp"] = (
        out["entry_half_spread_bp"]
        + out["exit_half_spread_bp"]
        + out["fees_bp"]
    )

    return out


def non_overlapping(
    frame: pd.DataFrame,
    direction_threshold: float,
    magnitude_threshold: float,
) -> pd.DataFrame:
    x = frame.copy()

    long_mask = (
        (x["p_up"] >= direction_threshold)
        & (x["p_large"] >= magnitude_threshold)
    )

    short_mask = (
        (
            x["p_up"]
            <= 1.0 - direction_threshold
        )
        & (
            x["p_large"]
            >= magnitude_threshold
        )
    )

    x["side"] = 0

    x.loc[
        long_mask,
        "side",
    ] = 1

    x.loc[
        short_mask,
        "side",
    ] = -1

    x = x[
        x["side"] != 0
    ].copy()

    x = x.sort_values(
        ["time", "ticker"]
    )

    accepted = []
    busy_until = {}

    for idx, row in x.iterrows():
        ticker = str(row["ticker"])
        now = pd.Timestamp(row["time"])

        if (
            ticker in busy_until
            and now < busy_until[ticker]
        ):
            continue

        busy_until[ticker] = (
            now
            + pd.Timedelta(
                minutes=HORIZON
            )
        )

        accepted.append(idx)

    trades = x.loc[
        accepted
    ].copy()

    if trades.empty:
        return trades

    trades["gross_bp"] = (
        trades["side"]
        * trades["future_ret_bp"]
    )

    trades["net_bp"] = (
        trades["gross_bp"]
        - trades["execution_cost_bp"]
    )

    return trades


def winsorized_mean(
    s: pd.Series,
    q: float = WINSOR_Q,
) -> float:
    if s.empty:
        return np.nan

    lo = s.quantile(q)
    hi = s.quantile(1.0 - q)

    return float(
        s.clip(lo, hi).mean()
    )


def fit_models(
    train: pd.DataFrame,
    buffer_bp: float,
):
    y_direction = (
        train["future_ret_bp"] > 0
    ).astype(int)

    required_move = (
        train["expected_cost_bp"]
        + buffer_bp
    )

    y_large = (
        train["future_ret_bp"].abs()
        >= required_move
    ).astype(int)

    if (
        y_direction.nunique() < 2
        or y_large.nunique() < 2
    ):
        return None, None

    direction_model = (
        HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    magnitude_model = (
        HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=43,
        )
    )

    direction_model.fit(
        train[FEATURES_BASIC],
        y_direction,
    )

    magnitude_model.fit(
        train[FEATURES_BASIC],
        y_large,
    )

    return (
        direction_model,
        magnitude_model,
    )


def predict(
    frame: pd.DataFrame,
    direction_model,
    magnitude_model,
) -> pd.DataFrame:
    out = frame.copy()

    out["p_up"] = (
        direction_model.predict_proba(
            out[FEATURES_BASIC]
        )[:, 1]
    )

    out["p_large"] = (
        magnitude_model.predict_proba(
            out[FEATURES_BASIC]
        )[:, 1]
    )

    return out


def evaluate_candidate(
    trades: pd.DataFrame,
) -> dict:
    daily = (
        trades.groupby("date")["net_bp"]
        .mean()
    )

    return {
        "trades": int(len(trades)),
        "gross_mean": float(
            trades["gross_bp"].mean()
        ),
        "cost_mean": float(
            trades["execution_cost_bp"].mean()
        ),
        "net_mean": float(
            trades["net_bp"].mean()
        ),
        "net_median": float(
            trades["net_bp"].median()
        ),
        "winsorized_net_mean": (
            winsorized_mean(
                trades["net_bp"]
            )
        ),
        "positive_days": float(
            (daily > 0).mean()
        ),
        "median_daily_net": float(
            daily.median()
        ),
        "worst_trade": float(
            trades["net_bp"].min()
        ),
        "best_trade": float(
            trades["net_bp"].max()
        ),
        "hit_rate": float(
            (
                trades["net_bp"] > 0
            ).mean()
        ),
    }


def passes_robustness(
    row: dict,
) -> bool:
    return (
        row["trades"]
        >= MIN_VALID_TRADES
        and row["net_mean"] > 0
        and row["winsorized_net_mean"]
        > MIN_WINSORIZED_NET_BP
        and row["positive_days"]
        >= MIN_POSITIVE_DAYS
        and row["median_daily_net"]
        > MIN_MEDIAN_DAILY_NET_BP
    )


def main() -> None:
    print("=" * 96)
    print("FREEZE SHADOW STRATEGY V1")
    print("=" * 96)

    base = pd.read_pickle(CACHE)

    base["time"] = pd.to_datetime(
        base["time"],
        utc=True,
    )

    df = add_exact_future(base)
    df = add_execution_fields(df)

    df = df[
        df["spread_bp"]
        <= MAX_ENTRY_SPREAD_BP
    ].copy()

    df["date"] = (
        df["time"].dt.date
    )

    dates = sorted(
        df["date"].unique()
    )

    needed = (
        TRAIN_DAYS
        + VALID_DAYS
    )

    if len(dates) < needed:
        raise RuntimeError(
            f"Нужно минимум {needed} дней, "
            f"есть {len(dates)}"
        )

    valid_dates = dates[
        -VALID_DAYS:
    ]

    train_dates = dates[
        -(TRAIN_DAYS + VALID_DAYS):
        -VALID_DAYS
    ]

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

    valid_start = valid["time"].min()

    train = train[
        train["time"]
        < (
            valid_start
            - pd.Timedelta(
                minutes=HORIZON
            )
        )
    ].copy()

    print(
        "TRAIN:",
        train_dates[0],
        "->",
        train_dates[-1],
    )

    print(
        "VALID:",
        valid_dates[0],
        "->",
        valid_dates[-1],
    )

    print(
        f"Train rows: {len(train):,}"
    )

    print(
        f"Valid rows: {len(valid):,}"
    )

    candidates = []
    models = {}

    for buffer_bp in BUFFERS_BP:
        (
            direction_model,
            magnitude_model,
        ) = fit_models(
            train,
            buffer_bp,
        )

        if direction_model is None:
            continue

        models[buffer_bp] = (
            direction_model,
            magnitude_model,
        )

        valid_pred = predict(
            valid,
            direction_model,
            magnitude_model,
        )

        for direction_threshold in (
            DIRECTION_THRESHOLDS
        ):
            for magnitude_threshold in (
                MAGNITUDE_THRESHOLDS
            ):
                trades = non_overlapping(
                    valid_pred,
                    direction_threshold,
                    magnitude_threshold,
                )

                if (
                    len(trades)
                    < MIN_VALID_TRADES
                ):
                    continue

                stats = evaluate_candidate(
                    trades
                )

                row = {
                    "buffer_bp": buffer_bp,
                    "direction_threshold": (
                        direction_threshold
                    ),
                    "magnitude_threshold": (
                        magnitude_threshold
                    ),
                    **stats,
                }

                row["passes"] = (
                    passes_robustness(row)
                )

                candidates.append(row)

    if not candidates:
        raise RuntimeError(
            "Нет validation-кандидатов "
            "с достаточным числом сделок."
        )

    table = pd.DataFrame(
        candidates
    )

    table = table.sort_values(
        [
            "passes",
            "winsorized_net_mean",
            "median_daily_net",
            "net_mean",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    print()
    print("TOP VALIDATION CANDIDATES")
    print("-" * 96)

    cols = [
        "buffer_bp",
        "direction_threshold",
        "magnitude_threshold",
        "trades",
        "net_mean",
        "winsorized_net_mean",
        "positive_days",
        "median_daily_net",
        "hit_rate",
        "passes",
    ]

    print(
        table[cols]
        .head(15)
        .to_string(
            index=False,
            float_format=lambda x:
            f"{x:.3f}",
        )
    )

    passed = table[
        table["passes"]
    ].copy()

    if passed.empty:
        print()
        print("=" * 96)
        print("STRATEGY NOT ARMED")
        print("=" * 96)
        print(
            "Ни одна конфигурация не прошла "
            "robust validation gate."
        )

        config = {
            "strategy_version": "v1",
            "armed": False,
            "created_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "reason": (
                "no_candidate_passed_robustness"
            ),
            "horizon_min": HORIZON,
            "max_entry_spread_bp": (
                MAX_ENTRY_SPREAD_BP
            ),
            "features": FEATURES_BASIC,
            "train_dates": [
                str(x)
                for x in train_dates
            ],
            "valid_dates": [
                str(x)
                for x in valid_dates
            ],
        }

        with open(
            STATE
            / "shadow_strategy_v1.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                config,
                f,
                ensure_ascii=False,
                indent=2,
            )

        table.to_csv(
            STATE
            / "shadow_strategy_v1_candidates.csv",
            index=False,
        )

        return

    best = passed.iloc[0].to_dict()

    buffer_bp = float(
        best["buffer_bp"]
    )

    direction_threshold = float(
        best["direction_threshold"]
    )

    magnitude_threshold = float(
        best["magnitude_threshold"]
    )

    (
        direction_model,
        magnitude_model,
    ) = models[
        buffer_bp
    ]

    direction_path = (
        STATE
        / "shadow_v1_direction.joblib"
    )

    magnitude_path = (
        STATE
        / "shadow_v1_magnitude.joblib"
    )

    joblib.dump(
        direction_model,
        direction_path,
    )

    joblib.dump(
        magnitude_model,
        magnitude_path,
    )

    config = {
        "strategy_version": "v1",
        "armed": True,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "horizon_min": HORIZON,
        "max_entry_spread_bp": (
            MAX_ENTRY_SPREAD_BP
        ),
        "fee_per_side_bp": (
            FEE_PER_SIDE_BP
        ),
        "buffer_bp": (
            buffer_bp
        ),
        "direction_threshold": (
            direction_threshold
        ),
        "magnitude_threshold": (
            magnitude_threshold
        ),
        "features": (
            FEATURES_BASIC
        ),
        "train_dates": [
            str(x)
            for x in train_dates
        ],
        "valid_dates": [
            str(x)
            for x in valid_dates
        ],
        "validation": {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (np.bool_, bool),
                )
                else (
                    int(value)
                    if isinstance(
                        value,
                        (np.integer,)
                    )
                    else (
                        float(value)
                        if isinstance(
                            value,
                            (
                                np.floating,
                                float,
                                int,
                            ),
                        )
                        else value
                    )
                )
            )
            for key, value
            in best.items()
        },
        "direction_model": (
            str(direction_path)
        ),
        "magnitude_model": (
            str(magnitude_path)
        ),
    }

    config_path = (
        STATE
        / "shadow_strategy_v1.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    table.to_csv(
        STATE
        / "shadow_strategy_v1_candidates.csv",
        index=False,
    )

    print()
    print("=" * 96)
    print("FROZEN STRATEGY V1")
    print("=" * 96)

    print(
        "buffer:",
        buffer_bp,
    )

    print(
        "direction threshold:",
        direction_threshold,
    )

    print(
        "magnitude threshold:",
        magnitude_threshold,
    )

    print(
        "validation trades:",
        int(best["trades"]),
    )

    print(
        "validation net:",
        f"{best['net_mean']:+.3f} bp",
    )

    print(
        "validation winsor net:",
        f"{best['winsorized_net_mean']:+.3f} bp",
    )

    print(
        "positive days:",
        f"{best['positive_days']:.1%}",
    )

    print(
        "median daily:",
        f"{best['median_daily_net']:+.3f} bp",
    )

    print()
    print(
        "Direction model:",
        direction_path,
    )

    print(
        "Magnitude model:",
        magnitude_path,
    )

    print(
        "Config:",
        config_path,
    )

    print("=" * 96)


if __name__ == "__main__":
    main()

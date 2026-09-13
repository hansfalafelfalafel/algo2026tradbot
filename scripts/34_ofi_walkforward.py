"""Walk-forward проверка OFI-сигнала.

20 предыдущих торговых дней -> train
5 следующих                 -> validation
1 следующий                 -> untouched test

Все торговые правила выбираются только на validation.
Test-день не используется для настройки стратегии.

Скрипт не отправляет заявки брокеру.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.lob.dataset import FEATURES_BASIC, build_dataset

load_dotenv(PROJECT_ROOT / ".env")


TRAIN_DAYS = 20
VALID_DAYS = 5

QUANTILES = (
    0.80,
    0.90,
    0.95,
    0.97,
    0.99,
)

FEE_PER_SIDE_BP = 0.5
MIN_VALID_TRADES = 50


def arg(flag: str, default, cast=float):
    if flag not in sys.argv:
        return default

    return cast(
        sys.argv[
            sys.argv.index(flag) + 1
        ]
    )


def add_trade_pnl(
    df: pd.DataFrame,
    side: int,
) -> pd.DataFrame:
    out = df.copy()

    out["gross_bp"] = (
        side
        * out["fwd_ret"]
        * 10_000
    )

    # Пока proxy для taker execution:
    # один текущий spread на round trip
    # + комиссия на входе и выходе.
    out["cost_bp"] = (
        out["spread_bp"]
        + 2 * FEE_PER_SIDE_BP
    )

    out["net_bp"] = (
        out["gross_bp"]
        - out["cost_bp"]
    )

    return out



def load_or_build_dataset(
    lob_dir: Path,
    horizon: int,
    rebuild: bool = False,
) -> pd.DataFrame:
    cache = (
        PROJECT_ROOT
        / "state"
        / f"ofi_dataset_h{horizon}.pkl"
    )

    if cache.exists() and not rebuild:
        print("Загружаю кэш датасета:", cache)
        df = pd.read_pickle(cache)

        print(
            f"Кэш: {len(df):,} баров, "
            f"{df['ticker'].nunique()} тикеров"
        )
        return df

    print("Строю датасет из сырых LOB-файлов...")

    df = build_dataset(
        lob_dir,
        horizon_min=horizon,
    )

    if df.empty:
        raise RuntimeError("Датасет пуст")

    cache.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_pickle(cache)

    print("Кэш датасета сохранён:", cache)

    return df


def non_overlapping_trades(
    frame: pd.DataFrame,
    side: int,
    threshold: float,
    horizon: int,
) -> pd.DataFrame:
    if side == 1:
        selected = frame[
            frame["p"] >= threshold
        ].copy()
    else:
        selected = frame[
            frame["p"] <= threshold
        ].copy()

    selected = selected.sort_values(
        ["time", "ticker"]
    )

    accepted_idx = []
    busy_until: dict[str, pd.Timestamp] = {}

    for idx, row in selected.iterrows():
        ticker = str(row["ticker"])
        now = pd.Timestamp(row["time"])

        if (
            ticker in busy_until
            and now < busy_until[ticker]
        ):
            continue

        busy_until[ticker] = (
            now
            + pd.Timedelta(minutes=horizon)
        )

        accepted_idx.append(idx)

    trades = selected.loc[
        accepted_idx
    ].copy()

    if trades.empty:
        return trades

    return add_trade_pnl(
        trades,
        side,
    )


def select_validation_rule(
    valid: pd.DataFrame,
    horizon: int,
) -> tuple[dict | None, pd.DataFrame]:
    candidates = []

    for side, side_name in (
        (1, "LONG"),
        (-1, "SHORT"),
    ):
        for q in QUANTILES:
            if side == 1:
                threshold = float(
                    valid["p"].quantile(q)
                )
            else:
                threshold = float(
                    valid["p"].quantile(1 - q)
                )

            trades = non_overlapping_trades(
                valid,
                side,
                threshold,
                horizon,
            )

            if len(trades) < MIN_VALID_TRADES:
                continue

            daily = (
                trades.groupby("date")["net_bp"]
                .mean()
            )

            positive_days = float(
                (daily > 0).mean()
            )

            candidates.append(
                {
                    "side": side,
                    "side_name": side_name,
                    "q": q,
                    "threshold": threshold,
                    "trades": len(trades),
                    "gross_mean": float(
                        trades["gross_bp"].mean()
                    ),
                    "net_mean": float(
                        trades["net_bp"].mean()
                    ),
                    "net_median": float(
                        trades["net_bp"].median()
                    ),
                    "hit_rate": float(
                        (trades["net_bp"] > 0).mean()
                    ),
                    "positive_days": positive_days,
                }
            )

    if not candidates:
        return None, pd.DataFrame()

    table = (
        pd.DataFrame(candidates)
        .sort_values(
            [
                "net_mean",
                "positive_days",
                "trades",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    best = table.iloc[0].to_dict()

    # Если даже лучший вариант на validation
    # отрицательный после proxy-издержек,
    # на test-день не торгуем.
    if best["net_mean"] <= 0:
        return None, table

    return best, table


def apply_rule(
    test: pd.DataFrame,
    rule: dict,
    horizon: int,
) -> pd.DataFrame:
    side = int(rule["side"])
    threshold = float(
        rule["threshold"]
    )

    if side == 1:
        candidates = test[
            test["p"] >= threshold
        ].copy()
    else:
        candidates = test[
            test["p"] <= threshold
        ].copy()

    candidates = candidates.sort_values(
        ["time", "ticker"]
    )

    # Не разрешаем перекрывающиеся позиции
    # по одному тикеру.
    accepted_idx = []
    busy_until: dict[
        str,
        pd.Timestamp,
    ] = {}

    for idx, row in candidates.iterrows():
        ticker = str(
            row["ticker"]
        )
        current_time = pd.Timestamp(
            row["time"]
        )

        if (
            ticker in busy_until
            and current_time
            < busy_until[ticker]
        ):
            continue

        busy_until[ticker] = (
            current_time
            + pd.Timedelta(
                minutes=horizon
            )
        )

        accepted_idx.append(idx)

    trades = candidates.loc[
        accepted_idx
    ].copy()

    if trades.empty:
        return trades

    trades = add_trade_pnl(
        trades,
        side,
    )

    trades["side"] = side
    trades["side_name"] = (
        rule["side_name"]
    )
    trades["validation_q"] = (
        rule["q"]
    )
    trades["prob_threshold"] = (
        threshold
    )

    return trades


def main() -> None:
    cfg = load_config()

    horizon = int(
        arg(
            "--horizon",
            5,
        )
    )

    max_spread = float(
        arg(
            "--max-spread",
            0,
        )
    )

    rebuild = "--rebuild" in sys.argv

    lob_dir = cfg.abs_path(
        cfg.data["cache_dir"],
        "lob",
    )

    print("=" * 88)
    print("OFI WALK-FORWARD")
    print("=" * 88)
    print("LOB:", lob_dir)
    print(
        "Horizon:",
        horizon,
        "min",
    )
    print(
        "Window:",
        TRAIN_DAYS,
        "train +",
        VALID_DAYS,
        "validation + 1 test",
    )

    df = load_or_build_dataset(
        lob_dir,
        horizon,
        rebuild=rebuild,
    )

    df = df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
    )

    df["date"] = (
        df["time"]
        .dt.date
    )

    if max_spread > 0:
        before = len(df)

        df = df[
            df["spread_bp"]
            <= max_spread
        ].copy()

        print(
            f"spread <= {max_spread:g} bp: "
            f"{len(df):,}/{before:,}"
        )

    df = (
        df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=(
                FEATURES_BASIC
                + [
                    "fwd_ret",
                    "spread_bp",
                    "time",
                    "ticker",
                ]
            )
        )
        .sort_values(
            ["time", "ticker"]
        )
        .reset_index(
            drop=True
        )
    )

    dates = sorted(
        df["date"].unique()
    )

    print(
        f"Датасет: "
        f"{len(df):,} баров, "
        f"{df['ticker'].nunique()} тикеров, "
        f"{len(dates)} дней"
    )

    required_days = (
        TRAIN_DAYS
        + VALID_DAYS
        + 1
    )

    if len(dates) < required_days:
        raise RuntimeError(
            f"Нужно минимум "
            f"{required_days} "
            f"торговых дней, "
            f"есть {len(dates)}"
        )

    test_summaries = []
    all_test_trades = []
    all_validation_candidates = []

    first_test_pos = (
        TRAIN_DAYS
        + VALID_DAYS
    )

    for test_pos in range(
        first_test_pos,
        len(dates),
    ):
        train_dates = dates[
            test_pos
            - VALID_DAYS
            - TRAIN_DAYS:
            test_pos
            - VALID_DAYS
        ]

        valid_dates = dates[
            test_pos
            - VALID_DAYS:
            test_pos
        ]

        test_date = dates[
            test_pos
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

        test = df[
            df["date"]
            == test_date
        ].copy()

        if (
            train.empty
            or valid.empty
            or test.empty
        ):
            continue

        # Purge из train последних horizon минут
        # перед validation.
        valid_start = (
            valid["time"].min()
        )

        train = train[
            train["time"]
            < (
                valid_start
                - pd.Timedelta(
                    minutes=horizon
                )
            )
        ].copy()

        # Purge из validation последних horizon минут
        # перед test-днём.
        test_start = (
            test["time"].min()
        )

        valid = valid[
            valid["time"]
            < (
                test_start
                - pd.Timedelta(
                    minutes=horizon
                )
            )
        ].copy()

        if (
            train.empty
            or valid.empty
        ):
            continue

        y_train = (
            train["fwd_ret"] > 0
        ).astype(int)

        if y_train.nunique() < 2:
            continue

        model = (
            HistGradientBoostingClassifier(
                max_depth=4,
                max_iter=200,
                learning_rate=0.05,
                random_state=42,
            )
        )

        model.fit(
            train[FEATURES_BASIC],
            y_train,
        )

        valid["p"] = (
            model.predict_proba(
                valid[
                    FEATURES_BASIC
                ]
            )[:, 1]
        )

        test["p"] = (
            model.predict_proba(
                test[
                    FEATURES_BASIC
                ]
            )[:, 1]
        )

        valid_y = (
            valid["fwd_ret"] > 0
        ).astype(int)

        test_y = (
            test["fwd_ret"] > 0
        ).astype(int)

        if (
            valid_y.nunique() < 2
            or test_y.nunique() < 2
        ):
            continue

        valid_auc = (
            roc_auc_score(
                valid_y,
                valid["p"],
            )
        )

        test_auc = (
            roc_auc_score(
                test_y,
                test["p"],
            )
        )

        rule, candidate_table = (
            select_validation_rule(
                valid,
                horizon,
            )
        )

        if not candidate_table.empty:
            candidate_table = candidate_table.copy()
            candidate_table.insert(
                0,
                "test_date",
                str(test_date),
            )
            candidate_table["valid_auc"] = valid_auc
            candidate_table["test_auc"] = test_auc

            all_validation_candidates.append(
                candidate_table
            )

        if rule is None:
            if candidate_table.empty:
                diagnostic = "NO TRADE | no candidates"
            else:
                best_diag = candidate_table.iloc[0]
                diagnostic = (
                    f"NO TRADE | best="
                    f"{best_diag['side_name']} "
                    f"q={best_diag['q']:.2f} | "
                    f"gross={best_diag['gross_mean']:+.2f} | "
                    f"net={best_diag['net_mean']:+.2f} | "
                    f"positive_days="
                    f"{best_diag['positive_days']:.2f}"
                )

            print(
                f"{test_date} | "
                f"valid AUC="
                f"{valid_auc:.3f} | "
                f"test AUC="
                f"{test_auc:.3f} | "
                f"{diagnostic}"
            )

            test_summaries.append(
                {
                    "date": str(
                        test_date
                    ),
                    "valid_auc": (
                        valid_auc
                    ),
                    "test_auc": (
                        test_auc
                    ),
                    "side": "NONE",
                    "q": np.nan,
                    "threshold": (
                        np.nan
                    ),
                    "trades": 0,
                    "gross_mean": (
                        np.nan
                    ),
                    "net_mean": (
                        np.nan
                    ),
                    "hit_rate": (
                        np.nan
                    ),
                }
            )

            continue

        trades = apply_rule(
            test,
            rule,
            horizon,
        )

        if trades.empty:
            gross_mean = np.nan
            net_mean = np.nan
            hit_rate = np.nan
        else:
            gross_mean = float(
                trades[
                    "gross_bp"
                ].mean()
            )

            net_mean = float(
                trades[
                    "net_bp"
                ].mean()
            )

            hit_rate = float(
                (
                    trades[
                        "net_bp"
                    ] > 0
                ).mean()
            )

            trades[
                "test_date"
            ] = str(
                test_date
            )

            all_test_trades.append(
                trades
            )

        print(
            f"{test_date} | "
            f"valid AUC="
            f"{valid_auc:.3f} | "
            f"test AUC="
            f"{test_auc:.3f} | "
            f"{rule['side_name']} "
            f"q={rule['q']:.2f} "
            f"thr="
            f"{rule['threshold']:.4f} | "
            f"n={len(trades):4d} | "
            f"gross="
            f"{gross_mean:+.2f} | "
            f"net="
            f"{net_mean:+.2f}"
        )

        test_summaries.append(
            {
                "date": str(
                    test_date
                ),
                "valid_auc": (
                    valid_auc
                ),
                "test_auc": (
                    test_auc
                ),
                "side": (
                    rule[
                        "side_name"
                    ]
                ),
                "q": rule["q"],
                "threshold": (
                    rule[
                        "threshold"
                    ]
                ),
                "trades": len(
                    trades
                ),
                "gross_mean": (
                    gross_mean
                ),
                "net_mean": (
                    net_mean
                ),
                "hit_rate": (
                    hit_rate
                ),
            }
        )

    summary = pd.DataFrame(
        test_summaries
    )

    print()
    print("=" * 88)
    print(
        "WALK-FORWARD SUMMARY"
    )
    print("=" * 88)

    if summary.empty:
        print(
            "Нет ни одного "
            "walk-forward окна."
        )
        return

    print(
        summary.to_string(
            index=False,
            float_format=(
                lambda x:
                f"{x:.4f}"
            ),
        )
    )

    traded = summary[
        summary["trades"] > 0
    ]

    print()
    print(
        "Test-дней:",
        len(summary),
    )
    print(
        "Дней с сигналом:",
        len(traded),
    )

    if all_test_trades:
        trades = pd.concat(
            all_test_trades,
            ignore_index=True,
        )

        print(
            "Всего сделок:",
            len(trades),
        )

        print(
            "Gross mean:",
            f"{trades['gross_bp'].mean():+.3f} bp",
        )

        print(
            "Net proxy mean:",
            f"{trades['net_bp'].mean():+.3f} bp",
        )

        print(
            "Net hit:",
            f"{(trades['net_bp'] > 0).mean() * 100:.1f}%",
        )

        by_ticker = (
            trades
            .groupby(
                [
                    "ticker",
                    "side_name",
                ]
            )
            .agg(
                trades=(
                    "ticker",
                    "size",
                ),
                gross_mean=(
                    "gross_bp",
                    "mean",
                ),
                net_mean=(
                    "net_bp",
                    "mean",
                ),
                hit=(
                    "net_bp",
                    lambda x:
                    (x > 0).mean(),
                ),
            )
            .sort_values(
                "net_mean",
                ascending=False,
            )
        )

        print()
        print(
            "BY TICKER"
        )

        print(
            by_ticker.to_string(
                float_format=(
                    lambda x:
                    f"{x:.3f}"
                )
            )
        )

    else:
        trades = pd.DataFrame()

        print(
            "Ни один validation-период "
            "не разрешил торговлю."
        )

    state = (
        PROJECT_ROOT
        / "state"
    )

    state.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        state
        / "ofi_walkforward_daily.csv",
        index=False,
    )

    if all_validation_candidates:
        validation_candidates = pd.concat(
            all_validation_candidates,
            ignore_index=True,
        )

        validation_candidates.to_csv(
            state
            / "ofi_walkforward_candidates.csv",
            index=False,
        )

        print()
        print("BEST VALIDATION CANDIDATES:")

        best_by_day = (
            validation_candidates
            .sort_values(
                ["test_date", "net_mean"],
                ascending=[True, False],
            )
            .groupby(
                "test_date",
                as_index=False,
            )
            .first()
        )

        print(
            best_by_day[
                [
                    "test_date",
                    "side_name",
                    "q",
                    "trades",
                    "gross_mean",
                    "net_mean",
                    "hit_rate",
                    "positive_days",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    if not trades.empty:
        trades.to_csv(
            state
            / "ofi_walkforward_trades.csv",
            index=False,
        )

    print()
    print(
        "Сохранено в state/"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()

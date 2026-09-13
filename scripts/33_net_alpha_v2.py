#!/usr/bin/env python3
"""LOB v2 stage 1: экономически ориентированная модель на текущих признаках.

Что меняется относительно scripts/32_exec_sim.py:
- строит базовый датасет один раз и кэширует его;
- добавляет лаги, rolling-признаки, относительные рыночные признаки и взаимодействия;
- обучает отдельные модели для каждого тикера;
- регрессор прогнозирует размер и знак будущего движения в bp;
- классификатор оценивает вероятность движения, способного покрыть расходы;
- пороги выбираются только на validation;
- итог измеряется на отдельном test с purge и запретом пересекающихся позиций.

Скрипт не отправляет заявки и не изменяет сборщик.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from src.config import load_config
from src.lob.dataset import FEATURES as BASE_FEATURES
from src.lob.dataset import build_dataset

load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOB v2: per-ticker net-alpha model."
    )
    parser.add_argument("--lob-dir", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--grid", default="1s")
    parser.add_argument("--min-seconds", type=int, default=5)
    parser.add_argument("--max-spread", type=float, default=2.0)
    parser.add_argument("--side-cost-bp", type=float, default=7.0)
    parser.add_argument("--buffer-bp", type=float, default=2.0)
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "state" / "lob_v2_base_h15.pkl",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-trades", action="store_true")
    parser.add_argument("--min-validation-trades", type=int, default=20)
    return parser.parse_args()


def dataset_kwargs(args: argparse.Namespace) -> dict:
    signature = inspect.signature(build_dataset)
    params = signature.parameters
    kwargs: dict = {}

    if "horizon_min" in params:
        kwargs["horizon_min"] = args.horizon
    elif "horizon" in params:
        kwargs["horizon"] = args.horizon
    else:
        raise RuntimeError(f"Не найден аргумент horizon: {signature}")

    for name in ("book_grid", "grid", "grid_freq", "resample_grid"):
        if name in params:
            kwargs[name] = args.grid
            break
    else:
        raise RuntimeError(f"Не найден аргумент секундной сетки: {signature}")

    for name in (
        "min_seconds_per_minute",
        "min_seconds",
        "min_seconds_per_bar",
        "min_active_seconds",
    ):
        if name in params:
            kwargs[name] = args.min_seconds
            break
    else:
        raise RuntimeError(f"Не найден аргумент min-seconds: {signature}")

    for name in ("include_trades", "use_trades"):
        if name in params:
            kwargs[name] = not args.no_trades
            break
    else:
        if args.no_trades:
            raise RuntimeError(
                f"build_dataset не умеет отключать trades: {signature}"
            )

    print("build_dataset:", signature)
    print("Аргументы датасета:", kwargs)
    return kwargs


def load_or_build_base(args: argparse.Namespace) -> pd.DataFrame:
    cfg = load_config()
    lob_dir = (
        args.lob_dir.resolve()
        if args.lob_dir is not None
        else cfg.abs_path(cfg.data["cache_dir"], "lob")
    )
    cache = args.cache.resolve()
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and not args.rebuild:
        print("Загружаю кэш:", cache)
        df = pd.read_pickle(cache)
    else:
        print("Строю базовый датасет из:", lob_dir)
        df = build_dataset(lob_dir, **dataset_kwargs(args))
        if df.empty:
            raise RuntimeError("Базовый датасет пуст")
        df.to_pickle(cache)
        print("Кэш сохранён:", cache)

    required = {"time", "ticker", "fwd_ret", "spread_bp"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"В датасете нет колонок: {missing}")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["ticker"] = df["ticker"].astype(str)
    df["fwd_bp"] = pd.to_numeric(df["fwd_ret"], errors="coerce") * 10_000
    df["spread_bp"] = pd.to_numeric(df["spread_bp"], errors="coerce")

    if args.max_spread > 0:
        before = len(df)
        df = df[df["spread_bp"] <= args.max_spread].copy()
        print(
            f"Фильтр spread <= {args.max_spread:g} bp: "
            f"{len(df):,} из {before:,}"
        )

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["time", "ticker", "fwd_bp", "spread_bp"])
        .sort_values(["ticker", "time"])
        .reset_index(drop=True)
    )

    print(
        f"База: {len(df):,} строк, {df.ticker.nunique()} тикеров, "
        f"{df.time.min()} → {df.time.max()}"
    )
    return df


def rolling_transform(
    group: pd.core.groupby.generic.SeriesGroupBy,
    window: int,
    kind: str,
) -> pd.Series:
    min_periods = max(2, window // 2)
    if kind == "mean":
        return group.transform(
            lambda s: s.rolling(window, min_periods=min_periods).mean()
        )
    if kind == "std":
        return group.transform(
            lambda s: s.rolling(window, min_periods=min_periods).std()
        )
    raise ValueError(kind)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    base = [column for column in BASE_FEATURES if column in df.columns]

    desired = [
        "ofi",
        "tfi",
        "imb1",
        "imb5",
        "micro_dev",
        "spread_bp",
        "ret_1m",
        "ret_5m",
        "ofi_5",
        "ofi_15",
        "ofi_30",
        "tfi_5",
        "mkt_ofi",
        "tod",
        "evening",
    ]
    numeric = [column for column in desired if column in df.columns]

    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    group = df.groupby("ticker", sort=False)
    derived: list[str] = []

    lag_columns = [
        column
        for column in (
            "ofi",
            "tfi",
            "imb1",
            "imb5",
            "micro_dev",
            "spread_bp",
            "ret_1m",
            "ret_5m",
            "mkt_ofi",
        )
        if column in df.columns
    ]

    for column in lag_columns:
        series_group = group[column]

        diff_name = f"{column}_d1"
        df[diff_name] = series_group.diff()
        derived.append(diff_name)

        for lag in (1, 2, 3, 5):
            name = f"{column}_lag{lag}"
            df[name] = series_group.shift(lag)
            derived.append(name)

        for window in (3, 5, 15):
            mean_name = f"{column}_mean{window}"
            std_name = f"{column}_std{window}"
            df[mean_name] = rolling_transform(series_group, window, "mean")
            df[std_name] = rolling_transform(series_group, window, "std")
            derived.extend([mean_name, std_name])

        mean60 = rolling_transform(series_group, 60, "mean")
        std60 = rolling_transform(series_group, 60, "std")
        z_name = f"{column}_z60"
        df[z_name] = ((df[column] - mean60) / std60.replace(0, np.nan)).clip(
            -10, 10
        )
        derived.append(z_name)

    def add(name: str, values: pd.Series) -> None:
        df[name] = values.replace([np.inf, -np.inf], np.nan)
        derived.append(name)

    if {"imb1", "imb5"} <= set(df.columns):
        add("imb_gap", df["imb1"] - df["imb5"])
        add("imb_agreement", np.sign(df["imb1"]) * np.sign(df["imb5"]))

    if {"micro_dev", "spread_bp"} <= set(df.columns):
        add(
            "micro_to_spread",
            df["micro_dev"] / df["spread_bp"].replace(0, np.nan),
        )

    if {"ofi_z60", "tfi_z60"} <= set(df.columns):
        add("ofi_tfi_interaction", df["ofi_z60"] * df["tfi_z60"])
        add(
            "flow_agreement",
            np.sign(df["ofi_z60"]) * np.sign(df["tfi_z60"]),
        )
        add("flow_strength", df["ofi_z60"].abs() + df["tfi_z60"].abs())

    if {"ofi", "spread_bp"} <= set(df.columns):
        add("ofi_per_spread", df["ofi"] / (df["spread_bp"] + 0.05))

    if {"tfi", "spread_bp"} <= set(df.columns):
        add("tfi_per_spread", df["tfi"] / (df["spread_bp"] + 0.05))

    market_source = [
        column
        for column in (
            "ret_1m",
            "ret_5m",
            "ofi_z60",
            "tfi_z60",
            "imb1",
            "micro_dev",
        )
        if column in df.columns
    ]
    for column in market_source:
        market_name = f"market_{column}"
        relative_name = f"relative_{column}"
        market = df.groupby("time")[column].transform("median")
        df[market_name] = market
        df[relative_name] = df[column] - market
        derived.extend([market_name, relative_name])

    if {"relative_ret_1m", "relative_ofi_z60"} <= set(df.columns):
        add(
            "relative_flow_return",
            df["relative_ret_1m"] * df["relative_ofi_z60"],
        )

    features = []
    seen = set()
    for column in base + derived:
        if column in df.columns and column not in seen:
            features.append(column)
            seen.add(column)

    # Удаляем колонки, в которых почти нет данных или нет вариативности.
    usable = []
    for column in features:
        valid = pd.to_numeric(df[column], errors="coerce")
        if valid.notna().mean() < 0.40:
            continue
        if valid.nunique(dropna=True) <= 1:
            continue
        df[column] = valid
        usable.append(column)

    print(f"Признаков v2: {len(usable)}")
    return df, usable


def time_split(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    times = (
        pd.Series(df["time"].drop_duplicates())
        .sort_values()
        .reset_index(drop=True)
    )
    if len(times) < 30:
        raise RuntimeError("Слишком мало уникальных временных точек")

    train_boundary = times.iloc[int(len(times) * 0.60)]
    test_boundary = times.iloc[int(len(times) * 0.80)]
    purge = pd.Timedelta(minutes=horizon)

    train = df[df["time"] < train_boundary - purge].copy()
    valid = df[
        (df["time"] >= train_boundary)
        & (df["time"] < test_boundary - purge)
    ].copy()
    test = df[df["time"] >= test_boundary].copy()

    print("Временное разделение:")
    print("  train до:", train_boundary - purge)
    print("  validation:", train_boundary, "→", test_boundary - purge)
    print("  test от:", test_boundary)
    print(
        f"  строк: train={len(train):,}, "
        f"validation={len(valid):,}, test={len(test):,}"
    )
    return train, valid, test


def fit_per_ticker(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    event_hurdle_bp: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_predictions = []
    test_predictions = []

    tickers = sorted(
        set(train["ticker"])
        & set(valid["ticker"])
        & set(test["ticker"])
    )

    for ticker in tickers:
        tr = train[train["ticker"] == ticker]
        va = valid[valid["ticker"] == ticker]
        te = test[test["ticker"] == ticker]

        if min(len(tr), len(va), len(te)) < 150:
            print(
                f"{ticker}: пропуск — мало данных "
                f"({len(tr)}/{len(va)}/{len(te)})"
            )
            continue

        x_train = tr[features]
        y_train = tr["fwd_bp"]

        regressor = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=5,
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        )
        regressor.fit(x_train, y_train)

        event_train = (tr["fwd_bp"].abs() >= event_hurdle_bp).astype(int)
        classifier = None
        if event_train.nunique() == 2 and event_train.sum() >= 20:
            classifier = HistGradientBoostingClassifier(
                max_depth=4,
                max_iter=200,
                learning_rate=0.05,
                l2_regularization=1.0,
                random_state=42,
            )
            classifier.fit(x_train, event_train)

        for frame, target in (
            (va, valid_predictions),
            (te, test_predictions),
        ):
            output = frame[
                ["time", "ticker", "fwd_bp", "spread_bp"]
            ].copy()
            output["pred_bp"] = regressor.predict(frame[features])

            if classifier is None:
                output["event_p"] = 1.0
            else:
                output["event_p"] = classifier.predict_proba(
                    frame[features]
                )[:, 1]

            target.append(output)

        rate = event_train.mean() * 100
        print(
            f"{ticker}: train={len(tr):,}, valid={len(va):,}, "
            f"test={len(te):,}, крупных движений={rate:.1f}%"
        )

    if not valid_predictions or not test_predictions:
        raise RuntimeError("Не удалось обучить модели ни для одного тикера")

    return (
        pd.concat(valid_predictions, ignore_index=True),
        pd.concat(test_predictions, ignore_index=True),
    )


def simulate(
    predictions: pd.DataFrame,
    pred_threshold: float,
    event_threshold: float,
    horizon: int,
    side_cost_bp: float,
) -> pd.DataFrame:
    candidates = predictions[
        (predictions["pred_bp"].abs() >= pred_threshold)
        & (predictions["event_p"] >= event_threshold)
    ].sort_values(["time", "ticker"])

    rows = []
    busy_until: dict[str, pd.Timestamp] = {}

    for row in candidates.itertuples(index=False):
        ticker = str(row.ticker)
        current_time = pd.Timestamp(row.time)

        if ticker in busy_until and current_time < busy_until[ticker]:
            continue

        side = 1 if row.pred_bp > 0 else -1
        busy_until[ticker] = current_time + pd.Timedelta(minutes=horizon)

        gross = side * float(row.fwd_bp)
        spread = float(row.spread_bp)

        rows.append(
            {
                "time": current_time,
                "date": current_time.date(),
                "ticker": ticker,
                "side": side,
                "pred_bp": float(row.pred_bp),
                "event_p": float(row.event_p),
                "actual_fwd_bp": float(row.fwd_bp),
                "spread_bp": spread,
                "gross": gross,
                "maker": gross - 2 * side_cost_bp,
                "taker": gross - spread - 2 * side_cost_bp,
            }
        )

    return pd.DataFrame(rows)


def choose_thresholds(
    validation_predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[float, float, pd.DataFrame]:
    abs_prediction = validation_predictions["pred_bp"].abs()

    quantiles = sorted(
        {
            float(abs_prediction.quantile(q))
            for q in (0.70, 0.80, 0.90, 0.95, 0.98)
            if np.isfinite(abs_prediction.quantile(q))
        }
    )
    event_thresholds = (0.40, 0.50, 0.60, 0.70, 0.80)

    trials = []
    best = None

    for pred_threshold in quantiles:
        for event_threshold in event_thresholds:
            trades = simulate(
                validation_predictions,
                pred_threshold,
                event_threshold,
                args.horizon,
                args.side_cost_bp,
            )
            if len(trades) < args.min_validation_trades:
                continue

            mean_net = trades["maker"].mean()
            sum_net = trades["maker"].sum()
            hit_rate = (trades["maker"] > 0).mean()

            record = {
                "pred_threshold": pred_threshold,
                "event_threshold": event_threshold,
                "trades": len(trades),
                "maker_mean": mean_net,
                "maker_sum": sum_net,
                "maker_hit": hit_rate,
                "gross_mean": trades["gross"].mean(),
            }
            trials.append(record)

            score = (mean_net, sum_net)
            if best is None or score > best[0]:
                best = (score, pred_threshold, event_threshold)

    if best is None:
        raise RuntimeError(
            "На validation не нашлось конфигурации с достаточным "
            "количеством сделок"
        )

    table = pd.DataFrame(trials).sort_values(
        ["maker_mean", "maker_sum"],
        ascending=False,
    )

    return best[1], best[2], table


def describe(name: str, trades: pd.DataFrame) -> None:
    print()
    print("=" * 88)
    print(name)
    print("=" * 88)

    if trades.empty:
        print("Сделок нет")
        return

    days = trades["date"].nunique()
    print(f"Сделок: {len(trades):,}; торговых дат: {days}")

    for column in ("gross", "maker", "taker"):
        series = trades[column]
        print(
            f"{column:>6}: mean={series.mean():+8.3f} bp | "
            f"median={series.median():+8.3f} | "
            f"hit={(series > 0).mean() * 100:5.1f}% | "
            f"sum={series.sum():+10.1f}"
        )

    print("Средний spread:", f"{trades.spread_bp.mean():.3f} bp")

    by_ticker = (
        trades.groupby("ticker")
        .agg(
            trades=("ticker", "size"),
            gross_mean=("gross", "mean"),
            maker_mean=("maker", "mean"),
            taker_mean=("taker", "mean"),
            maker_hit=("maker", lambda s: (s > 0).mean()),
        )
        .sort_values("maker_mean", ascending=False)
    )
    print()
    print(by_ticker.to_string(float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon должен быть больше нуля")
    if args.side_cost_bp < 0:
        raise ValueError("--side-cost-bp не может быть отрицательным")

    print("=" * 88)
    print("LOB V2 — NET ALPHA")
    print("=" * 88)
    print("Горизонт:", args.horizon, "мин")
    print("Расходы на сторону:", args.side_cost_bp, "bp")
    print("Buffer:", args.buffer_bp, "bp")
    print("Trades:", not args.no_trades)

    base = load_or_build_base(args)
    data, features = engineer_features(base)
    data = data.dropna(subset=["fwd_bp"]).sort_values(
        ["time", "ticker"]
    )

    train, valid, test = time_split(data, args.horizon)

    # Для event-классификатора крупным считается движение, которое
    # покрывает полный maker-оборот плюс небольшой запас.
    hurdle = 2 * args.side_cost_bp + args.buffer_bp
    print("Event hurdle:", hurdle, "bp")

    validation_predictions, test_predictions = fit_per_ticker(
        train,
        valid,
        test,
        features,
        hurdle,
    )

    pred_threshold, event_threshold, search = choose_thresholds(
        validation_predictions,
        args,
    )

    print()
    print("Лучшие validation-конфигурации:")
    print(
        search.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print()
    print("Выбрано только на validation:")
    print("  abs(pred_bp) >=", f"{pred_threshold:.4f}")
    print("  event_p >=", f"{event_threshold:.2f}")

    validation_trades = simulate(
        validation_predictions,
        pred_threshold,
        event_threshold,
        args.horizon,
        args.side_cost_bp,
    )
    test_trades = simulate(
        test_predictions,
        pred_threshold,
        event_threshold,
        args.horizon,
        args.side_cost_bp,
    )

    describe("VALIDATION", validation_trades)
    describe("UNTOUCHED TEST", test_trades)

    state = PROJECT_ROOT / "state"
    state.mkdir(parents=True, exist_ok=True)

    validation_predictions.to_csv(
        state / "lob_v2_validation_predictions.csv",
        index=False,
    )
    test_predictions.to_csv(
        state / "lob_v2_test_predictions.csv",
        index=False,
    )
    test_trades.to_csv(
        state / "lob_v2_test_trades.csv",
        index=False,
    )
    search.to_csv(
        state / "lob_v2_validation_search.csv",
        index=False,
    )

    print()
    print("Файлы сохранены в:", state)

    if test_trades.empty:
        print("В test нет сделок — пороги слишком жёсткие.")
    elif test_trades["taker"].mean() > 0:
        print("Результат положителен даже в TAKER.")
    elif test_trades["maker"].mean() > 0:
        print("Результат положителен только в условном MAKER.")
    else:
        print("V2 пока отрицательна после расходов.")


if __name__ == "__main__":
    main()

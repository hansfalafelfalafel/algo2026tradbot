#!/usr/bin/env python3
"""Честная OOS-симуляция исполнения LOB/OFI-сигналов."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import load_config
from src.lob.dataset import FEATURES, build_dataset

load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Честная временная OOS-симуляция LOB/OFI."
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--q", type=float, default=0.10)
    parser.add_argument("--grid", default="1s")
    parser.add_argument("--min-seconds", type=int, default=5)
    parser.add_argument("--max-spread", type=float, default=0.0)
    parser.add_argument("--fees-bp", type=float, default=None)
    parser.add_argument("--no-trades", action="store_true")
    parser.add_argument("--lob-dir", type=Path, default=None)
    return parser.parse_args()


def config_side_cost_bp() -> float:
    """Комиссия + проскальзывание на одну сторону сделки."""
    commission = 0.0005
    slippage = 0.0002

    try:
        import yaml

        raw = yaml.safe_load(
            (PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8")
        )
        env = (raw or {}).get("env", {})
        commission = float(env.get("commission", commission))
        slippage = float(env.get("slippage", slippage))
    except Exception as exc:
        print(
            "Предупреждение: не удалось прочитать расходы из config.yaml:",
            exc,
        )

    return (commission + slippage) * 10_000


def builder_kwargs(args: argparse.Namespace) -> dict:
    """Подбирает имена аргументов под установленную версию dataset.py."""
    signature = inspect.signature(build_dataset)
    params = signature.parameters
    kwargs: dict = {}

    if "horizon_min" in params:
        kwargs["horizon_min"] = args.horizon
    elif "horizon" in params:
        kwargs["horizon"] = args.horizon
    else:
        raise RuntimeError(
            f"build_dataset не поддерживает horizon: {signature}"
        )

    grid_name = next(
        (
            name
            for name in (
                "grid",
                "book_grid",
                "grid_freq",
                "resample_grid",
            )
            if name in params
        ),
        None,
    )
    if grid_name is None:
        raise RuntimeError(
            "Новый dataset.py не поддерживает секундную сетку. "
            f"Сигнатура: {signature}"
        )
    kwargs[grid_name] = args.grid

    min_seconds_name = next(
        (
            name
            for name in (
                "min_seconds",
                "min_seconds_per_bar",
                "min_seconds_per_minute",
                "min_active_seconds",
            )
            if name in params
        ),
        None,
    )
    if min_seconds_name is None:
        raise RuntimeError(
            "Новый dataset.py не поддерживает min-seconds. "
            f"Сигнатура: {signature}"
        )
    kwargs[min_seconds_name] = args.min_seconds

    trades_name = next(
        (
            name
            for name in (
                "use_trades",
                "include_trades",
            )
            if name in params
        ),
        None,
    )
    if trades_name is None and args.no_trades:
        raise RuntimeError(
            "Новый dataset.py не поддерживает отключение trades. "
            f"Сигнатура: {signature}"
        )
    if trades_name is not None:
        kwargs[trades_name] = not args.no_trades

    print("build_dataset:", signature)
    print("Аргументы датасета:", kwargs)

    return kwargs


def time_split(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """60% train, 20% validation, 20% test с purge по горизонту."""
    times = (
        pd.Series(df["time"].drop_duplicates())
        .sort_values()
        .reset_index(drop=True)
    )

    if len(times) < 30:
        raise RuntimeError(
            f"Слишком мало уникальных временных точек: {len(times)}"
        )

    train_boundary = times.iloc[int(len(times) * 0.60)]
    test_boundary = times.iloc[int(len(times) * 0.80)]
    purge = pd.Timedelta(minutes=horizon)

    train = df[df["time"] < train_boundary - purge].copy()
    valid = df[
        (df["time"] >= train_boundary)
        & (df["time"] < test_boundary - purge)
    ].copy()
    test = df[df["time"] >= test_boundary].copy()

    print()
    print("Временное разделение:")
    print("  train до:", train_boundary - purge)
    print("  validation:", train_boundary, "→", test_boundary - purge)
    print("  test от:", test_boundary)
    print(
        f"  строк: train={len(train):,}, "
        f"validation={len(valid):,}, test={len(test):,}"
    )

    if min(len(train), len(valid), len(test)) < 100:
        raise RuntimeError(
            "После временного разделения слишком мало данных."
        )

    return train, valid, test


def describe_series(series: pd.Series) -> str:
    return (
        f"средн {series.mean():+.2f} bp | "
        f"медиана {series.median():+.2f} | "
        f"hit-rate {(series > 0).mean() * 100:.1f}% | "
        f"сумма {series.sum():+.1f} bp"
    )


def main() -> None:
    args = parse_args()

    if args.horizon <= 0:
        raise ValueError("--horizon должен быть больше нуля")
    if not 0 < args.q < 0.5:
        raise ValueError("--q должен находиться между 0 и 0.5")

    cfg = load_config()
    lob_dir = (
        args.lob_dir.resolve()
        if args.lob_dir is not None
        else cfg.abs_path(cfg.data["cache_dir"], "lob")
    )

    side_cost_bp = (
        args.fees_bp
        if args.fees_bp is not None
        else config_side_cost_bp()
    )

    print("=" * 78)
    print("ЧЕСТНАЯ LOB/OFI-СИМУЛЯЦИЯ")
    print("=" * 78)
    print("LOB:", lob_dir)
    print("Горизонт:", args.horizon, "мин")
    print("Сигнальный хвост:", f"{args.q:.0%}")
    print("Сетка стакана:", args.grid)
    print("Минимум активных секунд:", args.min_seconds)
    print("Использовать trades:", not args.no_trades)
    print("Расходы на одну сторону:", f"{side_cost_bp:.2f} bp")

    kwargs = builder_kwargs(args)
    df = build_dataset(lob_dir, **kwargs)

    if df.empty:
        raise RuntimeError("Датасет пуст")

    required = set(FEATURES + ["time", "ticker", "fwd_ret", "spread_bp"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"В датасете отсутствуют колонки: {missing}")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=FEATURES + ["fwd_ret", "spread_bp"])
        .sort_values(["time", "ticker"])
        .reset_index(drop=True)
    )

    if args.max_spread > 0:
        before = len(df)
        df = df[df["spread_bp"] <= args.max_spread].copy()
        print(
            f"Фильтр spread <= {args.max_spread:g} bp: "
            f"{len(df):,} из {before:,}"
        )

    print(
        f"Датасет: {len(df):,} строк, "
        f"{df['ticker'].nunique()} тикеров, "
        f"{df['time'].min()} → {df['time'].max()}"
    )

    if len(df) < 3_000:
        raise RuntimeError(
            f"Пока мало данных: {len(df):,} строк"
        )

    train, valid, test = time_split(df, args.horizon)

    y_train = (train["fwd_ret"] > 0).astype(int)
    if y_train.nunique() < 2:
        raise RuntimeError("В train присутствует только один класс")

    model = HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=200,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(train[FEATURES], y_train)

    valid_probability = model.predict_proba(valid[FEATURES])[:, 1]
    lower = float(np.quantile(valid_probability, args.q))
    upper = float(np.quantile(valid_probability, 1 - args.q))

    print()
    print("Пороги рассчитаны только на validation:")
    print("  short <=", f"{lower:.6f}")
    print("  long  >=", f"{upper:.6f}")

    test = test.copy()
    test["p"] = model.predict_proba(test[FEATURES])[:, 1]

    trades: list[dict] = []
    busy_until: dict[str, pd.Timestamp] = {}

    for row in test.itertuples(index=False):
        if row.p >= upper:
            side = 1
        elif row.p <= lower:
            side = -1
        else:
            continue

        ticker = str(row.ticker)
        current_time = pd.Timestamp(row.time)

        if (
            ticker in busy_until
            and current_time < busy_until[ticker]
        ):
            continue

        busy_until[ticker] = current_time + pd.Timedelta(
            minutes=args.horizon
        )

        gross_bp = side * float(row.fwd_ret) * 10_000
        half_spread_bp = float(row.spread_bp) / 2

        taker_bp = gross_bp - 2 * (
            half_spread_bp + side_cost_bp
        )

        # Условный maker: выигрыш полуспреда на входе и
        # потеря полуспреда на выходе взаимно сокращаются.
        maker_bp = gross_bp - 2 * side_cost_bp

        trades.append(
            {
                "time": current_time,
                "date": current_time.date(),
                "ticker": ticker,
                "side": side,
                "p": float(row.p),
                "gross": gross_bp,
                "taker": taker_bp,
                "maker": maker_bp,
                "spread_bp": float(row.spread_bp),
            }
        )

    if not trades:
        print("На test-сегменте сделок не набралось.")
        return

    result = pd.DataFrame(trades)
    days = result["date"].nunique()

    print()
    print("=" * 78)
    print(
        f"TEST: {len(result):,} сделок за {days} торговых дат"
    )
    print("-" * 78)
    print("Без издержек: ", describe_series(result["gross"]))
    print("TAKER:        ", describe_series(result["taker"]))
    print("MAKER:        ", describe_series(result["maker"]))
    print(
        "Средний спред:",
        f"{result['spread_bp'].mean():.2f} bp",
    )

    print()
    print("По направлениям:")
    for side, name in ((1, "LONG"), (-1, "SHORT")):
        part = result[result["side"] == side]
        if not part.empty:
            print(
                f"  {name}: {len(part):,}; "
                f"taker={part['taker'].mean():+.2f} bp; "
                f"maker={part['maker'].mean():+.2f} bp"
            )

    by_ticker = (
        result.groupby("ticker")
        .agg(
            trades=("ticker", "size"),
            gross_bp=("gross", "mean"),
            taker_bp=("taker", "mean"),
            maker_bp=("maker", "mean"),
            hit_rate=("maker", lambda x: (x > 0).mean()),
        )
        .sort_values("maker_bp", ascending=False)
    )

    print()
    print("По тикерам:")
    print(by_ticker.to_string(float_format=lambda x: f"{x:.3f}"))

    output = PROJECT_ROOT / "state" / "lob_exec_sim_last.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print()
    print("Сделки сохранены:", output)
    print("=" * 78)

    if result["taker"].mean() > 0:
        print("Положителен даже консервативный TAKER.")
    elif result["maker"].mean() > 0:
        print("Плюс только в условном MAKER-режиме.")
    else:
        print("После расходов стратегия пока отрицательная.")


if __name__ == "__main__":
    main()

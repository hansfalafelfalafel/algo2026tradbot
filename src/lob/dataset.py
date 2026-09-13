"""Слой A: построение датасета OFI-признаков из стакана и ленты."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

BOOK_COLS = (
    ["time"]
    + [f"bid_p{i}" for i in range(10)]
    + [f"bid_q{i}" for i in range(10)]
    + [f"ask_p{i}" for i in range(10)]
    + [f"ask_q{i}" for i in range(10)]
)
TRADE_COLS = ["time", "price", "quantity", "direction"]


def _read_csv_robust(path: Path, cols) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
        if "time" not in df.columns:
            df = pd.read_csv(
                path, header=None, names=cols, low_memory=False,
                on_bad_lines="skip",
            )
        df = df[df["time"].astype(str).str.match(r"\d{4}-", na=False)].copy()
        for col in df.columns:
            if col != "time":
                df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
        return df.dropna().reset_index(drop=True)
    except Exception:
        return None


def _load_book_day(path: Path) -> Optional[pd.DataFrame]:
    df = _read_csv_robust(path, BOOK_COLS)
    if df is None or len(df) < 10:
        return None
    df["time"] = pd.to_datetime(
        df["time"], utc=True, format="ISO8601", errors="coerce"
    )
    return df.dropna(subset=["time"]).sort_values("time")


def _regularize_book(
    book: pd.DataFrame,
    grid: str = "1s",
    ffill_limit: int = 5,
) -> pd.DataFrame:
    value_cols = [col for col in book.columns if col != "time"]
    book = book.drop_duplicates(subset=["time"], keep="last")
    book = book.set_index("time")[value_cols].sort_index()
    book = book.resample(grid).last().ffill(limit=ffill_limit)
    return book.dropna(subset=["bid_p0", "ask_p0"]).reset_index()


def _dedupe_near_trades(
    trades: pd.DataFrame,
    window_ms: int = 20,
) -> pd.DataFrame:
    if trades.empty or window_ms <= 0:
        return trades
    trades = trades.sort_values("time").reset_index(drop=True)
    prev = trades.shift(1)
    dt_ms = (trades["time"] - prev["time"]).dt.total_seconds().mul(1000)
    same_payload = (
        trades["price"].eq(prev["price"])
        & trades["quantity"].eq(prev["quantity"])
        & trades["direction"].eq(prev["direction"])
    )
    duplicate = same_payload & dt_ms.ge(0) & dt_ms.le(window_ms)
    return trades.loc[~duplicate].reset_index(drop=True)


def build_ticker_day(
    book_path: Path,
    trades_path: Optional[Path],
    horizon_min: int = 5,
    book_grid: str = "1s",
    min_seconds_per_minute: int = 5,
    include_trades: bool = True,
    trade_dedupe_ms: int = 20,
) -> Optional[pd.DataFrame]:
    book = _load_book_day(book_path)
    if book is None:
        return None

    book = book[
        (book.bid_p0 > 0)
        & (book.ask_p0 > 0)
        & (book.ask_p0 > book.bid_p0)
    ].reset_index(drop=True)
    if len(book) < 10:
        return None

    book = (
        book.set_index("time")
        .between_time("07:00", "20:50")
        .reset_index()
    )
    if len(book) < 10:
        return None

    book = _regularize_book(book, grid=book_grid)
    if len(book) < 10:
        return None

    book["mid"] = (book.bid_p0 + book.ask_p0) / 2
    book["spread_bp"] = (book.ask_p0 - book.bid_p0) / book.mid * 1e4

    bidq5 = book[[f"bid_q{i}" for i in range(5)]].sum(axis=1)
    askq5 = book[[f"ask_q{i}" for i in range(5)]].sum(axis=1)
    book["imb1"] = (
        (book.bid_q0 - book.ask_q0)
        / (book.bid_q0 + book.ask_q0 + 1e-9)
    )
    book["imb5"] = (bidq5 - askq5) / (bidq5 + askq5 + 1e-9)

    micro = (
        book.bid_p0 * book.ask_q0 + book.ask_p0 * book.bid_q0
    ) / (book.bid_q0 + book.ask_q0 + 1e-9)
    book["micro_dev"] = (micro - book.mid) / book.mid * 1e4

    prev = book.shift(1)
    ofi = (
        (book.bid_p0 >= prev.bid_p0).astype(float) * book.bid_q0
        - (book.bid_p0 <= prev.bid_p0).astype(float) * prev.bid_q0
        - (book.ask_p0 <= prev.ask_p0).astype(float) * book.ask_q0
        + (book.ask_p0 >= prev.ask_p0).astype(float) * prev.ask_q0
    )
    book["ofi"] = ofi.fillna(0.0)

    g = book.set_index("time").resample("1min")
    bars = pd.DataFrame({
        "mid": g["mid"].last(),
        "spread_bp": g["spread_bp"].mean(),
        "imb1": g["imb1"].mean(),
        "imb5": g["imb5"].mean(),
        "micro_dev": g["micro_dev"].mean(),
        "ofi": g["ofi"].sum(),
        "n_snap": g["mid"].count(),
    })

    if include_trades and trades_path is not None and trades_path.exists():
        try:
            trades = _read_csv_robust(trades_path, TRADE_COLS)
            if trades is None or trades.empty:
                raise ValueError("empty trades")
            trades["time"] = pd.to_datetime(
                trades["time"], utc=True, format="ISO8601", errors="coerce"
            )
            trades = trades.dropna(subset=["time"]).sort_values("time")
            trades = _dedupe_near_trades(trades, trade_dedupe_ms)
            trades["signed"] = np.select(
                [trades.direction.eq(1), trades.direction.eq(2)],
                [trades.quantity, -trades.quantity],
                default=0.0,
            )
            tfi = trades.set_index("time").resample("1min")["signed"].sum()
            bars["tfi"] = tfi.reindex(bars.index).fillna(0.0)
        except Exception:
            bars["tfi"] = 0.0
    else:
        bars["tfi"] = 0.0

    # Только рабочие дни.
    bars = bars[bars.index.dayofweek < 5].copy()
    if bars.empty:
        return None

    mid = bars["mid"].astype(float)

    # Доходности только по точным временным отметкам.
    prev_1 = mid.reindex(
        bars.index - pd.Timedelta(minutes=1)
    ).to_numpy()
    bars["ret_1m"] = mid.to_numpy() / prev_1 - 1.0

    future_mid = mid.reindex(
        bars.index + pd.Timedelta(minutes=horizon_min)
    ).to_numpy()
    bars["fwd_ret"] = future_mid / mid.to_numpy() - 1.0

    # Временные окна не перескакивают длительные разрывы.
    for col in ("ofi", "tfi"):
        std = bars[col].rolling("60min", min_periods=20).std()
        bars[col] = (
            bars[col] / std.replace(0, np.nan)
        ).clip(-10, 10).fillna(0.0)

    for window in (5, 15, 30):
        bars[f"ofi_{window}"] = (
            bars["ofi"]
            .rolling(f"{window}min", min_periods=2)
            .sum()
            .fillna(0.0)
        )

    bars["tfi_5"] = (
        bars["tfi"]
        .rolling("5min", min_periods=2)
        .sum()
        .fillna(0.0)
    )

    prev_5 = mid.reindex(
        bars.index - pd.Timedelta(minutes=5)
    ).to_numpy()
    bars["ret_5m"] = mid.to_numpy() / prev_5 - 1.0

    idx = bars.index
    bars["tod"] = (idx.hour * 60 + idx.minute - 420) / 480.0
    bars["evening"] = (idx.hour >= 16).astype(float)

    bars = bars[bars.n_snap >= min_seconds_per_minute]
    bars = bars.replace([np.inf, -np.inf], np.nan)
    required = [
        "mid", "spread_bp", "imb1", "imb5", "micro_dev", "ofi",
        "tfi", "ret_1m", "ofi_5", "ofi_15", "ofi_30",
        "tfi_5", "ret_5m", "fwd_ret",
    ]
    bars = bars.dropna(subset=required)

    if len(bars) < horizon_min + 10:
        return None
    return bars.reset_index()


def build_dataset(
    lob_dir: Path,
    horizon_min: int = 5,
    book_grid: str = "1s",
    min_seconds_per_minute: int = 5,
    include_trades: bool = True,
    trade_dedupe_ms: int = 20,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    book_files = sorted(
        list(lob_dir.glob("*_book.csv"))
        + list(lob_dir.glob("*_book.csv.gz"))
    )

    for book_path in book_files:
        name = (
            book_path.name
            .replace("_book.csv.gz", "")
            .replace("_book.csv", "")
        )
        ticker = name.split("_", 1)[1]
        trades_path = book_path.with_name(name + "_trades.csv")
        if not trades_path.exists():
            gz = book_path.with_name(name + "_trades.csv.gz")
            trades_path = gz if gz.exists() else trades_path

        bars = build_ticker_day(
            book_path,
            trades_path,
            horizon_min=horizon_min,
            book_grid=book_grid,
            min_seconds_per_minute=min_seconds_per_minute,
            include_trades=include_trades,
            trade_dedupe_ms=trade_dedupe_ms,
        )
        if bars is not None and len(bars):
            bars["ticker"] = ticker
            rows.append(bars)

    if not rows:
        return pd.DataFrame()

    df = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["time", "ticker"])
        .reset_index(drop=True)
    )

    if "SBER" in set(df["ticker"]):
        market = (
            df[df.ticker == "SBER"][["time", "ofi_5"]]
            .drop_duplicates(subset=["time"], keep="last")
            .rename(columns={"ofi_5": "mkt_ofi"})
            .sort_values("time")
        )
        df = pd.merge_asof(
            df.sort_values("time"),
            market,
            on="time",
            direction="backward",
            tolerance=pd.Timedelta("5min"),
        )
        df["mkt_ofi"] = df["mkt_ofi"].fillna(0.0)
    else:
        df["mkt_ofi"] = 0.0

    return df.sort_values(["time", "ticker"]).reset_index(drop=True)


FEATURES_BASIC = [
    "ofi", "tfi", "imb1", "imb5",
    "micro_dev", "spread_bp", "ret_1m",
]
FEATURES = FEATURES_BASIC + [
    "ofi_5", "ofi_15", "ofi_30",
    "tfi_5", "ret_5m", "tod",
    "evening", "mkt_ofi",
]

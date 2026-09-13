from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.lob.dataset import (
    FEATURES_BASIC,
    TRADE_COLS,
    _dedupe_near_trades,
    _load_book_day,
    _read_csv_robust,
    _regularize_book,
)


STATE = PROJECT_ROOT / "state"
STRATEGY_CONFIG = STATE / "shadow_strategy_v1.json"
OUTPUT = STATE / "shadow_observer_features.csv"

BOOK_GRID = "1s"
MIN_SECONDS_PER_MINUTE = 5
TRADE_DEDUPE_MS = 20


def find_daily_file(
    lob_dir: Path,
    day: str,
    ticker: str,
    kind: str,
) -> Path | None:
    plain = lob_dir / f"{day}_{ticker}_{kind}.csv"
    gz = lob_dir / f"{day}_{ticker}_{kind}.csv.gz"

    if plain.exists():
        return plain

    if gz.exists():
        return gz

    return None


def build_live_features(
    book_path: Path,
    trades_path: Path | None,
) -> pd.DataFrame | None:
    """
    Causal counterpart of build_ticker_day().

    Important:
    - no fwd_ret;
    - no future_mid;
    - all seven FEATURES_BASIC use only data available
      up to the current minute.
    """

    book = _load_book_day(book_path)

    if book is None or len(book) < 10:
        return None

    book = book[
        (book.bid_p0 > 0)
        & (book.ask_p0 > 0)
        & (book.ask_p0 > book.bid_p0)
    ].reset_index(drop=True)

    if len(book) < 10:
        return None

    # Same trading-hours filter as research pipeline.
    book = (
        book.set_index("time")
        .between_time("07:00", "20:50")
        .reset_index()
    )

    if len(book) < 10:
        return None

    book = _regularize_book(
        book,
        grid=BOOK_GRID,
    )

    if len(book) < 10:
        return None

    # ---------- book features ----------
    book["mid"] = (
        book.bid_p0 + book.ask_p0
    ) / 2.0

    book["spread_bp"] = (
        (book.ask_p0 - book.bid_p0)
        / book.mid
        * 1e4
    )

    bidq5 = book[
        [f"bid_q{i}" for i in range(5)]
    ].sum(axis=1)

    askq5 = book[
        [f"ask_q{i}" for i in range(5)]
    ].sum(axis=1)

    book["imb1"] = (
        (book.bid_q0 - book.ask_q0)
        / (
            book.bid_q0
            + book.ask_q0
            + 1e-9
        )
    )

    book["imb5"] = (
        (bidq5 - askq5)
        / (bidq5 + askq5 + 1e-9)
    )

    micro = (
        book.bid_p0 * book.ask_q0
        + book.ask_p0 * book.bid_q0
    ) / (
        book.bid_q0
        + book.ask_q0
        + 1e-9
    )

    book["micro_dev"] = (
        (micro - book.mid)
        / book.mid
        * 1e4
    )

    prev = book.shift(1)

    ofi = (
        (book.bid_p0 >= prev.bid_p0).astype(float)
        * book.bid_q0
        - (book.bid_p0 <= prev.bid_p0).astype(float)
        * prev.bid_q0
        - (book.ask_p0 <= prev.ask_p0).astype(float)
        * book.ask_q0
        + (book.ask_p0 >= prev.ask_p0).astype(float)
        * prev.ask_q0
    )

    book["ofi"] = ofi.fillna(0.0)

    # ---------- minute aggregation ----------
    g = (
        book
        .set_index("time")
        .resample("1min")
    )

    bars = pd.DataFrame(
        {
            "mid": g["mid"].last(),
            "spread_bp": g["spread_bp"].mean(),
            "imb1": g["imb1"].mean(),
            "imb5": g["imb5"].mean(),
            "micro_dev": g["micro_dev"].mean(),
            "ofi": g["ofi"].sum(),
            "n_snap": g["mid"].count(),
        }
    )

    # ---------- trades / TFI ----------
    if (
        trades_path is not None
        and trades_path.exists()
    ):
        try:
            trades = _read_csv_robust(
                trades_path,
                TRADE_COLS,
            )

            if trades is None or trades.empty:
                raise ValueError("empty trades")

            trades["time"] = pd.to_datetime(
                trades["time"],
                utc=True,
                format="ISO8601",
                errors="coerce",
            )

            trades = (
                trades
                .dropna(subset=["time"])
                .sort_values("time")
            )

            trades = _dedupe_near_trades(
                trades,
                TRADE_DEDUPE_MS,
            )

            trades["signed"] = np.select(
                [
                    trades.direction.eq(1),
                    trades.direction.eq(2),
                ],
                [
                    trades.quantity,
                    -trades.quantity,
                ],
                default=0.0,
            )

            tfi = (
                trades
                .set_index("time")
                .resample("1min")["signed"]
                .sum()
            )

            bars["tfi"] = (
                tfi
                .reindex(bars.index)
                .fillna(0.0)
            )

        except Exception:
            bars["tfi"] = 0.0

    else:
        bars["tfi"] = 0.0

    # Do NOT filter by weekday here.
    # MOEX may have official weekend trading sessions.
    # If raw market data exists for the date, live causal features
    # are allowed to be produced for that session.

    if bars.empty:
        return None

    # Keep this BEFORE n_snap filtering,
    # exactly as in research pipeline.
    mid = bars["mid"].astype(float)

    prev_1 = mid.reindex(
        bars.index
        - pd.Timedelta(minutes=1)
    ).to_numpy()

    bars["ret_1m"] = (
        mid.to_numpy() / prev_1 - 1.0
    )

    # Same causal rolling normalization.
    for col in ("ofi", "tfi"):
        std = (
            bars[col]
            .rolling(
                "60min",
                min_periods=20,
            )
            .std()
        )

        bars[col] = (
            bars[col]
            / std.replace(0, np.nan)
        ).clip(-10, 10).fillna(0.0)

    bars = bars[
        bars.n_snap
        >= MIN_SECONDS_PER_MINUTE
    ].copy()

    bars = bars.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    bars = bars.dropna(
        subset=[
            "mid",
            "spread_bp",
            *FEATURES_BASIC,
        ]
    )

    if bars.empty:
        return None

    return bars.reset_index()


def load_strategy():
    if not STRATEGY_CONFIG.exists():
        return {
            "armed": False,
            "reason": "strategy_config_missing",
        }, None, None

    with open(
        STRATEGY_CONFIG,
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    if not config.get("armed", False):
        return config, None, None

    direction_path = Path(
        config["direction_model"]
    )

    magnitude_path = Path(
        config["magnitude_model"]
    )

    if (
        not direction_path.exists()
        or not magnitude_path.exists()
    ):
        raise RuntimeError(
            "Strategy armed, but model files are missing."
        )

    direction_model = joblib.load(
        direction_path
    )

    magnitude_model = joblib.load(
        magnitude_path
    )

    return (
        config,
        direction_model,
        magnitude_model,
    )


def decide(
    row: pd.Series,
    config: dict,
    direction_model,
    magnitude_model,
) -> dict:
    result = {
        "p_up": np.nan,
        "p_large": np.nan,
        "side": 0,
        "decision": "OBSERVE_UNARMED",
    }

    if not config.get("armed", False):
        return result

    max_spread = float(
        config["max_entry_spread_bp"]
    )

    if float(row["spread_bp"]) > max_spread:
        result["decision"] = "BLOCK_SPREAD"
        return result

    X = pd.DataFrame(
        [
            {
                f: float(row[f])
                for f in FEATURES_BASIC
            }
        ]
    )

    p_up = float(
        direction_model.predict_proba(X)[0, 1]
    )

    p_large = float(
        magnitude_model.predict_proba(X)[0, 1]
    )

    dir_thr = float(
        config["direction_threshold"]
    )

    mag_thr = float(
        config["magnitude_threshold"]
    )

    side = 0
    decision = "FLAT"

    if (
        p_up >= dir_thr
        and p_large >= mag_thr
    ):
        side = 1
        decision = "WOULD_LONG"

    elif (
        p_up <= 1.0 - dir_thr
        and p_large >= mag_thr
    ):
        side = -1
        decision = "WOULD_SHORT"

    result.update(
        {
            "p_up": p_up,
            "p_large": p_large,
            "side": side,
            "decision": decision,
        }
    )

    return result


def load_seen() -> set[tuple[str, str]]:
    if not OUTPUT.exists():
        return set()

    try:
        old = pd.read_csv(
            OUTPUT,
            usecols=["time", "ticker"],
        )

        return set(
            zip(
                old["time"].astype(str),
                old["ticker"].astype(str),
            )
        )
    except Exception:
        return set()


def append_rows(rows: list[dict]) -> None:
    if not rows:
        return

    out = pd.DataFrame(rows)

    header = not OUTPUT.exists()

    out.to_csv(
        OUTPUT,
        mode="a",
        header=header,
        index=False,
    )


def observe_once(
    lob_dir: Path,
    tickers: list[str],
    config: dict,
    direction_model,
    magnitude_model,
    seen: set[tuple[str, str]],
) -> int:
    day = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")

    now = pd.Timestamp.now(tz="UTC")

    # Current minute is incomplete.
    # Use latest fully completed minute.
    complete_cutoff = (
        now.floor("min")
        - pd.Timedelta(minutes=1)
    )

    rows = []

    for ticker in tickers:
        book_path = find_daily_file(
            lob_dir,
            day,
            ticker,
            "book",
        )

        if book_path is None:
            continue

        trades_path = find_daily_file(
            lob_dir,
            day,
            ticker,
            "trades",
        )

        bars = build_live_features(
            book_path,
            trades_path,
        )

        if bars is None or bars.empty:
            continue

        complete = bars[
            bars["time"]
            <= complete_cutoff
        ]

        if complete.empty:
            continue

        # Catch-up:
        # записываем ВСЕ завершённые минуты,
        # которых ещё нет в observer log.
        for _, row in complete.iterrows():
            key = (
                pd.Timestamp(
                    row["time"]
                ).isoformat(),
                ticker,
            )

            if key in seen:
                continue

            signal = decide(
                row,
                config,
                direction_model,
                magnitude_model,
            )

            rec = {
                "observed_at_utc": (
                    now.isoformat()
                ),
                "time": key[0],
                "ticker": ticker,
                "mid": float(row["mid"]),
                "n_snap": int(row["n_snap"]),
                **{
                    f: float(row[f])
                    for f in FEATURES_BASIC
                },
                **signal,
            }

            rows.append(rec)
            seen.add(key)

            print(
                f"{row['time']} "
                f"{ticker:6s} | "
                f"mid={row['mid']:.4f} | "
                f"spr={row['spread_bp']:.3f} | "
                f"ofi={row['ofi']:+.3f} | "
                f"tfi={row['tfi']:+.3f} | "
                f"{signal['decision']}"
            )

    append_rows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--once",
        action="store_true",
        help="One pass and exit.",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds.",
    )

    args = parser.parse_args()

    cfg = load_config()

    lob_dir = cfg.abs_path(
        cfg.data["cache_dir"],
        "lob",
    )

    tickers = list(
        cfg.data["figis"].keys()
    )

    (
        strategy,
        direction_model,
        magnitude_model,
    ) = load_strategy()

    print("=" * 92)
    print("SHADOW OBSERVER")
    print("=" * 92)

    print("LOB dir:", lob_dir)
    print("Tickers:", len(tickers))
    print(
        "Strategy armed:",
        bool(strategy.get("armed", False)),
    )

    if not strategy.get("armed", False):
        print(
            "Mode: OBSERVE ONLY — "
            "никаких торговых сигналов."
        )
        print(
            "Reason:",
            strategy.get(
                "reason",
                "not armed",
            ),
        )
    else:
        print(
            "Mode: SHADOW SIGNALS — "
            "никаких заявок брокеру."
        )

    print(
        "Output:",
        OUTPUT,
    )

    print("=" * 92)

    seen = load_seen()

    while True:
        try:
            n = observe_once(
                lob_dir,
                tickers,
                strategy,
                direction_model,
                magnitude_model,
                seen,
            )

            print(
                f"[observer] "
                f"{datetime.now():%Y-%m-%d %H:%M:%S} | "
                f"new rows={n}"
            )

            if args.once:
                break

            time.sleep(
                max(10, args.interval)
            )

        except KeyboardInterrupt:
            print(
                "\n[observer] stopped"
            )
            break

        except Exception as exc:
            print(
                f"[observer] ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            if args.once:
                raise

            time.sleep(
                max(10, args.interval)
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Forward-only shadow portfolio for the slow MOEX momentum candidate.

The script NEVER sends orders. It refreshes daily candles, records immutable
signals, executes pending target weights at the next trading day's OPEN in a
local paper ledger, applies turnover costs, and marks the portfolio at CLOSE.

Candidate fixed before forward testing:
  lookback=252, skip=0, top_n=4, regime_win=100,
  rebalance every 10 trading days, target volatility=15%, long-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOOKBACK = 252
SKIP = 0
TOP_N = 4
VOL_WIN = 60
TARGET_VOL = 0.15
REGIME_WIN = 100
REBALANCE_DAYS = 10
ANN = 252
VERSION = "shadow-momentum-v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--state-dir", default=str(ROOT / "state" / "shadow_momentum_v1"))
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--cost-bp", type=float, default=14.0,
                   help="Cost per unit of turnover, bp. 14 is deliberately conservative.")
    p.add_argument("--refresh", action="store_true",
                   help="Refresh the last 45 calendar days from T-Invest before calculation.")
    p.add_argument("--reset", action="store_true",
                   help="Start a new shadow ledger. Existing files are moved to a timestamped backup.")
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("data", {}).get("interval") != "day":
        raise RuntimeError("Shadow momentum requires config data.interval: day")
    return cfg


def refresh_one(figi: str, cache_file: Path) -> None:
    from src.data.loader import download_candles
    new = download_candles(figi, interval="day", history_days=45)
    if cache_file.exists():
        old = pd.read_csv(cache_file, parse_dates=["time"])
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new
    merged["time"] = pd.to_datetime(merged["time"], utc=True)
    merged = (merged.drop_duplicates("time", keep="last")
                    .sort_values("time").reset_index(drop=True))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_file, index=False)


def load_candles(cfg: dict, cache_dir: Path, refresh: bool) -> Dict[str, pd.DataFrame]:
    history_days = int(cfg["data"].get("history_days", 1825))
    out: Dict[str, pd.DataFrame] = {}
    for ticker, figi in cfg["data"]["figis"].items():
        path = cache_dir / f"{figi}_day_{history_days}d.csv"
        if refresh:
            try:
                refresh_one(figi, path)
                print(f"[refresh] {ticker}: OK")
            except Exception as e:  # noqa: BLE001
                print(f"[refresh] {ticker}: ERROR: {e}")
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, parse_dates=["time"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.drop_duplicates("time", keep="last").sort_values("time")
            if len(df) >= 320:
                out[ticker] = df
        except Exception as e:  # noqa: BLE001
            print(f"[load] {ticker}: ERROR: {e}")
    if not out:
        raise RuntimeError("No usable daily candle files")

    global_last = max(df["time"].max() for df in out.values())
    # Do not let delisted/renamed/stale instruments pollute the synthetic index.
    out = {
        tk: df for tk, df in out.items()
        if (global_last - df["time"].max()).days <= 7
    }
    if len(out) < 10:
        raise RuntimeError(f"Only {len(out)} fresh instruments remain; aborting")
    return out


def panels(candles: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_s, open_s = {}, {}
    for tk, df in candles.items():
        d = df.set_index("time").sort_index()
        close_s[tk] = pd.to_numeric(d["close"], errors="coerce")
        open_s[tk] = pd.to_numeric(d["open"], errors="coerce")
    close = pd.DataFrame(close_s).sort_index().ffill().dropna()
    opn = pd.DataFrame(open_s).sort_index().reindex(close.index).ffill()
    opn = opn[close.columns].dropna()
    common = close.index.intersection(opn.index)
    close, opn = close.loc[common], opn.loc[common]
    if len(close) < 320:
        raise RuntimeError(f"Only {len(close)} aligned days; need at least 320")
    return close, opn


def target_weights(close: pd.DataFrame) -> dict[str, float]:
    names = np.array(close.columns)
    P = close.to_numpy(dtype=float)
    if len(P) < max(LOOKBACK + SKIP, REGIME_WIN, VOL_WIN) + 2:
        raise RuntimeError("Insufficient aligned history")
    rets = np.diff(P, axis=0) / P[:-1]
    idx = (P / P[0]).mean(axis=1)
    idx_sma = pd.Series(idx).rolling(REGIME_WIN).mean().to_numpy()
    bull = bool(idx[-1] > idx_sma[-1])

    vol = rets[-VOL_WIN:].std(axis=0) * np.sqrt(ANN)
    frozen = (np.abs(rets[-VOL_WIN:]) < 1e-12).mean(axis=0)
    active = (frozen < 0.30) & (vol > 0.03)
    mom = P[-1 - SKIP] / P[-1 - LOOKBACK] - 1.0
    order = np.argsort(mom)[::-1]
    chosen = [i for i in order if mom[i] > 0 and active[i]][:TOP_N]

    w = np.zeros(len(names), dtype=float)
    basket_vol = 0.0
    scale = 0.0
    if bull and chosen:
        inv = np.array([1.0 / max(vol[i], 0.05) for i in chosen])
        raw = inv / inv.sum()
        basket = rets[-VOL_WIN:, chosen] @ raw
        basket_vol = float(basket.std() * np.sqrt(ANN))
        scale = float(min(1.0, TARGET_VOL / max(basket_vol, 1e-4)))
        w[chosen] = raw * scale

    return {
        "weights": {str(names[i]): float(w[i]) for i in np.where(w > 1e-10)[0]},
        "cash_weight": float(1.0 - w.sum()),
        "bull_regime": bull,
        "regime_strength": float(idx[-1] / idx_sma[-1] - 1.0),
        "basket_vol": basket_vol,
        "vol_scale": scale,
        "momentum": {str(names[i]): float(mom[i]) for i in chosen},
    }


def vec(weights: dict[str, float], columns: pd.Index) -> np.ndarray:
    return np.array([float(weights.get(str(c), 0.0)) for c in columns], dtype=float)


def data_fingerprint(close: pd.DataFrame) -> str:
    payload = f"{close.index[-1]}|{len(close)}|{','.join(close.columns)}|" + \
              ",".join(f"{x:.8g}" for x in close.iloc[-1].to_numpy())
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def reset_state(state_dir: Path) -> None:
    if state_dir.exists() and any(state_dir.iterdir()):
        backup = state_dir.with_name(state_dir.name + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        state_dir.rename(backup)
        print(f"Previous ledger moved to {backup}")


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_cfg(cfg_path)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (ROOT / cfg["data"]["cache_dir"])
    state_dir = Path(args.state_dir).resolve()
    if args.reset:
        reset_state(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    candles = load_candles(cfg, cache_dir, args.refresh)
    close, opn = panels(candles)
    latest_date = close.index[-1]
    state_file = state_dir / "state.json"
    signals_file = state_dir / "signals.jsonl"
    history_file = state_dir / "equity.csv"
    executions_file = state_dir / "executions.jsonl"
    cost = args.cost_bp / 1e4

    params = {
        "lookback": LOOKBACK, "skip": SKIP, "top_n": TOP_N,
        "vol_win": VOL_WIN, "target_vol": TARGET_VOL,
        "regime_win": REGIME_WIN, "rebalance_days": REBALANCE_DAYS,
        "cost_bp_per_turnover": args.cost_bp,
    }

    if not state_file.exists():
        signal = target_weights(close)
        state = {
            "version": VERSION,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "last_date": str(latest_date.date()),
            "equity": float(args.capital),
            "weights": {},
            "pending_weights": signal["weights"],
            "pending_signal_date": str(latest_date.date()),
            "days_since_rebalance": 0,
            "params": params,
            "data_fingerprint": data_fingerprint(close),
        }
        append_jsonl(signals_file, {
            "created_at": utc_now(), "signal_date": str(latest_date.date()),
            "effective": "next trading day open", "status": "pending",
            **signal, "params": params,
        })
        atomic_json(state_file, state)
        pd.DataFrame([{
            "date": str(latest_date.date()), "equity": args.capital,
            "daily_return": 0.0, "cost": 0.0, "gross_exposure": 0.0,
            "cash_weight": 1.0, "event": "initialized; first signal pending",
        }]).to_csv(history_file, index=False)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        print("Initialized. First target will execute at the next trading-day open.")
        return

    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state.get("version") != VERSION:
        raise RuntimeError(f"State version mismatch: {state.get('version')} != {VERSION}")
    last_date = pd.Timestamp(state["last_date"], tz="UTC")
    new_dates = close.index[close.index > last_date]
    if len(new_dates) == 0:
        state["updated_at"] = utc_now()
        state["data_fingerprint"] = data_fingerprint(close)
        atomic_json(state_file, state)
        print(f"No new daily bar. Latest cache date: {latest_date.date()}")
        return

    hist_rows = []
    weights = dict(state.get("weights", {}))
    pending = state.get("pending_weights")
    pending_signal_date = state.get("pending_signal_date")
    equity = float(state["equity"])
    days_since = int(state.get("days_since_rebalance", 0))
    columns = close.columns

    for d in new_dates:
        i = close.index.get_loc(d)
        prev_d = close.index[i - 1]
        old_w = vec(weights, columns)
        overnight = opn.loc[d].to_numpy(float) / close.loc[prev_d].to_numpy(float) - 1.0
        overnight_ret = float(old_w @ overnight)
        day_cost = 0.0
        event = "hold"

        if pending is not None:
            new_w = vec(pending, columns)
            turnover = float(np.abs(new_w - old_w).sum())
            day_cost = turnover * cost
            append_jsonl(executions_file, {
                "executed_at": utc_now(), "execution_date": str(d.date()),
                "execution_price": "daily open", "signal_date": pending_signal_date,
                "old_weights": weights, "new_weights": pending,
                "turnover": turnover, "cost_fraction": day_cost,
            })
            weights = {str(c): float(x) for c, x in zip(columns, new_w) if x > 1e-10}
            old_w = new_w
            pending = None
            pending_signal_date = None
            event = "executed pending target at open"

        intraday = close.loc[d].to_numpy(float) / opn.loc[d].to_numpy(float) - 1.0
        intraday_ret = float(old_w @ intraday)
        daily_ret = overnight_ret + intraday_ret - day_cost
        equity *= (1.0 + daily_ret)
        days_since += 1

        if days_since >= REBALANCE_DAYS:
            signal = target_weights(close.loc[:d])
            pending = signal["weights"]
            pending_signal_date = str(d.date())
            days_since = 0
            append_jsonl(signals_file, {
                "created_at": utc_now(), "signal_date": str(d.date()),
                "effective": "next trading day open", "status": "pending",
                **signal, "params": params,
            })
            event += "; new rebalance signal pending"

        gross = float(sum(weights.values()))
        hist_rows.append({
            "date": str(d.date()), "equity": equity,
            "daily_return": daily_ret, "overnight_return": overnight_ret,
            "intraday_return": intraday_ret, "cost": day_cost,
            "gross_exposure": gross, "cash_weight": 1.0 - gross,
            "event": event,
        })
        state["last_date"] = str(d.date())

    if history_file.exists():
        old_hist = pd.read_csv(history_file)
        new_hist = pd.concat([old_hist, pd.DataFrame(hist_rows)], ignore_index=True)
    else:
        new_hist = pd.DataFrame(hist_rows)
    new_hist = new_hist.drop_duplicates("date", keep="last").sort_values("date")
    tmp_hist = history_file.with_suffix(".csv.tmp")
    new_hist.to_csv(tmp_hist, index=False)
    os.replace(tmp_hist, history_file)

    state.update({
        "updated_at": utc_now(), "equity": equity, "weights": weights,
        "pending_weights": pending, "pending_signal_date": pending_signal_date,
        "days_since_rebalance": days_since, "params": params,
        "data_fingerprint": data_fingerprint(close),
    })
    atomic_json(state_file, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    print(f"Processed {len(new_dates)} new day(s). Ledger: {history_file}")


if __name__ == "__main__":
    main()

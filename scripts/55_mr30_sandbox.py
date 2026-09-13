#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path("/root/rl-trading-tbank")
STATE_DIR = ROOT / "state"

SIGNALS_FILE = STATE_DIR / "mr30_v3_shadow_signals.csv"

BOT_STATE_FILE = STATE_DIR / "mr30_sandbox_state.json"
JOURNAL_FILE = STATE_DIR / "mr30_sandbox_journal.csv"

ARM = "NO_BROAD_TREND"

HOLD_MINUTES = int(os.getenv("MR30_HOLD_MINUTES", "30"))
POLL_SEC = int(os.getenv("MR30_POLL_SEC", "10"))
LOTS_PER_TRADE = int(os.getenv("MR30_LOTS", "1"))
MAX_OPEN_POSITIONS = int(os.getenv("MR30_MAX_OPEN", "5"))

# Пока deliberately conservative.
MAX_DAILY_LOSS_PCT = float(
    os.getenv("MR30_MAX_DAILY_LOSS_PCT", "0.01")
)

load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.loader import get_lot_sizes
from src.notify.telegram import TelegramNotifier


def utcnow():
    return datetime.now(timezone.utc)


def moscow_time(dt=None):
    from zoneinfo import ZoneInfo

    if dt is None:
        dt = utcnow()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(
        ZoneInfo("Europe/Moscow")
    )


def fmt_moscow(dt=None):
    return moscow_time(dt).strftime(
        "%d.%m.%Y %H:%M:%S МСК"
    )


def load_state():
    if not BOT_STATE_FILE.exists():
        return {
            "bootstrapped": False,
            "processed": [],
            "positions": {},
            "account_id": None,
            "day": None,
            "day_start_equity": None,
            "paused": False,
        }

    try:
        return json.loads(
            BOT_STATE_FILE.read_text(encoding="utf-8")
        )
    except Exception:
        return {
            "bootstrapped": False,
            "processed": [],
            "positions": {},
            "account_id": None,
            "day": None,
            "day_start_equity": None,
            "paused": False,
        }


def save_state(st):
    tmp = BOT_STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            st,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(BOT_STATE_FILE)


JOURNAL_COLUMNS = [
    "event",
    "time",
    "ticker",
    "side",
    "lots",
    "signal_time",
    "entry_price",
    "exit_price",
    "entry_total_amount",
    "exit_total_amount",
    "entry_commission",
    "exit_commission",
    "commission_rub",
    "gross_pnl_rub",
    "net_pnl_rub",
    "gross_pnl_bp",
    "net_pnl_bp",
    "close_at",
    "arm",
]


def append_journal(row):
    df = pd.DataFrame([row])

    df = df.reindex(
        columns=JOURNAL_COLUMNS
    )

    if JOURNAL_FILE.exists():
        df.to_csv(
            JOURNAL_FILE,
            mode="a",
            header=False,
            index=False,
        )
    else:
        df.to_csv(
            JOURNAL_FILE,
            index=False,
        )


def q2f(q):
    return float(q.units) + float(q.nano) / 1e9


def detect_col(df, names):
    for c in names:
        if c in df.columns:
            return c
    return None


def load_no_broad_signals():
    if not SIGNALS_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(SIGNALS_FILE)
    except Exception as e:
        print("[signals] read error:", repr(e))
        return pd.DataFrame()

    if df.empty:
        return df

    arm_col = detect_col(
        df,
        ["arm", "strategy", "variant"],
    )

    if arm_col:
        df = df[
            df[arm_col].astype(str).eq(ARM)
        ].copy()

    if df.empty:
        return df

    time_col = detect_col(
        df,
        [
            "entry_time",
            "signal_time",
            "time",
            "timestamp",
        ],
    )

    ticker_col = detect_col(
        df,
        ["ticker", "symbol"],
    )

    side_col = detect_col(
        df,
        [
            "side",
            "direction",
            "position",
            "signal",
        ],
    )

    if not time_col or not ticker_col or not side_col:
        print()
        print("[FATAL] Не удалось определить схему signals CSV")
        print("columns:", list(df.columns))
        raise SystemExit(2)

    df["_time"] = pd.to_datetime(
        df[time_col],
        errors="coerce",
        utc=True,
    )

    df["_ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    def parse_side(v):
        s = str(v).strip().upper()

        if s in {
            "1",
            "1.0",
            "+1",
            "LONG",
            "BUY",
        }:
            return 1

        if s in {
            "-1",
            "-1.0",
            "SHORT",
            "SELL",
        }:
            return -1

        try:
            x = float(v)
            if x > 0:
                return 1
            if x < 0:
                return -1
        except Exception:
            pass

        return 0

    df["_side"] = df[side_col].map(parse_side)

    df = df[
        df["_time"].notna()
        & df["_ticker"].ne("")
        & df["_side"].isin([-1, 1])
    ].copy()

    df = df.sort_values("_time")

    df["_key"] = (
        df["_time"].astype(str)
        + "|"
        + df["_ticker"]
        + "|"
        + df["_side"].astype(str)
    )

    return df


def portfolio_value(client, account_id):
    pf = client.sandbox.get_sandbox_portfolio(
        account_id=account_id
    )
    return q2f(pf.total_amount_portfolio)


def get_prices(client, figis):
    ids = list(figis.values())

    if not ids:
        return {}

    resp = client.market_data.get_last_prices(
        instrument_id=ids
    )

    reverse = {
        instrument_id: ticker
        for ticker, instrument_id in figis.items()
    }

    out = {}

    for lp in resp.last_prices:
        instrument_id = getattr(
            lp,
            "instrument_uid",
            None,
        ) or getattr(lp, "figi", None)

        ticker = reverse.get(instrument_id)

        # SDK может вернуть FIGI в lp.figi,
        # а config хранить instrument UID.
        if ticker is None:
            figi = getattr(lp, "figi", None)
            ticker = reverse.get(figi)

        if ticker:
            out[ticker] = q2f(lp.price)

    return out


def execute_market(
    client,
    account_id,
    figis,
    ticker,
    signed_lots,
):
    from tinkoff.invest import (
        OrderDirection,
        OrderType,
    )

    instrument_id = figis.get(ticker)

    if not instrument_id:
        print(
            f"[order] {ticker}: instrument id not found"
        )
        return False, {}

    if signed_lots == 0:
        return False, {}

    direction = (
        OrderDirection.ORDER_DIRECTION_BUY
        if signed_lots > 0
        else OrderDirection.ORDER_DIRECTION_SELL
    )

    try:
        resp = client.sandbox.post_sandbox_order(
            instrument_id=instrument_id,
            quantity=abs(int(signed_lots)),
            direction=direction,
            account_id=account_id,
            order_type=OrderType.ORDER_TYPE_MARKET,
            order_id=str(uuid.uuid4()),
        )

        def money_attr(name):
            q = getattr(resp, name, None)

            if q is None:
                return None

            try:
                return q2f(q)
            except Exception:
                return None

        price = (
            money_attr("executed_order_price")
            or money_attr("initial_order_price")
        )

        commission = money_attr("executed_commission")
        total_amount = money_attr("total_order_amount")

        details = {
            "price": price,
            "commission": commission,
            "total_amount": total_amount,
            "order_id": getattr(resp, "order_id", None),
            "execution_report_status": str(
                getattr(
                    resp,
                    "execution_report_status",
                    "",
                )
            ),
        }

        return True, details

    except Exception as e:
        print(
            f"[order] {ticker} ERROR:",
            repr(e),
        )
        return False, {}

def notify(notifier, text):
    print(text)

    try:
        notifier.send(text)
    except Exception as e:
        print("[telegram] error:", repr(e))


def bootstrap_existing_signals(st):
    df = load_no_broad_signals()

    if df.empty:
        print(
            "[bootstrap] no NO_BROAD signals yet"
        )
        st["bootstrapped"] = True
        save_state(st)
        return

    keys = df["_key"].tolist()

    st["processed"] = keys[-5000:]
    st["bootstrapped"] = True

    save_state(st)

    print(
        "[bootstrap] existing signals marked processed:",
        len(keys),
    )

    print(
        "[bootstrap] latest:",
        df["_time"].max(),
    )

    print(
        "[bootstrap] IMPORTANT: historical signals "
        "will NOT be traded."
    )


def reset_day_if_needed(
    st,
    client,
    account_id,
):
    day = utcnow().date().isoformat()

    if st.get("day") != day:
        equity = portfolio_value(
            client,
            account_id,
        )

        st["day"] = day
        st["day_start_equity"] = equity
        st["paused"] = False

        save_state(st)

        print(
            f"[risk] new day {day}, "
            f"start equity={equity:,.2f}"
        )


def check_daily_loss(
    st,
    client,
    account_id,
):
    start = st.get("day_start_equity")

    if not start:
        return False, 0.0

    equity = portfolio_value(
        client,
        account_id,
    )

    dd = equity / float(start) - 1.0

    return (
        dd <= -abs(MAX_DAILY_LOSS_PCT),
        dd,
    )


def close_due_positions(
    st,
    client,
    account_id,
    figis,
    notifier,
):
    now = utcnow()

    positions = st.get("positions", {})

    to_close = []

    for ticker, p in positions.items():
        close_at = pd.to_datetime(
            p["close_at"],
            utc=True,
        ).to_pydatetime()

        if now >= close_at:
            to_close.append(ticker)

    for ticker in to_close:
        p = positions[ticker]

        side = int(p["side"])
        lots = int(p["lots"])

        signed_close = -side * lots

        ok, exit_exec = execute_market(
            client,
            account_id,
            figis,
            ticker,
            signed_close,
        )

        exit_price = exit_exec.get("price")
        exit_commission = exit_exec.get("commission")
        exit_total_amount = exit_exec.get("total_amount")

        if not ok:
            print(
                f"[close] {ticker}: failed; "
                f"will retry"
            )
            continue

        entry_price = p.get("entry_price")
        entry_commission = p.get("entry_commission")
        entry_total_amount = p.get("entry_total_amount")

        pnl_bp = None

        if (
            entry_price is not None
            and exit_price is not None
            and float(entry_price) > 0
        ):
            pnl_bp = (
                side
                * (
                    float(exit_price)
                    / float(entry_price)
                    - 1.0
                )
                * 10000.0
            )

        gross_pnl_rub = None

        if (
            entry_total_amount is not None
            and exit_total_amount is not None
        ):
            if side > 0:
                gross_pnl_rub = (
                    float(exit_total_amount)
                    - float(entry_total_amount)
                )
            else:
                gross_pnl_rub = (
                    float(entry_total_amount)
                    - float(exit_total_amount)
                )

        total_commission = None

        if (
            entry_commission is not None
            or exit_commission is not None
        ):
            total_commission = (
                float(entry_commission or 0.0)
                + float(exit_commission or 0.0)
            )

        net_pnl_rub = None

        if gross_pnl_rub is not None:
            net_pnl_rub = float(gross_pnl_rub)

            if total_commission is not None:
                net_pnl_rub -= total_commission

        net_pnl_bp = None

        if (
            net_pnl_rub is not None
            and entry_total_amount is not None
            and float(entry_total_amount) != 0
        ):
            net_pnl_bp = (
                net_pnl_rub
                / abs(float(entry_total_amount))
                * 10000.0
            )

        append_journal({
            "event": "CLOSE",
            "time": now.isoformat(),
            "ticker": ticker,
            "side": side,
            "lots": lots,
            "signal_time": p["signal_time"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_total_amount": entry_total_amount,
            "exit_total_amount": exit_total_amount,
            "entry_commission": entry_commission,
            "exit_commission": exit_commission,
            "commission_rub": total_commission,
            "gross_pnl_rub": gross_pnl_rub,
            "net_pnl_rub": net_pnl_rub,
            "gross_pnl_bp": pnl_bp,
            "net_pnl_bp": net_pnl_bp,
            "arm": ARM,
        })

        icon = "✅" if (net_pnl_rub or 0) > 0 else "❌" if (net_pnl_rub or 0) < 0 else "➖"

        msg = (
            f"{icon} <b>MR30 — СДЕЛКА ЗАКРЫТА</b>\n\n"
            f"Инструмент: <b>{ticker}</b>\n"
            f"Направление: <b>{'ЛОНГ' if side > 0 else 'ШОРТ'}</b>\n"
            f"Объём: <b>{lots} лот</b>\n"
            f"Время удержания: <b>{HOLD_MINUTES} мин</b>\n"
            f"Время закрытия: <b>{fmt_moscow(now)}</b>\n\n"
        )

        if entry_price is not None:
            msg += f"Цена входа: <b>{entry_price:.4f} ₽</b>\n"

        if exit_price is not None:
            msg += f"Цена выхода: <b>{exit_price:.4f} ₽</b>\n"

        if entry_total_amount is not None:
            msg += (
                f"Сумма входа: "
                f"<b>{entry_total_amount:,.2f} ₽</b>\n"
            )

        if gross_pnl_rub is not None:
            msg += (
                f"\nВаловый результат: "
                f"<b>{gross_pnl_rub:+,.2f} ₽</b>"
            )

        if pnl_bp is not None:
            msg += f" ({pnl_bp:+.2f} б.п.)"

        msg += "\n"

        if total_commission is not None:
            msg += (
                f"Комиссия: "
                f"<b>−{abs(total_commission):,.2f} ₽</b>\n"
            )
        else:
            msg += "Комиссия: <b>нет данных от API</b>\n"

        if net_pnl_rub is not None:
            msg += (
                f"Чистый результат: "
                f"<b>{net_pnl_rub:+,.2f} ₽</b>"
            )

            if net_pnl_bp is not None:
                msg += f" ({net_pnl_bp:+.2f} б.п.)"

            msg += "\n"

        notify(
            notifier,
            msg,
        )

        del positions[ticker]

        save_state(st)


def flatten_all(
    st,
    client,
    account_id,
    figis,
    notifier,
    reason,
):
    positions = st.get("positions", {})

    for ticker in list(positions):
        p = positions[ticker]

        side = int(p["side"])
        lots = int(p["lots"])

        ok, exit_price = execute_market(
            client,
            account_id,
            figis,
            ticker,
            -side * lots,
        )

        if not ok:
            continue

        append_journal({
            "event": "FLATTEN",
            "time": utcnow().isoformat(),
            "ticker": ticker,
            "side": side,
            "lots": lots,
            "signal_time": p.get("signal_time"),
            "entry_price": p.get("entry_price"),
            "exit_price": exit_price,
            "reason": reason,
            "arm": ARM,
        })

        del positions[ticker]

    save_state(st)

    notify(
        notifier,
        f"⛔ MR30 SANDBOX FLATTEN\nreason={reason}",
    )


def process_new_signals(
    st,
    client,
    account_id,
    figis,
    notifier,
):
    if st.get("paused"):
        return

    df = load_no_broad_signals()

    if df.empty:
        return

    processed = set(
        st.get("processed", [])
    )

    new = df[
        ~df["_key"].isin(processed)
    ].copy()

    if new.empty:
        return

    now = utcnow()

    for _, r in new.iterrows():
        key = r["_key"]
        signal_time = r["_time"]
        ticker = r["_ticker"]
        side = int(r["_side"])

        # Сразу помечаем обработанным, чтобы при API error
        # не спамить одной и той же заявкой бесконечно.
        processed.add(key)

        # Не торгуем stale signal.
        age = (
            now
            - signal_time.to_pydatetime()
        ).total_seconds()

        if age > 180:
            print(
                f"[signal] stale skip "
                f"{ticker} age={age:.0f}s"
            )
            continue

        if ticker in st["positions"]:
            print(
                f"[signal] {ticker}: "
                f"already open, skip"
            )
            continue

        if (
            len(st["positions"])
            >= MAX_OPEN_POSITIONS
        ):
            print(
                "[signal] max open positions reached"
            )
            continue

        if ticker not in figis:
            print(
                f"[signal] {ticker}: "
                f"not present in cfg.data.figIs"
            )
            continue

        signed_lots = side * LOTS_PER_TRADE

        ok, entry_exec = execute_market(
            client,
            account_id,
            figis,
            ticker,
            signed_lots,
        )

        entry_price = entry_exec.get("price")
        entry_commission = entry_exec.get("commission")
        entry_total_amount = entry_exec.get("total_amount")

        if not ok:
            append_journal({
                "event": "OPEN_FAILED",
                "time": now.isoformat(),
                "ticker": ticker,
                "side": side,
                "lots": LOTS_PER_TRADE,
                "signal_time": signal_time.isoformat(),
                "arm": ARM,
            })
            continue

        close_at = (
            signal_time.to_pydatetime()
            + timedelta(minutes=HOLD_MINUTES)
        )

        st["positions"][ticker] = {
            "side": side,
            "lots": LOTS_PER_TRADE,
            "signal_time": signal_time.isoformat(),
            "opened_at": now.isoformat(),
            "close_at": close_at.isoformat(),
            "entry_price": entry_price,
            "entry_commission": entry_commission,
            "entry_total_amount": entry_total_amount,
        }

        append_journal({
            "event": "OPEN",
            "time": now.isoformat(),
            "ticker": ticker,
            "side": side,
            "lots": LOTS_PER_TRADE,
            "signal_time": signal_time.isoformat(),
            "entry_price": entry_price,
            "entry_commission": entry_commission,
            "entry_total_amount": entry_total_amount,
            "close_at": close_at.isoformat(),
            "arm": ARM,
        })

        notify(
            notifier,
            (
                f"🟢 <b>MR30 — ПОЗИЦИЯ ОТКРЫТА</b>\n\n"
                f"Инструмент: <b>{ticker}</b>\n"
                f"Направление: <b>{'ЛОНГ' if side > 0 else 'ШОРТ'}</b>\n"
                f"Объём: <b>{LOTS_PER_TRADE} лот</b>\n"
                f"Время: <b>{fmt_moscow(now)}</b>\n"
                f"Цена входа: <b>{entry_price:.4f} ₽</b>\n"
                if entry_price is not None else
                f"🟢 <b>MR30 — ПОЗИЦИЯ ОТКРЫТА</b>\n\n"
                f"Инструмент: <b>{ticker}</b>\n"
                f"Направление: <b>{'ЛОНГ' if side > 0 else 'ШОРТ'}</b>\n"
                f"Объём: <b>{LOTS_PER_TRADE} лот</b>\n"
                f"Цена входа: <b>нет данных</b>\n"
            ),
        )

    st["processed"] = list(processed)[-5000:]

    save_state(st)


def get_or_create_account(
    st,
    client,
    pay_in,
):
    from tinkoff.invest import MoneyValue

    account_id = st.get("account_id")

    if account_id:
        try:
            portfolio_value(
                client,
                account_id,
            )
            print(
                "[sandbox] reuse account:",
                account_id,
            )
            return account_id
        except Exception:
            print(
                "[sandbox] stored account invalid, "
                "creating new"
            )

    acc = client.sandbox.open_sandbox_account()

    account_id = acc.account_id

    client.sandbox.sandbox_pay_in(
        account_id=account_id,
        amount=MoneyValue(
            units=int(pay_in),
            nano=0,
            currency="rub",
        ),
    )

    st["account_id"] = account_id

    save_state(st)

    print(
        f"[sandbox] new account={account_id}, "
        f"pay_in={pay_in:,.0f}"
    )

    return account_id


def main():
    token = os.getenv("TINKOFF_TOKEN")

    if not token:
        print(
            "FATAL: TINKOFF_TOKEN not found in .env"
        )
        raise SystemExit(1)

    cfg = load_config()

    figis = dict(cfg.data["figis"])

    print("=" * 100)
    print("MR30 SANDBOX TRADER")
    print("=" * 100)
    print("ARM               :", ARM)
    print("MODE              : TBANK SANDBOX ONLY")
    print("REAL MONEY        : NO")
    print("HOLD              :", HOLD_MINUTES, "minutes")
    print("LOTS / TRADE      :", LOTS_PER_TRADE)
    print("MAX OPEN          :", MAX_OPEN_POSITIONS)
    print(
        "DAILY LOSS LIMIT  :",
        f"{MAX_DAILY_LOSS_PCT:.2%}",
    )
    print("SIGNALS           :", SIGNALS_FILE)
    print()

    st = load_state()

    # Первый запуск только ставит watermark на уже имеющиеся сигналы.
    if not st.get("bootstrapped"):
        bootstrap_existing_signals(st)
        print()
        print(
            "BOOTSTRAP COMPLETE."
        )
        print(
            "Запусти скрипт ещё раз — "
            "после этого он будет ждать НОВЫЕ сигналы."
        )
        return

    from tinkoff.invest.sandbox.client import (
        SandboxClient,
    )

    notifier = TelegramNotifier()

    pay_in = float(
        cfg.sandbox.get(
            "pay_in",
            1_000_000,
        )
    )

    with SandboxClient(token) as client:
        account_id = get_or_create_account(
            st,
            client,
            pay_in,
        )

        reset_day_if_needed(
            st,
            client,
            account_id,
        )

        print(
            "[sandbox] equity:",
            f"{portfolio_value(client, account_id):,.2f}",
        )

        print()
        print(
            "Waiting for NEW NO_BROAD_TREND signals..."
        )

        while True:
            try:
                reset_day_if_needed(
                    st,
                    client,
                    account_id,
                )

                # Сначала закрываем то, чему исполнилось 30 минут.
                close_due_positions(
                    st,
                    client,
                    account_id,
                    figis,
                    notifier,
                )

                triggered, dd = check_daily_loss(
                    st,
                    client,
                    account_id,
                )

                if triggered:
                    if not st.get("paused"):
                        st["paused"] = True
                        save_state(st)

                        flatten_all(
                            st,
                            client,
                            account_id,
                            figis,
                            notifier,
                            reason=(
                                f"daily loss "
                                f"{dd:.2%}"
                            ),
                        )

                    time.sleep(POLL_SEC)
                    continue

                process_new_signals(
                    st,
                    client,
                    account_id,
                    figis,
                    notifier,
                )

                print(
                    f"[{utcnow().strftime('%H:%M:%S')} UTC] "
                    f"open={len(st['positions'])} "
                    f"paused={st.get('paused')} "
                    f"dd={dd:.3%}"
                )

                time.sleep(POLL_SEC)

            except KeyboardInterrupt:
                print()
                print(
                    "Stopped by user. "
                    "Existing sandbox positions were NOT "
                    "automatically flattened."
                )
                break

            except Exception as e:
                print(
                    "[loop] ERROR:",
                    repr(e),
                )
                time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()

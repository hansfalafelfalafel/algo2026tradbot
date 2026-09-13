from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
from dotenv import load_dotenv

from src.notify.telegram import TelegramNotifier


SHADOW = ROOT / "state" / "mr30_v3_shadow_trades.csv"
SANDBOX = ROOT / "state" / "mr30_sandbox_journal.csv"

POLL_SEC = 3


def today_msk():
    return pd.Timestamp.now(
        tz="Europe/Moscow"
    ).date()


def shadow_today():
    if not SHADOW.exists():
        return None

    try:
        df = pd.read_csv(SHADOW)
    except Exception:
        return None

    if df.empty:
        return pd.DataFrame()

    df = df[
        df["arm"].astype(str)
        == "NO_BROAD_TREND"
    ].copy()

    t = pd.to_datetime(
        df["exit_time"],
        utc=True,
        errors="coerce",
    )

    df["_date_msk"] = (
        t.dt.tz_convert(
            "Europe/Moscow"
        ).dt.date
    )

    return df[
        df["_date_msk"] == today_msk()
    ].copy()


def sandbox_today():
    if not SANDBOX.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(SANDBOX)
    except Exception as e:
        print(
            "[sandbox csv error]",
            repr(e),
        )
        return pd.DataFrame()

    if df.empty or "time" not in df.columns:
        return pd.DataFrame()

    t = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    df["_date_msk"] = (
        t.dt.tz_convert(
            "Europe/Moscow"
        ).dt.date
    )

    return df[
        df["_date_msk"] == today_msk()
    ].copy()


def summary():
    lines = [
        "📊 <b>MR30 — СВОДКА ЗА СЕГОДНЯ</b>",
        "",
        "🧠 <b>SHADOW — NO_BROAD_TREND</b>",
    ]

    sh = shadow_today()

    if sh is None or sh.empty:
        lines += [
            "Закрытых сделок: <b>0</b>",
            "Результат: <b>0.00 б.п.</b>",
        ]

    else:
        pnl = pd.to_numeric(
            sh["net_bp"],
            errors="coerce",
        ).dropna()

        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())
        flat = int((pnl == 0).sum())

        wr = (
            100.0 * wins / len(pnl)
            if len(pnl)
            else 0.0
        )

        lines += [
            f"Закрыто сделок: <b>{len(pnl)}</b>",
            f"✅ Прибыльных: <b>{wins}</b>",
            f"❌ Убыточных: <b>{losses}</b>",
            f"➖ В ноль: <b>{flat}</b>",
            f"Доля прибыльных: <b>{wr:.1f}%</b>",
            "",
            f"Суммарный net: <b>{pnl.sum():+.2f} б.п.</b>",
            f"Средняя сделка: <b>{pnl.mean():+.2f} б.п.</b>",
            f"Лучшая: <b>{pnl.max():+.2f} б.п.</b>",
            f"Худшая: <b>{pnl.min():+.2f} б.п.</b>",
        ]

    lines += [
        "",
        "💼 <b>T-BANK SANDBOX</b>",
    ]

    sb = sandbox_today()

    if sb.empty:
        lines += [
            "Открытий сегодня: <b>0</b>",
            "Закрытий сегодня: <b>0</b>",
            "Чистый результат: <b>0.00 ₽</b>",
        ]

    else:
        events = (
            sb["event"]
            .astype(str)
            .str.upper()
        )

        opens = int(
            (events == "OPEN").sum()
        )

        closes = sb[
            events == "CLOSE"
        ].copy()

        failed = int(
            (events == "OPEN_FAILED").sum()
        )

        lines += [
            f"Открытий: <b>{opens}</b>",
            f"Закрытий: <b>{len(closes)}</b>",
            f"Ошибок открытия: <b>{failed}</b>",
        ]

        if not closes.empty:
            pnl = pd.to_numeric(
                closes["net_pnl_rub"],
                errors="coerce",
            ).dropna()

            commissions = pd.to_numeric(
                closes["commission_rub"],
                errors="coerce",
            ).dropna()

            if len(pnl):
                wins = int((pnl > 0).sum())
                losses = int((pnl < 0).sum())

                wr = (
                    100.0 * wins / len(pnl)
                    if len(pnl)
                    else 0.0
                )

                lines += [
                    f"✅ Прибыльных: <b>{wins}</b>",
                    f"❌ Убыточных: <b>{losses}</b>",
                    f"Доля прибыльных: <b>{wr:.1f}%</b>",
                    "",
                    f"💰 Чистый PnL: <b>{pnl.sum():+,.2f} ₽</b>",
                    f"Средняя сделка: <b>{pnl.mean():+,.2f} ₽</b>",
                ]
            else:
                lines += [
                    "",
                    "💰 Чистый PnL: <b>нет данных</b>",
                ]

            if len(commissions):
                lines.append(
                    f"💸 Комиссий: <b>{commissions.sum():,.2f} ₽</b>"
                )

        else:
            lines += [
                "",
                "💰 Реализованный PnL: <b>0.00 ₽</b>",
            ]

    lines += [
        "",
        "<i>Shadow — модельный результат после "
        "расчётных издержек. Sandbox — фактическое "
        "исполнение в песочнице T-Bank.</i>",
    ]

    return "\n".join(lines)


def main():
    load_dotenv(ROOT / ".env")

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN missing"
        )

    notifier = TelegramNotifier()

    allowed = set(
        notifier.chat_ids
    )

    url = (
        f"https://api.telegram.org/"
        f"bot{token}/getUpdates"
    )

    offset = None

    print("=" * 80)
    print("MR30 — TELEGRAM COMMAND BOT")
    print("=" * 80)
    print("allowed:", sorted(allowed))
    print(
        "commands: "
        "/сводка /summary /status"
    )

    while True:
        try:
            params = {
                "timeout": 20,
            }

            if offset is not None:
                params["offset"] = offset

            r = requests.get(
                url,
                params=params,
                timeout=30,
            )

            data = r.json()

            for upd in data.get(
                "result",
                [],
            ):
                offset = (
                    upd["update_id"]
                    + 1
                )

                msg = (
                    upd.get("message")
                    or {}
                )

                chat = (
                    msg.get("chat")
                    or {}
                )

                chat_id = str(
                    chat.get("id")
                )

                text = str(
                    msg.get("text")
                    or ""
                ).strip().lower()

                if chat_id not in allowed:
                    continue

                if text in {
                    "/сводка",
                    "сводка",
                    "/summary",
                    "/status",
                    "статус",
                }:
                    TelegramNotifier(
                        chat_id=chat_id
                    ).send(
                        summary()
                    )

                    print(
                        "[command]",
                        chat_id,
                        text,
                    )

        except requests.exceptions.Timeout:
            pass

        except Exception as e:
            print(
                "[command bot error]",
                type(e).__name__,
            )

            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()

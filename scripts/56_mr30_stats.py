from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv

from src.notify.telegram import TelegramNotifier


JOURNAL = ROOT / "state" / "mr30_sandbox_journal.csv"
STATE = ROOT / "state" / "mr30_sandbox_state.json"

POLL_SEC = 10


def money(v):
    try:
        if pd.isna(v):
            return "нет данных"
        return f"{float(v):+,.2f} ₽"
    except Exception:
        return "нет данных"


def bp(v):
    try:
        if pd.isna(v):
            return "нет данных"
        return f"{float(v):+.2f} б.п."
    except Exception:
        return "нет данных"


def load_journal():
    if not JOURNAL.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(JOURNAL)
    except Exception as e:
        print("[journal error]", repr(e))
        return pd.DataFrame()


def load_closed():
    df = load_journal()

    if df.empty or "event" not in df.columns:
        return pd.DataFrame()

    df = df[
        df["event"].astype(str).str.upper() == "CLOSE"
    ].copy()

    numeric = [
        "gross_pnl_bp",
        "net_pnl_bp",
        "gross_pnl_rub",
        "net_pnl_rub",
        "commission_rub",
        "entry_price",
        "exit_price",
        "entry_total_amount",
        "exit_total_amount",
    ]

    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


def today_stats(df):
    if df.empty or "time" not in df.columns:
        return (
            "📅 <b>СЕГОДНЯ</b>\n"
            "Закрытых сделок пока нет."
        )

    x = df.copy()

    t = pd.to_datetime(
        x["time"],
        utc=True,
        errors="coerce",
    )

    x["_msk_date"] = (
        t.dt.tz_convert("Europe/Moscow").dt.date
    )

    today = pd.Timestamp.now(
        tz="Europe/Moscow"
    ).date()

    x = x[
        x["_msk_date"] == today
    ].copy()

    if x.empty:
        return (
            "📅 <b>СЕГОДНЯ</b>\n"
            "Закрытых сделок пока нет."
        )

    lines = [
        "📅 <b>СЕГОДНЯ</b>",
        "",
        f"Закрыто сделок: <b>{len(x)}</b>",
    ]

    if (
        "net_pnl_rub" in x.columns
        and x["net_pnl_rub"].notna().any()
    ):
        pnl = pd.to_numeric(
            x["net_pnl_rub"],
            errors="coerce",
        ).dropna()

        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())

        win_rate = (
            100.0 * wins / len(pnl)
            if len(pnl)
            else 0.0
        )

        lines += [
            f"✅ Прибыльных: <b>{wins}</b>",
            f"❌ Убыточных: <b>{losses}</b>",
            f"Доля прибыльных: <b>{win_rate:.1f}%</b>",
            "",
            f"💰 Чистый результат: <b>{pnl.sum():+,.2f} ₽</b>",
            f"Средняя сделка: <b>{pnl.mean():+,.2f} ₽</b>",
        ]

    if (
        "commission_rub" in x.columns
        and x["commission_rub"].notna().any()
    ):
        c = pd.to_numeric(
            x["commission_rub"],
            errors="coerce",
        ).dropna().sum()

        lines.append(
            f"💸 Комиссий: <b>{c:,.2f} ₽</b>"
        )

    if (
        "net_pnl_bp" in x.columns
        and x["net_pnl_bp"].notna().any()
    ):
        b = pd.to_numeric(
            x["net_pnl_bp"],
            errors="coerce",
        ).dropna()

        lines.append(
            f"📈 Сумма: <b>{b.sum():+.2f} б.п.</b>"
        )

    return "\n".join(lines)


def build_stats(df):
    if df.empty:
        return (
            "📊 <b>СТАТИСТИКА MR30</b>\n\n"
            "Закрытых сделок пока нет."
        )

    # Для win/loss приоритет — NET ₽.
    if (
        "net_pnl_rub" in df.columns
        and df["net_pnl_rub"].notna().any()
    ):
        pnl = df["net_pnl_rub"].dropna()
        mode = "rub"
    elif (
        "net_pnl_bp" in df.columns
        and df["net_pnl_bp"].notna().any()
    ):
        pnl = df["net_pnl_bp"].dropna()
        mode = "bp"
    else:
        pnl = df["gross_pnl_bp"].dropna()
        mode = "bp"

    n = len(pnl)

    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    flat = int((pnl == 0).sum())

    win_rate = (
        100.0 * wins / n
        if n
        else 0.0
    )

    lines = [
        "📊 <b>СТАТИСТИКА MR30</b>",
        "",
        f"Закрыто сделок: <b>{n}</b>",
        f"✅ Прибыльных: <b>{wins}</b>",
        f"❌ Убыточных: <b>{losses}</b>",
        f"➖ В ноль: <b>{flat}</b>",
        f"Доля прибыльных: <b>{win_rate:.1f}%</b>",
    ]

    if (
        "net_pnl_rub" in df.columns
        and df["net_pnl_rub"].notna().any()
    ):
        rub = df["net_pnl_rub"].dropna()

        lines += [
            "",
            "💰 <b>Деньги</b>",
            f"Чистый результат: <b>{rub.sum():+,.2f} ₽</b>",
            f"Средняя сделка: <b>{rub.mean():+,.2f} ₽</b>",
            f"Медианная сделка: <b>{rub.median():+,.2f} ₽</b>",
            f"Лучшая сделка: <b>{rub.max():+,.2f} ₽</b>",
            f"Худшая сделка: <b>{rub.min():+,.2f} ₽</b>",
        ]

    if (
        "commission_rub" in df.columns
        and df["commission_rub"].notna().any()
    ):
        comm = df["commission_rub"].dropna().sum()

        lines.append(
            f"💸 Комиссий: <b>{comm:,.2f} ₽</b>"
        )

    if (
        "net_pnl_bp" in df.columns
        and df["net_pnl_bp"].notna().any()
    ):
        p = df["net_pnl_bp"].dropna()

        lines += [
            "",
            "📈 <b>Результат в базисных пунктах</b>",
            f"Сумма: <b>{p.sum():+.2f} б.п.</b>",
            f"Среднее: <b>{p.mean():+.2f} б.п.</b>",
            f"Медиана: <b>{p.median():+.2f} б.п.</b>",
        ]

    return "\n".join(lines)


def last_trade(df):
    if df.empty:
        return ""

    r = df.iloc[-1]

    ticker = str(r.get("ticker", "?"))

    try:
        side = (
            "ЛОНГ"
            if int(float(r.get("side"))) > 0
            else "ШОРТ"
        )
    except Exception:
        side = "?"

    net = r.get("net_pnl_rub")

    try:
        x = float(net)

        icon = (
            "✅"
            if x > 0
            else "❌"
            if x < 0
            else "➖"
        )

    except Exception:
        icon = "ℹ️"

    lines = [
        f"{icon} <b>ПОСЛЕДНЯЯ СДЕЛКА</b>",
        "",
        f"Инструмент: <b>{ticker}</b>",
        f"Направление: <b>{side}</b>",
    ]

    if "gross_pnl_rub" in r:
        lines.append(
            f"Валовый результат: <b>{money(r.get('gross_pnl_rub'))}</b>"
        )

    if "commission_rub" in r:
        c = r.get("commission_rub")

        try:
            if not pd.isna(c):
                lines.append(
                    f"Комиссия: <b>−{abs(float(c)):,.2f} ₽</b>"
                )
            else:
                lines.append(
                    "Комиссия: <b>нет данных</b>"
                )
        except Exception:
            pass

    if "net_pnl_rub" in r:
        lines.append(
            f"Чистый результат: <b>{money(r.get('net_pnl_rub'))}</b>"
        )

    if "net_pnl_bp" in r:
        lines.append(
            f"Результат: <b>{bp(r.get('net_pnl_bp'))}</b>"
        )

    return "\n".join(lines)


def main():
    load_dotenv(ROOT / ".env")

    notifier = TelegramNotifier()

    print("=" * 80)
    print("MR30 — МОНИТОР СТАТИСТИКИ")
    print("=" * 80)
    print("journal:", JOURNAL)
    print("poll   :", POLL_SEC, "sec")

    previous_count = None

    while True:
        try:
            df = load_closed()
            count = len(df)

            if previous_count is None:
                previous_count = count

                print(
                    f"[init] closed={count}"
                )

            elif count > previous_count:
                added = (
                    count
                    - previous_count
                )

                print(
                    f"[new close] +{added}, "
                    f"total={count}"
                )

                message = (
                    last_trade(df)
                    + "\n\n"
                    + today_stats(df)
                    + "\n\n"
                    + build_stats(df)
                )

                ok = notifier.send(message)

                print(
                    "[telegram]",
                    ok,
                )

                previous_count = count

            elif count < previous_count:
                previous_count = count

                print(
                    f"[reset] closed={count}"
                )

        except Exception as e:
            print(
                "[stats error]",
                repr(e),
            )

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()

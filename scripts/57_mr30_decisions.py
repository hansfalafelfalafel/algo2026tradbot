from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv

from src.notify.telegram import TelegramNotifier


SIGNALS = ROOT / "state" / "mr30_v3_shadow_signals.csv"
STATE = ROOT / "state" / "mr30_decisions_state.json"

POLL_SEC = 10
BREADTH_LIMIT = 0.70


def load_state():
    if not STATE.exists():
        return {
            "bootstrapped": False,
            "processed": [],
        }

    try:
        return json.loads(
            STATE.read_text(encoding="utf-8")
        )
    except Exception:
        return {
            "bootstrapped": False,
            "processed": [],
        }


def save_state(st):
    tmp = STATE.with_suffix(".tmp")

    tmp.write_text(
        json.dumps(
            st,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tmp.replace(STATE)


def side_name(v):
    try:
        return "ЛОНГ" if int(float(v)) > 0 else "ШОРТ"
    except Exception:
        return str(v)


def fmt(v, digits=3):
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "нет данных"


def to_moscow(v):
    try:
        ts = pd.to_datetime(
            v,
            utc=True,
            errors="coerce",
        )

        if pd.isna(ts):
            return "нет данных"

        return ts.tz_convert(
            "Europe/Moscow"
        ).strftime(
            "%d.%m.%Y %H:%M МСК"
        )

    except Exception:
        return "нет данных"


def make_key(row):
    return (
        f"{row.get('ticker')}|"
        f"{row.get('entry_time')}|"
        f"{row.get('side')}"
    )


def load_signals():
    if not SIGNALS.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(SIGNALS)
    except Exception as e:
        print("[read error]", repr(e))
        return pd.DataFrame()

    if df.empty:
        return df

    return df


def main():
    load_dotenv(ROOT / ".env")

    notifier = TelegramNotifier()
    last_heartbeat = 0.0

    print("=" * 80)
    print("MR30 — МОНИТОР РЕШЕНИЙ")
    print("=" * 80)
    print("signals :", SIGNALS)
    print("breadth :", BREADTH_LIMIT)
    print("poll    :", POLL_SEC, "sec")

    st = load_state()

    while True:
        try:
            df = load_signals()

            if df.empty:
                time.sleep(POLL_SEC)
                continue

            base = df[
                df["arm"].astype(str) == "BASE"
            ].copy()

            no_broad = df[
                df["arm"].astype(str) == "NO_BROAD_TREND"
            ].copy()

            # Heartbeat только в лог, чтобы было видно,
            # что монитор жив и продолжает читать сигналы.
            now_ts = time.time()

            if now_ts - last_heartbeat >= 60:
                latest_base = "нет"

                if not base.empty and "entry_time" in base.columns:
                    try:
                        latest_base = str(
                            pd.to_datetime(
                                base["entry_time"],
                                utc=True,
                                errors="coerce",
                            ).max()
                        )
                    except Exception:
                        pass

                print(
                    "[heartbeat]",
                    f"BASE={len(base)}",
                    f"NO_BROAD={len(no_broad)}",
                    f"processed={len(st.get('processed', []))}",
                    f"latest_BASE={to_moscow(latest_base)}",
                    flush=True,
                )

                last_heartbeat = now_ts

            # Первый запуск:
            # все старые BASE сигналы отмечаем обработанными,
            # чтобы Telegram не засыпало историей.
            if not st.get("bootstrapped"):
                st["processed"] = [
                    make_key(r)
                    for _, r in base.iterrows()
                ][-5000:]

                st["bootstrapped"] = True
                save_state(st)

                print(
                    "[bootstrap] old BASE signals processed:",
                    len(st["processed"]),
                )

                time.sleep(POLL_SEC)
                continue

            processed = set(
                st.get("processed", [])
            )

            for _, r in base.iterrows():
                key = make_key(r)

                if key in processed:
                    continue

                ticker = str(r.get("ticker", "?"))
                entry_time = str(r.get("entry_time", "?"))
                side = r.get("side")
                score = r.get("score")
                breadth = r.get("breadth_extreme")
                confirm = r.get("confirm_count")
                spread = r.get("entry_spread_bp")

                # Нормализуем время и направление.
                # iterrows() может превратить side=1 в 1.0,
                # поэтому строковое сравнение здесь ненадёжно.
                nb_time = pd.to_datetime(
                    no_broad["entry_time"],
                    utc=True,
                    errors="coerce",
                )

                row_time = pd.to_datetime(
                    entry_time,
                    utc=True,
                    errors="coerce",
                )

                nb_side = pd.to_numeric(
                    no_broad["side"],
                    errors="coerce",
                )

                try:
                    row_side = int(float(side))
                except Exception:
                    row_side = None

                accepted = no_broad[
                    (no_broad["ticker"].astype(str) == ticker)
                    & (nb_time == row_time)
                    & (nb_side == row_side)
                ]

                if not accepted.empty:
                    msg = (
                        "🟢 <b>MR30 — СИГНАЛ ОДОБРЕН</b>\n\n"
                        f"Инструмент: <b>{ticker}</b>\n"
                        f"Направление: <b>{side_name(side)}</b>\n"
                        f"Время сигнала: <b>{to_moscow(entry_time)}</b>\n"
                        f"Время сигнала: <b>{to_moscow(entry_time)}</b>\n"
                        f"Score: <b>{fmt(score)}</b>\n"
                        f"Спред: <b>{fmt(spread)} б.п.</b>\n\n"
                        f"breadth_extreme: <b>{fmt(breadth)}</b>\n"
                        f"Лимит: <b>≤ {BREADTH_LIMIT:.2f}</b>\n"
                        f"confirm_count: <b>{confirm}</b>\n\n"
                        "Решение стратегии: <b>БЕРЁМ</b>\n\n"
                        "<i>Фактическое открытие дополнительно "
                        "зависит от технических ограничений "
                        "sandbox-бота.</i>"
                    )

                    print(
                        f"[TAKE] {ticker} "
                        f"{side_name(side)} "
                        f"breadth={fmt(breadth)}"
                    )

                else:
                    try:
                        breadth_value = float(breadth)
                    except Exception:
                        breadth_value = None

                    if (
                        breadth_value is not None
                        and breadth_value > BREADTH_LIMIT
                    ):
                        reason = (
                            "Широкий рыночный тренд слишком сильный"
                        )
                    else:
                        reason = (
                            "Сигнал не прошёл фильтр "
                            "NO_BROAD_TREND"
                        )

                    msg = (
                        "🔴 <b>MR30 — СИГНАЛ ПРОПУЩЕН</b>\n\n"
                        f"Инструмент: <b>{ticker}</b>\n"
                        f"Направление: <b>{side_name(side)}</b>\n"
                        f"Score: <b>{fmt(score)}</b>\n"
                        f"Спред: <b>{fmt(spread)} б.п.</b>\n\n"
                        f"Причина: <b>{reason}</b>\n\n"
                        f"breadth_extreme: <b>{fmt(breadth)}</b>\n"
                        f"Лимит стратегии: "
                        f"<b>≤ {BREADTH_LIMIT:.2f}</b>\n"
                        f"confirm_count: <b>{confirm}</b>\n\n"
                        "Решение: <b>НЕ БЕРЁМ</b>"
                    )

                    print(
                        f"[SKIP] {ticker} "
                        f"{side_name(side)} "
                        f"breadth={fmt(breadth)}"
                    )

                try:
                    notifier.send(msg)
                except Exception as e:
                    print(
                        "[telegram error]",
                        repr(e),
                    )

                processed.add(key)

            st["processed"] = list(processed)[-5000:]
            save_state(st)

        except Exception as e:
            print(
                "[decision error]",
                repr(e),
            )

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()

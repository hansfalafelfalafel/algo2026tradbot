"""Telegram-уведомления для MR30."""

from __future__ import annotations

import os
import time
import requests


class TelegramNotifier:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")

        if chat_id is not None:
            self.chat_ids = [str(chat_id)]
        else:
            raw = (
                os.environ.get("TELEGRAM_CHAT_IDS")
                or os.environ.get("TELEGRAM_CHAT_ID")
                or ""
            )

            self.chat_ids = [
                x.strip()
                for x in raw.split(",")
                if x.strip()
            ]

        # Обратная совместимость
        self.chat_id = self.chat_ids[0] if self.chat_ids else None

    def send(self, text: str) -> bool:
        if not self.token or not self.chat_ids:
            print("[telegram] TOKEN/CHAT_ID не настроены")
            return False

        # Никогда не выводим URL: в нём содержится секретный bot token.
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"

        all_ok = True

        for chat_id in self.chat_ids:
            sent = False

            for attempt in range(1, 4):
                try:
                    resp = requests.post(
                        url,
                        json={
                            "chat_id": chat_id,
                            "text": text,
                            "parse_mode": "HTML",
                        },
                        timeout=(5, 15),
                    )

                    if resp.status_code == 200:
                        print(
                            f"[telegram] chat_id={chat_id} "
                            f"отправлено"
                        )
                        sent = True
                        break

                    print(
                        f"[telegram] chat_id={chat_id} "
                        f"HTTP {resp.status_code}, "
                        f"попытка {attempt}/3"
                    )

                except requests.exceptions.Timeout:
                    print(
                        f"[telegram] chat_id={chat_id} "
                        f"таймаут, попытка {attempt}/3"
                    )

                except requests.exceptions.ConnectionError:
                    print(
                        f"[telegram] chat_id={chat_id} "
                        f"ошибка соединения, "
                        f"попытка {attempt}/3"
                    )

                except Exception as e:
                    # repr(e) специально не выводим:
                    # некоторые исключения requests могут содержать URL,
                    # а URL содержит bot token.
                    print(
                        f"[telegram] chat_id={chat_id} "
                        f"ошибка {type(e).__name__}, "
                        f"попытка {attempt}/3"
                    )

                if attempt < 3:
                    time.sleep(2 * attempt)

            if not sent:
                print(
                    f"[telegram] chat_id={chat_id}: "
                    "не удалось отправить после 3 попыток"
                )
                all_ok = False

        return all_ok


def format_recommendation(rec: dict) -> str:
    lines = ["<b>Рекомендация</b>"]

    if "capital" in rec:
        try:
            lines.append(
                f"Капитал: {float(rec['capital']):,.0f} ₽"
            )
        except Exception:
            pass

    positions = rec.get("positions") or []

    if positions:
        lines.append("")
        lines.append("<b>Позиции:</b>")

        for p in positions:
            ticker = p.get("ticker", "?")
            weight = p.get("weight")

            if weight is not None:
                try:
                    lines.append(
                        f"{ticker}: {float(weight):+.1%}"
                    )
                except Exception:
                    lines.append(str(ticker))
            else:
                lines.append(str(ticker))

    if rec.get("risk_off"):
        lines.append("")
        lines.append("⚠️ Risk-off")

    return "\n".join(lines)

"""Планировщик авто-отправки рекомендаций в Telegram по расписанию.

Держите этот скрипт запущенным (в терминале, в screen/tmux или как сервис на
VPS). В заданные времена он присылает «портфель на сегодня».

Запуск:
    python scripts/10_scheduler.py

Времена настраиваются в config.yaml -> schedule.daily_times (и/или every_minutes).
Альтернатива без постоянно запущенного скрипта — системный планировщик
(cron в Linux/Mac или «Планировщик заданий» в Windows), вызывающий
scripts/08_recommend.py. См. README, раздел «Расписание».
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import schedule
from dotenv import load_dotenv

from src.config import load_config
from src.live.daily import send_daily_recommendation

load_dotenv()


def main() -> None:
    cfg = load_config()

    def job():
        try:
            print("[scheduler] Отправляю рекомендацию...")
            send_daily_recommendation(cfg, refresh_news=True)
            print("[scheduler] Готово.")
        except Exception as e:  # noqa: BLE001
            print(f"[scheduler] Ошибка: {e}")

    times = cfg.raw.get("schedule", {}).get("daily_times", [])
    for t in times:
        schedule.every().day.at(t).do(job)
        print(f"[scheduler] Запланировано ежедневно в {t}")

    every = int(cfg.raw.get("schedule", {}).get("every_minutes", 0) or 0)
    if every > 0:
        schedule.every(every).minutes.do(job)
        print(f"[scheduler] Запланировано каждые {every} мин.")

    if not times and every <= 0:
        print("Расписание пустое. Задайте schedule.daily_times в config.yaml.")
        return

    print("[scheduler] Работаю. Ctrl+C для выхода.")
    while True:
        schedule.run_pending()
        time.sleep(20)


if __name__ == "__main__":
    main()

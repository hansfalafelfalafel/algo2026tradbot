"""Сформировать рекомендацию на сегодня, показать её, записать в state/ и
отправить в Telegram (разовый запуск).

Запуск:
    python scripts/08_recommend.py                 # свежие новости из RSS
    python scripts/08_recommend.py --sample-news    # демо без сети
    python scripts/08_recommend.py --no-refresh     # из кэша новостей

Требует обученный ансамбль (scripts/06_train_ensemble.py). Telegram включится
автоматически, если в .env заданы TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.config import load_config
from src.live.daily import send_daily_recommendation
from src.notify.telegram import format_recommendation

load_dotenv()


def main() -> None:
    cfg = load_config()
    rec = send_daily_recommendation(
        cfg,
        refresh_news="--no-refresh" not in sys.argv and "--sample-news" not in sys.argv,
        sample_news="--sample-news" in sys.argv,
    )
    # Печать в консоль (без HTML-тегов).
    txt = format_recommendation(rec)
    for tag in ("<b>", "</b>", "<i>", "</i>"):
        txt = txt.replace(tag, "")
    print(txt)


if __name__ == "__main__":
    main()

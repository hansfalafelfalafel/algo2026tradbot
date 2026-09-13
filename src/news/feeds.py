"""Загрузка новостных заголовков из RSS-лент.

По умолчанию используются открытые RSS российских деловых СМИ (не требуют ключа).
Ленты настраиваются в config.yaml -> news.feeds. Если сеть недоступна или ленты
пусты, возвращается пустой список — вызывающий код умеет это обрабатывать.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

# Открытые RSS-ленты по умолчанию (деловые/экономические новости).
DEFAULT_FEEDS = [
    "https://www.finam.ru/analysis/conews/rsspoint/",
    "https://www.vedomosti.ru/rss/rubric/economics/markets.xml",
    "https://quote.rbc.ru/v1/news.rss",
]


@dataclass
class Headline:
    title: str
    summary: str
    published: Optional[datetime]
    source: str

    @property
    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()


def _parse_date(entry) -> Optional[datetime]:
    import time

    for key in ("published_parsed", "updated_parsed"):
        val = getattr(entry, key, None) or entry.get(key) if hasattr(entry, "get") else None
        if val:
            return datetime.fromtimestamp(time.mktime(val), tz=timezone.utc)
    return None


def fetch_headlines(feeds: Optional[List[str]] = None, limit_per_feed: int = 100) -> List[Headline]:
    """Скачать заголовки из списка RSS-лент.

    :param feeds: список URL RSS; если None — используются DEFAULT_FEEDS.
    :param limit_per_feed: максимум записей с одной ленты.
    """
    import feedparser  # ленивый импорт

    feeds = feeds or DEFAULT_FEEDS
    headlines: List[Headline] = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            source = parsed.feed.get("title", url) if parsed.feed else url
            for entry in parsed.entries[:limit_per_feed]:
                headlines.append(
                    Headline(
                        title=entry.get("title", ""),
                        summary=entry.get("summary", ""),
                        published=_parse_date(entry),
                        source=source,
                    )
                )
        except Exception as e:  # noqa: BLE001
            print(f"[news/feeds] Не удалось загрузить {url}: {e}")
    return headlines


def sample_headlines() -> List[Headline]:
    """Оффлайн-набор заголовков для демонстрации и тестов (без сети)."""
    now = datetime.now(tz=timezone.utc)
    data = [
        ("ЦБ повысил ключевую ставку на фоне ускорения инфляции", "Регулятор предупредил о рисках рецессии", -0.0),
        ("Рынок акций вырос на рекордной прибыли экспортеров", "Индекс МосБиржи обновил максимум", 0.0),
        ("Новые санкции усилили давление на рубль", "Аналитики ждут распродажи и обвала котировок", 0.0),
        ("Компания объявила дивиденды выше прогноза", "Позитив поддержал акции", 0.0),
        ("Экономисты предупреждают о спаде в промышленности", "Слабый спрос и риск дефолтов", 0.0),
    ]
    return [Headline(title=t, summary=s, published=now, source="sample") for t, s, _ in data]

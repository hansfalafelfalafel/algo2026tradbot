"""Загрузить новости, оценить тональность и сохранить сигнал в кэш.

Запуск:
    python scripts/05_fetch_news.py            # реальные RSS-ленты
    python scripts/05_fetch_news.py --sample   # оффлайн-демо без сети

Результат: data_cache/news_signal.csv с колонками date, sentiment (+сглаженное).
Бэкенд тональности берётся из config.yaml (news.backend): lexicon | hf | llm.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.config import load_config
from src.news.feeds import fetch_headlines, sample_headlines
from src.news.sentiment import get_backend
from src.news.signal import build_daily_series, score_headlines

load_dotenv()


def main() -> None:
    cfg = load_config()
    use_sample = "--sample" in sys.argv

    if use_sample:
        headlines = sample_headlines()
        print(f"Демо-режим: {len(headlines)} заголовков (без сети).")
    else:
        feeds = cfg.news.get("feeds") or None
        headlines = fetch_headlines(feeds)
        print(f"Загружено заголовков: {len(headlines)}")
        if not headlines:
            print("Ленты пусты/недоступны. Попробуйте --sample или проверьте сеть.")
            return

    backend_name = cfg.news.get("backend", "lexicon")
    kwargs = {}
    if backend_name == "hf":
        kwargs["model_name"] = cfg.news.get("hf_model")
    elif backend_name == "llm":
        kwargs["model"] = cfg.news.get("llm_model", "gpt-4o-mini")
    backend = get_backend(backend_name, **kwargs)
    print(f"Бэкенд тональности: {backend_name}")

    daily = score_headlines(headlines, backend)
    daily = build_daily_series(daily, ema_span=cfg.news.get("ema_span", 3))

    out = cfg.abs_path(cfg.data["cache_dir"], "news_signal.csv")
    daily.to_csv(out, index=False)
    print(f"\nСохранено: {out}")
    print(daily.tail(10).to_string(index=False))
    if not daily.empty:
        print(f"\nСредний сентимент: {daily['sentiment'].mean():+.3f}")


if __name__ == "__main__":
    main()

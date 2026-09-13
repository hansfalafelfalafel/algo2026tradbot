"""Оценка тональности новостей (sentiment) с подключаемыми бэкендами.

Возвращает по каждому тексту число в диапазоне [-1, 1]:
    -1 — резко негативно (риск падения),
     0 — нейтрально,
    +1 — позитивно.

Бэкенды (выбираются в config.yaml -> news.backend):
  * "lexicon" — офлайн-словарь (по умолчанию): работает без ключей и интернета,
                удобно для запуска «из коробки» и для тестов.
  * "hf"      — модель тональности из HuggingFace Transformers (напр., RuBERT),
                качественнее словаря, требует пакет transformers.
  * "llm"     — большая языковая модель через OpenAI-совместимый API
                (OpenAI / GigaChat / YandexGPT / локальный сервер). Умеет не
                только тональность, но и оценку риска экономического спада.

Все бэкенды реализуют один интерфейс score_texts(texts) -> list[float].
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import List


class SentimentBackend(ABC):
    @abstractmethod
    def score_texts(self, texts: List[str]) -> List[float]:
        """Оценить список текстов -> список чисел в [-1, 1]."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Офлайн-словарь (по умолчанию)
# ---------------------------------------------------------------------------
# Небольшой финансово-экономический словарь (RU + EN). Не претендует на
# точность LLM, но даёт разумный сигнал без внешних зависимостей и ключей.
_POSITIVE = {
    # рост и позитив
    "рост", "растет", "вырос", "выросли", "прибыль", "рекорд", "рекордный",
    "укрепление", "укрепляется", "подъем", "восстановление", "оптимизм",
    "позитив", "позитивный", "поддержка", "стимул", "дивиденды", "buyback",
    "beat", "surge", "rally", "profit", "gain", "growth", "record", "upgrade",
    "bullish", "recovery", "stimulus", "outperform", "strong",
}
_NEGATIVE = {
    # падение, кризис, риск
    "падение", "падает", "упал", "упали", "обвал", "кризис", "рецессия",
    "спад", "убыток", "убытки", "дефолт", "санкции", "инфляция", "девальвация",
    "распродажа", "паника", "снижение", "сокращение", "bankротство", "банкротство",
    "риск", "угроза", "негатив", "негативный", "слабый", "просадка", "давление",
    "crash", "crisis", "recession", "loss", "default", "sanctions", "inflation",
    "selloff", "panic", "downgrade", "bearish", "plunge", "slump", "weak",
    "miss", "cut", "warning", "layoff", "layoffs",
}
# Усилители/ослабители негатива для «предвосхищения спада»
_RISK_WORDS = {
    "рецессия", "кризис", "дефолт", "обвал", "санкции", "паника", "девальвация",
    "recession", "crisis", "default", "crash", "sanctions", "panic",
}

_TOKEN_RE = re.compile(r"[а-яё]+|[a-z]+", re.IGNORECASE)


class LexiconBackend(SentimentBackend):
    """Простой лексиконный анализатор тональности (RU/EN)."""

    def score_texts(self, texts: List[str]) -> List[float]:
        scores = []
        for text in texts:
            tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
            if not tokens:
                scores.append(0.0)
                continue
            pos = sum(1 for t in tokens if t in _POSITIVE)
            neg = sum(1 for t in tokens if t in _NEGATIVE)
            risk = sum(1 for t in tokens if t in _RISK_WORDS)
            # Риск-слова дополнительно «тянут» вниз (предвосхищение спада).
            raw = (pos - neg - 0.5 * risk)
            # Нормируем на длину значимой части, ограничиваем в [-1, 1].
            denom = max(1.0, pos + neg + risk)
            score = max(-1.0, min(1.0, raw / denom))
            scores.append(float(score))
        return scores


# ---------------------------------------------------------------------------
# 2. HuggingFace Transformers (опционально)
# ---------------------------------------------------------------------------
class HFBackend(SentimentBackend):
    """Тональность через модель Transformers (напр., RuBERT-sentiment).

    Модель по умолчанию — многоязычная/русская sentiment-модель. Требует
    установленный пакет ``transformers`` и загрузку весов при первом запуске.
    """

    def __init__(self, model_name: str = "blanchefort/rubert-base-cased-sentiment"):
        from transformers import pipeline  # ленивый импорт

        self.pipe = pipeline("sentiment-analysis", model=model_name, truncation=True)

    def score_texts(self, texts: List[str]) -> List[float]:
        if not texts:
            return []
        results = self.pipe(list(texts))
        out = []
        for r in results:
            label = r.get("label", "").upper()
            conf = float(r.get("score", 0.0))
            if "POS" in label:
                out.append(conf)
            elif "NEG" in label:
                out.append(-conf)
            else:  # NEUTRAL
                out.append(0.0)
        return out


# ---------------------------------------------------------------------------
# 3. LLM через OpenAI-совместимый API (опционально)
# ---------------------------------------------------------------------------
class LLMBackend(SentimentBackend):
    """Оценка тональности и риска через большую языковую модель.

    Работает с любым OpenAI-совместимым API. Параметры берутся из окружения:
      * NEWS_LLM_API_KEY   — ключ API;
      * NEWS_LLM_BASE_URL  — базовый URL (для GigaChat/YandexGPT/локального сервера);
      * NEWS_LLM_MODEL     — имя модели (по умолчанию из аргумента).

    LLM просит оценить, насколько новость позитивна/негативна для рынка и
    предвещает ли она ухудшение экономических показателей, и вернуть число.
    """

    SYSTEM_PROMPT = (
        "Ты финансовый аналитик. Оцени влияние новости на российский фондовый "
        "рынок числом от -1 до 1: -1 — крайне негативно (предвещает падение, "
        "рецессию, ухудшение экономических показателей), 0 — нейтрально, "
        "1 — крайне позитивно. Верни ТОЛЬКО число."
    )

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI  # ленивый импорт

        api_key = os.environ.get("NEWS_LLM_API_KEY")
        base_url = os.environ.get("NEWS_LLM_BASE_URL")  # None -> OpenAI по умолчанию
        self.model = os.environ.get("NEWS_LLM_MODEL", model)
        if not api_key:
            raise RuntimeError(
                "Для backend='llm' задайте NEWS_LLM_API_KEY (и при необходимости "
                "NEWS_LLM_BASE_URL / NEWS_LLM_MODEL) в .env."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _score_one(self, text: str) -> float:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": (text or "")[:2000]},
                ],
                temperature=0.0,
                max_tokens=8,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"-?\d+(\.\d+)?", raw)
            val = float(m.group()) if m else 0.0
            return max(-1.0, min(1.0, val))
        except Exception as e:  # noqa: BLE001
            print(f"[news/llm] Ошибка запроса к LLM: {e}. Ставлю 0.")
            return 0.0

    def score_texts(self, texts: List[str]) -> List[float]:
        return [self._score_one(t) for t in texts]


# ---------------------------------------------------------------------------
# Фабрика
# ---------------------------------------------------------------------------
def get_backend(name: str = "lexicon", **kwargs) -> SentimentBackend:
    name = (name or "lexicon").lower()
    if name == "lexicon":
        return LexiconBackend()
    if name == "hf":
        return HFBackend(**kwargs)
    if name == "llm":
        return LLMBackend(**kwargs)
    raise ValueError(f"Неизвестный backend тональности: {name!r}")

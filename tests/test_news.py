"""Тесты новостного модуля (офлайн-словарь) и выравнивания сентимента."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.news.feeds import sample_headlines
from src.news.sentiment import get_backend
from src.news.signal import align_to_df, build_daily_series, score_headlines


def test_lexicon_polarity():
    b = get_backend("lexicon")
    neg, pos, neu = b.score_texts([
        "Обвал и кризис на фоне санкций и рецессии",
        "Рекордная прибыль и рост дивидендов",
        "Обычный день",
    ])
    assert neg < 0 < pos
    assert abs(neu) < 1e-9


def test_score_and_align():
    b = get_backend("lexicon")
    daily = build_daily_series(score_headlines(sample_headlines(), b))
    assert "sentiment" in daily.columns

    candles = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=50, freq="h"),
        "close": range(100, 150),
    })
    out = align_to_df(candles, daily)
    assert "sentiment" in out.columns
    assert "risk_off" in out.columns
    assert len(out) == len(candles)


if __name__ == "__main__":
    test_lexicon_polarity()
    test_score_and_align()
    print("Тесты новостного модуля пройдены ✓")

"""Тест оценки переобучения PBO (CSCV)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.backtest.walkforward import pbo_cscv


def test_pbo_low_for_persistent_skill():
    """Одна конфигурация стабильно лучше -> низкий PBO."""
    rng = np.random.default_rng(0)
    T, N = 400, 6
    perf = rng.normal(0.0, 0.01, (T, N))
    perf[:, 0] += 0.004  # конфигурация 0 имеет устойчивое преимущество
    res = pbo_cscv(perf, n_splits=8)
    assert res["pbo"] < 0.2


def test_pbo_high_for_noise():
    """Чистый шум без устойчивого лидера -> PBO в среднем около 0.5.

    Оценка PBO на отдельной выборке шумит, поэтому усредняем по нескольким сидам.
    """
    pbos = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        perf = rng.normal(0.0, 0.01, (400, 8))
        pbos.append(pbo_cscv(perf, n_splits=8)["pbo"])
    assert 0.3 <= float(np.mean(pbos)) <= 0.7


def test_pbo_skill_below_noise():
    """У стратегии с устойчивым скиллом PBO ниже, чем у шума (в среднем)."""
    rng = np.random.default_rng(7)
    perf = rng.normal(0.0, 0.01, (400, 6))
    perf[:, 0] += 0.004
    skill = pbo_cscv(perf, n_splits=8)["pbo"]
    noise = np.mean([pbo_cscv(np.random.default_rng(s).normal(0, 0.01, (400, 6)), 8)["pbo"]
                     for s in range(6)])
    assert skill < noise


if __name__ == "__main__":
    test_pbo_low_for_persistent_skill()
    test_pbo_high_for_noise()
    test_pbo_skill_below_noise()
    print("Тесты PBO пройдены ✓")

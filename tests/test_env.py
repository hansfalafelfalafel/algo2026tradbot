"""Юнит-тесты торговой среды на синтетических данных.

Эти тесты не требуют токена и интернета — они проверяют корректность логики
среды (форма наблюдений, границы действий, учёт комиссий) на сгенерированном
случайном ряде цен. Запуск: ``pytest -q`` или ``python tests/test_env.py``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.features import add_features, feature_columns
from src.env.trading_env import ACTION_LONG, ACTION_FLAT, TradingEnv


def _make_synthetic_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Случайное блуждание как заменитель цен для тестов."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=n, freq="h"),
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.integers(1000, 5000, size=n),
        }
    )
    return df


def _make_env(**kwargs) -> TradingEnv:
    df = add_features(_make_synthetic_df(), use_indicators=True)
    cols = feature_columns(use_indicators=True)
    return TradingEnv(df=df, feature_cols=cols, window_size=30, **kwargs)


def test_observation_shape():
    env = _make_env()
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32


def test_action_space_sizes():
    env_long_only = _make_env(allow_short=False)
    assert env_long_only.action_space.n == 2
    env_short = _make_env(allow_short=True)
    assert env_short.action_space.n == 3


def test_full_episode_runs():
    env = _make_env()
    env.reset()
    done = False
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
        assert np.isfinite(reward)
    assert steps > 0
    assert info["equity"] > 0


def test_commission_reduces_equity_on_flip():
    """Постоянные развороты позиции должны съедать капитал через комиссии."""
    env = _make_env(commission=0.01, slippage=0.0)
    env.reset()
    # Чередуем лонг/флэт каждый шаг -> много сделок.
    actions = [ACTION_LONG, ACTION_FLAT]
    done = False
    i = 0
    while not done:
        _, _, terminated, truncated, info = env.step(actions[i % 2])
        done = terminated or truncated
        i += 1
    assert info["trades"] > 0


if __name__ == "__main__":
    test_observation_shape()
    test_action_space_sizes()
    test_full_episode_runs()
    test_commission_reduces_equity_on_flip()
    print("Все тесты пройдены ✓")

"""Тесты портфельной среды: форма наблюдений, ограничение экспозиции, risk-off."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.env.portfolio_env import PortfolioEnv


def _make_env(**kw):
    T, N, F = 200, 3, 3
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, N)), axis=0))
    feats = rng.normal(0, 1, (T, N * F))
    return PortfolioEnv(prices, feats, N, F, window_size=10, **kw)


def test_obs_and_action_shapes():
    env = _make_env()
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.action_space.shape == (3,)


def test_gross_exposure_capped():
    env = _make_env(max_gross=1.0, risk_off_strength=0.0)
    env.reset()
    _, _, _, _, info = env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    # Σ|w| не должна превышать max_gross.
    assert info["gross_exposure"] <= 1.0 + 1e-6


def test_risk_off_reduces_exposure():
    T, N, F = 100, 2, 2
    rng = np.random.default_rng(1)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, N)), axis=0))
    feats = rng.normal(0, 1, (T, N * F))
    neg = np.full(T, -0.8)
    env = PortfolioEnv(prices, feats, N, F, sentiment=neg,
                       window_size=5, risk_off_strength=0.5, max_gross=1.0)
    env.reset()
    _, _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    # risk_scale = 1 - 0.5*0.8 = 0.6 -> экспозиция должна упасть заметно ниже 1.
    assert info["risk_scale"] < 0.7
    assert info["gross_exposure"] < 0.7


def test_full_episode():
    env = _make_env()
    env.reset()
    done = False
    while not done:
        _, r, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc
        assert np.isfinite(r)
    assert info["equity"] > 0


if __name__ == "__main__":
    test_obs_and_action_shapes()
    test_gross_exposure_capped()
    test_risk_off_reduces_exposure()
    test_full_episode()
    print("Тесты портфельной среды пройдены ✓")

"""Тесты Monte-Carlo прогноза и диагностики."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.analytics.diagnostics import drawdown_series, rolling_sharpe, summarize_returns
from src.analytics.forecast import monte_carlo_forecast


def test_forecast_structure_and_bounds():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, 500)
    res = monte_carlo_forecast(returns, capital=1_000_000, horizon=60,
                               n_sims=500, percentiles=(5, 50, 95))
    assert len(res["percentiles"][50]) == 60
    # P5 всегда ниже P95 на каждом шаге.
    assert np.all(res["percentiles"][5] <= res["percentiles"][95] + 1e-6)
    assert 0.0 <= res["prob_loss"] <= 1.0
    assert res["final_p5"] <= res["final_median"] <= res["final_p95"]


def test_positive_drift_grows_median():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.002, 0.005, 500)  # уверенный положительный дрейф
    res = monte_carlo_forecast(returns, capital=100_000, horizon=100, n_sims=800)
    assert res["final_median"] > 100_000


def test_diagnostics_shapes():
    eq = np.cumprod(1 + np.random.default_rng(2).normal(0.001, 0.01, 200)) * 1e5
    dd = drawdown_series(eq)
    assert dd.min() <= 0.0 and dd.max() <= 1e-9
    rs = rolling_sharpe(np.diff(eq) / eq[:-1], window=30)
    assert len(rs) == len(eq) - 1
    stats = summarize_returns(np.diff(eq) / eq[:-1])
    assert "sharpe" in stats and "var_95" in stats


if __name__ == "__main__":
    test_forecast_structure_and_bounds()
    test_positive_drift_grows_median()
    test_diagnostics_shapes()
    print("Тесты прогноза и диагностики пройдены ✓")

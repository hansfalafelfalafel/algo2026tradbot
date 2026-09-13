"""Бэктест портфельного агента/ансамбля на отложенной выборке."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agent.ensemble import make_portfolio_env
from src.backtest.backtest import compute_metrics


def run_portfolio_backtest(agent, arrays: dict, cfg) -> dict:
    """Прогнать агента (одиночного или ансамбль) по массивам и собрать метрики."""
    env = make_portfolio_env(arrays, cfg)
    obs, _ = env.reset()
    done = False
    gross, risk_scales = [], []
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        gross.append(info["gross_exposure"])
        risk_scales.append(info["risk_scale"])
        done = terminated or truncated

    equity = env.equity_curve
    metrics = compute_metrics(equity, cfg.data["interval"])

    # Бенчмарк: равновзвешенный «купи и держи» по всем инструментам.
    prices = arrays["prices"][env.window_size:]
    norm = prices / prices[0]
    eq_bh = cfg.env["initial_balance"] * norm.mean(axis=1)
    bh_metrics = compute_metrics(eq_bh, cfg.data["interval"])

    return {
        "metrics": metrics,
        "buy_hold_metrics": bh_metrics,
        "equity_curve": equity,
        "buy_hold_curve": eq_bh,
        "avg_gross_exposure": float(np.mean(gross)) if gross else 0.0,
        "avg_risk_scale": float(np.mean(risk_scales)) if risk_scales else 1.0,
    }


def format_portfolio_report(result: dict, cfg) -> str:
    m = result["metrics"]
    bh = result["buy_hold_metrics"]

    def pct(x: float) -> str:
        return f"{x * 100:+.2f}%"

    return "\n".join([
        "=" * 62,
        f"ПОРТФЕЛЬНЫЙ БЭКТЕСТ — {', '.join(cfg.data.get('tickers', []))}",
        f"Алгоритм: ансамбль {cfg.agent.get('algo', 'PPO+SAC+A2C')}",
        "=" * 62,
        f"{'Метрика':<30}{'Агент':>15}{'Buy&Hold':>15}",
        "-" * 62,
        f"{'Итоговая доходность':<30}{pct(m['total_return']):>15}{pct(bh['total_return']):>15}",
        f"{'Коэффициент Шарпа':<30}{m['sharpe']:>15.2f}{bh['sharpe']:>15.2f}",
        f"{'Макс. просадка':<30}{pct(m['max_drawdown']):>15}{pct(bh['max_drawdown']):>15}",
        f"{'Итоговый капитал, руб.':<30}{m['final_equity']:>15,.0f}{bh['final_equity']:>15,.0f}",
        "-" * 62,
        f"Средняя экспозиция (gross):   {result['avg_gross_exposure']:.2f}",
        f"Средний риск-множитель новостей: {result['avg_risk_scale']:.2f}",
        "=" * 62,
    ])

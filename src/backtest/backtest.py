"""Бэктест обученного агента на отложенной (test) выборке.

Считает ключевые метрики стратегии и строит график кривой капитала в сравнении
с пассивным «купи и держи» (buy & hold).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Примечание: make_env (и через него stable_baselines3) импортируется ЛЕНИВО
# внутри run_backtest, чтобы модуль метрик (compute_metrics) можно было
# использовать без установленного RL-стека (напр., в парном трейдинге).


def _annualization_factor(interval: str) -> float:
    """Сколько баров приходится примерно на один торговый год.

    Нужно для приведения метрик (Sharpe) к годовому виду.
    """
    bars_per_year = {
        "1min": 60 * 8 * 250,
        "5min": 12 * 8 * 250,
        "15min": 4 * 8 * 250,
        "hour": 8 * 250,
        "day": 250,
    }
    return float(bars_per_year.get(interval, 250))


def compute_metrics(equity_curve: np.ndarray, interval: str) -> dict:
    """Рассчитать метрики по кривой капитала."""
    equity = np.asarray(equity_curve, dtype=np.float64)
    returns = np.diff(equity) / equity[:-1]

    total_return = equity[-1] / equity[0] - 1.0
    ann = _annualization_factor(interval)

    if returns.std() > 1e-12:
        sharpe = np.sqrt(ann) * returns.mean() / returns.std()
    else:
        sharpe = 0.0

    # Максимальная просадка.
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    # Доля прибыльных шагов.
    win_rate = float((returns > 0).mean()) if len(returns) else 0.0

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": win_rate,
        "final_equity": float(equity[-1]),
    }


def run_backtest(model, df_test: pd.DataFrame, cfg) -> dict:
    """Прогнать модель по test-выборке детерминированно и вернуть результаты."""
    from src.agent.train import make_env  # ленивый импорт (тянет stable_baselines3)
    env = make_env(df_test, cfg)
    obs, _ = env.reset()
    done = False
    positions = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        positions.append(info["position"])
        done = terminated or truncated

    equity_curve = env.equity_curve
    metrics = compute_metrics(equity_curve, cfg.data["interval"])

    # Buy & hold для сравнения: держим +1 лонг весь период.
    prices = env.prices[env.window_size:]
    bh_curve = cfg.env["initial_balance"] * (prices / prices[0])
    bh_metrics = compute_metrics(bh_curve, cfg.data["interval"])

    return {
        "metrics": metrics,
        "buy_hold_metrics": bh_metrics,
        "equity_curve": equity_curve,
        "buy_hold_curve": bh_curve,
        "positions": np.asarray(positions),
    }


def plot_results(result: dict, cfg, save_path: str | Path) -> Path:
    """Сохранить график кривой капитала стратегии vs buy & hold."""
    import matplotlib
    matplotlib.use("Agg")  # без GUI
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    strat = result["equity_curve"]
    bh = result["buy_hold_curve"]
    # Выравниваем длины (у стратегии на 1 точку больше из-за стартового значения).
    n = min(len(strat), len(bh))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(strat[:n], label="RL-агент", linewidth=1.6)
    ax.plot(bh[:n], label="Buy & Hold", linewidth=1.2, alpha=0.8)
    ax.set_title(
        f"Кривая капитала: {cfg.data['ticker']} ({cfg.data['interval']}), "
        f"алгоритм {cfg.agent['algo']}"
    )
    ax.set_xlabel("Бар (шаг)")
    ax.set_ylabel("Капитал, руб.")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def format_report(result: dict, cfg) -> str:
    """Человекочитаемый текстовый отчёт с метриками."""
    m = result["metrics"]
    bh = result["buy_hold_metrics"]

    def pct(x: float) -> str:
        return f"{x * 100:+.2f}%"

    lines = [
        "=" * 60,
        f"РЕЗУЛЬТАТЫ БЭКТЕСТА — {cfg.data['ticker']} ({cfg.data['interval']})",
        f"Алгоритм: {cfg.agent['algo']}",
        "=" * 60,
        f"{'Метрика':<28}{'RL-агент':>16}{'Buy & Hold':>16}",
        "-" * 60,
        f"{'Итоговая доходность':<28}{pct(m['total_return']):>16}{pct(bh['total_return']):>16}",
        f"{'Коэффициент Шарпа':<28}{m['sharpe']:>16.2f}{bh['sharpe']:>16.2f}",
        f"{'Макс. просадка':<28}{pct(m['max_drawdown']):>16}{pct(bh['max_drawdown']):>16}",
        f"{'Доля прибыльных шагов':<28}{pct(m['win_rate']):>16}{pct(bh['win_rate']):>16}",
        f"{'Итоговый капитал, руб.':<28}{m['final_equity']:>16,.0f}{bh['final_equity']:>16,.0f}",
        "=" * 60,
    ]
    return "\n".join(lines)

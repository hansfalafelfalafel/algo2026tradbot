"""Диагностика модели и стратегии.

Считает по кривой капитала и доходностям: скользящий Шарп, серию просадок,
распределение доходностей, а также читает логи обучения агентов (loss,
value_loss, explained_variance) — это и есть «ошибки модели».
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def rolling_sharpe(returns: np.ndarray, window: int = 50, ann: float = 252) -> np.ndarray:
    """Скользящий коэффициент Шарпа."""
    s = pd.Series(returns)
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    sharpe = np.sqrt(ann) * mean / std.replace(0, np.nan)
    return sharpe.to_numpy()


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Серия просадок (доля от локального максимума, <=0)."""
    eq = np.asarray(equity, dtype=np.float64)
    running_max = np.maximum.accumulate(eq)
    return (eq - running_max) / running_max


def returns_from_equity(equity: np.ndarray) -> np.ndarray:
    eq = np.asarray(equity, dtype=np.float64)
    return np.diff(eq) / eq[:-1]


def summarize_returns(returns: np.ndarray, ann: float = 252) -> dict:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {}
    downside = r[r < 0]
    sortino = (np.sqrt(ann) * r.mean() / downside.std()) if downside.std() > 1e-12 else 0.0
    sharpe = (np.sqrt(ann) * r.mean() / r.std()) if r.std() > 1e-12 else 0.0
    return {
        "mean": float(r.mean()),
        "std": float(r.std()),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "skew": float(pd.Series(r).skew()),
        "kurtosis": float(pd.Series(r).kurtosis()),
        "win_rate": float((r > 0).mean()),
        "best": float(r.max()),
        "worst": float(r.min()),
        "var_95": float(np.percentile(r, 5)),  # 5%-квантиль (VaR)
    }


def read_training_logs(model_dir: Path, model_name: str, algos) -> dict:
    """Прочитать progress.csv по каждому агенту ансамбля.

    Возвращает {algo -> DataFrame} с метриками обучения (loss, value_loss,
    explained_variance, ep_rew_mean и т.п.), если логи есть.
    """
    out = {}
    for algo in algos:
        p = Path(model_dir) / "logs" / f"{model_name}_{algo}" / "progress.csv"
        if p.exists():
            try:
                out[algo] = pd.read_csv(p)
            except Exception:  # noqa: BLE001
                pass
    return out

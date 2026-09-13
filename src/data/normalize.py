"""Нормализация признаков (z-score) для устойчивого обучения RL.

Признаки в среде имеют разный масштаб (доходности ~0.01 рядом с RSI 0–1), из-за
чего нейросети трудно обучаться — агент часто вырождается в «всегда лонг».
Стандартизация приводит все признаки к нулевому среднему и единичному разбросу.

ВАЖНО: статистики (среднее, std) считаются ТОЛЬКО по обучающей части, иначе
произойдёт «подглядывание в будущее» (data leakage). Эти же статистики затем
применяются к валидации, тесту и live-данным.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def fit_feature_stats(features: np.ndarray) -> dict:
    """Посчитать среднее и std по каждому столбцу-признаку."""
    feats = np.asarray(features, dtype=np.float64)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # защита от деления на ноль
    return {"mean": mean, "std": std}


def apply_feature_stats(features: np.ndarray, stats: dict) -> np.ndarray:
    """Стандартизовать признаки заданными статистиками, обрезав выбросы."""
    feats = np.asarray(features, dtype=np.float64)
    z = (feats - stats["mean"]) / stats["std"]
    return np.clip(z, -10.0, 10.0)  # отсечь экстремальные выбросы


def normalize_fit(arrays: dict, split: int) -> dict:
    """Обучить нормализатор на первых split барах и применить ко всем.

    Меняет arrays['features'] на месте, возвращает статистики.
    """
    stats = fit_feature_stats(arrays["features"][:split])
    arrays["features"] = apply_feature_stats(arrays["features"], stats)
    return stats


def normalize_apply(arrays: dict, stats: dict) -> None:
    """Применить готовые статистики к arrays['features'] (на месте)."""
    arrays["features"] = apply_feature_stats(arrays["features"], stats)


def save_stats(path: str | Path, stats: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=stats["mean"], std=stats["std"])


def load_stats(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path)
    return {"mean": data["mean"], "std": data["std"]}

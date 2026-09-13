"""Загрузка конфигурации проекта из YAML-файла.

Все параметры системы вынесены в ``config.yaml`` в корне проекта, чтобы менять
поведение (инструмент, таймфрейм, гиперпараметры) без правки кода.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

# Корень проекта = папка на уровень выше src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    """Обёртка над словарём конфигурации с удобным доступом по секциям."""

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(raw=data)

    # Доступ к секциям как к атрибутам: cfg.data, cfg.env, ...
    def __getattr__(self, item: str) -> Any:
        # __getattr__ вызывается только если обычный атрибут не найден
        raw = self.__dict__.get("raw", {})
        if item in raw:
            return raw[item]
        raise AttributeError(item)

    def abs_path(self, *parts: str) -> Path:
        """Абсолютный путь относительно корня проекта."""
        return PROJECT_ROOT.joinpath(*parts)


def load_config(path: str | os.PathLike | None = None) -> Config:
    return Config.load(path)

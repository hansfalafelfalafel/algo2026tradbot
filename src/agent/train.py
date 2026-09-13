"""Обучение RL-агента на исторических данных.

Поддерживаются три алгоритма из Stable-Baselines3: PPO (по умолчанию), A2C, DQN.
Выбор алгоритма и гиперпараметры берутся из config.yaml (секция agent).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data.features import add_features, feature_columns
from src.env.trading_env import TradingEnv

ALGOS = {"PPO": PPO, "A2C": A2C, "DQN": DQN}


def make_env(df: pd.DataFrame, cfg) -> TradingEnv:
    """Создать TradingEnv из «сырого» датафрейма свечей и конфига."""
    feat_df = add_features(df, use_indicators=cfg.features["use_indicators"])
    cols = feature_columns(use_indicators=cfg.features["use_indicators"])
    env = TradingEnv(
        df=feat_df,
        feature_cols=cols,
        window_size=cfg.features["window_size"],
        initial_balance=cfg.env["initial_balance"],
        commission=cfg.env["commission"],
        slippage=cfg.env["slippage"],
        allow_short=cfg.env["allow_short"],
        reward_scaling=cfg.env["reward_scaling"],
    )
    return env


def train(df_train: pd.DataFrame, cfg) -> Path:
    """Обучить агента и сохранить модель. Возвращает путь к файлу модели."""
    algo_name = cfg.agent["algo"].upper()
    if algo_name not in ALGOS:
        raise ValueError(f"Неизвестный алгоритм {algo_name}. Доступно: {list(ALGOS)}")
    Algo = ALGOS[algo_name]

    # Оборачиваем среду в Monitor + DummyVecEnv (требование SB3).
    env = DummyVecEnv([lambda: Monitor(make_env(df_train, cfg))])

    model = Algo(
        "MlpPolicy",
        env,
        learning_rate=cfg.agent["learning_rate"],
        gamma=cfg.agent["gamma"],
        seed=cfg.agent["seed"],
        verbose=1,
    )
    model.learn(total_timesteps=cfg.agent["total_timesteps"])

    model_dir = cfg.abs_path(cfg.agent["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{cfg.agent['model_name']}.zip"
    model.save(model_path)
    return model_path


def load_model(cfg):
    """Загрузить ранее обученную модель по имени из конфига."""
    algo_name = cfg.agent["algo"].upper()
    Algo = ALGOS[algo_name]
    model_path = cfg.abs_path(cfg.agent["model_dir"], f"{cfg.agent['model_name']}.zip")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}. Сначала запустите обучение "
            f"(scripts/02_train.py)."
        )
    return Algo.load(model_path)

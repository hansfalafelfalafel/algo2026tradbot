"""Ансамбль актор-критик агентов (PPO + SAC + A2C) для портфельной среды.

Идея (по мотивам ensemble-стратегии AI4Finance/FinRL): обучаем несколько
разнотипных агентов и агрегируем их действия с весами по риск-скорректированной
доходности (коэффициенту Шарпа) на валидации. Ансамбль снижает дисперсию и
устойчивее отдельного агента.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.portfolio_env import PortfolioEnv

# Для непрерывного действия (веса портфеля) подходят PPO, SAC, A2C.
ENSEMBLE_ALGOS = {"PPO": PPO, "SAC": SAC, "A2C": A2C}


def make_portfolio_env(arrays: dict, cfg) -> PortfolioEnv:
    """Создать PortfolioEnv из собранных массивов и конфига."""
    e = cfg.env
    return PortfolioEnv(
        prices=arrays["prices"],
        features=arrays["features"],
        n_assets=arrays["n_assets"],
        n_features=arrays["n_features"],
        sentiment=arrays.get("sentiment"),
        window_size=cfg.features["window_size"],
        initial_balance=e["initial_balance"],
        commission=e["commission"],
        slippage=e["slippage"],
        allow_short=e["allow_short"],
        max_gross=e.get("max_gross", 1.0),
        turnover_penalty=e.get("turnover_penalty", 0.001),
        dd_penalty=e.get("dd_penalty", 1.0),
        dd_limit=e.get("dd_limit", 0.1),
        risk_off_strength=e.get("risk_off_strength", 0.5),
    )


def evaluate_sharpe(model, arrays: dict, cfg) -> float:
    """Прогнать модель по данным и вернуть (несглаженный) коэффициент Шарпа."""
    env = make_portfolio_env(arrays, cfg)
    obs, _ = env.reset()
    rets: List[float] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        rets.append(info["portfolio_return"])
        done = terminated or truncated
    r = np.asarray(rets)
    if r.std() < 1e-12:
        return 0.0
    return float(np.sqrt(252) * r.mean() / r.std())


def train_ensemble(arrays_train: dict, cfg, algos: List[str] | None = None) -> Dict[str, Path]:
    """Обучить каждый алгоритм ансамбля и сохранить модели."""
    algos = algos or list(ENSEMBLE_ALGOS.keys())
    model_dir = cfg.abs_path(cfg.agent["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    for name in algos:
        Algo = ENSEMBLE_ALGOS[name]
        env = DummyVecEnv([lambda: Monitor(make_portfolio_env(arrays_train, cfg))])
        # SAC не принимает n_steps-подобные параметры; общие параметры совместимы.
        # SAC — off-policy: у него есть буфер воспроизведения. При большом числе
        # инструментов наблюдение крупное, поэтому ограничиваем буфер, иначе не
        # хватит оперативной памяти (буфер по умолчанию — 1e6 переходов).
        extra = {"buffer_size": int(cfg.agent.get("sac_buffer_size", 50000))} if name == "SAC" else {}
        model = Algo(
            "MlpPolicy",
            env,
            learning_rate=cfg.agent["learning_rate"],
            gamma=cfg.agent["gamma"],
            seed=cfg.agent["seed"],
            verbose=0,
            **extra,
        )
        # CSV-логгер: пишет loss, value_loss, explained_variance и др. —
        # это «ошибки модели» для DS-вкладки дашборда.
        log_dir = model_dir / "logs" / f"{cfg.agent['model_name']}_{name}"
        log_dir.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(log_dir), ["csv"]))
        print(f"[ensemble] Обучаю {name} ...")
        model.learn(total_timesteps=cfg.agent["total_timesteps"])
        path = model_dir / f"{cfg.agent['model_name']}_{name}.zip"
        model.save(path)
        paths[name] = path
        print(f"[ensemble] {name} сохранён: {path.name}")
    return paths


class EnsembleAgent:
    """Агрегатор обученных агентов с весами по Шарпу."""

    def __init__(self, models: Dict[str, object], weights: Dict[str, float]):
        self.models = models
        self.weights = weights

    @classmethod
    def load_and_weight(cls, paths: Dict[str, Path], arrays_val: dict, cfg) -> "EnsembleAgent":
        """Загрузить модели и рассчитать веса по Шарпу на валидации."""
        models, sharpes = {}, {}
        for name, path in paths.items():
            model = ENSEMBLE_ALGOS[name].load(path)
            models[name] = model
            sharpes[name] = evaluate_sharpe(model, arrays_val, cfg)
            print(f"[ensemble] {name}: Sharpe(val) = {sharpes[name]:.3f}")

        # Softmax по неотрицательным Шарпам (переобученных/убыточных не усиливаем).
        s = np.array([max(0.0, sharpes[n]) for n in models])
        if s.sum() < 1e-9:
            w = np.ones(len(models)) / len(models)  # все плохи -> равные веса
        else:
            e = np.exp(s - s.max())
            w = e / e.sum()
        weights = {n: float(w[i]) for i, n in enumerate(models)}
        print(f"[ensemble] Веса агрегации: {weights}")
        return cls(models, weights)

    def predict(self, obs, deterministic: bool = True):
        """Взвешенное усреднение действий агентов."""
        acc = None
        for name, model in self.models.items():
            action, _ = model.predict(obs, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float64)
            acc = action * self.weights[name] if acc is None else acc + action * self.weights[name]
        return acc, None

"""Торговая среда в стандарте Gymnasium для обучения RL-агента.

Идея: на каждом шаге агент видит окно из последних ``window_size`` баров
(признаки) и свою текущую позицию, а затем выбирает действие — встать в лонг,
в шорт или выйти из рынка. Награда равна изменению стоимости портфеля за шаг
за вычетом торговых издержек.

Среда совместима со Stable-Baselines3 (PPO/A2C/DQN).
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# Действия агента
ACTION_FLAT = 0   # вне рынка (позиция 0)
ACTION_LONG = 1   # длинная позиция (+1)
ACTION_SHORT = 2  # короткая позиция (-1)

# Целевая позиция для каждого действия
ACTION_TO_POSITION = {ACTION_FLAT: 0, ACTION_LONG: 1, ACTION_SHORT: -1}


class TradingEnv(gym.Env):
    """Среда торговли одним инструментом.

    :param df: датафрейм с колонками OHLCV и признаками (после add_features).
    :param feature_cols: какие колонки использовать как признаки состояния.
    :param window_size: сколько прошлых баров показывать агенту.
    :param initial_balance: стартовый капитал.
    :param commission: комиссия за сделку (доля, напр. 0.0005 = 0.05%).
    :param slippage: проскальзывание (доля).
    :param allow_short: разрешены ли короткие позиции.
    :param reward_scaling: множитель награды для стабильности обучения.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        window_size: int = 30,
        initial_balance: float = 100_000.0,
        commission: float = 0.0005,
        slippage: float = 0.0002,
        allow_short: bool = True,
        reward_scaling: float = 100.0,
    ) -> None:
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.allow_short = bool(allow_short)
        self.reward_scaling = float(reward_scaling)

        self.prices = self.df["close"].to_numpy(dtype=np.float64)
        self.features = self.df[feature_cols].to_numpy(dtype=np.float64)
        self.n_steps = len(self.df)

        if self.n_steps <= window_size + 1:
            raise ValueError(
                "Слишком мало данных для выбранного window_size: "
                f"{self.n_steps} баров при окне {window_size}."
            )

        # Пространство действий: 3 дискретных действия (либо 2, если шорт запрещён).
        n_actions = 3 if allow_short else 2
        self.action_space = spaces.Discrete(n_actions)

        # Наблюдение: окно признаков (window_size * n_features) + позиция.
        n_features = len(feature_cols)
        obs_dim = window_size * n_features + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Внутреннее состояние (заполняется в reset)
        self._t = 0
        self.position = 0            # -1 / 0 / +1
        self.equity = self.initial_balance
        self._equity_curve: list[float] = []
        self._trades = 0

    # ------------------------------------------------------------------ utils
    def _get_observation(self) -> np.ndarray:
        window = self.features[self._t - self.window_size : self._t]
        obs = np.concatenate([window.flatten(), [float(self.position)]])
        return obs.astype(np.float32)

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._t = self.window_size
        self.position = 0
        self.equity = self.initial_balance
        self._equity_curve = [self.equity]
        self._trades = 0
        return self._get_observation(), {}

    def step(self, action: int):
        # 1. Определяем целевую позицию по действию.
        target_position = ACTION_TO_POSITION[int(action)]
        if not self.allow_short and target_position == -1:
            target_position = 0  # подстраховка, если шорт запрещён

        # 2. Издержки при смене позиции (комиссия + проскальзывание).
        trade_cost = 0.0
        if target_position != self.position:
            # объём сделки в «долях капитала» = |смена позиции| (0->1, 1->-1=2)
            traded = abs(target_position - self.position)
            trade_cost = traded * (self.commission + self.slippage)
            self._trades += 1
        self.position = target_position

        # 3. Доходность рынка за следующий бар и переоценка портфеля.
        price_now = self.prices[self._t]
        price_next = self.prices[self._t + 1]
        market_ret = (price_next - price_now) / price_now

        # P&L шага = позиция * доходность рынка - издержки на ре-балансировку.
        step_return = self.position * market_ret - trade_cost
        self.equity *= (1.0 + step_return)
        self._equity_curve.append(self.equity)

        # 4. Награда — доходность шага (масштабированная).
        reward = step_return * self.reward_scaling

        # 5. Переход к следующему бару.
        self._t += 1
        terminated = self._t >= self.n_steps - 1
        truncated = False

        info = {
            "equity": self.equity,
            "position": self.position,
            "trades": self._trades,
            "step_return": step_return,
        }
        obs = self._get_observation()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(
            f"t={self._t} pos={self.position:+d} "
            f"equity={self.equity:,.2f} trades={self._trades}"
        )

    # -------------------------------------------------------------- helpers
    @property
    def equity_curve(self) -> np.ndarray:
        return np.asarray(self._equity_curve, dtype=np.float64)

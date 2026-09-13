"""Портфельная торговая среда с риск-ориентированной наградой и новостным сигналом.

Отличия от простой TradingEnv:
  * торгуется портфель из N инструментов (действие — веса, а не «лонг/шорт»);
  * награда — дифференциальный коэффициент Шарпа (Moody & Saffell) со штрафами
    за оборот (комиссии) и за просадку;
  * учитывается новостной сентимент: он входит в состояние агента и включает
    риск-слой, снижающий экспозицию при негативном фоне (предвосхищение спада).

Среда совместима со Stable-Baselines3 (PPO, A2C, SAC — непрерывные действия).
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """Распределение капитала между N инструментами.

    :param prices: массив цен закрытия формы (T, N).
    :param features: признаки формы (T, N * F) — по каждому активу F признаков.
    :param n_assets: число инструментов N.
    :param n_features: число признаков на инструмент F.
    :param sentiment: (опц.) массив (T,) новостного настроения в [-1, 1].
    :param window_size: длина окна наблюдения.
    :param commission, slippage: издержки на оборот (доли).
    :param allow_short: разрешать отрицательные веса.
    :param max_gross: максимальная суммарная экспозиция Σ|w| (плечо).
    :param turnover_penalty: штраф за оборот в награде.
    :param dd_penalty: штраф за превышение лимита просадки.
    :param dd_limit: допустимая просадка без штрафа (напр. 0.1 = 10%).
    :param risk_off_strength: насколько сильно негативные новости срезают экспозицию.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        n_assets: int,
        n_features: int,
        sentiment: Optional[np.ndarray] = None,
        window_size: int = 30,
        initial_balance: float = 100_000.0,
        commission: float = 0.0005,
        slippage: float = 0.0002,
        allow_short: bool = True,
        max_gross: float = 1.0,
        turnover_penalty: float = 0.001,
        dd_penalty: float = 1.0,
        dd_limit: float = 0.1,
        risk_off_strength: float = 0.5,
        reward_scaling: float = 1.0,
        dsr_eta: float = 0.01,
    ) -> None:
        super().__init__()

        self.prices = np.asarray(prices, dtype=np.float64)
        self.features = np.asarray(features, dtype=np.float64)
        self.n_assets = int(n_assets)
        self.n_features = int(n_features)
        self.T = self.prices.shape[0]

        self.sentiment = (
            np.asarray(sentiment, dtype=np.float64)
            if sentiment is not None
            else np.zeros(self.T)
        )

        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.allow_short = bool(allow_short)
        self.max_gross = float(max_gross)
        self.turnover_penalty = float(turnover_penalty)
        self.dd_penalty = float(dd_penalty)
        self.dd_limit = float(dd_limit)
        self.risk_off_strength = float(risk_off_strength)
        self.reward_scaling = float(reward_scaling)
        self.dsr_eta = float(dsr_eta)

        if self.T <= window_size + 1:
            raise ValueError("Слишком мало данных для выбранного window_size.")

        # Действие: вектор «сырых» весов по активам в [-1, 1] (нормируется внутри).
        low = -1.0 if allow_short else 0.0
        self.action_space = spaces.Box(
            low=low, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Наблюдение: окно признаков + текущие веса + сентимент.
        obs_dim = window_size * self.n_assets * self.n_features + self.n_assets + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._reset_state()

    # ------------------------------------------------------------------ utils
    def _reset_state(self):
        self._t = self.window_size
        self.weights = np.zeros(self.n_assets)  # текущие эффективные веса
        self.equity = self.initial_balance
        self._peak = self.equity
        self._equity_curve = [self.equity]
        # Параметры дифференциального Шарпа.
        self._A = 0.0
        self._B = 0.0

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Привести действие к валидным весам: Σ|w| <= max_gross."""
        w = np.asarray(action, dtype=np.float64).copy()
        if not self.allow_short:
            w = np.clip(w, 0.0, None)
        gross = np.sum(np.abs(w))
        if gross > self.max_gross and gross > 1e-9:
            w = w * (self.max_gross / gross)
        return w

    def _risk_scale(self) -> float:
        """Множитель экспозиции по новостям: при негативе < 1."""
        s = self.sentiment[self._t]
        # negative sentiment -> уменьшаем экспозицию; позитив не увеличивает.
        scale = 1.0 - self.risk_off_strength * max(0.0, -s)
        return float(np.clip(scale, 0.0, 1.0))

    def _get_observation(self) -> np.ndarray:
        window = self.features[self._t - self.window_size : self._t].flatten()
        obs = np.concatenate([window, self.weights, [self.sentiment[self._t]]])
        return obs.astype(np.float32)

    def _diff_sharpe(self, R: float) -> float:
        """Дифференциальный коэффициент Шарпа (онлайн-обновление A, B)."""
        eta = self.dsr_eta
        dA = R - self._A
        dB = R * R - self._B
        denom = (self._B - self._A ** 2) ** 1.5
        if denom > 1e-12:
            dsr = (self._B * dA - 0.5 * self._A * dB) / denom
        else:
            dsr = R  # первые шаги: пока нет дисперсии, ведём себя как по прибыли
        self._A += eta * dA
        self._B += eta * dB
        return float(dsr)

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        # 1. Целевые веса из действия + риск-слой по новостям.
        target = self._normalize_action(action) * self._risk_scale()

        # 2. Оборот и издержки при ребалансировке.
        turnover = float(np.sum(np.abs(target - self.weights)))
        trade_cost = turnover * (self.commission + self.slippage)
        self.weights = target

        # 3. Доходности активов за следующий бар.
        p_now = self.prices[self._t]
        p_next = self.prices[self._t + 1]
        asset_ret = (p_next - p_now) / np.where(p_now == 0, 1e-9, p_now)

        # 4. Доходность портфеля и переоценка капитала.
        R = float(np.dot(self.weights, asset_ret) - trade_cost)
        self.equity *= (1.0 + R)
        self._equity_curve.append(self.equity)
        self._peak = max(self._peak, self.equity)
        drawdown = (self.equity - self._peak) / self._peak  # <= 0

        # 5. Награда: дифференциальный Шарп - штрафы.
        dsr = self._diff_sharpe(R)
        dd_excess = max(0.0, -drawdown - self.dd_limit)
        reward = (
            dsr
            - self.turnover_penalty * turnover
            - self.dd_penalty * dd_excess
        ) * self.reward_scaling

        # 6. Шаг вперёд.
        self._t += 1
        terminated = self._t >= self.T - 1
        truncated = False
        info = {
            "equity": self.equity,
            "portfolio_return": R,
            "drawdown": drawdown,
            "turnover": turnover,
            "risk_scale": self._risk_scale() if not terminated else 1.0,
            "gross_exposure": float(np.sum(np.abs(self.weights))),
        }
        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self):
        print(
            f"t={self._t} equity={self.equity:,.0f} "
            f"gross={np.sum(np.abs(self.weights)):.2f} "
            f"sent={self.sentiment[self._t]:+.2f}"
        )

    @property
    def equity_curve(self) -> np.ndarray:
        return np.asarray(self._equity_curve, dtype=np.float64)

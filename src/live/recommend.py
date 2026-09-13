"""Формирование рекомендации «куда и сколько вкладывать сегодня».

Берёт обученный ансамбль, свежие данные и новостной сентимент, и выдаёт целевое
распределение капитала по инструментам — в процентах, рублях и лотах. Результат
используется дашбордом и Telegram-уведомлениями.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.agent.ensemble import make_portfolio_env
from src.state import now_iso


def get_recommendation(
    agent,
    arrays: dict,
    cfg,
    capital: float,
    lot_sizes: Optional[Dict[str, int]] = None,
) -> dict:
    """Посчитать целевое распределение портфеля на текущий момент.

    :param agent: одиночная модель или EnsembleAgent (метод predict).
    :param arrays: собранные массивы (см. build_portfolio_arrays).
    :param cfg: конфиг.
    :param capital: доступный капитал в рублях.
    :param lot_sizes: словарь {тикер -> размер лота}; если None — лоты не считаем.
    :return: словарь-рекомендация (сериализуемый в JSON).
    """
    env = make_portfolio_env(arrays, cfg)
    env.reset()
    # Встаём на последний доступный бар.
    env._t = len(arrays["prices"]) - 1

    obs = env._get_observation()
    action, _ = agent.predict(obs, deterministic=True)

    # Нормируем и применяем риск-слой по новостям — ровно как в среде.
    weights = env._normalize_action(action) * env._risk_scale()
    sentiment = float(env.sentiment[env._t])
    risk_off = sentiment < cfg.news.get("risk_off_threshold", -0.2)
    prices_now = arrays["prices"][env._t]

    positions = []
    for i, ticker in enumerate(arrays["assets"]):
        w = float(weights[i])
        rub = w * capital
        lots = None
        if lot_sizes and ticker in lot_sizes and lot_sizes[ticker] > 0:
            per_lot_rub = prices_now[i] * lot_sizes[ticker]
            if per_lot_rub > 0:
                lots = int(rub / per_lot_rub)  # знак сохраняется (шорт < 0)
        positions.append(
            {
                "ticker": ticker,
                "weight": w,
                "rub": rub,
                "price": float(prices_now[i]),
                "lots": lots,
            }
        )

    gross = float(np.sum(np.abs(weights)))
    return {
        "updated_at": now_iso(),
        "capital": float(capital),
        "sentiment": sentiment,
        "risk_off": bool(risk_off),
        "gross_exposure": gross,
        "cash_weight": float(max(0.0, 1.0 - gross)),
        "positions": positions,
    }

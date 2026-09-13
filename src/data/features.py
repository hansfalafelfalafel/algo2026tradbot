"""Формирование признаков (feature engineering) из свечей OHLCV.

Агент RL принимает решения не по «сырой» цене, а по набору признаков, которые
описывают состояние рынка в безразмерном (нормализованном) виде. Это помогает
нейросети обучаться устойчивее и переносить знания между разными ценовыми
уровнями.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Индекс относительной силы (RSI)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)


def _macd(close: pd.Series) -> pd.Series:
    """MACD-гистограмма (разница быстрой и медленной EMA минус сигнальная)."""
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def add_features(df: pd.DataFrame, use_indicators: bool = True) -> pd.DataFrame:
    """Добавить к DataFrame колонки признаков.

    Возвращает новый DataFrame; исходные OHLCV сохраняются (нужны среде для
    расчёта P&L по колонке ``close``).
    """
    out = df.copy().reset_index(drop=True)

    # Логарифмическая доходность — базовый безразмерный признак.
    out["log_ret"] = np.log(out["close"] / out["close"].shift(1))

    if use_indicators:
        # Доходности за разные горизонты.
        out["ret_5"] = out["close"].pct_change(5)
        out["ret_10"] = out["close"].pct_change(10)
        out["ret_20"] = out["close"].pct_change(20)
        # Отклонение цены от скользящих средних (в долях).
        out["sma_10"] = out["close"].rolling(10).mean() / out["close"] - 1.0
        out["sma_30"] = out["close"].rolling(30).mean() / out["close"] - 1.0
        # Волатильность (стандартное отклонение доходностей).
        out["vol_10"] = out["log_ret"].rolling(10).std()
        # Технические осцилляторы.
        out["rsi"] = _rsi(out["close"]) / 100.0            # -> [0, 1]
        out["macd"] = _macd(out["close"]) / out["close"]   # нормировка на цену
        # Нормированный объём.
        out["vol_norm"] = (
            out["volume"] / out["volume"].rolling(20).mean() - 1.0
        )

        # --- Признаки РЕЖИМА рынка (чтобы агент различал рост/падение/волатильность) ---
        # Долгосрочный тренд: отклонение от 50-барной средней.
        out["trend_50"] = out["close"].rolling(50).mean() / out["close"] - 1.0
        # Сила тренда: разница коротких и длинных средних.
        out["trend_str"] = (
            out["close"].rolling(10).mean() - out["close"].rolling(50).mean()
        ) / out["close"]
        # Просадка от локального максимума за ~63 бара (<=0): «мы в падении?».
        out["dd_63"] = out["close"] / out["close"].rolling(63).max() - 1.0
        # Режим волатильности: текущая волатильность относительно средней.
        out["vol_regime"] = (
            out["vol_10"] / out["vol_10"].rolling(60).mean() - 1.0
        )

    # Убираем строки с NaN, возникшие из-за окон индикаторов.
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out


def feature_columns(use_indicators: bool = True) -> list[str]:
    """Список колонок-признаков, которые попадут в наблюдение агента."""
    cols = ["log_ret"]
    if use_indicators:
        cols += [
            "ret_5", "ret_10", "ret_20", "sma_10", "sma_30",
            "vol_10", "rsi", "macd", "vol_norm",
            "trend_50", "trend_str", "dd_63", "vol_regime",
        ]
    return cols

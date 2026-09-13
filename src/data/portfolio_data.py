"""Сборка данных нескольких инструментов в массивы для PortfolioEnv.

Выравнивает свечи разных инструментов по общим временным меткам, считает
признаки по каждому и (опционально) добавляет новостной сентимент.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.features import add_features, feature_columns


def build_portfolio_arrays(
    candle_dfs: Dict[str, pd.DataFrame],
    use_indicators: bool = True,
    sentiment_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Собрать массивы prices, features, sentiment для среды.

    :param candle_dfs: словарь {тикер/figi -> DataFrame свечей OHLCV}.
    :param use_indicators: считать ли технические индикаторы.
    :param sentiment_df: (опц.) DataFrame с колонками time и sentiment,
        выровненный или подлежащий выравниванию по времени.
    :return: dict с ключами prices, features, sentiment, times, assets,
        n_assets, n_features, feature_cols.
    """
    assets = list(candle_dfs.keys())
    cols = feature_columns(use_indicators)

    # 1. Признаки по каждому активу, индекс по времени.
    per_asset = {}
    for name, df in candle_dfs.items():
        feat = add_features(df, use_indicators=use_indicators)
        feat = feat[["time", "close"] + cols].copy()
        feat = feat.rename(
            columns={c: f"{name}__{c}" for c in ["close"] + cols}
        )
        per_asset[name] = feat.set_index("time")

    # 2. Пересечение по времени (inner join), чтобы все активы были синхронны.
    merged = None
    for name in assets:
        merged = per_asset[name] if merged is None else merged.join(
            per_asset[name], how="inner"
        )
    merged = merged.sort_index()

    if len(merged) == 0:
        raise RuntimeError(
            "После выравнивания по времени не осталось общих баров. "
            "Проверьте, что у инструментов один таймфрейм и пересекающийся период."
        )

    times = merged.index.to_numpy()

    # 3. Матрица цен (T, N).
    prices = np.column_stack([merged[f"{name}__close"].to_numpy() for name in assets])

    # 4. Признаки (T, N*F): по каждому бару подряд признаки всех активов.
    feat_blocks = []
    for name in assets:
        block = merged[[f"{name}__{c}" for c in cols]].to_numpy()
        feat_blocks.append(block)
    features = np.concatenate(feat_blocks, axis=1)  # (T, N*F)

    # 5. Сентимент, выровненный по времени.
    # Котировки API приходят с таймзоной (UTC), а даты новостей — без неё.
    # Приводим оба индекса к «наивному» UTC-времени, иначе pandas не сопоставит.
    idx = pd.DatetimeIndex(merged.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    if sentiment_df is not None and len(sentiment_df) > 0:
        # Убираем возможные дублирующиеся колонки (напр. две 'sentiment').
        s = sentiment_df.loc[:, ~sentiment_df.columns.duplicated(keep="last")].copy()
        s_time = pd.to_datetime(s["time"], utc=True).dt.tz_localize(None)
        sent_vals = np.asarray(s["sentiment"], dtype=float).reshape(-1)
        s = pd.Series(sent_vals, index=s_time).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        sent = s.reindex(idx, method="ffill").to_numpy()
        sent = np.nan_to_num(sent, nan=0.0)
    else:
        sent = np.zeros(len(idx))

    return {
        "prices": prices,
        "features": features,
        "sentiment": sent,
        "times": times,
        "assets": assets,
        "n_assets": len(assets),
        "n_features": len(cols),
        "feature_cols": cols,
    }

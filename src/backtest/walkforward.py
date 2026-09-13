"""Walk-forward валидация и оценка переобучения (PBO).

Два инструмента честной проверки стратегии:

1. Walk-forward: модель многократно переобучается на скользящем окне прошлого и
   проверяется на следующем (ещё не виденном) отрезке. Итоговые метрики считаются
   только по склеенным out-of-sample участкам — так почти невозможно «подогнать»
   результат под историю.

2. PBO (Probability of Backtest Overfitting) методом CSCV (Bailey, López de Prado):
   оценивает вероятность того, что выбранная «лучшая» конфигурация окажется ниже
   медианы на независимых данных. Высокий PBO (→0.5 и выше) — тревожный сигнал
   переобучения; низкий (→0) — стратегия устойчива.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List

import numpy as np


def _sharpe_cols(x: np.ndarray) -> np.ndarray:
    """Коэффициент Шарпа по каждому столбцу (конфигурации)."""
    m = x.mean(axis=0)
    s = x.std(axis=0)
    return np.where(s > 1e-12, m / s, 0.0)


def pbo_cscv(perf: np.ndarray, n_splits: int = 8,
             metric: Callable[[np.ndarray], np.ndarray] = _sharpe_cols) -> Dict:
    """Оценка вероятности переобучения бэктеста (PBO) методом CSCV.

    :param perf: матрица (T наблюдений × N конфигураций) пошаговой доходности.
    :param n_splits: на сколько частей делить по времени (чётное).
    :return: {'pbo': float, 'logits': np.ndarray, 'n_configs': int}.
    """
    perf = np.asarray(perf, dtype=np.float64)
    T, N = perf.shape
    if N < 2:
        raise ValueError("Нужно минимум 2 конфигурации для PBO.")
    if n_splits % 2 != 0:
        n_splits += 1
    Tt = (T // n_splits) * n_splits
    perf = perf[:Tt]
    parts = np.array_split(perf, n_splits, axis=0)
    idx = list(range(n_splits))

    logits: List[float] = []
    for is_sel in combinations(idx, n_splits // 2):
        oos_sel = [i for i in idx if i not in is_sel]
        IS = np.vstack([parts[i] for i in is_sel])
        OOS = np.vstack([parts[i] for i in oos_sel])
        r_is = metric(IS)
        r_oos = metric(OOS)
        n_star = int(np.argmax(r_is))            # лучшая конфигурация на IS
        order = np.argsort(r_oos)                 # ранжирование по OOS (возр.)
        rank = int(np.where(order == n_star)[0][0])
        omega = (rank + 1) / (N + 1)              # относительный ранг в (0,1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1 - omega))))

    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0.0))           # доля, где IS-лучший ниже медианы OOS
    return {"pbo": pbo, "logits": logits, "n_configs": N}


def run_walkforward(arrays: dict, cfg, n_folds: int = 4, train_frac: float = 0.6) -> Dict:
    """Walk-forward: переобучение ансамбля по скользящим окнам + сбор OOS.

    Дополнительно собирает матрицу пошаговой доходности кандидатов
    (PPO, SAC, A2C, ансамбль, buy&hold) для расчёта PBO.
    """
    # Ленивые импорты (тяжёлые зависимости RL).
    from src.agent.ensemble import (ENSEMBLE_ALGOS, EnsembleAgent,
                                     make_portfolio_env, train_ensemble)

    T = len(arrays["prices"])
    fold_len = T // (n_folds + 1)      # +1: первый блок только на обучение
    train_len = int(fold_len * (1 + train_frac) * n_folds / n_folds)  # окно обучения
    train_len = max(fold_len, int(T * train_frac))

    def slice_arrays(a, lo, hi):
        o = dict(a)
        for k in ("prices", "features", "sentiment"):
            o[k] = a[k][lo:hi]
        return o

    oos_returns_ensemble: List[float] = []
    # Для PBO: доходности кандидатов на общем OOS-таймлайне.
    cand_names = list(ENSEMBLE_ALGOS) + ["ENSEMBLE", "BuyHold"]
    cand_returns: Dict[str, List[float]] = {c: [] for c in cand_names}

    step = (T - train_len) // n_folds
    if step <= cfg.features["window_size"] + 2:
        raise ValueError("Мало данных для walk-forward: увеличьте историю или уменьшите n_folds.")

    for f in range(n_folds):
        tr_lo = f * step
        tr_hi = tr_lo + train_len
        te_lo = tr_hi
        te_hi = min(te_lo + step, T)
        if te_hi - te_lo <= cfg.features["window_size"] + 2:
            break

        arr_tr = slice_arrays(arrays, tr_lo, tr_hi)
        arr_te = slice_arrays(arrays, te_lo, te_hi)
        val_lo = int(tr_lo + train_len * 0.8)
        arr_val = slice_arrays(arrays, val_lo, tr_hi)

        # Нормализация: статистики по train этого фолда -> на train/val/test.
        from src.data.normalize import apply_feature_stats, fit_feature_stats
        _stats = fit_feature_stats(arr_tr["features"])
        for _a in (arr_tr, arr_val, arr_te):
            _a["features"] = apply_feature_stats(_a["features"], _stats)

        paths = train_ensemble(arr_tr, cfg)
        models = {n: ENSEMBLE_ALGOS[n].load(p) for n, p in paths.items()}
        # веса ансамбля — по последней части train как валидации
        agent = EnsembleAgent.load_and_weight(paths, arr_val, cfg)

        # Прогон кандидатов по тест-фолду, сбор пошаговых доходностей.
        def run_returns(a, arr):
            env = make_portfolio_env(arr, cfg)
            obs, _ = env.reset()
            rr, done = [], False
            while not done:
                act, _ = a.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(act)
                rr.append(info["portfolio_return"])
                done = term or trunc
            return rr

        ens_r = run_returns(agent, arr_te)
        oos_returns_ensemble.extend(ens_r)
        cand_returns["ENSEMBLE"].extend(ens_r)
        for n in ENSEMBLE_ALGOS:
            cand_returns[n].extend(run_returns(models[n], arr_te))
        # buy&hold равновзвешенный
        prices = arr_te["prices"][env_ws(cfg):]
        bh = (prices[1:] / prices[:-1] - 1).mean(axis=1)
        cand_returns["BuyHold"].extend(list(bh)[: len(ens_r)])

        print(f"[walkforward] фолд {f+1}/{n_folds}: OOS баров {len(ens_r)}")

    # Выравниваем длины кандидатов для матрицы PBO.
    min_len = min(len(v) for v in cand_returns.values() if v)
    matrix = np.column_stack([np.array(cand_returns[c][:min_len]) for c in cand_names])
    pbo = pbo_cscv(matrix, n_splits=min(8, (min_len // 5) * 2 if min_len >= 10 else 2))

    equity = float(cfg.env["initial_balance"]) * np.cumprod(1 + np.array(oos_returns_ensemble))
    return {
        "oos_equity": equity,
        "oos_returns": np.array(oos_returns_ensemble),
        "pbo": pbo["pbo"],
        "candidates": cand_names,
        "n_oos_bars": len(oos_returns_ensemble),
    }


def env_ws(cfg) -> int:
    return int(cfg.features["window_size"])

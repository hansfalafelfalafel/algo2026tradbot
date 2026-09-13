"""Найти коинтегрированные пары среди инструментов из config.yaml.

Запуск:
    python scripts/20_find_pairs.py

Скачивает историю по инструментам (data.figis), считает тест коинтеграции для
всех пар и печатает лучшие (низкий p-value = устойчивая связь = кандидат на
парный трейдинг). Результат сохраняется в data_cache/pairs.csv.

Совет: для хороших пар добавьте в config.yaml инструменты ОДНОГО сектора
(нефть: LKOH, ROSN, TATN, SNGS; банки/финансы и т.д.) — они чаще коинтегрированы.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.config import load_config
from src.data.loader import load_many
from src.pairs.find_pairs import find_cointegrated_pairs

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))


def main() -> None:
    cfg = load_config()
    candle_dfs = load_many(cfg.data["figis"], cfg.data["interval"],
                           cfg.data["history_days"], cfg.abs_path(cfg.data["cache_dir"]))
    print(f"Инструментов: {len(candle_dfs)} — ищу коинтегрированные пары...")

    pairs = find_cointegrated_pairs(candle_dfs)
    out = cfg.abs_path(cfg.data["cache_dir"], "pairs.csv")
    pairs.to_csv(out, index=False)

    print("\nЛучшие пары (сортировка по p-value, меньше = лучше):")
    print(pairs.head(10).to_string(index=False))
    good = pairs[pairs["pvalue"] < 0.05]
    print(f"\nКоинтегрированных пар (p<0.05): {len(good)} из {len(pairs)}")
    if len(good) == 0:
        print("Устойчивых пар не найдено. Добавьте в config.yaml инструменты "
              "одного сектора (напр. нефтяные: LKOH, ROSN, TATN, SNGS).")
    else:
        print("Дальше: python scripts/21_backtest_pairs.py — бэктест лучшей пары.")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()

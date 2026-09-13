"""Прогнать обученного агента на отложенной (test) выборке и оценить метрики.

Запуск:
    python scripts/03_backtest.py

Строит график reports/equity_<model_name>.png и печатает таблицу метрик.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.agent.train import load_model
from src.backtest.backtest import format_report, plot_results, run_backtest
from src.config import load_config
from src.data.loader import load_or_download

load_dotenv()


def main() -> None:
    cfg = load_config()
    df = load_or_download(
        figi=cfg.data["figi"],
        interval=cfg.data["interval"],
        history_days=cfg.data["history_days"],
        cache_dir=cfg.abs_path(cfg.data["cache_dir"]),
    )
    split = int(len(df) * cfg.data["train_ratio"])
    df_test = df.iloc[split:].reset_index(drop=True)
    print(f"Тестовая выборка: {len(df_test)} баров")

    model = load_model(cfg)
    result = run_backtest(model, df_test, cfg)

    print("\n" + format_report(result, cfg))

    plot_path = cfg.abs_path("reports", f"equity_{cfg.agent['model_name']}.png")
    plot_results(result, cfg, plot_path)
    print(f"\nГрафик сохранён: {plot_path}")


if __name__ == "__main__":
    main()

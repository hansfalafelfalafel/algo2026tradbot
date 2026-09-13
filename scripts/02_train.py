"""Обучить RL-агента на train-части исторических данных.

Запуск:
    python scripts/02_train.py

Данные берутся из кэша (сначала запустите 01_download_data.py). Обученная
модель сохраняется в models/<model_name>.zip.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.agent.train import train
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
    # Делим на train/test по времени (без перемешивания!).
    split = int(len(df) * cfg.data["train_ratio"])
    df_train = df.iloc[:split].reset_index(drop=True)
    print(f"Обучающая выборка: {len(df_train)} баров из {len(df)}")

    model_path = train(df_train, cfg)
    print(f"\nМодель сохранена: {model_path}")


if __name__ == "__main__":
    main()

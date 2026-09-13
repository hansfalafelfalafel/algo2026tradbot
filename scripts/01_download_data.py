"""Скачать исторические свечи и сохранить в кэш (data_cache/).

Запуск:
    python scripts/01_download_data.py
    python scripts/01_download_data.py --force   # перекачать заново
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.config import load_config
from src.data.loader import load_or_download

load_dotenv()


def main() -> None:
    force = "--force" in sys.argv
    cfg = load_config()
    df = load_or_download(
        figi=cfg.data["figi"],
        interval=cfg.data["interval"],
        history_days=cfg.data["history_days"],
        cache_dir=cfg.abs_path(cfg.data["cache_dir"]),
        force=force,
    )
    print(f"Загружено свечей: {len(df)}")
    print(f"Период: {df['time'].min()} — {df['time'].max()}")
    print(df.tail())


if __name__ == "__main__":
    main()

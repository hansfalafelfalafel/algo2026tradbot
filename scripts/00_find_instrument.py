"""Найти FIGI инструмента по тикеру.

Запуск:
    python scripts/00_find_instrument.py SBER
    python scripts/00_find_instrument.py GAZP

FIGI затем прописывается в config.yaml (секция data.figi).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.data.loader import find_figi_by_ticker

load_dotenv()


def main() -> None:
    if len(sys.argv) < 2:
        print("Использование: python scripts/00_find_instrument.py <ТИКЕР> [CLASS_CODE]")
        sys.exit(1)
    ticker = sys.argv[1].upper()
    class_code = sys.argv[2] if len(sys.argv) > 2 else "TQBR"
    figi = find_figi_by_ticker(ticker, class_code)
    print(f"{ticker} ({class_code}) -> FIGI: {figi}")


if __name__ == "__main__":
    main()

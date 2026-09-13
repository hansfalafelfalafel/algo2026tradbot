"""Запустить обученного агента в песочнице Т-Банка (бумажная торговля).

Запуск:
    python scripts/04_run_sandbox.py            # 5 итераций (демо)
    python scripts/04_run_sandbox.py --loop     # бесконечный цикл (Ctrl+C для выхода)

Требуется переменная окружения TINKOFF_TOKEN (файл .env). Торговля идёт на
виртуальные деньги — реальный счёт не затрагивается.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.agent.train import load_model
from src.config import load_config
from src.live.sandbox_trader import SandboxTrader

load_dotenv()


def main() -> None:
    cfg = load_config()
    token = os.environ.get("TINKOFF_TOKEN")
    if not token:
        print("Ошибка: не задан TINKOFF_TOKEN. Создайте файл .env (см. .env.example).")
        sys.exit(1)

    model = load_model(cfg)
    trader = SandboxTrader(model, cfg, token)

    loop = "--loop" in sys.argv
    n_iter = None if loop else 5
    trader.run(n_iterations=n_iter)


if __name__ == "__main__":
    main()

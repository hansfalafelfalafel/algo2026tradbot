"""Live-торговля в песочнице: быстрый цикл, режимы manual/semi/auto, лимиты,
управление из дашборда и Telegram.

Запуск:
    python scripts/09_run_live.py            # цикл до Ctrl+C
    python scripts/09_run_live.py --once      # один проход (для проверки)

Режим и лимиты берутся из config.yaml (секция execution) и могут
переопределяться на лету из дашборда (вкладка «Управление»).
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.agent.ensemble import ENSEMBLE_ALGOS, EnsembleAgent
from src.config import load_config
from src.data.loader import get_lot_sizes, load_many
from src.data.portfolio_data import build_portfolio_arrays
from src.live.live_trader import LiveTrader
from src.live.recommend import get_recommendation
from src.live.sandbox_broker import SandboxBroker

load_dotenv()


def _news_df(cfg):
    path = cfg.abs_path(cfg.data["cache_dir"], "news_signal.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    col = "sentiment_smooth" if "sentiment_smooth" in df.columns else "sentiment"
    return df[["date", col]].rename(columns={"date": "time", col: "sentiment"})


def _target_weights(agent, cfg, lot_sizes):
    """Скачать свежие свечи, посчитать целевые веса портфеля и сентимент."""
    candle_dfs = load_many(cfg.data["figis"], cfg.data["interval"],
                           history_days=max(30, cfg.features["window_size"] // 4 + 10),
                           cache_dir=cfg.abs_path(cfg.data["cache_dir"]), force=True)
    arrays = build_portfolio_arrays(candle_dfs, cfg.features["use_indicators"], _news_df(cfg))
    from src.data.normalize import load_stats, normalize_apply
    _stats = load_stats(cfg.abs_path(cfg.agent["model_dir"], f"{cfg.agent['model_name']}_norm.npz"))
    if _stats is not None:
        normalize_apply(arrays, _stats)
    rec = get_recommendation(agent, arrays, cfg, capital=1.0, lot_sizes=None)
    weights = {p["ticker"]: p["weight"] for p in rec["positions"]}
    return weights, rec["sentiment"], rec["risk_off"]


def main() -> None:
    cfg = load_config()
    token = os.environ.get("TINKOFF_TOKEN")
    if not token:
        print("Нужен TINKOFF_TOKEN в .env.")
        sys.exit(1)

    # Загружаем ансамбль (равные веса; точные — из 07_backtest_portfolio).
    model_dir = cfg.abs_path(cfg.agent["model_dir"])
    paths = {n: model_dir / f"{cfg.agent['model_name']}_{n}.zip"
             for n in ENSEMBLE_ALGOS
             if (model_dir / f"{cfg.agent['model_name']}_{n}.zip").exists()}
    if not paths:
        print("Модели ансамбля не найдены. Запустите scripts/06_train_ensemble.py.")
        sys.exit(1)
    models = {n: ENSEMBLE_ALGOS[n].load(p) for n, p in paths.items()}
    agent = EnsembleAgent(models, {n: 1.0 / len(models) for n in models})

    from tinkoff.invest import MoneyValue
    from tinkoff.invest.sandbox.client import SandboxClient

    once = "--once" in sys.argv
    poll = int(cfg.execution.get("poll_interval_sec", 10))
    lot_sizes = get_lot_sizes(cfg.data["figis"])

    with SandboxClient(token) as client:
        acc = client.sandbox.open_sandbox_account()
        pay = float(cfg.sandbox.get("pay_in", 1_000_000))
        client.sandbox.sandbox_pay_in(
            account_id=acc.account_id,
            amount=MoneyValue(units=int(pay), nano=0, currency="rub"))
        print(f"[live] Счёт песочницы {acc.account_id}, режим {cfg.execution.get('mode')}")

        broker = SandboxBroker(client, acc.account_id, cfg.data["figis"], lot_sizes)
        trader = LiveTrader(agent, cfg, broker)
        trader.controller.start_day(broker.portfolio_value(), time.strftime("%Y-%m-%d"))

        # Опциональный стриминг котировок (быстрая реакция).
        feed = None
        if "--stream" in sys.argv:
            from src.live.streaming import StreamingPriceFeed
            feed = StreamingPriceFeed(client, cfg.data["figis"])
            feed.start()
            print("[live] Стриминг котировок включён.")

        try:
            while True:
                weights, sent, risk_off = _target_weights(agent, cfg, lot_sizes)
                # Цены: из стрима (мгновенно) либо опросом.
                prices = feed.get_prices() if (feed and feed.ready()) else broker.prices()
                res = trader.step(weights, prices, lot_sizes, sentiment=sent, risk_off=risk_off)
                print(f"[live] {time.strftime('%H:%M:%S')} {res}")
                if once:
                    break
                time.sleep(poll)
        except KeyboardInterrupt:
            print("\n[live] Остановлено пользователем.")
        finally:
            if feed:
                feed.stop()


if __name__ == "__main__":
    main()

"""АВТОПОРТФЕЛЬ (слой B): еженедельная рекомендация «что держать».

Логика — единственная, обыгравшая рынок в наших тестах 2021–2026:
моментум top-N + веса 1/волатильность + режимный фильтр (медвежий рынок ->
кэш) + volatility targeting 15% годовых.

Запуск (вручную или еженедельным таймером):
    python scripts/50_portfolio_recommend.py
    python scripts/50_portfolio_recommend.py --capital 300000

Что делает: качает свежие дневные свечи (кэш обновляется), считает целевой
портфель, печатает таблицу, шлёт в Telegram и обновляет дашборд (вкладка
«Обзор» оживает).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.config import load_config
from src.data.loader import get_lot_sizes, load_many
from src.factor.live_portfolio import current_target_portfolio
from src.factor.momentum import align_prices
from src.notify.telegram import TelegramNotifier
from src.state import append_equity, write_latest

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))


def _arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main() -> None:
    cfg = load_config()
    if cfg.data["interval"] != "day":
        print(f"Внимание: interval={cfg.data['interval']}, автопортфель работает "
              f"на дневных данных (config.yaml -> data.interval: day)")

    capital = _arg("--capital", float(cfg.sandbox.get("pay_in", 1_000_000)))

    # Свежие дневные свечи (force=True — дотягиваем последние дни).
    candle_dfs = load_many(cfg.data["figis"], cfg.data["interval"],
                           cfg.data["history_days"],
                           cfg.abs_path(cfg.data["cache_dir"]), force=True)
    prices = align_prices(candle_dfs)
    res = current_target_portfolio(prices)

    lot_sizes = {}
    if os.environ.get("TINKOFF_TOKEN"):
        try:
            lot_sizes = get_lot_sizes(cfg.data["figis"])
        except Exception as e:  # noqa: BLE001
            print(f"[лоты] не получены: {e}")

    # ---------- формируем отчёт ----------
    lines = ["<b>📊 АВТОПОРТФЕЛЬ — рекомендация недели</b>",
             f"<i>по данным на {res['asof']}</i>", ""]
    if not res["bull_regime"]:
        lines.append("🔴 Режим рынка: МЕДВЕЖИЙ (индекс ниже 200-дневной средней).")
        lines.append("Рекомендация: <b>весь портфель в кэше</b>. Это защитный "
                     "режим — именно он спасал капитал в обвалы в бэктесте.")
    else:
        lines.append(f"🟢 Режим рынка: бычий (+{res['regime_strength']*100:.1f}% "
                     f"к 200-дн. средней) | вол-таргетинг: {res['vol_scale']*100:.0f}% "
                     f"экспозиции")
        lines.append("")
        lines.append("<b>Что держать (доли капитала):</b>")
        for tk, w in sorted(res["weights"].items(), key=lambda x: -x[1]):
            rub = w * capital
            last = float(prices[tk].iloc[-1])
            lots = ""
            if tk in lot_sizes and lot_sizes[tk] > 0 and last > 0:
                lots = f" (~{int(rub / (last * lot_sizes[tk]))} лот.)"
            mom = res["momentum"].get(tk, 0.0)
            lines.append(f"📈 <b>{tk}</b>: {w*100:.1f}%  ≈ {rub:,.0f} ₽{lots} "
                         f"| моментум {mom*100:+.0f}%")
        lines.append(f"💰 Кэш: {res['cash_weight']*100:.1f}%")
    lines.append("")
    lines.append("Ребаланс — раз в неделю. Не инвестиционная рекомендация.")
    msg = "\n".join(lines)

    print(msg.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    TelegramNotifier().send(msg)

    # ---------- состояние для дашборда ----------
    positions = [{"ticker": tk, "weight": w, "rub": w * capital,
                  "lots": (int(w * capital / (float(prices[tk].iloc[-1]) * lot_sizes[tk]))
                           if tk in lot_sizes and lot_sizes[tk] > 0 else None)}
                 for tk, w in res["weights"].items()]
    write_latest({
        "capital": capital,
        "mode": "portfolio-weekly",
        "note": ("медвежий режим -> кэш" if not res["bull_regime"]
                 else f"бычий режим, vol-scale {res['vol_scale']:.2f}"),
        "sentiment": 0.0,
        "risk_off": not res["bull_regime"],
        "gross_exposure": float(sum(res["weights"].values())),
        "cash_weight": res["cash_weight"],
        "daily_drawdown": 0.0,
        "trades_today": 0,
        "positions": positions,
    })
    append_equity(capital)
    print("\nСостояние дашборда обновлено (вкладка «Обзор»).")


if __name__ == "__main__":
    main()

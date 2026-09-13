"""Тесты движка исполнения: лимиты, режимы, ограничение экспозиции."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live.execution import ExecutionController, RiskLimits


def _controller(mode="auto", **lim):
    base = dict(
        max_invest_rub=300_000, max_order_rub=50_000, max_position_weight=0.35,
        max_gross_exposure=0.8, max_trades_per_day=20, min_seconds_between_orders=0,
        max_daily_loss_pct=0.05, confirm_above_rub=100_000, allow_short=False,
        whitelist=["SBER", "GAZP", "LKOH"],
    )
    base.update(lim)
    return ExecutionController(RiskLimits(**base), mode=mode)


def test_max_invest_caps_gross():
    c = _controller()
    # capital 1e6, max_invest 300k -> суммарная экспозиция <= 0.30
    w = c.clamp_target({"SBER": 0.3, "GAZP": 0.3, "LKOH": 0.2}, capital=1_000_000)
    assert sum(abs(v) for v in w.values()) <= 0.30 + 1e-6


def test_per_position_cap():
    c = _controller()
    w = c.clamp_target({"SBER": 0.9}, capital=1_000_000)
    assert abs(w["SBER"]) <= 0.35 + 1e-6


def test_no_short_when_disabled():
    c = _controller(allow_short=False)
    w = c.clamp_target({"SBER": -0.3}, capital=1_000_000)
    assert w["SBER"] >= 0.0


def test_whitelist_filters():
    c = _controller()
    w = c.clamp_target({"TSLA": 0.3, "SBER": 0.2}, capital=1_000_000)
    assert "TSLA" not in w and "SBER" in w


def test_order_size_capped():
    c = _controller(mode="auto")
    orders = c.plan_orders(
        capital=1_000_000, current_weights={},
        target_weights={"SBER": 0.3}, prices={"SBER": 300.0},
        rebalance_threshold=0.01,
    )
    assert orders and abs(orders[0]["rub"]) <= 50_000 + 1e-6


def test_manual_mode_pending():
    c = _controller(mode="manual")
    orders = c.plan_orders(
        capital=1_000_000, current_weights={},
        target_weights={"SBER": 0.2}, prices={"SBER": 300.0}, rebalance_threshold=0.01,
    )
    assert all(o["decision"] == "pending" for o in orders)


def test_kill_switch():
    c = _controller()
    c.start_day(1_000_000, "2026-01-01")
    assert not c.kill_switch_triggered(990_000)     # -1% ок
    assert c.kill_switch_triggered(940_000)          # -6% -> стоп


def test_trade_throttling():
    c = _controller(max_trades_per_day=1, min_seconds_between_orders=100)
    c.start_day(1_000_000, "2026-01-01")
    assert c.can_trade_now(now_ts=1000)
    c.register_trade(now_ts=1000)
    assert not c.can_trade_now(now_ts=1050)  # лимит сделок исчерпан


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("Тесты движка исполнения пройдены ✓")

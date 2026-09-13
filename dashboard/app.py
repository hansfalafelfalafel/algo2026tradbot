"""Portable Streamlit dashboard for the GitHub repository.

This version is designed for a clean clone of the repository:
- it does not require runtime state/ or model logs;
- it reads only committed reproducible artifacts under results/;
- it restores the Layer A (order book) demo from data/sample/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SAMPLE = ROOT / "data" / "sample"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            x = json.load(f)
        return x if isinstance(x, dict) else {}
    except Exception:
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_orderbook_features(book: pd.DataFrame) -> pd.DataFrame:
    if book.empty:
        return pd.DataFrame()

    df = book.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).sort_values("time")

    df["mid"] = (df["bid_p0"] + df["ask_p0"]) / 2
    df["spread_bp"] = (df["ask_p0"] - df["bid_p0"]) / df["mid"] * 10000

    den1 = df["bid_q0"] + df["ask_q0"]
    df["imb1"] = np.where(
        den1 != 0,
        (df["bid_q0"] - df["ask_q0"]) / den1,
        np.nan,
    )

    bid5 = sum(df[f"bid_q{i}"] for i in range(5))
    ask5 = sum(df[f"ask_q{i}"] for i in range(5))
    den5 = bid5 + ask5
    df["imb5"] = np.where(den5 != 0, (bid5 - ask5) / den5, np.nan)

    prev = df.shift(1)
    ofi = (
        (df["bid_p0"] >= prev["bid_p0"]).astype(float) * df["bid_q0"]
        - (df["bid_p0"] <= prev["bid_p0"]).astype(float) * prev["bid_q0"]
        - (df["ask_p0"] <= prev["ask_p0"]).astype(float) * df["ask_q0"]
        + (df["ask_p0"] >= prev["ask_p0"]).astype(float) * prev["ask_q0"]
    )
    df["ofi"] = ofi.fillna(0.0)

    return df


def build_tfi(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    x = trades.copy()
    x["time"] = pd.to_datetime(x["time"], errors="coerce", utc=True)
    x = x.dropna(subset=["time"]).sort_values("time")
    x["signed_qty"] = np.select(
        [x["direction"].eq(1), x["direction"].eq(2)],
        [x["quantity"], -x["quantity"]],
        default=0.0,
    )
    x["minute"] = x["time"].dt.floor("min")
    return (
        x.groupby("minute", as_index=False)["signed_qty"]
        .sum()
        .rename(columns={"signed_qty": "tfi"})
    )


ofi = read_json(RESULTS / "json" / "ofi_results.json")
forward = read_json(RESULTS / "json" / "forward_eval_multiday_summary.json")
sandbox = read_csv(RESULTS / "sandbox" / "mr30_sandbox_journal.csv")
ablation = read_csv(RESULTS / "tables" / "mr30_exact_ablation_leaderboard.csv")

orderbook = build_orderbook_features(read_csv(SAMPLE / "orderbook_sample.csv"))
trades = read_csv(SAMPLE / "lob_trades_sample.csv")
tfi = build_tfi(trades)

closed = (
    sandbox[sandbox["event"].eq("CLOSE")].copy()
    if not sandbox.empty and "event" in sandbox.columns
    else pd.DataFrame()
)

st.set_page_config(
    page_title="Algo2026 • Research Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Интеллектуальная торговая система")
st.caption(
    "Демонстрационная версия итогового проекта: воспроизводимые результаты "
    "исследования + пример микроструктуры рынка из data/sample/."
)

overview, research_tab, validation_tab, layer_tab, architecture_tab = st.tabs(
    [
        "📊 Обзор",
        "🔬 Исследование",
        "🧪 Forward & Sandbox",
        "🧬 Слой A (стакан)",
        "🧩 Архитектура",
    ]
)

with overview:
    st.subheader("Главный вопрос проекта")
    st.write(
        "Можно ли получить торговый сигнал на данных MOEX, который сохраняется "
        "на новых временных периодах и после учёта торговых издержек?"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Наблюдений микроструктуры", f"{ofi.get('bars', 0):,}")
    c2.metric("Дней OFI", ofi.get("days", 0))
    c3.metric(
        "ROC-AUC OFI",
        f"{ofi.get('models', {}).get('базовые-7', {}).get('auc', 0):.3f}",
    )
    c4.metric("Forward MR30", f"{forward.get('net_mean_bp', 0):+.2f} bp")

    st.markdown("### Путь исследования")
    st.markdown(
        "**RL → пары и факторы → микроструктура → MR/Momentum → "
        "freeze → forward → T-Bank Sandbox**"
    )

    st.info(
        "Главный результат — не обещание прибыльности, а рабочая end-to-end система, "
        "которая позволяет находить и отбраковывать стратегии, хорошо выглядящие "
        "только на истории."
    )

with research_tab:
    st.subheader("Микроструктура рынка")

    r1, r2, r3 = st.columns(3)
    r1.metric("Наблюдений", f"{ofi.get('bars', 0):,}")
    r2.metric(
        "Лучший ROC-AUC",
        f"{ofi.get('models', {}).get('базовые-7', {}).get('auc', 0):.3f}",
    )
    r3.metric("Оценка издержек", f"{ofi.get('cost_bp', 0):.1f} bp")

    daily = pd.DataFrame(ofi.get("daily", []))
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        fig = go.Figure(
            go.Scatter(
                x=daily["date"],
                y=daily["auc"],
                mode="lines+markers",
                name="ROC-AUC",
            )
        )
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="OFI: качество модели по дням",
            yaxis_title="ROC-AUC",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Mean Reversion")
    st.latex(r"score_{MR}=-0.75\cdot z(ret_{5m})-0.25\cdot z(ret_{1m})")
    st.caption(
        "Сильное недавнее падение повышает MR-score и формирует потенциальный "
        "long-сигнал; сильный рост — противоположный сигнал. Затем применяются "
        "фильтры силы сигнала, spread и состояния рынка."
    )

    if not ablation.empty:
        st.markdown("#### Сравнение вариантов MR30")
        st.dataframe(ablation.head(15), use_container_width=True, hide_index=True)

with validation_tab:
    st.subheader("Forward после фиксации параметров")

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Сделок", forward.get("trades", 0))
    f2.metric("Средний net", f"{forward.get('net_mean_bp', 0):+.2f} bp")
    f3.metric(
        "Положительных дней",
        f"{forward.get('positive_days', 0)}/{forward.get('days', 0)}",
    )
    f4.metric("Hit rate", f"{forward.get('hit_rate', 0) * 100:.1f}%")

    st.warning(
        "После freeze результат ухудшился на следующем временном участке. "
        "Это показывает, почему forward-проверка важнее одного красивого backtest."
    )

    st.markdown("### T-Bank Sandbox")
    if not closed.empty:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Закрытых сделок", len(closed))
        s2.metric("Gross P&L", f"{closed['gross_pnl_rub'].sum():+.2f} ₽")
        s3.metric("Комиссии", f"{closed['commission_rub'].sum():.2f} ₽")
        s4.metric("Net P&L", f"{closed['net_pnl_rub'].sum():+.2f} ₽")

        curve = closed[["time", "net_pnl_rub"]].copy()
        curve["time"] = pd.to_datetime(curve["time"], errors="coerce")
        curve["cum_net"] = curve["net_pnl_rub"].cumsum()

        fig = go.Figure(
            go.Scatter(
                x=curve["time"],
                y=curve["cum_net"],
                mode="lines",
                name="Net P&L",
            )
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Накопленный результат Sandbox",
            yaxis_title="₽",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        cols = [
            c
            for c in [
                "time",
                "ticker",
                "side",
                "entry_price",
                "exit_price",
                "commission_rub",
                "net_pnl_rub",
                "net_pnl_bp",
            ]
            if c in closed.columns
        ]
        st.dataframe(closed[cols].tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("Журнал Sandbox отсутствует в results/sandbox/.")

with layer_tab:
    st.subheader("🧬 Слой A — пример стакана и потока сделок")
    st.caption(
        "В GitHub хранится небольшой sample, поэтому эту вкладку можно открыть "
        "даже в чистом клоне без большого локального кэша."
    )

    if orderbook.empty:
        st.warning("data/sample/orderbook_sample.csv не найден.")
    else:
        b = orderbook.copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Последний mid", f"{b['mid'].iloc[-1]:.4f}")
        c2.metric("Spread", f"{b['spread_bp'].iloc[-1]:.2f} bp")
        c3.metric("Imbalance L1", f"{b['imb1'].iloc[-1]:+.3f}")
        c4.metric("OFI", f"{b['ofi'].iloc[-1]:+.0f}")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=b["time"],
                y=b["bid_p0"],
                mode="lines",
                name="best bid",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=b["time"],
                y=b["ask_p0"],
                mode="lines",
                name="best ask",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=b["time"],
                y=b["mid"],
                mode="lines",
                name="mid",
            )
        )
        fig.update_layout(
            title="Лучшие bid / ask и mid",
            yaxis_title="Цена",
            height=330,
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)

        fig_imb = go.Figure()
        fig_imb.add_trace(
            go.Scatter(x=b["time"], y=b["imb1"], mode="lines", name="imb1")
        )
        fig_imb.add_trace(
            go.Scatter(x=b["time"], y=b["imb5"], mode="lines", name="imb5")
        )
        fig_imb.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_imb.update_layout(title="Дисбаланс стакана", height=300)
        left.plotly_chart(fig_imb, use_container_width=True)

        fig_flow = go.Figure(
            go.Scatter(x=b["time"], y=b["ofi"], mode="lines", name="OFI")
        )
        fig_flow.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_flow.update_layout(title="Order Flow Imbalance", height=300)
        right.plotly_chart(fig_flow, use_container_width=True)

        if not tfi.empty:
            fig_tfi = go.Figure(
                go.Bar(x=tfi["minute"], y=tfi["tfi"], name="TFI")
            )
            fig_tfi.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_tfi.update_layout(
                title="Trade Flow Imbalance по минутам",
                yaxis_title="Подписанный объём",
                height=300,
            )
            st.plotly_chart(fig_tfi, use_container_width=True)

        st.markdown("#### Снимок лучших уровней стакана")
        last = b.iloc[-1]
        depth = pd.DataFrame(
            {
                "Уровень": list(range(1, 6)),
                "Bid price": [last[f"bid_p{i}"] for i in range(5)],
                "Bid qty": [last[f"bid_q{i}"] for i in range(5)],
                "Ask price": [last[f"ask_p{i}"] for i in range(5)],
                "Ask qty": [last[f"ask_q{i}"] for i in range(5)],
            }
        )
        st.dataframe(depth, use_container_width=True, hide_index=True)

with architecture_tab:
    st.subheader("End-to-end архитектура")
    st.markdown(
        """
        **Данные** → **Признаки** → **Стратегии / ML / RL** →
        **Backtest / walk-forward** → **Freeze / forward** →
        **Sandbox / риск-контроль** → **Мониторинг и результаты**
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Данные**  \n`src/data/`, `src/lob/`, `data/sample/`")
    c2.markdown("**Исследование**  \n`src/agent/`, `src/pairs/`, `src/factor/`")
    c3.markdown("**Проверка**  \n`src/backtest/`, `src/live/`, `results/`")

    st.success(
        "Эта версия дашборда специально не зависит от runtime state: "
        "после `git clone` она сразу показывает сохранённые результаты проекта."
    )

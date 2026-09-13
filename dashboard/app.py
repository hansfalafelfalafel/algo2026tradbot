"""Веб-дашборд (Streamlit): наблюдение, DS-аналитика, прогноз и управление.

Запуск:
    streamlit run dashboard/app.py

Читает состояние из state/ (пишут скрипты 08_recommend.py и 09_run_live.py) и
логи обучения из models/logs/. Обновляется автоматически.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Ансамбль (stable_baselines3) может быть не установлен на лёгком сервере —
# дашборду от него нужны только имена алгоритмов для логов.
try:
    from src.agent.ensemble import ENSEMBLE_ALGOS  # noqa: F401
    ENSEMBLE_ALGOS = list(ENSEMBLE_ALGOS)
except Exception:  # noqa: BLE001
    ENSEMBLE_ALGOS = ["PPO", "SAC", "A2C"]
from src.analytics.diagnostics import (drawdown_series, read_training_logs,
                                       returns_from_equity, rolling_sharpe,
                                       summarize_returns)
from src.analytics.forecast import monte_carlo_forecast
from src.config import load_config
from src import state

st.set_page_config(page_title="RL-Трейдинг • Дашборд", page_icon="📈", layout="wide")
cfg = load_config()

refresh_sec = st.sidebar.slider("Автообновление, сек", 5, 120, 15)
st_autorefresh(interval=refresh_sec * 1000, key="auto")
st.sidebar.markdown("### RL-Трейдинг")
st.sidebar.caption("Ансамбль PPO+SAC+A2C · портфель MOEX · новостной сигнал")

latest = state.read_latest()
eq_df = state.read_equity()

st.title("📈 Интеллектуальная торговая система")

tab_overview, tab_ds, tab_forecast, tab_control, tab_lob = st.tabs(
    ["📊 Обзор", "🔬 Аналитика (DS)", "🔮 Прогноз капитала", "🎛 Управление",
     "🧬 Слой A (стакан)"]
)

# ============================================================ ОБЗОР
with tab_overview:
    if not latest:
        st.info("Нет данных. Запустите `scripts/08_recommend.py` или `scripts/09_run_live.py`.")
    else:
        st.caption(f"Обновлено: {latest.get('updated_at','—')}  ·  режим: "
                   f"**{latest.get('mode','—')}**  ·  {latest.get('note','')}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Капитал, ₽", f"{latest.get('capital',0):,.0f}")
        c2.metric("Экспозиция", f"{latest.get('gross_exposure',0)*100:.0f}%")
        dd = latest.get("daily_drawdown", 0.0)
        c3.metric("Дневной P&L", f"{dd*100:+.2f}%", delta_color="inverse")
        sent = latest.get("sentiment", 0.0)
        c4.metric("Новости", f"{sent:+.2f}",
                  delta="risk-off" if latest.get("risk_off") else "норма",
                  delta_color="inverse" if latest.get("risk_off") else "normal")
        c5.metric("Сделок сегодня", latest.get("trades_today", 0))

        if latest.get("risk_off"):
            st.warning("⚠️ RISK-OFF: негативный новостной фон — экспозиция снижена.")

        st.subheader("💡 Куда и сколько вкладывать сейчас")
        positions = [p for p in latest.get("positions", []) if abs(p.get("weight", 0)) > 1e-4]
        if positions:
            rows = [{
                "Инструмент": p["ticker"],
                "Направление": "LONG 📈" if p["weight"] > 0 else "SHORT 📉",
                "Доля, %": round(p["weight"] * 100, 1),
                "Сумма, ₽": round(p["rub"]),
                "Лоты": p.get("lots"),
            } for p in positions]
            left, right = st.columns([3, 2])
            left.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            fig = go.Figure(go.Bar(
                x=[p["ticker"] for p in positions], y=[p["weight"] * 100 for p in positions],
                marker_color=["#2E9E5B" if p["weight"] > 0 else "#D64545" for p in positions],
                text=[f"{p['weight']*100:.1f}%" for p in positions], textposition="outside"))
            fig.update_layout(title="Целевые веса", yaxis_title="%", height=320, margin=dict(t=40, b=20))
            right.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Сейчас рекомендуется быть вне рынка (кэш).")

        st.subheader("📊 Динамика капитала")
        if len(eq_df) > 1:
            fig2 = go.Figure(go.Scatter(x=eq_df["time"], y=eq_df["equity"], mode="lines",
                                        line=dict(color="#2E6BE9", width=2)))
            fig2.update_layout(height=300, margin=dict(t=20, b=20), yaxis_title="₽")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("История капитала появится по мере работы системы.")

# ============================================================ АНАЛИТИКА (DS)
with tab_ds:
    st.subheader("🔬 Диагностика — взгляд дата-сайентиста")

    logs = read_training_logs(cfg.abs_path(cfg.agent["model_dir"]),
                              cfg.agent["model_name"], list(ENSEMBLE_ALGOS))
    if logs:
        st.markdown("**Ошибки и метрики обучения агентов**")
        metric_map = {
            "train/loss": "Loss",
            "train/value_loss": "Ошибка функции ценности (value loss)",
            "train/explained_variance": "Explained variance (качество оценки)",
            "rollout/ep_rew_mean": "Средняя награда за эпизод",
        }
        available = set()
        for df in logs.values():
            available |= set(df.columns)
        for key, title in metric_map.items():
            if key not in available:
                continue
            fig = go.Figure()
            for algo, df in logs.items():
                if key in df.columns and "time/total_timesteps" in df.columns:
                    fig.add_trace(go.Scatter(x=df["time/total_timesteps"], y=df[key],
                                             mode="lines", name=algo))
            fig.update_layout(title=title, height=260, margin=dict(t=40, b=20),
                              xaxis_title="шаги обучения")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Логи обучения не найдены. Обучите ансамбль: `scripts/06_train_ensemble.py`.")

    if len(eq_df) > 5:
        rets = returns_from_equity(eq_df["equity"].to_numpy())
        st.markdown("**Поведение стратегии**")
        cc1, cc2 = st.columns(2)
        # Скользящий Шарп
        rs = rolling_sharpe(rets, window=min(50, max(5, len(rets) // 3)))
        f1 = go.Figure(go.Scatter(y=rs, mode="lines", line=dict(color="#2E9E5B")))
        f1.add_hline(y=0, line_dash="dot", line_color="gray")
        f1.update_layout(title="Скользящий коэффициент Шарпа", height=260, margin=dict(t=40, b=20))
        cc1.plotly_chart(f1, use_container_width=True)
        # Просадка
        dd = drawdown_series(eq_df["equity"].to_numpy()) * 100
        f2 = go.Figure(go.Scatter(y=dd, mode="lines", fill="tozeroy", line=dict(color="#D64545")))
        f2.update_layout(title="Просадка, %", height=260, margin=dict(t=40, b=20))
        cc2.plotly_chart(f2, use_container_width=True)
        # Распределение доходностей + статистика
        st.markdown("**Распределение доходностей и статистика**")
        d1, d2 = st.columns([3, 2])
        fh = go.Figure(go.Histogram(x=rets * 100, nbinsx=40, marker_color="#8A5CF6"))
        fh.update_layout(title="Гистограмма доходностей шага, %", height=280, margin=dict(t=40, b=20))
        d1.plotly_chart(fh, use_container_width=True)
        stats = summarize_returns(rets)
        if stats:
            d2.dataframe(pd.DataFrame({
                "Метрика": ["Шарп", "Сортино", "Win-rate", "Асимметрия", "Эксцесс",
                            "Лучший шаг", "Худший шаг", "VaR 95%"],
                "Значение": [f"{stats['sharpe']:.2f}", f"{stats['sortino']:.2f}",
                             f"{stats['win_rate']*100:.0f}%", f"{stats['skew']:.2f}",
                             f"{stats['kurtosis']:.2f}", f"{stats['best']*100:+.2f}%",
                             f"{stats['worst']*100:+.2f}%", f"{stats['var_95']*100:+.2f}%"],
            }), use_container_width=True, hide_index=True)
    else:
        st.caption("Для аналитики стратегии нужна история капитала (запустите торговлю/бэктест).")

# ============================================================ ПРОГНОЗ
with tab_forecast:
    st.subheader("🔮 Прогноз роста капитала (Monte-Carlo)")
    st.caption("Честный прогноз: множество сценариев на основе исторической доходности "
               "стратегии, с доверительными коридорами. Не гарантия, а распределение исходов.")
    if len(eq_df) > 10:
        rets = returns_from_equity(eq_df["equity"].to_numpy())
        cap = float(eq_df["equity"].iloc[-1])
        h = st.slider("Горизонт, шагов", 20, 500, int(cfg.forecast.get("horizon_days", 90)))
        nsim = st.select_slider("Сценариев", [500, 1000, 2000, 5000],
                                value=int(cfg.forecast.get("n_simulations", 2000)))
        try:
            fc = monte_carlo_forecast(rets, cap, horizon=h, n_sims=nsim, percentiles=(5, 50, 95))
            x = list(range(h))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=fc["percentiles"][95], mode="lines",
                                     line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=fc["percentiles"][5], mode="lines", fill="tonexty",
                                     fillcolor="rgba(46,107,233,0.15)", line=dict(width=0),
                                     name="коридор P5–P95"))
            fig.add_trace(go.Scatter(x=x, y=fc["percentiles"][50], mode="lines",
                                     line=dict(color="#2E6BE9", width=2), name="медиана (P50)"))
            fig.add_hline(y=cap, line_dash="dot", line_color="gray")
            fig.update_layout(height=380, margin=dict(t=20, b=20),
                              xaxis_title="шаги вперёд", yaxis_title="капитал, ₽")
            st.plotly_chart(fig, use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Медианный исход, ₽", f"{fc['final_median']:,.0f}",
                      f"{fc['expected_total_return']*100:+.1f}%")
            m2.metric("Пессимистичный (P5), ₽", f"{fc['final_p5']:,.0f}")
            m3.metric("Оптимистичный (P95), ₽", f"{fc['final_p95']:,.0f}")
            m4.metric("Вероятность убытка", f"{fc['prob_loss']*100:.0f}%")
        except Exception as e:  # noqa: BLE001
            st.error(f"Не удалось построить прогноз: {e}")
    else:
        st.info("Нужно больше истории капитала для прогноза (хотя бы ~10 точек).")

# ============================================================ УПРАВЛЕНИЕ
with tab_control:
    st.subheader("🎛 Управление торговлей")
    ctrl = state.read_control()

    cc1, cc2, cc3 = st.columns(3)
    mode = cc1.selectbox("Режим исполнения",
                         ["manual", "semi", "auto"],
                         index=["manual", "semi", "auto"].index(
                             ctrl.get("mode")
                             or cfg.execution.get("mode", "manual")
                         ),
                         help="manual — только подтверждённые вручную; semi — авто в лимитах, "
                              "крупные на подтверждение; auto — полностью авто в лимитах.")
    paused = cc2.toggle("Пауза", value=bool(ctrl.get("paused")))
    if cc3.button("⛔️ Закрыть все позиции", use_container_width=True):
        state.write_control({"flatten_requested": True})
        st.success("Команда на закрытие отправлена.")

    st.markdown("**Тонкая настройка лимитов** (переопределяет config.yaml на лету)")
    lim = dict(cfg.execution.get("limits", {}))
    ov = ctrl.get("limits_override") or {}
    l1, l2, l3 = st.columns(3)
    max_invest = l1.number_input("Макс. автоввод, ₽", 0, 100_000_000,
                                 int(ov.get("max_invest_rub", lim.get("max_invest_rub", 300000))), step=10000)
    max_order = l2.number_input("Макс. на заявку, ₽", 0, 10_000_000,
                                int(ov.get("max_order_rub", lim.get("max_order_rub", 50000))), step=5000)
    max_pos = l3.number_input("Макс. доля инструмента", 0.0, 1.0,
                              float(ov.get("max_position_weight", lim.get("max_position_weight", 0.35))), step=0.05)
    l4, l5, l6 = st.columns(3)
    max_gross = l4.number_input("Макс. экспозиция", 0.0, 3.0,
                                float(ov.get("max_gross_exposure", lim.get("max_gross_exposure", 0.8))), step=0.1)
    max_loss = l5.number_input("Дневной стоп-лосс", 0.0, 0.5,
                               float(ov.get("max_daily_loss_pct", lim.get("max_daily_loss_pct", 0.05))), step=0.01)
    max_trades = l6.number_input("Сделок в день", 0, 1000,
                                 int(ov.get("max_trades_per_day", lim.get("max_trades_per_day", 20))))

    if st.button("💾 Применить настройки", type="primary"):
        state.write_control({
            "mode": mode, "paused": paused,
            "limits_override": {
                "max_invest_rub": max_invest, "max_order_rub": max_order,
                "max_position_weight": max_pos, "max_gross_exposure": max_gross,
                "max_daily_loss_pct": max_loss, "max_trades_per_day": max_trades,
            },
        })
        st.success("Настройки применены — торговый цикл подхватит их на следующем шаге.")

    # Очередь заявок на подтверждение (режимы manual/semi).
    st.markdown("**Заявки на подтверждение**")
    pending = state.read_pending()
    waiting = [o for o in pending if o.get("decision") in (None, "pending")]
    if not waiting:
        st.caption("Очередь пуста.")
    else:
        for o in waiting:
            pc1, pc2, pc3, pc4 = st.columns([3, 2, 1, 1])
            pc1.write(f"**{o['side']} {o['ticker']}** — {abs(o['rub']):,.0f} ₽"
                      + (f" (~{o['lots']} лот.)" if o.get("lots") is not None else ""))
            pc2.write(f"вес {o['weight_from']*100:.0f}% → {o['weight_to']*100:.0f}%")
            if pc3.button("✅", key=f"ok_{o['id']}"):
                for x in pending:
                    if x["id"] == o["id"]:
                        x["decision"] = "approved"
                state.write_pending(pending)
                st.rerun()
            if pc4.button("❌", key=f"no_{o['id']}"):
                for x in pending:
                    if x["id"] == o["id"]:
                        x["decision"] = "rejected"
                state.write_pending(pending)
                st.rerun()

# ============================================================ СЛОЙ A (СТАКАН)
with tab_lob:
    st.subheader("🧬 Слой A: сбор стакана и OFI-модель")

    lob_dir = cfg.abs_path(cfg.data["cache_dir"], "lob")
    files = (sorted(list(lob_dir.glob("*_book.csv")) + list(lob_dir.glob("*_book.csv.gz")))
             if lob_dir.exists() else [])
    if not files:
        st.info("Коллектор ещё не записал данных (файлы появляются в торговые "
                "часы MOEX). Проверьте сервис: systemctl status lob-collector")
    else:
        rows = {}
        for f in files:
            day = f.name.split("_")[0]
            r = rows.setdefault(day, {"Дата": day, "Файлов": 0, "МБ": 0.0})
            r["Файлов"] += 1
            r["МБ"] += f.stat().st_size / 1e6
        for f in (list(lob_dir.glob("*_trades.csv")) + list(lob_dir.glob("*_trades.csv.gz"))):
            day = f.name.split("_")[0]
            if day in rows:
                rows[day]["Файлов"] += 1
                rows[day]["МБ"] += f.stat().st_size / 1e6
        tbl = pd.DataFrame(sorted(rows.values(), key=lambda r: r["Дата"]))
        tbl["МБ"] = tbl["МБ"].round(1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Торговых дней собрано", len(tbl))
        c2.metric("Объём данных, МБ", f"{tbl['МБ'].sum():.0f}")
        c3.metric("Последнее обновление",
                  pd.Timestamp(max(f.stat().st_mtime for f in files),
                               unit="s").strftime("%d.%m %H:%M"))
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown("**Результаты OFI-модели** (обновляются запуском "
                "`scripts/31_ofi_model.py`)")
    import json as _json
    ofi_path = state.STATE_DIR / "ofi_results.json"
    if not ofi_path.exists():
        st.caption("Пока нет — запустите 31_ofi_model.py после 1+ торгового дня.")
    else:
        with open(ofi_path, encoding="utf-8") as f:
            ofi = _json.load(f)
        st.caption(f"Обновлено: {ofi.get('updated_at','—')} · дней: {ofi.get('days')}"
                   f" · баров: {ofi.get('bars'):,} · издержки ~{ofi.get('cost_bp')} б.п.")
        mtbl = pd.DataFrame([
            {"Модель": n, "OOS AUC": m["auc"], "Edge вверх, б.п.": m["edge_up"],
             "Edge вниз, б.п.": m["edge_dn"]}
            for n, m in ofi.get("models", {}).items()])
        st.dataframe(mtbl, use_container_width=True, hide_index=True)
        daily = ofi.get("daily", [])
        if daily:
            d = pd.DataFrame(daily)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d["date"], y=d["auc"], mode="lines+markers",
                                     name="AUC", line=dict(color="#2E6BE9")))
            fig.add_hline(y=0.55, line_dash="dash", line_color="#2E9E5B",
                          annotation_text="порог сигнала 0.55")
            fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
            fig.update_layout(title="Стабильность AUC по дням (главный график "
                                    "ближайших недель)", height=300,
                              margin=dict(t=40, b=20))
            fig.update_xaxes(type="category")
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=d["date"], y=d["edge_dn"], name="edge вниз",
                                  marker_color="#D64545"))
            fig2.add_trace(go.Bar(x=d["date"], y=d["edge_up"], name="edge вверх",
                                  marker_color="#2E9E5B"))
            fig2.add_hline(y=0, line_color="gray")
            fig2.update_layout(title="Чистый edge по дням, б.п. (после издержек)",
                               height=280, barmode="group", margin=dict(t=40, b=20))
            fig2.update_xaxes(type="category")
            fig2.update_xaxes(type="category")
            st.plotly_chart(fig2, use_container_width=True)
